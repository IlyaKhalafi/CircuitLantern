from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from circuit_lantern.services.generation import decode_generation


@pytest.mark.parametrize(
    ("model", "decoder", "prompt_ids", "sequence_ids", "prompt", "text", "suffix"),
    [
        (
            models.WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"),
            None,
            [1],
            [1, 2],
            "hello",
            "hello world",
            " world",
        ),
        (
            models.Unigram([("<unk>", 0.0), ("\u2581hello", -1.0), ("\u2581world", -1.0)], 0),
            decoders.Metaspace(),
            [1],
            [1, 2],
            "hello",
            "hello world",
            " world",
        ),
        (
            models.WordPiece({"[UNK]": 0, "hello": 1, "##s": 2}),
            decoders.WordPiece(),
            [1],
            [1, 2],
            "hello",
            "hellos",
            "s",
        ),
        (
            models.WordLevel({"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2}, unk_token="<unk>"),
            decoders.ByteFallback(),
            [1],
            [1, 2],
            "\ufffd",
            "\u00e9",
            None,
        ),
    ],
)
def test_decode_in_context_preserves_boundaries_and_handles_rewritten_prefixes(
    model, decoder, prompt_ids, sequence_ids, prompt, text, suffix
):
    tokenizer = Tokenizer(model)
    if decoder is not None:
        tokenizer.decoder = decoder
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    result = decode_generation(tokenizer, prompt_ids, sequence_ids)
    assert result["decoded_prompt"] == prompt
    assert result["generated_text"] == text
    assert result["continuation_text"] == suffix
    assert result["generated_token_ids"] == sequence_ids[len(prompt_ids) :]


def test_eos_ids_survive_while_special_tokens_are_hidden_in_decoded_text(runtime):
    tokenizer = runtime.tokenizer
    result = decode_generation(tokenizer, [3], [3, 4, tokenizer.eos_token_id])
    assert result["generated_text"] == "the capital"
    assert result["continuation_text"] == " capital"
    assert result["generated_token_ids"] == [4, tokenizer.eos_token_id]
    assert result["generated_tokens"][-1] == "<eos>"
    assert result["generated_pieces"][-1] == "<eos>"


def test_trace_bounds_generation_and_overrides_checkpoint_output_defaults(runtime):
    config = runtime.model.generation_config
    config.return_dict_in_generate = True
    config.output_scores = True
    config.output_logits = True
    config.token_healing = True
    config.num_beams = 2
    config.num_return_sequences = 2
    before = config.to_dict()
    with patch.object(runtime.model, "generate", wraps=runtime.model.generate) as generate:
        result = runtime.trace("the capital", 4, False, None)
    assert runtime.model.generation_config.to_dict() == before
    assert result["capture_attention"] is False
    assert result["generation_settings"]["max_new_tokens"] == 4
    assert result["generation_settings"]["num_beams"] == 1
    assert result["generation_settings"]["return_dict_in_generate"] is False
    assert generate.call_args.kwargs["output_scores"] is False
    ids = result["token_ids"] + result["generated_token_ids"]
    assert result["generated_text"] == runtime.tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert 1 <= len(result["generated_token_ids"]) <= 4


def test_trace_records_early_eos_and_requested_generation_allowance(runtime):
    ids = runtime.tokenizer("the capital").input_ids + [runtime.tokenizer.eos_token_id]
    with patch.object(runtime.model, "generate", return_value=torch.tensor([ids])):
        result = runtime.trace("the capital", 8, True, None)
    assert result["generation_settings"]["max_new_tokens"] == 8
    assert result["generated_token_ids"] == [runtime.tokenizer.eos_token_id]
    assert result["continuation_text"] == ""
    assert result["capture_attention"] is True


@pytest.mark.parametrize(
    ("layer", "head"),
    [(None, None), (1, None), (None, {"layer": 0, "head": 0})],
)
def test_prompt_capture_skips_generation_and_retains_predictions_and_interventions(
    runtime, layer, head
):
    with (
        patch.object(
            runtime.model, "generate", side_effect=AssertionError("Generation ran")
        ) as generate,
        patch.object(runtime.model, "forward", wraps=runtime.model.forward) as forward,
    ):
        result = runtime.trace("the capital", 0, True, layer, head)
    generate.assert_not_called()
    assert forward.call_count == 1
    assert result["generation_settings"] is None
    assert result["generation_performed"] is False
    assert result["requested_max_new_tokens"] == 0
    assert (
        result["generated_token_ids"]
        == result["generated_tokens"]
        == result["generated_pieces"]
        == []
    )
    assert result["generated_text"] == result["decoded_prompt"] == "the capital"
    assert result["continuation_text"] == result["continuation"] == ""
    assert result["predictions"] and result["layers"][-1]["predictions"]
    assert len(result["attention"]) == 2
    assert len(result["geometry"]["labels"]) == 3
    if layer is not None:
        assert result["layers"][layer + 1]["update_norm"] == 0
    if head is not None:
        assert result["ablate_head"]["scope"] == "all_prompt_tokens"
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_capture_only_uses_the_full_prompt_allowance_and_matches_token_preview(runtime):
    runtime.metadata["execution_context_length"] = 16
    prompt = "the " * 16
    preview = runtime.tokenize([prompt], 0)
    result = runtime.trace(prompt, 0, False, None)
    assert preview["token_limit"] == 16
    assert preview["prompts"][0]["fits"]
    assert preview["prompts"][0]["token_ids"] == result["token_ids"]
    assert len(result["tokens"]) == 16
    with pytest.raises(ValueError, match="at most 15 tokens"):
        runtime.trace(prompt, 1, False, None)
    with pytest.raises(ValueError, match="at most 16 tokens"):
        runtime.trace(prompt + "the", 0, False, None)


def test_failed_prompt_capture_removes_intervention_and_capture_hooks(runtime):
    with (
        patch.object(runtime.model, "forward", side_effect=RuntimeError("capture failed")),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        runtime.trace("the capital", 0, True, None, {"layer": 0, "head": 0})
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_capture_only_request_is_valid_and_generation_bounds_remain_enforced():
    from circuit_lantern.schemas import TraceRequest

    assert TraceRequest(prompt="the capital", max_new_tokens=0).max_new_tokens == 0
    for count in (-1, 65):
        with pytest.raises(ValueError):
            TraceRequest(prompt="the capital", max_new_tokens=count)
