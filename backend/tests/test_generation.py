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
