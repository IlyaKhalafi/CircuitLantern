from unittest.mock import patch

import pytest

from circuit_lantern.schemas import TokenizeRequest
from circuit_lantern.services.runtime import RuntimeBusyError
from circuit_lantern.services.tokens import MAX_PREVIEW_TOKENS, capture_token_limit


def test_preview_matches_trace_tokenization_and_reserves_generation(runtime):
    preview = runtime.tokenize(["the capital of France is"], 4)
    trace = runtime.trace("the capital of France is", 4, False, None)
    assert preview["model_id"] == trace["model_id"]
    assert preview["revision"] == trace["revision"]
    assert preview["prompts"][0]["token_ids"] == trace["token_ids"]
    assert preview["prompts"][0]["tokens"] == trace["tokens"]
    assert preview["token_limit"] == 60
    assert preview["prompts"][0]["fits"]


def test_batched_preview_compares_token_positions_and_marks_special_tokens(runtime):
    preview = runtime.tokenize(["the capital of France", "the capital"], 0)
    clean, corrupted = preview["prompts"]
    assert clean["token_count"] == 4
    assert corrupted["token_count"] == 2
    assert clean["token_ids"][:2] == corrupted["token_ids"]
    special = runtime.tokenize(["<eos> the"], 0)["prompts"][0]
    assert special["special_tokens"] == [True, False]


def test_preview_count_is_complete_when_display_is_bounded_and_budget_is_exceeded(runtime):
    preview = runtime.tokenize(["the " * 700], 0)
    result = preview["prompts"][0]
    assert result["token_count"] == 700
    assert len(result["token_ids"]) == MAX_PREVIEW_TOKENS
    assert result["truncated"]
    assert not result["fits"]


def test_empty_preview_and_exhausted_context_are_reported_without_running_a_model(runtime):
    runtime.metadata["execution_context_length"] = 16
    with patch.object(
        runtime.model, "forward", side_effect=AssertionError("Preview ran the model")
    ):
        preview = runtime.tokenize(["", "the capital"], 16)
    assert preview["token_limit"] == 0
    assert all(not prompt["fits"] for prompt in preview["prompts"])
    assert preview["prompts"][0]["token_count"] == 0
    assert capture_token_limit(runtime.metadata, 128, 4) == 12


def test_preview_respects_model_operation_lock(runtime):
    with runtime.lock, pytest.raises(RuntimeBusyError, match="busy"):
        runtime.tokenize(["the capital"], 0)


@pytest.mark.parametrize(
    "payload",
    [
        {"prompts": []},
        {"prompts": ["a", "b", "c"]},
        {"prompts": ["x" * 8001]},
        {"prompts": ["a"], "reserved_tokens": -1},
        {"prompts": ["a"], "reserved_tokens": 65},
    ],
)
def test_preview_request_bounds(payload):
    with pytest.raises(ValueError):
        TokenizeRequest.model_validate(payload)


def test_tokenizer_added_bos_is_counted_and_matches_the_actual_capture(runtime):
    from tokenizers.processors import TemplateProcessing

    runtime.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="<pad> $A", special_tokens=[("<pad>", 0)]
    )
    preview = runtime.tokenize(["the capital"], 1)["prompts"][0]
    trace = runtime.trace("the capital", 1, False, None)
    assert preview["token_count"] == 3
    assert preview["token_ids"] == trace["token_ids"] == [0, 3, 4]
    assert preview["special_tokens"] == [True, False, False]


def test_added_special_tokens_outside_the_public_map_are_marked_in_all_previews(runtime):
    from tokenizers import AddedToken

    from circuit_lantern.services.targets import preview_target_tokens
    from circuit_lantern.services.tokens import special_token_ids

    tokenizer = runtime.tokenizer
    identifier = tokenizer.convert_tokens_to_ids("France")
    tokenizer.add_tokens([AddedToken("France", special=True)], special_tokens=True)
    assert identifier not in tokenizer.all_special_ids
    assert identifier in special_token_ids(tokenizer)
    preview = runtime.tokenize(["France the"], 0)["prompts"][0]
    assert preview["token_ids"] == [identifier, 3]
    assert preview["special_tokens"] == [True, False]
    target = preview_target_tokens(tokenizer, ["France"], 16)[0]
    assert target["valid"] and target["special_tokens"] == [True]
