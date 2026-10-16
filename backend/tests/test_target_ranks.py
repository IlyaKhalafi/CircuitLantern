import json
from unittest.mock import patch

import pytest
import torch

from circuit_lantern.services import token_logits
from circuit_lantern.services.token_logits import summarize_token_logits


def expected_ranks(logits, targets):
    # Independent competition ranking with equal logits assigned their first sorted place.
    return [
        [sorted(row, reverse=True).index(row[target]) + 1 for target in targets]
        for row in logits.cpu().tolist()
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chunked_ranks_match_full_vocabulary_order_with_ties_and_eight_targets(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.randint(-4, 5, (37, 17), generator=torch.Generator().manual_seed(71)).to(device)
    targets = (0, 16, 4, 9, 3, 1, 15, 8)
    with patch.object(token_logits, "MAX_LOGIT_ELEMENTS", 34):
        measured = summarize_token_logits(logits, targets)
    assert measured["token_target_ranks"] == expected_ranks(logits, targets)
    baseline = summarize_token_logits(logits)
    assert all(measured[key] == value for key, value in baseline.items())
    assert all(
        1 <= rank <= logits.shape[1] for row in measured["token_target_ranks"] for rank in row
    )
    json.dumps(measured, allow_nan=False)


def test_ranks_distinguish_logits_when_normalization_cancels_small_differences():
    largest = torch.finfo(torch.float32).max
    logits = torch.tensor([[largest, 0, 1, -largest, 0], [5, 5, 1, -2, 1]])
    result = summarize_token_logits(logits, (0, 1, 2, 3, 4))
    assert result["token_target_ranks"] == [[1, 3, 2, 5, 3], [1, 1, 3, 5, 3]]
    assert (
        result["token_target_log_probabilities"][0][1]
        == result["token_target_log_probabilities"][0][2]
    )
    assert result["token_target_ranks"] == expected_ranks(logits, (0, 1, 2, 3, 4))


@pytest.mark.parametrize("intervention", ["baseline", "first block", "last block", "head"])
def test_captured_ranks_match_each_real_logit_lens_projection(runtime, intervention):
    prompt = "the capital of France is"
    captured_logits = []
    original = runtime.model.lm_head.forward

    def record(values):
        output = original(values)
        captured_logits.append(output.detach().float()[0].clone())
        return output

    with (
        patch.object(runtime.model.lm_head, "forward", side_effect=record),
        patch.object(runtime.model, "forward", wraps=runtime.model.forward) as forward,
    ):
        result = runtime.trace(
            prompt,
            0,
            False,
            0
            if intervention == "first block"
            else len(runtime._blocks()) - 1
            if intervention == "last block"
            else None,
            {"layer": 0, "head": 0} if intervention == "head" else None,
            tracked_tokens=["France", "Poland", "<pad>"],
        )
    forward.assert_called_once()
    # The first output is the actual model forward, followed by each diagnostic projection.
    assert len(captured_logits) == len(result["layers"]) + 1
    for layer, logits in zip(result["layers"], captured_logits[1:], strict=True):
        assert layer["token_target_ranks"] == expected_ranks(logits, (6, 9, 0))
    assert result["layers"][-1]["token_target_ranks"] == expected_ranks(
        captured_logits[0], (6, 9, 0)
    )
    assert "strictly greater" in result["target_rank_method"]
    assert all(not block._forward_hooks for block in runtime._blocks())
    json.dumps(result, allow_nan=False)


def test_untracked_captures_keep_their_original_shape_and_prefix_causality(runtime):
    baseline = runtime.trace("the capital", 0, False, None)
    assert "target_rank_method" not in baseline
    assert all("token_target_ranks" not in layer for layer in baseline["layers"])
    a = runtime.trace(
        "the capital of France is", 0, False, None, tracked_tokens=["France", "Poland"]
    )
    b = runtime.trace(
        "the capital of Poland is", 0, False, None, tracked_tokens=["France", "Poland"]
    )
    for first, second in zip(a["layers"], b["layers"], strict=True):
        assert first["token_target_ranks"][:3] == second["token_target_ranks"][:3]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("column", [0, 1])
def test_invalid_target_or_competing_logits_reject_the_entire_capture(value, column):
    logits = torch.tensor([[2.0, 1.0], [3.0, 4.0]])
    logits[1, column] = value
    with (
        patch.object(token_logits, "MAX_LOGIT_ELEMENTS", 2),
        pytest.raises(ValueError, match="non finite logits"),
    ):
        summarize_token_logits(logits, (0,))
