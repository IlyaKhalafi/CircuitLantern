import json
import math
from unittest.mock import patch

import pytest
import torch

from circuit_lantern.schemas import TraceRequest
from circuit_lantern.services import token_logits
from circuit_lantern.services.causal import replace_hidden
from circuit_lantern.services.interventions import install_head_ablation
from circuit_lantern.services.store import ExperimentStore
from circuit_lantern.services.token_logits import summarize_token_logits


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_ids", [(0,), (3, 0, 2)])
def test_tracked_scores_preserve_order_and_match_full_vocabulary_distributions(device, target_ids):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.tensor(
        [[math.log(2), 0, -1, 3], [1000, 1000, -1000, 0], [-1000, 0, 1000, 1]],
        dtype=torch.float64,
        device=device,
    )
    with patch.object(token_logits, "MAX_LOGIT_ELEMENTS", 4):
        result = summarize_token_logits(logits, target_ids)
    measured = torch.tensor(result["token_target_log_probabilities"], dtype=torch.float64)
    expected = logits.log_softmax(-1)[:, target_ids].cpu()
    torch.testing.assert_close(measured, expected, rtol=1e-14, atol=1e-14)
    if len(target_ids) > 1:
        torch.testing.assert_close(
            measured[:, 0] - measured[:, 1],
            (logits[:, target_ids[0]] - logits[:, target_ids[1]]).cpu(),
            rtol=1e-13,
            atol=1e-13,
        )
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_extreme_finite_targets_keep_log_probabilities_when_probability_underflows():
    largest = torch.finfo(torch.float32).max
    result = summarize_token_logits(torch.tensor([[largest, -largest]]), (0, 1))
    scores = result["token_target_log_probabilities"][0]
    assert scores[0] == 0
    assert math.isfinite(scores[1]) and scores[1] < -1e38
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("ids", [(-1,), (4,), (1, 1), (True,), (0.5,), tuple(range(9))])
def test_invalid_target_ids_are_rejected_before_distribution_work(ids):
    with (
        patch.object(token_logits, "log_probabilities") as score,
        pytest.raises(ValueError, match="distinct vocabulary IDs"),
    ):
        summarize_token_logits(torch.zeros((1, 4)), ids)
    score.assert_not_called()


def test_optional_tracking_preserves_the_original_summary_shape(runtime):
    result = runtime.trace("the capital", 0, False, None, tracked_tokens=[])
    assert "tracked_targets" not in result
    assert "target_score_method" not in result
    assert all("token_target_log_probabilities" not in layer for layer in result["layers"])


def test_every_tracked_position_matches_actual_intermediate_projections(runtime):
    prompt = "the capital of France is"
    expected = []
    inputs = runtime.tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = runtime.model(**inputs, use_cache=False, output_hidden_states=True)
        for index, hidden in enumerate(outputs.hidden_states):
            projected = (
                runtime.model.transformer.ln_f(hidden)
                if index < len(outputs.hidden_states) - 1
                else hidden
            )
            logits = runtime.model.lm_head(projected).float()[0].double()
            expected.append(logits.log_softmax(-1)[:, (6, 9, 0)])
    result = runtime.trace(prompt, 0, False, None, tracked_tokens=["France", "Poland", "<pad>"])
    assert result["tracked_targets"] == [
        {"requested_text": "France", "token_id": 6, "token": "France"},
        {"requested_text": "Poland", "token_id": 9, "token": "Poland"},
        {"requested_text": "<pad>", "token_id": 0, "token": "<pad>"},
    ]
    for capture, scores in zip(result["layers"], expected, strict=True):
        measured = torch.tensor(capture["token_target_log_probabilities"], dtype=torch.float64)
        torch.testing.assert_close(measured, scores, rtol=1e-12, atol=1e-14)
        assert measured.shape == (len(result["tokens"]), 3)
    assert "float64" in result["target_score_method"]
    store = ExperimentStore(runtime.settings.data_dir / "tracked.sqlite3")
    saved = store.create("Target trajectories", "Compare France and Poland.", result)
    runtime.unload()
    reopened = ExperimentStore(runtime.settings.data_dir / "tracked.sqlite3")
    assert reopened.get(saved["id"])["trace"] == result


@pytest.mark.parametrize("intervention", ["block", "last_block", "head"])
def test_intervened_final_targets_match_the_intervened_model(runtime, intervention):
    block_index = len(runtime._blocks()) - 1 if intervention == "last_block" else 0
    if intervention in ("block", "last_block"):
        handle = runtime._blocks()[block_index].register_forward_hook(
            lambda module, args, output: replace_hidden(output, args[0])
        )
    else:
        handle, _ = install_head_ablation(runtime._blocks(), 0, 0)
    try:
        inputs = runtime.tokenizer("the capital of France", return_tensors="pt")
        with torch.inference_mode():
            expected = runtime.model(**inputs, use_cache=False).logits[0].double().log_softmax(-1)
    finally:
        handle.remove()
    result = runtime.trace(
        "the capital of France",
        0,
        False,
        block_index if intervention in ("block", "last_block") else None,
        {"layer": 0, "head": 0} if intervention == "head" else None,
        tracked_tokens=["France", "Poland"],
    )
    torch.testing.assert_close(
        torch.tensor(result["layers"][-1]["token_target_log_probabilities"], dtype=torch.float64),
        expected[:, [6, 9]],
        rtol=1e-12,
        atol=1e-14,
    )
    assert all(not block._forward_hooks for block in runtime._blocks())


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ([""], "between 1 and 100"),
        (["France", " France"], "distinct vocabulary IDs"),
        (["the capital"], "exactly one token"),
        (["a" * 101], "between 1 and 100"),
        (["France"] * 9, "at most 8"),
    ],
)
def test_invalid_targets_do_not_run_the_model_or_leave_hooks(runtime, targets, message):
    before = runtime.metadata.copy()
    with (
        patch.object(runtime.model, "forward") as forward,
        pytest.raises(ValueError, match=message),
    ):
        runtime.trace("the capital", 0, False, None, tracked_tokens=targets)
    forward.assert_not_called()
    assert runtime.metadata == before
    assert all(not block._forward_hooks for block in runtime._blocks())


@pytest.mark.parametrize("targets", [[""], ["x" * 101], ["France"] * 9, [1]])
def test_request_schema_rejects_unbounded_or_invalid_targets(targets):
    with pytest.raises(ValueError):
        TraceRequest(prompt="the capital", tracked_tokens=targets)


def test_target_text_is_preserved_without_stripping_leading_spaces(runtime):
    result = runtime.trace("the capital", 0, False, None, tracked_tokens=[" France"])
    assert result["tracked_targets"][0] == {
        "requested_text": " France",
        "token_id": 6,
        "token": "France",
    }


def test_target_scores_follow_the_causal_prefix_without_using_later_tokens(runtime):
    first = runtime.trace(
        "the capital of France is", 0, False, None, tracked_tokens=["France", "Poland"]
    )
    second = runtime.trace(
        "the capital of Poland is", 0, False, None, tracked_tokens=["France", "Poland"]
    )
    for a, b in zip(first["layers"], second["layers"], strict=True):
        assert a["token_target_log_probabilities"][:3] == b["token_target_log_probabilities"][:3]
    assert (
        first["layers"][0]["token_target_log_probabilities"][3]
        != second["layers"][0]["token_target_log_probabilities"][3]
    )
