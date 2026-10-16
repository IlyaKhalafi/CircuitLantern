import time
from unittest.mock import patch

import pytest
import torch

from circuit_lantern.services.head_sweep import head_ablation_sweep, select_head_targets
from circuit_lantern.services.jobs import JobRecord


def sweep(runtime, **overrides):
    arguments = {
        "model": runtime.model,
        "tokenizer": runtime.tokenizer,
        "blocks": runtime._blocks(),
        "metadata": runtime.metadata,
        "device": runtime.device,
        "prompt": "the capital of France",
        "target": "",
        "layers": None,
        "job": {},
        "max_tokens": 128,
    }
    arguments.update(overrides)
    return head_ablation_sweep(**arguments)


def assert_no_head_hooks(runtime):
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_sweep_matches_individual_head_traces_and_preserves_weights(runtime):
    weights = {name: tensor.clone() for name, tensor in runtime.model.state_dict().items()}
    result = sweep(runtime)
    assert result["last_token_logits_only"] is True
    assert result["logits_dtype"] == "float32"
    assert result["total_heads"] == 4
    assert len(result["heads"]) == 4
    assert result["scope"] == "all_prompt_tokens"
    assert result["token_ids"] == runtime.tokenizer(result["prompt"]).input_ids
    for head in result["heads"]:
        trace = runtime.trace(
            result["prompt"],
            1,
            False,
            None,
            {"layer": head["layer"], "head": head["head"]},
        )
        prediction = next(
            item for item in trace["predictions"] if item["token_id"] == result["target_token_id"]
        )
        assert head["probability"] == pytest.approx(prediction["probability"], abs=1e-7)
        assert head["delta_log_probability"] == pytest.approx(
            head["log_probability"] - result["baseline"]["log_probability"]
        )
        assert head["kl_divergence"] >= 0
    assert all(
        torch.equal(weights[name], tensor) for name, tensor in runtime.model.state_dict().items()
    )
    assert_no_head_hooks(runtime)


def test_layer_selection_and_explicit_target_are_recorded(runtime):
    result = sweep(runtime, layers=[1], target="capital")
    assert len(result["heads"]) == 2
    assert {head["layer"] for head in result["heads"]} == {1}
    assert result["requested_target"] == "capital"
    assert result["target_token_id"] == runtime.tokenizer.encode("capital")[0]


@pytest.mark.parametrize("layers", [[], [0, 0], [99]])
def test_invalid_layers_fail_before_forward(runtime, layers):
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError),
    ):
        sweep(runtime, layers=layers)
    assert_no_head_hooks(runtime)


def test_sweep_work_is_bounded_before_model_work(runtime):
    with (
        patch(
            "circuit_lantern.services.head_sweep.attention_head_targets",
            return_value=[
                {"layer": 0, "heads": 300, "head_dim": 16},
                {"layer": 1, "heads": 300, "head_dim": 16},
            ],
        ),
        pytest.raises(ValueError, match="at most 512 heads"),
    ):
        select_head_targets(runtime._blocks(), None)


def test_failed_ablated_forward_removes_temporary_hook(runtime):
    original = runtime.model.forward
    count = 0

    def fail_ablated(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("ablated forward failed")
        return original(*args, **kwargs)

    with patch.object(runtime.model, "forward", side_effect=fail_ablated):
        result = sweep(runtime)
    assert result["complete"] is False
    assert result["error"] == "ablated forward failed"
    assert result["failed_head"] == {"layer": 0, "head": 0}
    assert result["heads"] == []
    assert_no_head_hooks(runtime)


def test_cancellation_keeps_completed_heads_and_correct_terminal_message(runtime):
    class CancelAfterOne(JobRecord):
        def update(self, **changes):
            super().update(**changes)
            if changes.get("message", "").startswith("Tested head"):
                self.request_cancel()

    job = CancelAfterOne({"status": "running", "progress": 0})
    result = sweep(runtime, job=job)
    assert len(result["heads"]) == 1
    job.finish(result, label="Head sweep", unit="heads")
    snapshot = job.snapshot()
    assert snapshot["status"] == "cancelled"
    assert snapshot["result"]["cancelled"] is True
    assert snapshot["progress"] == 25
    assert "Completed heads" in snapshot["message"]
    assert_no_head_hooks(runtime)


def test_prompt_and_target_validation(runtime):
    runtime.metadata["execution_context_length"] = 16
    with pytest.raises(ValueError, match="16 prompt tokens"):
        sweep(runtime, prompt="the " * 17)
    with pytest.raises(ValueError, match="exactly one token"):
        sweep(runtime, target="the capital")
    assert_no_head_hooks(runtime)


def test_background_sweep_records_execution_and_remains_cancellable(runtime):
    runtime.metadata["execution_id"] = "a" * 32
    job = runtime.head_sweep(
        {
            "prompt": "the capital",
            "target": "",
            "layers": [0],
            "expected_execution_id": "a" * 32,
        }
    )
    assert job["kind"] == "head_sweep"
    for _ in range(100):
        snapshot = runtime.job(job["id"])
        if snapshot["status"] in ("ready", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert snapshot["status"] == "ready", snapshot["message"]
    assert snapshot["result"]["execution_id"] == "a" * 32
    assert len(snapshot["result"]["heads"]) == 2
    assert runtime.cancel_job(job["id"])["cancel_requested"] is False


def test_queued_head_sweep_rejects_replaced_model_session(runtime):
    job = JobRecord(
        {
            "kind": "head_sweep",
            "model_id": runtime.metadata["id"],
            "execution_id": "b" * 32,
            "status": "queued",
            "progress": 0,
        }
    )
    runtime.metadata["execution_id"] = "a" * 32
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        runtime._run_sweep(job, {}, runtime.metadata["revision"])
    assert job["status"] == "failed"
    assert "active model changed" in job["message"]


def test_cancelling_a_queued_head_sweep_avoids_any_forward(runtime):
    job = JobRecord(
        {
            "kind": "head_sweep",
            "status": "queued",
            "progress": 0,
            "cancel_requested": True,
        }
    )
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        runtime._run_sweep(job, {}, runtime.metadata["revision"])
    assert job["status"] == "cancelled"
    assert "Head sweep" in job["message"]


def test_failed_sweep_retains_completed_heads_in_failed_job(runtime):
    original = runtime.model.forward
    count = 0

    def fail_after_one_head(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 2:
            raise RuntimeError("head failed")
        return original(*args, **kwargs)

    job = JobRecord({"status": "running", "progress": 0})
    with patch.object(runtime.model, "forward", side_effect=fail_after_one_head):
        result = sweep(runtime, job=job)
    job.finish(result, label="Head sweep", unit="heads")
    snapshot = job.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["message"] == "head failed"
    assert len(snapshot["result"]["heads"]) == 1
    assert snapshot["result"]["failed_head"] == {"layer": 0, "head": 1}
    assert snapshot["progress"] == 25
    assert_no_head_hooks(runtime)


def test_training_mode_is_rejected_before_model_work(runtime):
    runtime.model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        sweep(runtime)
    assert_no_head_hooks(runtime)


def test_omitted_layers_in_request_schema_tests_all_supported_heads():
    from circuit_lantern.schemas import HeadSweepRequest

    payload = HeadSweepRequest(prompt="the capital").model_dump()
    assert payload["layers"] is None
