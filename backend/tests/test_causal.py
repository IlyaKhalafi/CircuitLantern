from unittest.mock import patch

import pytest

from circuit_lantern.services.causal import activation_patch_sweep, normalized_recovery
from circuit_lantern.services.jobs import JobRecord


def sweep(runtime, **overrides):
    arguments = {
        "model": runtime.model,
        "tokenizer": runtime.tokenizer,
        "blocks": runtime._blocks(),
        "metadata": runtime.metadata,
        "device": runtime.device,
        "clean_prompt": "the capital of France is",
        "corrupted_prompt": "the capital of the is",
        "target": "",
        "position": -1,
        "job": {},
        "max_tokens": 128,
    }
    arguments.update(overrides)
    return activation_patch_sweep(**arguments)


def test_last_block_patch_restores_clean_target_score(runtime):
    result = sweep(runtime)
    assert len(result["layers"]) == 2
    assert result["layers"][-1]["log_probability"] == pytest.approx(
        result["clean"]["log_probability"]
    )
    if result["recovery_defined"]:
        assert result["layers"][-1]["recovery"] == pytest.approx(1, abs=1e-4)
    else:
        assert result["layers"][-1]["recovery"] is None
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_all_position_patch_restores_clean_computation_at_every_block(runtime):
    result = sweep(runtime, position=None)
    for layer in result["layers"]:
        assert layer["log_probability"] == pytest.approx(
            result["clean"]["log_probability"], abs=1e-5
        )


def test_identical_prompts_do_not_report_normalized_recovery(runtime):
    result = sweep(runtime, corrupted_prompt="the capital of France is")
    assert not result["recovery_defined"]
    assert all(layer["recovery"] is None for layer in result["layers"])


def test_unaligned_prompts_rejected_before_installing_hooks(runtime):
    with pytest.raises(ValueError, match="same length"):
        sweep(runtime, corrupted_prompt="the capital")
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_invalid_target_rejected_with_actionable_message(runtime):
    with pytest.raises(ValueError, match="exactly one token"):
        sweep(runtime, target="the capital")


def test_cancelled_sweep_keeps_partial_result_and_removes_hooks(runtime):
    job = JobRecord({"status": "running", "progress": 0, "cancel_requested": True})
    result = sweep(runtime, job=job)
    assert job["status"] == "running"
    job.finish(result)
    snapshot = job.snapshot()
    assert snapshot["status"] == "cancelled"
    assert snapshot["result"]["layers"] == []
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_failed_clean_forward_removes_capture_hooks(runtime):
    with (
        patch.object(runtime.model, "forward", side_effect=RuntimeError("forward failed")),
        pytest.raises(RuntimeError, match="forward failed"),
    ):
        sweep(runtime)
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_prompt_position_validation(runtime):
    with pytest.raises(ValueError, match="outside"):
        sweep(runtime, position=100)


def test_causal_request_blocks_model_replacement(runtime):
    runtime._state = {"loaded": True, "metadata": runtime.metadata, "device": runtime.device}
    runtime.jobs["active"] = {"status": "running", "kind": "causal"}
    with pytest.raises(ValueError, match="background operation"):
        runtime.load(runtime.metadata, "cpu")
    with pytest.raises(ValueError, match="background operation"):
        runtime.causal({})


def test_download_cancellation_is_explicitly_unsupported(runtime):
    runtime.jobs["download"] = {"status": "queued", "kind": "download"}
    with pytest.raises(ValueError, match="Only causal sweeps"):
        runtime.cancel_job("download")
    assert runtime.cancel_job("missing") is None


def test_recovery_requires_a_positive_measurable_corruption_gap():
    assert normalized_recovery(-1, -2, -1.5) == pytest.approx(0.5)
    assert normalized_recovery(-1, -2, -1) == pytest.approx(1)
    assert normalized_recovery(-1, -2, -3) == pytest.approx(-1)
    assert normalized_recovery(-1, -2, 0) == pytest.approx(2)
    assert normalized_recovery(-2, -1, -1.5) is None
    assert normalized_recovery(-1, -1.0001, -1) is None
