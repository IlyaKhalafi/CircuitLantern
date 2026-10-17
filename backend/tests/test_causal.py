from unittest.mock import patch

import pytest
from model_fixtures import assert_no_forward_hooks

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
    assert result["token_ids"] == runtime.tokenizer.encode(result["clean_prompt"])
    assert result["corrupted_token_ids"] == runtime.tokenizer.encode(result["corrupted_prompt"])
    assert result["tokens"] == [runtime.tokenizer.decode([i]) for i in result["token_ids"]]
    assert result["corrupted_tokens"] == [
        runtime.tokenizer.decode([i]) for i in result["corrupted_token_ids"]
    ]
    assert result["layers"][-1]["log_probability"] == pytest.approx(
        result["clean"]["log_probability"]
    )
    if result["recovery_defined"]:
        assert result["layers"][-1]["recovery"] == pytest.approx(1, abs=1e-4)
    else:
        assert result["layers"][-1]["recovery"] is None
    assert_no_forward_hooks(runtime._blocks())


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
    assert_no_forward_hooks(runtime._blocks())


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
    assert_no_forward_hooks(runtime._blocks())


def test_failed_clean_forward_removes_capture_hooks(runtime):
    with (
        patch.object(runtime.model, "forward", side_effect=RuntimeError("forward failed")),
        pytest.raises(RuntimeError, match="forward failed"),
    ):
        sweep(runtime)
    assert_no_forward_hooks(runtime._blocks())


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
    with pytest.raises(ValueError, match="Only research jobs"):
        runtime.cancel_job("download")
    assert runtime.cancel_job("missing") is None


def test_recovery_requires_a_positive_measurable_corruption_gap():
    assert normalized_recovery(-1, -2, -1.5) == pytest.approx(0.5)
    assert normalized_recovery(-1, -2, -1) == pytest.approx(1)
    assert normalized_recovery(-1, -2, -3) == pytest.approx(-1)
    assert normalized_recovery(-1, -2, 0) == pytest.approx(2)
    assert normalized_recovery(-2, -1, -1.5) is None
    assert normalized_recovery(-1, -1.0001, -1) is None


def test_causal_prompt_limit_uses_the_loaded_context_allowance(runtime):
    runtime.metadata["execution_context_length"] = 16
    with pytest.raises(ValueError, match="16 prompt tokens"):
        activation_patch_sweep(
            runtime.model,
            runtime.tokenizer,
            runtime._blocks(),
            runtime.metadata,
            "cpu",
            clean_prompt="the " * 17,
            corrupted_prompt="capital " * 17,
            target="",
            position=-1,
            job={},
            max_tokens=128,
        )


def test_failed_patch_keeps_completed_layers_and_removes_intervention_hooks(runtime):
    expected = sweep(runtime)
    forward = runtime.model.forward
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("The next patch forward failed")
        return forward(*args, **kwargs)

    job = JobRecord({"status": "running", "progress": 0})
    with patch.object(runtime.model, "forward", side_effect=fail_later):
        result = sweep(runtime, job=job)
    job.finish(result)
    assert result["layers"] == expected["layers"][:1]
    assert result["clean"] == expected["clean"]
    assert result["corrupted"] == expected["corrupted"]
    assert result["complete"] is False
    assert result["error"] == "The next patch forward failed"
    assert result["failed_layer"] == 1
    assert result["cancelled"] is False
    assert job["status"] == "failed"
    assert job["progress"] == 50
    assert job["result"] is result
    assert_no_forward_hooks(runtime._blocks())


@pytest.mark.parametrize("nonfinite", [False, True])
def test_first_patch_failure_retains_baselines_without_publishing_a_successful_block(
    runtime, nonfinite
):
    forward = runtime.model.forward
    calls = 0

    def fail_first_patch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3 and not nonfinite:
            raise RuntimeError()
        output = forward(*args, **kwargs)
        if calls == 3:
            output.logits.fill_(float("nan"))
        return output

    job = JobRecord({"status": "running", "progress": 0})
    with patch.object(runtime.model, "forward", side_effect=fail_first_patch):
        result = sweep(runtime, job=job)
    job.finish(result)
    assert result["failed_layer"] == 0
    assert result["layers"] == []
    assert result["complete"] is False
    assert result["clean"] and result["corrupted"]
    assert result["error"] == (
        "The model produced non finite logits. Choose a different checkpoint or prompt."
        if nonfinite
        else "RuntimeError"
    )
    assert job["status"] == "failed" and job["progress"] == 0
    assert_no_forward_hooks(runtime._blocks())


def test_patch_failure_wins_over_a_simultaneous_cancellation(runtime):
    forward = runtime.model.forward
    calls = 0
    job = JobRecord({"status": "running", "progress": 0})

    def fail_after_cancel(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            job.request_cancel()
            raise RuntimeError("Patch failed while cancellation was requested")
        return forward(*args, **kwargs)

    with patch.object(runtime.model, "forward", side_effect=fail_after_cancel):
        result = sweep(runtime, job=job)
    job.finish(result)
    assert job["status"] == "failed"
    assert len(result["layers"]) == 1
    assert result["cancelled"] is False
    assert result["failed_layer"] == 1
    assert_no_forward_hooks(runtime._blocks())
