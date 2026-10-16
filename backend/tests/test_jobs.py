from unittest.mock import patch

import pytest

from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.runtime import RuntimeBusyError


def test_cancellation_publishes_partial_results_with_the_terminal_status():
    job = JobRecord({"id": "test", "status": "running", "progress": 40})
    response = job.request_cancel()
    assert response == {"status": "running", "cancel_requested": True}
    assert "result" not in job.snapshot()
    job.finish({"layers": [{"layer": 0}]})
    snapshot = job.snapshot()
    assert snapshot["status"] == "cancelled"
    assert snapshot["result"]["layers"] == [{"layer": 0}]
    assert snapshot["result"]["cancelled"]
    assert snapshot["progress"] == 40


def test_completed_job_cannot_acknowledge_a_late_cancellation():
    job = JobRecord({"id": "test", "status": "running", "progress": 75})
    job.finish({"layers": []})
    response = job.request_cancel()
    assert response == {"status": "ready", "cancel_requested": False}
    assert job.snapshot()["result"]["cancelled"] is False
    assert job["progress"] == 100


def test_job_snapshots_do_not_expose_mutable_transition_state():
    job = JobRecord({"id": "test", "status": "queued"})
    snapshot = job.snapshot()
    snapshot["status"] = "ready"
    assert job["status"] == "queued"


def test_foreground_model_operations_fail_quickly_while_the_runtime_is_busy(runtime):
    with runtime.lock:
        with pytest.raises(RuntimeBusyError, match="busy"):
            runtime.graph()
        with pytest.raises(RuntimeBusyError, match="busy"):
            runtime.trace("the capital", 1, False, None)
    assert runtime.graph()["matrices"]


def test_failed_replacement_reports_the_released_model(runtime):
    runtime.device = "cuda"
    job = JobRecord({"status": "queued", "progress": 0})
    with (
        patch("circuit_lantern.services.runtime.snapshot_download", return_value="/unused"),
        patch(
            "circuit_lantern.services.runtime.AutoTokenizer.from_pretrained",
            side_effect=ValueError("invalid tokenizer"),
        ),
    ):
        runtime._load(job, {"id": "replacement", "revision": "test"}, "cpu")
    assert job["status"] == "failed"
    assert "previous model was released" in job["message"]
    assert runtime.state() == {
        "loaded": False,
        "metadata": None,
        "device": "cpu",
        "max_prompt_tokens": runtime.settings.max_prompt_tokens,
    }
