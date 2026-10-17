"""A terminal loader event must follow model-lock release and cleanup."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest
from model_fixtures import checkpoint_metadata

from circuit_lantern.services.jobs import JobRecord


@pytest.fixture
def checkpoint(runtime, tmp_path, monkeypatch):
    directory = tmp_path / "checkpoint"
    runtime.model.save_pretrained(directory, safe_serialization=True)
    runtime.tokenizer.save_pretrained(directory)
    metadata = checkpoint_metadata(runtime.model, directory)
    metadata["context_length"] = runtime.model.config.n_positions
    monkeypatch.setattr(
        "circuit_lantern.services.runtime.snapshot_download", lambda *args, **kwargs: str(directory)
    )
    monkeypatch.setattr(runtime, "_assess_fit", lambda *args, **kwargs: {"compatible": True})
    return metadata


def test_ready_event_allows_inspection_before_the_loader_thread_returns(runtime, checkpoint):
    published, release = Event(), Event()

    class ObservedJob(JobRecord):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            if kwargs.get("status") == "ready":
                published.set()
                assert release.wait(5)

    job = ObservedJob({"status": "queued"})
    with ThreadPoolExecutor(max_workers=1) as executor:
        loading = executor.submit(runtime._load, job, checkpoint, "cpu", "float32", 32)
        try:
            assert published.wait(5), job.snapshot()
            captured = runtime.trace("the capital", 1, False, None)
            assert captured["execution_id"] == runtime.metadata["execution_id"]
            assert captured["execution_precision"] == "float32"
            assert captured["token_ids"] == [3, 4]
        finally:
            release.set()
        loading.result(timeout=5)


def test_failed_event_waits_for_replacement_cleanup_to_finish(runtime, checkpoint, monkeypatch):
    cleaning, release = Event(), Event()
    original = runtime._unload
    calls = 0

    def unload():
        nonlocal calls
        calls += 1
        if calls == 2:
            cleaning.set()
            assert release.wait(5)
        original()

    monkeypatch.setattr(runtime, "_unload", unload)
    job = JobRecord({"status": "queued"})
    with (
        patch(
            "circuit_lantern.services.runtime.AutoModelForCausalLM.from_pretrained",
            side_effect=OSError("Checkpoint could not be read"),
        ),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        loading = executor.submit(runtime._load, job, checkpoint, "cpu", "float32", 32)
        try:
            assert cleaning.wait(5), job.snapshot()
            assert job["status"] == "loading"
        finally:
            release.set()
        loading.result(timeout=5)
    assert job["status"] == "failed"
    assert "previous model was released" in job["message"]
    assert not runtime.state()["loaded"]
    with runtime.operation():
        assert runtime.model is None


@pytest.mark.parametrize("initially_loaded", [True, False])
@pytest.mark.parametrize("failure_point", ["release", "cleanup"])
def test_cleanup_errors_still_publish_a_terminal_loader_failure(
    runtime, checkpoint, monkeypatch, initially_loaded, failure_point
):
    if not initially_loaded:
        runtime.unload()
    original = runtime._unload
    calls = 0

    def unload():
        nonlocal calls
        calls += 1
        original()
        if calls == (1 if failure_point == "release" else 2):
            raise RuntimeError("Cache cleanup could not finish")

    monkeypatch.setattr(runtime, "_unload", unload)
    job = JobRecord({"status": "queued"})
    with patch(
        "circuit_lantern.services.runtime.AutoModelForCausalLM.from_pretrained",
        side_effect=OSError("Checkpoint could not be read"),
    ):
        runtime._load(job, checkpoint, "cpu", "float32", 32)
    assert job["status"] == "failed"
    if failure_point == "cleanup":
        assert "Checkpoint could not be read" in job["message"]
    assert "Cache cleanup could not finish" in job["message"]
    assert ("previous model was released" in job["message"]) == initially_loaded
    assert not runtime.state()["loaded"]
    with runtime.operation():
        assert runtime.model is None


def test_long_load_error_preserves_cleanup_diagnostics(runtime, checkpoint, monkeypatch):
    original = runtime._unload
    calls = 0

    def unload():
        nonlocal calls
        calls += 1
        original()
        if calls == 2:
            raise RuntimeError("Cleanup details " + "z" * 180)

    monkeypatch.setattr(runtime, "_unload", unload)
    job = JobRecord({"status": "queued"})
    with patch(
        "circuit_lantern.services.runtime.AutoModelForCausalLM.from_pretrained",
        side_effect=OSError("x" * 1200),
    ):
        runtime._load(job, checkpoint, "cpu", "float32", 32)
    assert job["status"] == "failed"
    assert len(job["message"]) <= 1000
    assert job["message"].endswith("Cleanup details " + "z" * 180)
    assert "previous model was released" in job["message"]
