import asyncio
from unittest.mock import patch

import httpx
import pytest

from circuit_lantern.main import app
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.store import ExperimentStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(runtime, monkeypatch):
    runtime.metadata["execution_id"] = "a" * 32
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(
        app.state, "store", ExperimentStore(runtime.settings.data_dir / "records.db"), raising=False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as connection:
        yield connection


@pytest.mark.parametrize(
    ("path", "payload", "kind", "row_key"),
    [
        ("/api/head-sweep", {"prompt": "the capital", "layers": [0]}, "head_sweep", "heads"),
        (
            "/api/causal",
            {"clean_prompt": "the capital of France", "corrupted_prompt": "the capital of Poland"},
            "causal",
            "layers",
        ),
    ],
)
async def test_saved_sweep_survives_model_unload_and_store_reopen(
    client, runtime, path, payload, kind, row_key
):
    queued = await client.post(path, json=payload)
    assert queued.status_code == 202
    job_id = queued.json()["id"]
    for _ in range(100):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in ("ready", "failed"):
            break
        await asyncio.sleep(0.01)
    assert job["status"] == "ready", job["message"]
    runtime.unload()
    saved = await client.post(
        "/api/experiments/from-job",
        json={"job_id": job_id, "name": "Saved research", "notes": "A durable observation"},
    )
    assert saved.status_code == 201
    record = saved.json()
    assert record["kind"] == kind
    assert record["model_id"] == "local-test"
    stored = (await client.get(f"/api/experiments/{record['id']}")).json()
    assert "trace" not in stored
    assert stored["result"] == job["result"]
    assert len(stored["result"][row_key]) == 2
    assert stored["result"]["execution_id"] == "a" * 32
    reopened = ExperimentStore(runtime.settings.data_dir / "records.db")
    assert reopened.get(record["id"]) == stored
    updated = await client.patch(
        f"/api/experiments/{record['id']}", json={"notes": "Updated research notes"}
    )
    assert updated.json()["result"] == job["result"]
    assert updated.json()["notes"] == "Updated research notes"
    assert (await client.get("/api/experiments")).json()[0]["kind"] == kind


@pytest.mark.parametrize("status", ["cancelled", "failed"])
async def test_partial_sweep_results_are_saved_with_their_terminal_state(client, runtime, status):
    job_id = "b" * 32
    result = {
        "model_id": "historical-checkpoint",
        "revision": "historical-revision",
        "execution_id": "c" * 32,
        "heads": [{"layer": 0, "head": 0}],
        "complete": False,
        "cancelled": status == "cancelled",
        "error": "Forward failed" if status == "failed" else None,
    }
    runtime.jobs[job_id] = JobRecord(
        {"id": job_id, "kind": "head_sweep", "status": status, "result": result}
    )
    saved = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Partial capture"}
    )
    assert saved.status_code == 201
    stored = (await client.get(f"/api/experiments/{saved.json()['id']}")).json()
    assert stored["result"] == result
    assert stored["model_id"] == "historical-checkpoint"


async def test_unpublished_jobs_cannot_create_misleading_experiments(client, runtime):
    for status in ("queued", "running", "cancelled", "failed"):
        job_id = "d" * 32
        runtime.jobs[job_id] = JobRecord({"id": job_id, "kind": "head_sweep", "status": status})
        response = await client.post(
            "/api/experiments/from-job", json={"job_id": job_id, "name": "Not captured"}
        )
        assert response.status_code == 409
    assert (await client.get("/api/experiments")).json() == []


async def test_downloads_and_expired_jobs_return_useful_save_errors(client, runtime):
    job_id = "e" * 32
    runtime.jobs[job_id] = JobRecord({"id": job_id, "kind": "download", "status": "ready"})
    response = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Download"}
    )
    assert response.status_code == 400
    response = await client.post(
        "/api/experiments/from-job", json={"job_id": "f" * 32, "name": "Expired"}
    )
    assert response.status_code == 404
    assert "expired" in response.json()["detail"]


async def test_retry_after_job_expiry_returns_the_original_edited_experiment(client, runtime):
    job_id = "b" * 32
    result = {"model_id": "source-model", "revision": "source-revision", "heads": []}
    runtime.jobs[job_id] = JobRecord(
        {"id": job_id, "kind": "head_sweep", "status": "ready", "result": result}
    )
    body = {"job_id": job_id, "name": "First save"}
    first = await client.post("/api/experiments/from-job", json=body)
    record_id = first.json()["id"]
    await client.patch(
        f"/api/experiments/{record_id}", json={"name": "Edited title", "notes": "Preserved notes"}
    )
    del runtime.jobs[job_id]
    repeated = await client.post("/api/experiments/from-job", json=body)
    assert repeated.status_code == 201
    assert repeated.json()["id"] == record_id
    assert repeated.json()["name"] == "Edited title"
    assert repeated.json()["notes"] == "Preserved notes"
    assert len((await client.get("/api/experiments")).json()) == 1


@pytest.mark.parametrize("result", [None, [], {}, {"model_id": None}, {"model_id": "model"}])
async def test_missing_capture_identity_cannot_create_an_experiment(client, runtime, result):
    job_id = "b" * 32
    runtime.jobs[job_id] = JobRecord(
        {"id": job_id, "kind": "head_sweep", "status": "failed", "result": result}
    )
    response = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Missing identity"}
    )
    assert response.status_code == 409
    assert (await client.get("/api/experiments")).json() == []


async def test_nonfinite_results_are_rejected_with_a_useful_client_error(client, runtime):
    job_id = "b" * 32
    runtime.jobs[job_id] = JobRecord(
        {
            "id": job_id,
            "kind": "head_sweep",
            "status": "failed",
            "result": {"model_id": "model", "revision": "source", "value": float("nan")},
        }
    )
    response = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Non finite"}
    )
    assert response.status_code == 400
    assert (await client.get("/api/experiments")).json() == []


@pytest.mark.parametrize("failure_call", [3, 4])
async def test_actual_failed_patch_job_keeps_saveable_measurements_after_unload(
    client, runtime, failure_call
):
    forward = runtime.model.forward
    calls = 0

    def fail_patch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("A patch forward failed")
        return forward(*args, **kwargs)

    with patch.object(runtime.model, "forward", side_effect=fail_patch):
        response = await client.post(
            "/api/causal",
            json={
                "clean_prompt": "the capital of France",
                "corrupted_prompt": "the capital of Poland",
                "position": -1,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(100):
            job = (await client.get(f"/api/jobs/{job_id}")).json()
            if job["status"] in ("ready", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
    assert job["status"] == "failed"
    result = job["result"]
    assert result["error"] == "A patch forward failed"
    assert result["failed_layer"] == failure_call - 3
    assert len(result["layers"]) == failure_call - 3
    assert result["complete"] is False and result["cancelled"] is False
    assert all(not block._forward_hooks for block in runtime._blocks())
    runtime.unload()
    saved = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Retained partial patch"}
    )
    assert saved.status_code == 201
    record_id = saved.json()["id"]
    record = (await client.get(f"/api/experiments/{record_id}")).json()
    assert record["kind"] == "causal" and record["result"] == result
    notes = "Repeat the incomplete sweep before drawing a conclusion."
    await client.patch(f"/api/experiments/{record_id}", json={"notes": notes})
    del runtime.jobs[job_id]
    retry = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Stale retry name"}
    )
    assert retry.status_code == 201
    assert retry.json()["id"] == record_id
    assert retry.json()["name"] == "Retained partial patch"
    assert retry.json()["notes"] == notes
    assert (
        ExperimentStore(runtime.settings.data_dir / "records.db").get(record_id)["result"] == result
    )
