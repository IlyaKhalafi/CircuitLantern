import asyncio

import httpx
import pytest

from circuit_lantern.main import app
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


async def analyze(client, name="transformer.h.0.attn.c_attn.weight"):
    response = await client.get("/api/matrix", params={"name": name})
    assert response.status_code == 200
    return response.json()


async def test_saved_matrix_preserves_actual_measurements_after_unload_and_restart(client, runtime):
    metrics = await analyze(client)
    assert metrics["analysis_id"]
    assert metrics["device"] == "cpu"
    saved = await client.post(
        "/api/experiments/from-matrix",
        json={
            "analysis_id": metrics["analysis_id"],
            "expected_execution_id": "a" * 32,
            "name": "Attention projection geometry",
            "notes": "Inspect the spectrum.",
        },
    )
    assert saved.status_code == 201
    record = saved.json()
    assert record["kind"] == "matrix"
    assert record["source_analysis_id"] == metrics["analysis_id"]
    assert record["source_job_id"] is None
    runtime.unload()
    stored = (await client.get("/api/experiments/" + record["id"])).json()
    assert stored["result"] == metrics
    assert "trace" not in stored
    assert ExperimentStore(runtime.settings.data_dir / "records.db").get(record["id"]) == stored
    edited = await client.patch(
        "/api/experiments/" + record["id"],
        json={"name": "Edited analysis", "notes": "Keep this observation."},
    )
    assert edited.status_code == 200
    retry = await client.post(
        "/api/experiments/from-matrix",
        json={
            "analysis_id": metrics["analysis_id"],
            "name": "Old unsaved name",
            "expected_execution_id": "a" * 32,
        },
    )
    assert retry.status_code == 201
    assert retry.json()["id"] == record["id"]
    assert retry.json()["name"] == "Edited analysis"
    assert retry.json()["notes"] == "Keep this observation."
    assert (await client.get("/api/experiments")).json() == [retry.json()]


async def test_saving_does_not_recompute_or_read_replacement_weights(client, runtime, monkeypatch):
    metrics = await analyze(client)

    def unexpected_matrix(*args, **kwargs):
        raise AssertionError("Saving a captured analysis must not recompute it")

    monkeypatch.setattr(runtime, "matrix", unexpected_matrix)
    saved = await client.post(
        "/api/experiments/from-matrix",
        json={"analysis_id": metrics["analysis_id"], "name": "Cached source"},
    )
    assert saved.status_code == 201
    record_id = saved.json()["id"]
    assert (await client.get("/api/experiments/" + record_id)).json()["result"] == metrics


async def test_expired_analysis_and_stale_execution_cannot_save_new_records(client, runtime):
    metrics = await analyze(client)
    stale = await client.post(
        "/api/experiments/from-matrix",
        json={
            "analysis_id": metrics["analysis_id"],
            "name": "Stale",
            "expected_execution_id": "b" * 32,
        },
    )
    assert stale.status_code == 409
    runtime.metrics_cache.clear()
    expired = await client.post(
        "/api/experiments/from-matrix",
        json={"analysis_id": metrics["analysis_id"], "name": "Expired"},
    )
    assert expired.status_code == 409
    assert "Reopen the matrix" in expired.json()["detail"]
    assert (await client.get("/api/experiments")).json() == []


async def test_unsaved_analysis_cannot_be_reconstructed_after_unload(client, runtime):
    metrics = await analyze(client)
    runtime.unload()
    response = await client.post(
        "/api/experiments/from-matrix",
        json={"analysis_id": metrics["analysis_id"], "name": "Unavailable source"},
    )
    assert response.status_code == 409
    assert (await client.get("/api/experiments")).json() == []


async def test_each_new_computation_gets_its_own_source_identifier(client, runtime):
    first = await analyze(client)
    cached = await analyze(client)
    assert cached == first
    runtime.metrics_cache.clear()
    second = await analyze(client)
    assert second["analysis_id"] != first["analysis_id"]
    for key in ["singular_values", "frobenius_norm", "heatmap", "model_id", "revision"]:
        assert second[key] == first[key]


async def test_repeated_matrix_saves_create_one_record(client):
    metrics = await analyze(client)
    body = {"analysis_id": metrics["analysis_id"], "name": "Repeated save"}
    responses = await asyncio.gather(
        client.post("/api/experiments/from-matrix", json=body),
        client.post("/api/experiments/from-matrix", json=body),
    )
    assert all(response.status_code in (201, 409) for response in responses)
    repeated = await client.post("/api/experiments/from-matrix", json=body)
    assert repeated.status_code == 201
    assert len((await client.get("/api/experiments")).json()) == 1
    assert all(
        response.json()["id"] == repeated.json()["id"]
        for response in responses
        if response.status_code == 201
    )


@pytest.mark.parametrize("analysis_id", ["", "not-a-capture", "x" * 32, "a" * 33])
async def test_invalid_matrix_source_identifiers_are_rejected(client, analysis_id):
    response = await client.post(
        "/api/experiments/from-matrix", json={"analysis_id": analysis_id, "name": "Invalid"}
    )
    assert response.status_code == 422
    assert (await client.get("/api/experiments")).json() == []
