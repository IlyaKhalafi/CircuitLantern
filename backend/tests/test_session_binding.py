from unittest.mock import patch

import httpx
import pytest

from circuit_lantern.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


CURRENT = "a" * 32
STALE = "b" * 32


@pytest.fixture
async def bound_client(runtime, monkeypatch):
    runtime.metadata["execution_id"] = CURRENT
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", f"/api/graph?expected_execution_id={STALE}", None),
        ("GET", f"/api/matrix?name=transformer.wte.weight&expected_execution_id={STALE}", None),
        ("DELETE", f"/api/runtime?expected_execution_id={STALE}", None),
        ("POST", "/api/tokenize", {"prompts": ["the capital"], "expected_execution_id": STALE}),
        ("POST", "/api/trace", {"prompt": "the capital", "expected_execution_id": STALE}),
        ("POST", "/api/head-sweep", {"prompt": "the capital", "expected_execution_id": STALE}),
        (
            "POST",
            "/api/causal",
            {
                "clean_prompt": "the capital of France",
                "corrupted_prompt": "the capital of Poland",
                "expected_execution_id": STALE,
            },
        ),
    ],
)
async def test_stale_sessions_are_rejected_before_model_work_or_unloading(
    bound_client, runtime, method, path, body
):
    active = runtime.model
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        response = await bound_client.request(method, path, json=body)
    assert response.status_code == 409
    assert "active model changed" in response.json()["detail"]
    assert runtime.model is active
    assert not runtime.jobs
    assert not runtime.metrics_cache


async def test_matrix_export_uses_actual_identity_and_checks_even_cached_requests(
    bound_client, runtime
):
    path = "/api/matrix?name=transformer.wte.weight"
    legacy = await bound_client.get(path)
    assert legacy.status_code == 200
    result = legacy.json()
    assert result["model_id"] == runtime.metadata["id"]
    assert result["revision"] == runtime.metadata["revision"]
    assert result["execution_id"] == CURRENT
    assert result["frobenius_norm"] > 0
    assert (await bound_client.get(path + f"&expected_execution_id={CURRENT}")).json() == result
    assert (await bound_client.get(path + f"&expected_execution_id={STALE}")).status_code == 409


async def test_bound_tokenization_and_trace_record_the_same_session(bound_client):
    preview = await bound_client.post(
        "/api/tokenize",
        json={
            "prompts": ["the capital"],
            "reserved_tokens": 1,
            "expected_execution_id": CURRENT,
        },
    )
    capture = await bound_client.post(
        "/api/trace",
        json={
            "prompt": "the capital",
            "max_new_tokens": 1,
            "expected_execution_id": CURRENT,
        },
    )
    assert preview.status_code == capture.status_code == 200
    assert preview.json()["execution_id"] == capture.json()["execution_id"] == CURRENT
    assert preview.json()["prompts"][0]["token_ids"] == capture.json()["token_ids"]


async def test_execution_query_and_body_ids_are_validated(bound_client):
    assert (await bound_client.get("/api/graph?expected_execution_id=bad")).status_code == 422
    response = await bound_client.post(
        "/api/trace", json={"prompt": "the capital", "expected_execution_id": "bad"}
    )
    assert response.status_code == 422


async def test_bound_unload_clears_only_the_requested_active_session(bound_client, runtime):
    assert (
        await bound_client.delete(f"/api/runtime?expected_execution_id={CURRENT}")
    ).status_code == 204
    assert runtime.model is None
    assert (
        await bound_client.delete(f"/api/runtime?expected_execution_id={CURRENT}")
    ).status_code == 409
    assert (await bound_client.delete("/api/runtime")).status_code == 204


async def test_head_sweep_endpoint_completes_with_bound_identity(bound_client):
    import asyncio

    response = await bound_client.post(
        "/api/head-sweep",
        json={"prompt": "the capital", "layers": [0], "expected_execution_id": CURRENT},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    for _ in range(100):
        job = (await bound_client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in ("ready", "failed"):
            break
        await asyncio.sleep(0.01)
    assert job["status"] == "ready", job["message"]
    assert job["result"]["execution_id"] == CURRENT
    assert len(job["result"]["heads"]) == 2
