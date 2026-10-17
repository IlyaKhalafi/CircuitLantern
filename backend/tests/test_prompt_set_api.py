import time

from fastapi.testclient import TestClient

from circuit_lantern.main import app
from circuit_lantern.services.store import ExperimentStore


def test_prompt_set_jobs_save_and_reopen_offline_with_session_binding(
    runtime, tmp_path, monkeypatch
):
    runtime.metadata.update(execution_id="a" * 32, execution_precision="float32")
    store = ExperimentStore(tmp_path / "notebook.sqlite3")
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    client = TestClient(app)
    payload = {
        "expected_execution_id": "a" * 32,
        "pairs": [
            {
                "label": "literal {raw}",
                "clean_prompt": "the capital of France is",
                "corrupted_prompt": "the capital of Poland is",
                "target": "France",
                "alternative": "Poland",
            }
        ],
    }
    assert (
        client.post(
            "/api/causal/sets/plan", json={**payload, "expected_execution_id": "b" * 32}
        ).status_code
        == 409
    )
    assert client.post("/api/causal/sets/run", json={"pairs": payload["pairs"]}).status_code == 422
    plan = client.post("/api/causal/sets/plan", json=payload)
    assert plan.status_code == 200 and plan.json()["compatible"]
    assert plan.json()["forward_passes"] == 4
    started = client.post("/api/causal/sets/run", json=payload)
    assert started.status_code == 202
    identifier = started.json()["id"]
    for _ in range(200):
        job = client.get("/api/jobs/" + identifier).json()
        if job["status"] in ("ready", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert job["status"] == "ready", job
    saved = client.post(
        "/api/experiments/from-job",
        json={"job_id": identifier, "name": "Prompt set {literal}", "notes": "Test"},
    )
    assert saved.status_code == 201
    runtime.model = runtime.tokenizer = None
    runtime._state = {"loaded": False}
    record = client.get("/api/experiments/" + saved.json()["id"]).json()
    assert record["kind"] == "causal_batch"
    assert record["result"]["execution_id"] == "a" * 32
    assert record["result"]["requested_pairs"] == payload["pairs"]
    assert record["result"]["layers"][0]["mean_interval"] is None
    assert (
        client.post(
            "/api/experiments/from-job",
            json={"job_id": identifier, "name": "Repeated save", "notes": ""},
        ).json()["id"]
        == record["id"]
    )
