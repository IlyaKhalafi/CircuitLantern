import asyncio
import json
import sqlite3
from unittest.mock import patch

import httpx
import pytest

from circuit_lantern.induction_schemas import InductionProbeRequest
from circuit_lantern.main import app
from circuit_lantern.services.induction.lab import InductionLab
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.runtime import RuntimeBusyError
from circuit_lantern.services.store import ExperimentStore
from circuit_lantern.services.store_schema import SCHEMA_VERSION

HARDWARE = {"ram_available_bytes": 2**40, "experiment_disk_free_bytes": 2**40}
REQUEST = {"block_length": 4, "repeats": 2, "examples": 2, "causal_heads": 1, "seed": 19}


@pytest.fixture
def lab(runtime, monkeypatch):
    runtime.metadata.update(execution_id="a" * 32, execution_context_length=64)
    monkeypatch.setattr(
        "circuit_lantern.services.induction.lab.hardware_snapshot", lambda _: dict(HARDWARE)
    )
    return InductionLab(runtime)


def request(**overrides):
    return InductionProbeRequest(**{**REQUEST, **overrides})


def test_capabilities_recommend_valid_context_and_vocabulary_settings(lab, runtime):
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        caps = lab.capabilities()
        assert caps["loaded"] and caps["supported"]
        assert caps["vocabulary_pool_size"] == 13
        assert caps["start_token_id"] == 0
        assert caps["recommended"]["block_length"] == 13
        assert lab.plan(InductionProbeRequest(**caps["recommended"]))["compatible"]
        runtime.metadata["execution_context_length"] = 16
        caps = lab.capabilities()
        assert caps["recommended"]["block_length"] == 5
        assert caps["recommended"]["repeats"] == 3
        assert lab.plan(InductionProbeRequest(**caps["recommended"]))["tokens_per_probe"] == 16


def test_resource_blocking_precedes_tokenizer_vocabulary_allocation_and_forward(
    lab, runtime, monkeypatch
):
    monkeypatch.setattr(
        "circuit_lantern.services.induction.lab.hardware_snapshot",
        lambda _: {
            "ram_available_bytes": 1,
            "experiment_disk_free_bytes": 1,
        },
    )
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        patch.object(
            runtime.tokenizer, "get_vocab", side_effect=AssertionError("Unexpected allocation")
        ),
    ):
        plan = lab.plan(request())
        assert not plan["compatible"]
        assert "dataset_fingerprint" not in plan
        with pytest.raises(ValueError, match="RAM"):
            lab.start(request())
        assert not lab.capabilities()["supported"]
    assert runtime.jobs == {}


def test_plan_preserves_rng_and_rejects_stale_execution(lab, runtime):
    import torch

    before = torch.get_rng_state()
    plan = lab.plan(request(expected_execution_id="a" * 32))
    assert plan == lab.plan(request())
    assert torch.equal(before, torch.get_rng_state())
    assert plan["tokens_per_probe"] == 9 and plan["requested_forward_passes"] == 4
    assert plan["selected_heads"] == 4 and plan["causal_heads"] == 1
    assert plan["execution_id"] == "a" * 32
    with pytest.raises(ValueError, match="active model changed"):
        lab.plan(request(expected_execution_id="b" * 32))


@pytest.mark.parametrize(
    "overrides",
    [
        {"layers": [0, 0]},
        {"layers": [2]},
        {"block_length": 14},
        {"block_length": 32, "repeats": 4},
    ],
)
def test_invalid_model_specific_request_is_rejected_before_forward(lab, runtime, overrides):
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError),
    ):
        lab.start(request(**overrides))
    assert runtime.jobs == {}


def test_busy_runtime_rejects_planning_without_waiting_or_mutation(lab, runtime):
    with runtime.operation(), pytest.raises(RuntimeBusyError):
        lab.plan(request())
    assert runtime.jobs == {}


@pytest.mark.parametrize("mode", ["unloaded", "training", "sdpa"])
def test_unsupported_execution_has_useful_capabilities_without_forward(lab, runtime, mode):
    if mode == "unloaded":
        runtime.model = None
    elif mode == "training":
        runtime.model.train()
    else:
        runtime.model.config._attn_implementation = "sdpa"
    caps = lab.capabilities()
    assert not caps["supported"] and caps["recommended"] is None
    assert caps["loaded"] == (mode != "unloaded")
    if mode != "unloaded":
        assert caps["reason"]


def test_worker_rechecks_exact_probe_fingerprint_before_any_forward(lab, runtime):
    batch, _, _ = lab._prepare(request())
    payload = request().model_dump()
    payload["expected_comparison_identity"] = lab.plan(request())["comparison_identity_sha256"]
    payload["expected_probe_fingerprint"] = "0" * 64
    assert batch.fingerprint != payload["expected_probe_fingerprint"]
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError, match="probe vocabulary changed"),
    ):
        lab._measure(payload, {})
    assert runtime.jobs == {}


def test_queued_probe_rejects_replaced_execution_without_forward(lab, runtime):
    batch, _, _ = lab._prepare(request())
    payload = request().model_dump()
    payload["expected_comparison_identity"] = lab.plan(request())["comparison_identity_sha256"]
    payload["expected_probe_fingerprint"] = batch.fingerprint
    job = JobRecord(
        {
            "kind": "induction_probe",
            "model_id": runtime.metadata["id"],
            "execution_id": "b" * 32,
            "status": "queued",
            "progress": 0,
        }
    )
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        runtime._run_sweep(job, payload, runtime.metadata["revision"], lab._measure)
    assert job["status"] == "failed"
    assert "active model changed" in job["message"]


def test_queued_probe_is_cancellable_before_capture(lab, runtime):
    job = JobRecord(
        {
            "id": "c" * 32,
            "kind": "induction_probe",
            "status": "queued",
            "progress": 0,
            "cancel_requested": False,
        }
    )
    runtime.jobs[job["id"]] = job
    assert runtime.cancel_job(job["id"])["cancel_requested"]
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        runtime._run_sweep(job, {}, runtime.metadata["revision"], lab._measure)
    assert job["status"] == "cancelled"
    assert job["message"] == "Induction probe cancelled before capture"


def test_version_two_notebooks_upgrade_without_reencoding_prior_evidence(tmp_path):
    path = tmp_path / "old.db"
    payload = (
        json.dumps({"model_id": "model", "revision": "pinned", "values": [0.5]}, indent=3) + " \n"
    )
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
            created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'trace' CHECK (kind IN ('trace', 'causal', 'head_sweep', 'matrix')),
            source_job_id TEXT, source_analysis_id TEXT
        )""")
        connection.execute(
            "CREATE UNIQUE INDEX experiments_source_job ON experiments(source_job_id)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX experiments_source_analysis ON experiments(source_analysis_id)"
        )
        connection.execute("PRAGMA user_version = 2")
        for index, kind in enumerate(("trace", "causal", "head_sweep", "matrix")):
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(index),
                    kind,
                    "Retain notes",
                    "2026-01-01",
                    "model",
                    payload,
                    kind,
                    str(index) * 32 if kind in ("causal", "head_sweep") else None,
                    str(index) * 32 if kind == "matrix" else None,
                ),
            )
        original = connection.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    store = ExperimentStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM experiments ORDER BY id").fetchall() == original
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    saved = store.create_artifact(
        "Repeated tokens",
        "",
        "induction_probe",
        {
            "model_id": "model",
            "revision": "pinned",
            "dataset_fingerprint": "f" * 64,
        },
        source_job_id="f" * 32,
    )
    store.update(saved["id"], "Edited", "A research observation")
    reopened = ExperimentStore(path)
    repeated = reopened.create_artifact("Retry", "", "induction_probe", {}, source_job_id="f" * 32)
    assert repeated["id"] == saved["id"]
    assert repeated["name"] == "Edited" and repeated["notes"] == "A research observation"
    assert len(reopened.list()) == 5


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(lab, runtime, monkeypatch):
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(app.state, "induction", lab, raising=False)
    monkeypatch.setattr(
        app.state, "store", ExperimentStore(runtime.settings.data_dir / "records.db"), raising=False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as connection:
        yield connection


@pytest.mark.anyio
async def test_api_runs_and_saves_the_captured_execution_after_unloading(client, runtime):
    plan = await client.post("/api/induction/plan", json=REQUEST)
    assert plan.status_code == 200
    queued = await client.post(
        "/api/induction/probe",
        json={
            **REQUEST,
            "expected_execution_id": "a" * 32,
        },
    )
    assert queued.status_code == 202
    job_id = queued.json()["id"]
    for _ in range(200):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] in ("ready", "failed", "cancelled"):
            break
        await asyncio.sleep(0.01)
    assert job["status"] == "ready", job["message"]
    assert job["message"] == "Induction probe complete"
    result = job["result"]
    assert (
        result["complete"] and result["dataset_fingerprint"] == plan.json()["dataset_fingerprint"]
    )
    runtime.unload()
    assert not (await client.get("/api/induction/capabilities")).json()["loaded"]
    saved = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Head probe"}
    )
    assert saved.status_code == 201
    record = (await client.get(f"/api/experiments/{saved.json()['id']}")).json()
    assert record["kind"] == "induction_probe" and record["result"] == result
    reopened = ExperimentStore(runtime.settings.data_dir / "records.db")
    assert reopened.get(record["id"]) == record
    runtime.jobs.clear()
    repeated = await client.post(
        "/api/experiments/from-job", json={"job_id": job_id, "name": "Retry"}
    )
    assert repeated.json()["id"] == record["id"]
    assert (await client.get(f"/api/experiments/{record['id']}")).json()["result"] == result


@pytest.mark.anyio
async def test_api_rejects_stale_binding_and_strict_request_types(client, runtime):
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        for route in ("plan", "probe"):
            stale = await client.post(
                f"/api/induction/{route}",
                json={
                    **REQUEST,
                    "expected_execution_id": "b" * 32,
                },
            )
            assert stale.status_code == 409
        for invalid in (
            {"seed": True},
            {"repeats": "2"},
            {"layers": [False]},
            {"surprise": 1},
            {"causal_heads": 13},
        ):
            response = await client.post("/api/induction/probe", json={**REQUEST, **invalid})
            assert response.status_code == 422
    assert runtime.jobs == {}
