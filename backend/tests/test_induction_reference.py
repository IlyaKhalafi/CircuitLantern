import asyncio
from unittest.mock import patch

import httpx
import pytest
import torch
from induction_fixtures import capture

from circuit_lantern.induction_schemas import InductionProbeRequest
from circuit_lantern.main import app
from circuit_lantern.services.induction.lab import InductionLab
from circuit_lantern.services.induction.reference import ReferenceProbeNotFoundError
from circuit_lantern.services.store import ExperimentStore

HARDWARE = {"ram_available_bytes": 2**40, "experiment_disk_free_bytes": 2**40}


@pytest.fixture
def setup(runtime, monkeypatch):
    source = capture(runtime, causal=1)
    store = ExperimentStore(runtime.settings.data_dir / "reference.db")
    record = store.create_artifact("Base probe", "", "induction_probe", source)
    lab = InductionLab(runtime, store)
    request = InductionProbeRequest(
        block_length=4,
        repeats=2,
        examples=3,
        causal_heads=1,
        seed=19,
        reference_experiment_id=record["id"],
    )
    monkeypatch.setattr(
        "circuit_lantern.services.induction.lab.hardware_snapshot", lambda _: dict(HARDWARE)
    )
    return lab, store, record, source, request


def test_matching_plan_is_read_only_binds_exact_candidates_and_budgets_retained_evidence(
    setup, runtime
):
    lab, store, record, source, request = setup
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        matched = lab.plan(request)
        standalone = lab.plan(request.model_copy(update={"reference_experiment_id": None}))
    assert matched["compatible"]
    assert matched["dataset_fingerprint"] == source["dataset_fingerprint"]
    assert matched["comparison_identity_sha256"] == source["comparison_identity_sha256"]
    assert matched["causal_candidates"] == [
        {"layer": source["heads"][0]["layer"], "head": source["heads"][0]["head"]}
    ]
    assert matched["requested_forward_passes"] == 6
    size = store.payload_size(record["id"])
    assert size > 0
    assert matched["retained_reference_bytes"] == size
    assert matched["estimated_host_bytes"] - standalone["estimated_host_bytes"] == size * 12
    assert store.payload_size("missing") is None


def test_matching_replays_reference_ids_when_special_token_designations_change(setup, runtime):
    lab, _, _, source, request = setup
    sampled = {token for probe in source["dataset"]["probes"] for token in probe["token_ids"]}
    excluded = set(runtime.tokenizer.all_special_ids) | sampled
    token = next(
        name for name, index in runtime.tokenizer.get_vocab().items() if index not in excluded
    )
    original_mapping = runtime.tokenizer.get_vocab()
    runtime.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    assert runtime.tokenizer.get_vocab() == original_mapping
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        fresh = lab.plan(request.model_copy(update={"reference_experiment_id": None}))
        matched = lab.plan(request)
    assert fresh["dataset_fingerprint"] != source["dataset_fingerprint"]
    assert matched["dataset_fingerprint"] == source["dataset_fingerprint"]
    assert matched["comparison_identity_sha256"] == source["comparison_identity_sha256"]


def test_source_note_changes_do_not_invalidate_bound_scientific_settings(setup):
    lab, store, record, _, request = setup
    original = lab.plan(request)
    store.update(record["id"], "Renamed base", "New observation")
    changed = lab.plan(request)
    assert original["reference_signature"] == changed["reference_signature"]
    assert changed["reference_probe"]["name"] == "Renamed base"


def test_source_resource_check_precedes_json_decoding_and_tokenization(setup, runtime, monkeypatch):
    lab, store, _, _, request = setup
    monkeypatch.setattr(
        "circuit_lantern.services.induction.lab.hardware_snapshot",
        lambda _: {
            "ram_available_bytes": 1,
            "experiment_disk_free_bytes": 2**40,
        },
    )
    with (
        patch.object(store, "get", side_effect=AssertionError("Unexpected reference decode")),
        patch.object(
            runtime.tokenizer,
            "get_vocab",
            side_effect=AssertionError("Unexpected vocabulary allocation"),
        ),
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
    ):
        assert not lab.plan(request)["compatible"]
    assert runtime.jobs == {}


@pytest.mark.parametrize(
    "change,message",
    [
        ({"seed": 18}, "different inputs"),
        ({"include_start_token": False}, "different inputs"),
    ],
)
def test_reference_requires_exact_input_configuration_before_teacher_work(
    setup, runtime, change, message
):
    lab, _, _, _, request = setup
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError, match=message),
    ):
        lab.start(request.model_copy(update=change))
    assert runtime.jobs == {}


def test_deleting_reference_before_start_is_reported_without_forward(setup, runtime):
    lab, store, record, _, request = setup
    assert store.delete(record["id"])
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ReferenceProbeNotFoundError),
    ):
        lab.start(request)


def test_matching_candidates_remain_at_reference_coordinates_when_the_model_head_ranking_changes(
    setup, runtime
):
    lab, store, record, source, request = setup
    top = source["heads"][0]
    layer, head = top["layer"], top["head"]
    other = 1 - head
    attention = runtime._blocks()[layer].attn
    order = torch.arange(16)
    order[head * 8 : (head + 1) * 8] = torch.arange(other * 8, (other + 1) * 8)
    order[other * 8 : (other + 1) * 8] = torch.arange(head * 8, (head + 1) * 8)
    with torch.no_grad():
        weight, bias = attention.c_attn.weight.clone(), attention.c_attn.bias.clone()
        for offset in (0, 16, 32):
            attention.c_attn.weight[:, offset : offset + 16] = weight[:, offset + order]
            attention.c_attn.bias[offset : offset + 16] = bias[offset + order]
        attention.c_proj.weight.copy_(attention.c_proj.weight.clone()[order])
    runtime.metadata.update(id="permuted-checkpoint", revision="permuted", execution_id="c" * 32)
    planned = lab.plan(request)
    payload = request.model_dump()
    payload.update(
        expected_probe_fingerprint=planned["dataset_fingerprint"],
        expected_comparison_identity=planned["comparison_identity_sha256"],
        expected_reference_signature=planned["reference_signature"],
    )
    result = lab._measure(payload, {})
    assert result["complete"] and result["completed_forwards"] == 6
    assert (result["heads"][0]["layer"], result["heads"][0]["head"]) == (layer, other)
    assert [(row["layer"], row["head"]) for row in result["causal_results"]] == [(layer, head)]
    assert result["reference_experiment_id"] == record["id"]
    assert result["causal_candidate_policy"] == "reference_prefix_ranking"
    assert store.get(record["id"])["result"] == source


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_matching_reference_jobs_and_deleted_source_errors_through_the_api(
    setup, runtime, monkeypatch
):
    lab, store, record, source, request = setup
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setattr(app.state, "induction", lab, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        queued = await client.post("/api/induction/probe", json=request.model_dump())
        assert queued.status_code == 202, queued.text
        for _ in range(200):
            job = (await client.get("/api/jobs/" + queued.json()["id"])).json()
            if job["status"] in ("ready", "failed", "cancelled"):
                break
            await asyncio.sleep(0.01)
        assert job["status"] == "ready", job["message"]
        assert job["result"]["reference_experiment_id"] == record["id"]
        assert job["result"]["dataset_fingerprint"] == source["dataset_fingerprint"]
        store.delete(record["id"])
        missing = await client.post("/api/induction/plan", json=request.model_dump())
        assert missing.status_code == 404
        assert "may have been deleted" in missing.json()["detail"]


@pytest.mark.anyio
async def test_comparison_ram_check_precedes_evidence_decode(setup, monkeypatch):
    _, store, record, source, _ = setup
    second = store.create_artifact("Second", "", "induction_probe", source)
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setattr(
        "circuit_lantern.services.induction.planning.memory_budget",
        lambda *args: {"ram_available_bytes": 1},
    )
    with patch.object(store, "get", side_effect=AssertionError("Unexpected evidence decode")):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get(
                "/api/induction/comparison",
                params={"selected": record["id"], "reference": second["id"]},
            )
    assert response.status_code == 400
    assert "Insufficient available RAM" in response.json()["detail"]
