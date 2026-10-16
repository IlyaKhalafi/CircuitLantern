import asyncio
import uuid
from unittest.mock import patch

import httpx
import pytest
import torch

from circuit_lantern.feature_schemas import DictionaryTrainingRequest
from circuit_lantern.main import app
from circuit_lantern.services.execution import describe_execution
from circuit_lantern.services.features.inference import feature_intervention
from circuit_lantern.services.features.lab import FeatureLab
from circuit_lantern.services.features.store import DictionaryStore

DOCUMENTS = [
    "the capital of France is",
    "the capital of Poland is",
    "Poland is the capital of Poland",
    "France is the capital of France",
    "the capital is France",
    "the capital is Poland",
    "France of the is capital",
    "Poland of the is capital",
]
HARDWARE = {"ram_available_bytes": 8 * 1024**3, "experiment_disk_free_bytes": 8 * 1024**3}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def lab(runtime, monkeypatch):
    runtime.metadata.update(
        hidden_size=16,
        num_layers=2,
        architecture_supported=True,
        execution_id=uuid.uuid4().hex,
        execution_precision="float32",
        execution_config=describe_execution(runtime.model, "float32", "cpu"),
    )
    monkeypatch.setattr(
        "circuit_lantern.services.features.lab.hardware_snapshot", lambda _: dict(HARDWARE)
    )
    return FeatureLab(runtime, DictionaryStore(runtime.settings.data_dir / "dictionaries"))


@pytest.fixture
async def client(lab, monkeypatch):
    monkeypatch.setattr(app.state, "runtime", lab.runtime, raising=False)
    monkeypatch.setattr(app.state, "features", lab, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as connection:
        yield connection


def request(**changes):
    return DictionaryTrainingRequest(
        **dict(
            documents=DOCUMENTS, layers=[0], steps=16, batch_size=16, active_features=4, **changes
        )
    )


async def finished(client, job_id):
    for _ in range(500):
        response = await client.get("/api/jobs/" + job_id)
        result = response.json()
        if result["status"] in ("ready", "cancelled", "failed"):
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("The feature job did not finish.")


@pytest.mark.anyio
async def test_real_dictionary_training_api_preserves_teacher_and_restores_artifact(client, lab):
    before = {name: tensor.clone() for name, tensor in lab.runtime.model.state_dict().items()}
    random_state = torch.random.get_rng_state().clone()
    plan = await client.post("/api/features/plan", json=request().model_dump())
    assert plan.status_code == 200, plan.text
    assert plan.json()["compatible"]
    assert plan.json()["unique_train_tokens"] > 1
    started = await client.post("/api/features/train", json=request().model_dump())
    assert started.status_code == 202, started.text
    job = await finished(client, started.json()["id"])
    assert job["status"] == "ready", job["message"]
    assert job["result"]["complete"]
    listing = (await client.get("/api/features/dictionaries")).json()
    assert len(listing) == 1
    identifier = listing[0]["id"]
    saved = (await client.get("/api/features/dictionaries/" + identifier)).json()
    assert saved["training_summary"]["completed_steps"] == 16
    assert (
        saved["training_summary"]["tokens_seen"]
        == 16 * saved["training_summary"]["effective_batch_size"]
    )
    assert saved["corpus"]["fingerprint"] == plan.json()["corpus_fingerprint"]
    assert len(saved["feature_statistics"]) == 64
    assert len(saved["training_mean"]) == 16
    assert "feature_activation_counts" not in listing[0]["validation_metrics"]
    assert saved["method_record"]["implementation"] == "circuit_lantern_sae_v1"
    DictionaryStore(lab.store.directory).load_model(identifier)
    download = await client.get("/api/features/dictionaries/" + identifier + "/weights")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    import hashlib

    assert hashlib.sha256(download.content).hexdigest() == saved["weights_sha256"]

    torch.testing.assert_close(torch.random.get_rng_state(), random_state, rtol=0, atol=0)
    for name, tensor in lab.runtime.model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert all(not block._forward_hooks for block in lab.runtime._blocks())


@pytest.mark.anyio
async def test_prompt_inspection_unit_multiplier_is_identity_and_label_edits_preserve_measurements(
    client, lab
):
    job = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    identifier = job["result"]["dictionaries"][0]["id"]
    path = "/api/features/dictionaries/" + identifier
    inspected = await client.post(path + "/inspect", json={"prompt": "the capital of France"})
    assert inspected.status_code == 200, inspected.text
    data = inspected.json()
    assert len(data["tokens"]) == 4
    feature = data["features"][0]["index"]
    identity = await client.post(
        path + "/inspect",
        json={"prompt": "the capital of France", "feature": feature, "multiplier": 1},
    )
    intervention = identity.json()["intervention"]
    assert intervention["target_log_probability_change"] == 0
    assert intervention["kl_baseline_to_patched"] == 0
    before = (await client.get(path)).json()
    changed = await client.patch(
        path + f"/features/{feature}", json={"label": "France {raw}", "notes": "Human observation"}
    )
    assert changed.status_code == 200
    after = (await client.get(path)).json()
    assert after["feature_labels"][str(feature)]["label"] == "France {raw}"
    assert before["weights_sha256"] == after["weights_sha256"]
    assert before["feature_statistics"] == after["feature_statistics"]
    assert (
        await client.post(path + "/inspect", json={"prompt": "the capital", "position": 100})
    ).status_code == 400
    assert (
        await client.post(
            path + "/inspect", json={"prompt": "the capital", "target_token_id": 1000}
        )
    ).status_code == 400
    lab.runtime.metadata["execution_config"]["precision"] = "int4"
    assert (await client.post(path + "/inspect", json={"prompt": "the capital"})).status_code == 400


def test_feature_intervention_subtracts_only_its_decoded_contribution(lab):
    from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig

    dictionary = SparseAutoencoder(16, TrainingConfig(features=32))
    generator = torch.Generator().manual_seed(20)
    hidden = torch.randn(1, 4, 16, generator=generator)
    with torch.no_grad():
        encoded = dictionary.encode(hidden[0])
        feature = int(encoded.sum(0).argmax())
        changed = feature_intervention(dictionary, feature, 0, 2)(hidden)
        expected = hidden.clone()
        expected[0, 2] -= encoded[2, feature] * dictionary.feature_direction(feature)
    torch.testing.assert_close(changed, expected)
    torch.testing.assert_close(changed[0, :2], hidden[0, :2], rtol=0, atol=0)
    torch.testing.assert_close(
        hidden[0, 2] - dictionary.decode(encoded)[2],
        changed[0, 2]
        - dictionary.decode(encoded * torch.tensor([0 if i == feature else 1 for i in range(32)]))[
            2
        ],
    )


@pytest.mark.anyio
async def test_reconstruction_faithfulness_handles_single_token_without_inventing_loss(client):
    job = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    path = "/api/features/dictionaries/" + job["result"]["dictionaries"][0]["id"] + "/evaluate"
    single = (await client.post(path, json={"prompt": "France"})).json()
    assert single["scored_tokens"] == 0
    assert single["loss_recovered"] is None
    assert single["baseline_cross_entropy"] is None
    result = (await client.post(path, json={"prompt": "the capital of France"})).json()
    assert result["scored_tokens"] == 3
    assert result["baseline_cross_entropy"] > 0
    assert result["reconstructed_cross_entropy"] > 0
    gap = result["mean_ablated_cross_entropy"] - result["baseline_cross_entropy"]
    assert result["loss_recovered"] == pytest.approx(
        (result["mean_ablated_cross_entropy"] - result["reconstructed_cross_entropy"]) / gap
    )


@pytest.mark.anyio
async def test_feature_graph_uses_only_later_layer_edges_and_persists_after_unload(client, lab):
    payload = request().model_dump()
    payload["layers"] = [0, 1]
    trained = await finished(
        client, (await client.post("/api/features/train", json=payload)).json()["id"]
    )
    assert trained["status"] == "ready", trained["message"]
    identifiers = [item["id"] for item in trained["result"]["dictionaries"]]
    from circuit_lantern.services.features import circuits

    observed = []
    original_forward = circuits.forward

    def record_forward(*args, **kwargs):
        logits, activations = original_forward(*args, **kwargs)
        observed.append({key: values.clone() for key, values in activations.items()})
        return logits, activations

    with patch.object(circuits, "forward", side_effect=record_forward):
        started = await client.post(
            "/api/features/circuits",
            json={
                "dictionary_ids": identifiers,
                "prompt": "the capital of France",
                "features_per_layer": 2,
            },
        )
        assert started.status_code == 202, started.text
        job = await finished(client, started.json()["id"])
    assert job["status"] == "ready", job["message"]
    result = job["result"]
    assert result["complete"]
    assert len(result["measurements"]) == len(result["nodes"]) > 0
    nodes = {node["id"]: node for node in result["nodes"]}
    assert result["method"]["edge_position_scope"] == "destination_dictionary_eligible_positions"
    assert len(observed) == 1 + len(result["measurements"])
    differs_from_all_positions = False
    for node in nodes.values():
        positions = result["eligible_positions"][node["dictionary_id"]]
        assert 0 not in positions
        values = observed[0][node["dictionary_id"]][:, node["feature"]]
        expected = float(values[positions].mean())
        assert node["mean_activation"] == pytest.approx(expected)
        differs_from_all_positions |= abs(expected - float(values.mean())) > 1e-6
    assert differs_from_all_positions
    assert any(measurement["downstream_changes"] for measurement in result["measurements"])
    for index, measurement in enumerate(result["measurements"], start=1):
        for edge in measurement["downstream_changes"]:
            destination = nodes[edge["target"]]
            assert nodes[edge["source"]]["layer"] < destination["layer"]
            identifier, feature = destination["dictionary_id"], destination["feature"]
            positions = result["eligible_positions"][identifier]
            expected = float(observed[index][identifier][positions, feature].mean()) - float(
                observed[0][identifier][positions, feature].mean()
            )
            assert edge["activation_change"] == pytest.approx(expected, abs=1e-8)
    assert all(not block._forward_hooks for block in lab.runtime._blocks())
    lab.runtime.unload()
    durable = (await client.get("/api/features/runs/" + result["id"])).json()
    assert result == durable
    assert len((await client.get("/api/features/runs")).json()) == 1


@pytest.mark.anyio
async def test_low_memory_and_invalid_layers_fail_before_activation_allocation(
    client, lab, monkeypatch
):
    with patch("circuit_lantern.services.features.lab.capture_activations") as capture:
        monkeypatch.setattr(
            "circuit_lantern.services.features.lab.hardware_snapshot",
            lambda _: dict(HARDWARE, ram_available_bytes=1),
        )
        response = await client.post("/api/features/train", json=request().model_dump())
        assert response.status_code == 400
        assert "RAM" in response.json()["detail"]
        capture.assert_not_called()
    payload = request().model_dump()
    payload["layers"] = [3]
    assert (await client.post("/api/features/plan", json=payload)).status_code == 400


@pytest.mark.anyio
async def test_inference_excludes_untrained_positions_and_marks_prompt_split(client, lab):
    trained = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    identifier = trained["result"]["dictionaries"][0]["id"]
    path = "/api/features/dictionaries/" + identifier
    manifest = lab.store.get(identifier)
    training_text = next(
        document["text"]
        for document in manifest["corpus"]["documents"]
        if document["split"] == "train"
    )
    # A compatible reload receives a new execution identity. Evidence must keep
    # current inference and original dictionary training identities separate.
    lab.runtime.metadata["execution_id"] = uuid.uuid4().hex
    inspected = (await client.post(path + "/inspect", json={"prompt": training_text})).json()
    assert inspected["format"] == "circuit-lantern-feature-inspection-v1"
    assert inspected["execution_id"] != manifest["execution_id"]
    assert (
        inspected["dictionary_source"]["training_execution"]["execution_id"]
        == manifest["execution_id"]
    )
    assert inspected["dictionary_source"]["weights_sha256"] == manifest["weights_sha256"]
    assert inspected["dictionary_source"]["manifest_sha256"] == manifest["manifest_sha256"]
    assert inspected["dictionary_source"]["corpus_fingerprint"] == manifest["corpus"]["fingerprint"]
    assert inspected["dictionary_source"]["training"] == manifest["training"]
    assert inspected["dictionary_source"]["capture_point"] == "residual_post"
    assert inspected["prompt_split"] == "train"
    assert 0 not in inspected["eligible_positions"]
    assert inspected["active_features_per_token"][0] is None
    assert all(feature["activations"][0] is None for feature in inspected["features"])
    feature = inspected["features"][0]["index"]
    excluded = await client.post(
        path + "/inspect", json={"prompt": training_text, "feature": feature, "position": 0}
    )
    assert excluded.status_code == 400
    from circuit_lantern.services.features import inference

    original = inference.forward
    splices = []

    def checked_forward(runtime, inputs, dictionaries, intervention=None):
        if intervention:
            layer, operation = intervention

            def checked(hidden):
                changed = operation(hidden)
                torch.testing.assert_close(changed[:, 0], hidden[:, 0], rtol=0, atol=0)
                splices.append(True)
                return changed

            intervention = (layer, checked)
        return original(runtime, inputs, dictionaries, intervention)

    with patch.object(inference, "forward", side_effect=checked_forward):
        evaluated = await client.post(path + "/evaluate", json={"prompt": training_text})
    assert evaluated.status_code == 200
    assert evaluated.json()["prompt_split"] == "train"
    assert evaluated.json()["format"] == "circuit-lantern-feature-evaluation-v1"
    assert evaluated.json()["dictionary_source"] == inspected["dictionary_source"]
    from datetime import datetime

    for result in (inspected, evaluated.json()):
        assert datetime.fromisoformat(result["captured_at"]).tzinfo is not None
    assert len(splices) == 2


def test_bfloat16_rounding_is_measured_instead_of_assumed_exact():
    from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig

    dictionary = SparseAutoencoder(2, TrainingConfig(features=2))
    with torch.no_grad():
        dictionary.encoder.zero_()
        dictionary.encoder_bias.fill_(0.0001)
        operation = feature_intervention(dictionary, 0, 0, None)
        hidden = torch.ones(1, 3, 2, dtype=torch.bfloat16)
        changed = operation(hidden)
    torch.testing.assert_close(changed, hidden, rtol=0, atol=0)
    assert operation.measurement["intended_delta_norm"] > 0
    assert operation.measurement["realized_delta_norm"] == 0
    assert operation.measurement["relative_rounding_error"] == 1


def test_capabilities_remain_readable_during_a_model_operation(lab):
    with lab.runtime.lock:
        capabilities = lab.capabilities()
    assert capabilities["supported"]
    assert capabilities["layers"] == [0, 1]


@pytest.mark.anyio
async def test_shutdown_cancels_running_dictionary_jobs_before_waiting_for_workers(
    lab, monkeypatch
):
    from fastapi import FastAPI

    from circuit_lantern import main

    monkeypatch.setattr(main, "get_settings", lambda: lab.runtime.settings)
    monkeypatch.setattr(main, "ModelRuntime", lambda settings: lab.runtime)
    application = FastAPI()
    import threading

    entered = threading.Event()

    def worker(payload, job):
        import time

        entered.set()
        while not job.get("cancel_requested"):
            time.sleep(0.01)
        return {"complete": False}

    async with main.lifespan(application):
        queued = lab.runtime._start_sweep("sae_training", {}, worker)
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
    assert lab.runtime.job(queued["id"])["status"] == "cancelled"


@pytest.mark.anyio
async def test_cancellation_keeps_only_completed_optimizer_steps_and_marks_partial_dictionary(
    client, lab
):
    import threading

    from circuit_lantern.services.features import lab as module

    entered = threading.Event()
    release = threading.Event()
    original = module.train_dictionary

    def blocked(*args, **kwargs):
        original_progress = kwargs["progress"]

        def progress(row):
            original_progress(row)
            if row["step"] >= 5 and not entered.is_set():
                entered.set()
                assert release.wait(timeout=5)

        kwargs["progress"] = progress
        return original(*args, **kwargs)

    payload = request().model_dump()
    payload["steps"] = 128
    with patch.object(module, "train_dictionary", side_effect=blocked):
        started = await client.post("/api/features/train", json=payload)
        identifier = started.json()["id"]
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            response = await client.post("/api/jobs/" + identifier + "/cancel")
            assert response.json()["cancel_requested"]
        finally:
            release.set()
        job = await finished(client, identifier)
    assert job["status"] == "cancelled", job
    assert not job["result"]["complete"]
    dictionary_id = job["result"]["dictionaries"][0]["id"]
    manifest = lab.store.get(dictionary_id)
    assert manifest["training_summary"]["cancelled"]
    assert manifest["training_summary"]["completed_steps"] == 5
    assert (
        manifest["training_summary"]["tokens_seen"]
        == 5 * manifest["training_summary"]["effective_batch_size"]
    )
    capture = await client.post(
        "/api/features/dictionaries/" + dictionary_id + "/inspect",
        json={"prompt": "the capital of France"},
    )
    assert capture.json()["dictionary_training"]["cancelled"]


@pytest.mark.anyio
async def test_separate_dictionary_seed_keeps_the_corpus_split_fixed(client):
    first = request().model_dump()
    first["seed"] = 12
    first["split_seed"] = 7
    first["batch_size"] = 128
    second = dict(first, seed=13)
    a = await client.post("/api/features/plan", json=first)
    b = await client.post("/api/features/plan", json=second)
    assert a.json()["corpus_fingerprint"] == b.json()["corpus_fingerprint"]
    assert a.json()["warnings"]


@pytest.mark.anyio
async def test_missing_dictionary_weight_file_has_a_clean_client_error(client, lab):
    job = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    identifier = job["result"]["dictionaries"][0]["id"]
    lab.store._weights_path(identifier).unlink()
    response = await client.post(
        "/api/features/dictionaries/" + identifier + "/inspect",
        json={"prompt": "the capital of France"},
    )
    assert response.status_code == 400
    assert "missing" in response.json()["detail"]


@pytest.mark.anyio
async def test_malformed_circuit_identifiers_do_not_start_jobs(client, lab):
    before = len(lab.runtime.jobs)
    response = await client.post(
        "/api/features/circuits", json={"dictionary_ids": ["../weights"], "prompt": "the capital"}
    )
    assert response.status_code == 422
    assert len(lab.runtime.jobs) == before


@pytest.mark.anyio
async def test_empty_feature_graph_fails_without_publishing_a_complete_run(client, lab):
    trained = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    identifier = trained["result"]["dictionaries"][0]["id"]
    manifest, dictionary = lab.store.load_model(identifier)
    with torch.no_grad():
        dictionary.encoder.zero_()
    with patch.object(lab, "load_dictionaries", return_value=[(manifest, dictionary)]):
        queued = await client.post(
            "/api/features/circuits",
            json={"dictionary_ids": [identifier], "prompt": "the capital of France"},
        )
        job = await finished(client, queued.json()["id"])
    assert job["status"] == "failed"
    assert "No selected feature is active" in job["message"]
    assert not lab.store.runs()
    assert all(not block._forward_hooks for block in lab.runtime._blocks())


@pytest.mark.anyio
async def test_prompt_feature_pair_uses_one_baseline_forward_and_validates_before_inference(
    client, lab
):
    from circuit_lantern.services.features import inference

    trained = await finished(
        client, (await client.post("/api/features/train", json=request().model_dump())).json()["id"]
    )
    identifier = trained["result"]["dictionaries"][0]["id"]
    path = "/api/features/dictionaries/" + identifier + "/inspect"
    observed = []
    original = inference.forward

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    with patch.object(inference, "forward", side_effect=recording) as forward:
        response = await client.post(
            path, json={"prompt": "the capital of France", "comparison_features": [0, 1]}
        )
    assert response.status_code == 200, response.text
    assert forward.call_count == 1
    data = response.json()
    comparison = data["feature_comparison"]
    assert comparison["feature_ids"] == [0, 1]
    assert data["eligible_positions"] == [1, 2, 3]
    assert comparison["eligible_position_count"] == 3
    assert {0, 1}.issubset(row["index"] for row in data["features"])
    assert all(row["activations"][0] is None for row in data["features"])
    raw = observed[0][1][identifier].double().numpy()[1:, :2]
    import numpy as np

    expected_active = (raw > 0).sum(axis=0)
    assert comparison["active_counts"] == expected_active.tolist()
    if all(np.std(raw, axis=0) > 0):
        assert comparison["pearson_correlation"] == pytest.approx(
            np.corrcoef(raw.T)[0, 1], abs=1e-12
        )
    else:
        assert comparison["pearson_correlation"] is None
    assert "Baseline normalized" in comparison["scope"]
    assert data["dictionary_source"]["id"] == identifier
    width = lab.store.get(identifier)["features"]
    with patch.object(inference, "forward") as forward:
        for pair in ([0, 0], [0, width]):
            invalid = await client.post(
                path, json={"prompt": "the capital of France", "comparison_features": pair}
            )
            assert invalid.status_code == 400
        for pair in ([0], [0, 1, 2], [-1, 0], [True, 0], [0.5, 1]):
            invalid = await client.post(
                path, json={"prompt": "the capital of France", "comparison_features": pair}
            )
            assert invalid.status_code == 422
    forward.assert_not_called()
    assert all(not block._forward_hooks for block in lab.runtime._blocks())
