"""Real teacher capture, model-free joint training and immutable API evidence."""

import asyncio
import hashlib
import uuid

import httpx
import pytest
import torch
from safetensors.torch import load

from circuit_lantern.main import app
from circuit_lantern.services.crosscoders.lab import CrosscoderLab
from circuit_lantern.services.execution import describe_execution
from circuit_lantern.services.features.lab import FeatureLab
from circuit_lantern.services.features.store import DictionaryStore
from circuit_lantern.services.tensor_artifacts import TensorArtifactStore

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
        architecture="gpt2",
        architecture_supported=True,
        execution_id=uuid.uuid4().hex,
        execution_precision="float32",
        execution_config=describe_execution(runtime.model, "float32", "cpu"),
    )
    monkeypatch.setattr(
        "circuit_lantern.services.crosscoders.lab.hardware_snapshot",
        lambda _: dict(HARDWARE),
    )
    return CrosscoderLab(
        runtime, TensorArtifactStore(runtime.settings.data_dir / "shared", 1024**2)
    )


@pytest.fixture
async def client(lab, monkeypatch):
    monkeypatch.setattr(app.state, "runtime", lab.runtime, raising=False)
    monkeypatch.setattr(app.state, "crosscoders", lab, raising=False)
    monkeypatch.setattr(
        app.state,
        "features",
        FeatureLab(lab.runtime, DictionaryStore(lab.runtime.settings.data_dir / "dictionaries")),
        raising=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as value:
        yield value


async def finished(client, job):
    for _ in range(500):
        response = await client.get("/api/jobs/" + job["id"])
        value = response.json()
        if value["status"] in ("ready", "cancelled", "failed"):
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("The joint feature job did not finish.")


async def capture_pair(client, lab):
    body = {
        "name": "Reference",
        "documents": DOCUMENTS,
        "layer": 0,
        "max_samples": 64,
        "expected_execution_id": lab.runtime.metadata["execution_id"],
    }
    response = await client.post("/api/crosscoders/captures", json=body)
    assert response.status_code == 202, response.text
    first = await finished(client, response.json())
    assert first["status"] == "ready", first
    reference = first["result"]["captures"][0]
    response = await client.post(
        "/api/crosscoders/captures/" + reference["id"] + "/match",
        json={
            "name": "Compared",
            "layer": 0,
            "expected_execution_id": lab.runtime.metadata["execution_id"],
        },
    )
    assert response.status_code == 202, response.text
    second = await finished(client, response.json())
    assert second["status"] == "ready", second
    return reference, second["result"]["captures"][0]


@pytest.mark.anyio
async def test_capture_joint_training_and_exports_work_after_teacher_unload(client, lab):
    teacher = lab.runtime.model
    original = {name: value.clone() for name, value in teacher.state_dict().items()}
    forwards = []
    handle = teacher.base_model.register_forward_pre_hook(lambda *args: forwards.append(1))
    try:
        body = {"documents": DOCUMENTS, "layer": 0, "max_samples": 64}
        plan = await client.post("/api/crosscoders/captures/plan", json=body)
        assert plan.status_code == 200, plan.text
        assert plan.json()["compatible"] and not forwards
        started = await client.post("/api/crosscoders/captures", json=body)
        first_job = await finished(client, started.json())
        assert first_job["status"] == "ready", first_job
        reference = first_job["result"]["captures"][0]
        assert len(forwards) == plan.json()["teacher_forward_passes"]
        for name, value in teacher.state_dict().items():
            assert torch.equal(value, original[name])
        assert all(not block._forward_hooks for block in lab.runtime._blocks())
        first_execution = lab.runtime.metadata["execution_id"]

        with torch.no_grad():
            teacher.transformer.h[0].mlp.c_fc.weight.add_(0.02)
        compared_state = {name: value.clone() for name, value in teacher.state_dict().items()}
        lab.runtime.metadata.update(
            id="local-finetuned", revision="fine-revision", execution_id=uuid.uuid4().hex
        )
        compared_execution = lab.runtime.metadata["execution_id"]
        before_match = len(forwards)
        matched = {"name": "Compared", "layer": 0, "expected_execution_id": compared_execution}
        path = "/api/crosscoders/captures/" + reference["id"]
        second_plan = await client.post(path + "/match/plan", json=matched)
        assert second_plan.status_code == 200, second_plan.text
        assert len(forwards) == before_match
        started = await client.post(path + "/match", json=matched)
        second_job = await finished(client, started.json())
        assert second_job["status"] == "ready", second_job
        compared = second_job["result"]["captures"][0]
        assert len(forwards) == before_match + second_plan.json()["teacher_forward_passes"]
        assert reference["corpus_fingerprint"] == compared["corpus_fingerprint"]
        assert reference["source"]["execution_id"] == first_execution
        assert compared["source"]["execution_id"] == compared_execution
        for name, value in teacher.state_dict().items():
            assert torch.equal(value, compared_state[name])
        exported_capture = await client.get(path + "/tensors")
        assert hashlib.sha256(exported_capture.content).hexdigest() == reference["tensors_sha256"]
        activations = load(exported_capture.content)["activations"]
        assert activations.shape == (reference["sample_count"], 16)
        assert not activations.is_cuda
        assert (await client.delete("/api/runtime")).status_code == 204
        assert not lab.runtime.state()["loaded"]

        request = {
            "capture_ids": [reference["id"], compared["id"]],
            "expansion": 2,
            "steps": 16,
            "batch_size": 8,
            "l1_coefficient": 0.02,
        }
        planned = await client.post("/api/crosscoders/plan", json=request)
        assert planned.status_code == 200 and planned.json()["compatible"], planned.text
        started = await client.post("/api/crosscoders/train", json=request)
        assert started.status_code == 202, started.text
        assert "execution_id" not in started.json() and "model_id" not in started.json()
        trained = await finished(client, started.json())
        assert trained["status"] == "ready", trained
        assert len(forwards) == before_match + second_plan.json()["teacher_forward_passes"]
        identifier = trained["result"]["crosscoders"][0]["id"]
        url = "/api/crosscoders/" + identifier
        overview = (await client.get(url + "/overview")).json()
        assert "feature_statistics" not in overview
        assert all("training_mean" not in bank for bank in overview["banks"])
        assert all(
            "training_mean" not in bank
            for bank in (await client.get("/api/crosscoders")).json()[0]["banks"]
        )
        assert overview["training_summary"]["completed_steps"] == 16
        assert overview["banks"][0]["source"]["execution_id"] == first_execution
        assert overview["banks"][1]["source"]["execution_id"] == compared_execution
        page = (await client.get(url + "/features?limit=5")).json()
        assert page["total"] == 32 and len(page["features"]) == 5
        feature = page["features"][0]["index"]
        detail = (await client.get(url + f"/features/{feature}")).json()
        for example in detail["examples"]:
            assert 0 <= example["active_offset"] < len(example["tokens"])
            assert example["split"] in ("train", "validation")
        label = await client.patch(
            url + f"/features/{feature}",
            json={"label": "Shared {raw}", "notes": "Observed context"},
        )
        assert label.status_code == 200
        searched = (await client.get(url + "/features?q=shared")).json()
        assert searched["total"] == 1 and searched["features"][0]["index"] == feature
        bundle = (await client.get(url + "/export")).json()
        assert bundle["artifact"]["tensors_sha256"] == overview["tensors_sha256"]
        assert str(feature) in bundle["annotations"]
        weights = await client.get(url + "/weights")
        assert hashlib.sha256(weights.content).hexdigest() == overview["tensors_sha256"]
        manifest, restored = lab.load_model(identifier)
        assert restored.config.features == 32 and all(
            value.device.type == "cpu" for value in restored.parameters()
        )
        assert manifest["feature_statistics"][feature]["frequency"] == detail["frequency"]

        assert (await client.delete(path)).status_code == 204
        assert (await client.get(url + f"/features/{feature}")).status_code == 200
        assert (await client.get(url + "/export")).status_code == 200
        assert (await client.delete(url)).status_code == 204
        assert (await client.get(url + "/overview")).status_code == 404
    finally:
        handle.remove()


@pytest.mark.anyio
async def test_mismatched_tokenization_and_stale_execution_stop_before_forward(client, lab):
    from tokenizers.processors import TemplateProcessing

    reference, _ = await capture_pair(client, lab)
    forwards = []
    handle = lab.runtime.model.base_model.register_forward_pre_hook(
        lambda *args: forwards.append(1)
    )
    try:
        lab.runtime.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="<eos> $A",
            special_tokens=[("<eos>", lab.runtime.tokenizer.eos_token_id)],
        )
        path = "/api/crosscoders/captures/" + reference["id"] + "/match/plan"
        response = await client.post(path, json={"layer": 0})
        assert response.status_code == 400 and "Tokenization" in response.json()["detail"]
        assert not forwards
        response = await client.post(
            "/api/crosscoders/captures",
            json={
                "documents": DOCUMENTS,
                "layer": 0,
                "expected_execution_id": "a" * 32,
            },
        )
        assert response.status_code == 409 and not forwards
    finally:
        handle.remove()


@pytest.mark.anyio
async def test_capture_cancellation_never_publishes_partial_activation_rows(client, lab):
    forwards = []

    def cancel(*args):
        forwards.append(1)
        for job in lab.runtime.jobs.values():
            if job.get("kind") == "activation_capture":
                job.request_cancel()

    handle = lab.runtime.model.base_model.register_forward_pre_hook(cancel)
    try:
        response = await client.post(
            "/api/crosscoders/captures", json={"documents": DOCUMENTS, "layer": 0}
        )
        job = await finished(client, response.json())
        assert job["status"] == "cancelled" and len(forwards) == 1
        assert (await client.get("/api/crosscoders/captures")).json() == []
        assert all(not block._forward_hooks for block in lab.runtime._blocks())
    finally:
        handle.remove()


@pytest.mark.anyio
async def test_preflight_rejects_resource_and_request_errors_without_forward(
    client, lab, monkeypatch
):
    forwards = []
    handle = lab.runtime.model.base_model.register_forward_pre_hook(
        lambda *args: forwards.append(1)
    )
    try:
        monkeypatch.setattr(
            "circuit_lantern.services.crosscoders.lab.hardware_snapshot",
            lambda _: dict(HARDWARE, ram_available_bytes=1),
        )
        request = {"documents": DOCUMENTS, "layer": 0}
        response = await client.post("/api/crosscoders/captures/plan", json=request)
        assert response.status_code == 200 and not response.json()["compatible"]
        assert (await client.post("/api/crosscoders/captures", json=request)).status_code == 400
        assert (
            await client.post(
                "/api/crosscoders/train",
                json={
                    "capture_ids": ["a" * 32, "b" * 32],
                    "l1_coefficient": True,
                },
            )
        ).status_code == 422
        assert not forwards and not lab.runtime.jobs
    finally:
        handle.remove()


@pytest.mark.anyio
async def test_capture_capabilities_and_missing_tensor_download(client, lab):
    capabilities = (await client.get("/api/crosscoders/capabilities")).json()
    assert capabilities["max_capture_tokens"] == lab.runtime.settings.max_prompt_tokens
    assert capabilities["references"][0]["url"].endswith("/2024/crosscoders/index.html")
    reference, _ = await capture_pair(client, lab)
    path = lab.store.directory / (reference["id"] + ".safetensors")
    path.unlink()
    response = await client.get("/api/crosscoders/captures/" + reference["id"] + "/tensors")
    assert response.status_code == 404
    assert "tensor file" in response.json()["detail"]


@pytest.mark.anyio
async def test_cancelled_joint_training_retains_a_checksummed_partial_dictionary(
    client, lab, monkeypatch
):
    from circuit_lantern.services.crosscoders.training import train_crosscoder

    reference, compared = await capture_pair(client, lab)
    lab.runtime.unload()

    def stop_after_observed_steps(train, validation, config, *, device, progress, cancelled):
        def observed(row):
            progress(row)
            if row["step"] >= 3:
                for job in lab.runtime.jobs.values():
                    if job["kind"] == "crosscoder_training":
                        job.request_cancel()

        return train_crosscoder(
            train, validation, config, device=device, progress=observed, cancelled=cancelled
        )

    monkeypatch.setattr(
        "circuit_lantern.services.crosscoders.lab.train_crosscoder", stop_after_observed_steps
    )
    response = await client.post(
        "/api/crosscoders/train",
        json={
            "capture_ids": [reference["id"], compared["id"]],
            "expansion": 1,
            "steps": 16,
        },
    )
    assert response.status_code == 202
    job = await finished(client, response.json())
    assert job["status"] == "cancelled"
    assert job["result"]["complete"] is False
    artifact = job["result"]["crosscoders"][0]
    assert artifact["training_summary"]["completed_steps"] == 3
    assert artifact["training_summary"]["cancelled"] is True
    manifest, dictionary = lab.load_model(artifact["id"])
    assert manifest["training_summary"] == artifact["training_summary"]
    assert dictionary.config.features == artifact["features"]
    assert (await client.get("/api/crosscoders/" + artifact["id"] + "/weights")).status_code == 200


@pytest.mark.anyio
async def test_matched_capture_reuses_effective_reference_limit_with_larger_target_context(
    client, lab
):
    runtime = lab.runtime
    runtime.metadata["execution_context_length"] = 16
    documents = [text + " the capital of France is" * 6 for text in DOCUMENTS]
    started = await client.post(
        "/api/crosscoders/captures",
        json={
            "name": "Context reference",
            "documents": documents,
            "layer": 0,
            "max_tokens": 128,
            "max_samples": 64,
        },
    )
    assert started.status_code == 202, started.text
    reference = (await finished(client, started.json()))["result"]["captures"][0]
    manifest = lab.store.get(reference["id"], kind="capture")
    assert manifest["capture_config"]["max_tokens"] == 128
    assert manifest["corpus"]["token_limit"] == 16
    runtime.metadata.update(execution_context_length=32, execution_id=uuid.uuid4().hex)
    path = "/api/crosscoders/captures/" + reference["id"] + "/match"
    forwards = []
    handle = runtime.model.base_model.register_forward_pre_hook(lambda *args: forwards.append(1))
    try:
        body = {"layer": 0, "expected_execution_id": runtime.metadata["execution_id"]}
        plan = await client.post(path + "/plan", json=body)
        assert plan.status_code == 200, plan.text
        assert not forwards
        assert plan.json()["token_limit"] == 16
        assert plan.json()["corpus_fingerprint"] == reference["corpus_fingerprint"]
        response = await client.post(path, json=body)
        assert response.status_code == 202, response.text
        compared_job = await finished(client, response.json())
        assert compared_job["status"] == "ready"
        compared = compared_job["result"]["captures"][0]
        assert compared["sample_count"] == reference["sample_count"]
        assert compared["corpus_fingerprint"] == reference["corpus_fingerprint"]
        forwards.clear()
        runtime.metadata.update(execution_context_length=8, execution_id=uuid.uuid4().hex)
        blocked = await client.post(
            path + "/plan",
            json={
                "layer": 0,
                "expected_execution_id": runtime.metadata["execution_id"],
            },
        )
        assert blocked.status_code == 400
        assert "limits differ" in blocked.json()["detail"]
        assert not forwards
    finally:
        handle.remove()
