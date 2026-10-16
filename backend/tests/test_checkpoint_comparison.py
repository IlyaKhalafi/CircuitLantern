from unittest.mock import patch

import pytest
import torch

from circuit_lantern.checkpoint_schemas import CheckpointCaptureRequest, CheckpointCompareRequest
from circuit_lantern.services.checkpoint_store import CheckpointStore
from circuit_lantern.services.checkpoints import CheckpointLab, weight_change


@pytest.fixture
def lab(runtime, tmp_path):
    return CheckpointLab(runtime, CheckpointStore(tmp_path / "evidence"))


def capture(lab):
    return lab.capture(
        CheckpointCaptureRequest(
            name="Base checkpoint",
            prompt="the capital of France is",
            matrices=["transformer.h.0.attn.c_attn.weight"],
        )
    )


def test_identical_checkpoint_has_zero_actual_weight_residual_and_prediction_drift(lab):
    before = {name: tensor.clone() for name, tensor in lab.runtime.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    reference = capture(lab)
    result = lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    assert result["weights"][0]["distance"] == 0
    assert all(item["distance"] == 0 for item in result["layers"])
    assert result["kl_reference_to_selected"] == 0 and result["total_variation"] == 0
    assert result["warnings"]
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in lab.runtime.model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert all(not block._forward_hooks for block in lab.runtime._blocks())
    assert CheckpointStore(lab.store.directory).comparison(result["id"]) == result


def test_base_and_finetuned_repository_comparison_measures_actual_delta(lab):
    reference = capture(lab)
    with torch.no_grad():
        lab.runtime.model.transformer.h[0].attn.c_attn.weight.add_(0.1)
    lab.runtime.metadata = dict(
        lab.runtime.metadata, id="author/finetuned", revision="fine-revision"
    )
    result = lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    assert result["selected"]["model_id"] == "author/finetuned"
    assert result["reference"]["model_id"] == "local-test"
    assert result["weights"][0]["rms_change"] == pytest.approx(0.1, abs=1e-7)
    assert result["weights"][0]["distance"] > 0
    assert any(item["distance"] > 0 for item in result["layers"])
    lab.store.delete(reference["id"])
    assert lab.store.comparison(result["id"]) == result


def test_incompatible_tokenizer_is_rejected_before_a_model_forward(lab):
    reference = capture(lab)
    lab.runtime.tokenizer.add_tokens(["new-vocabulary-token"])
    with (
        patch.object(lab.runtime.model, "forward") as forward,
        pytest.raises(ValueError, match="vocabulary"),
    ):
        lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    forward.assert_not_called()


def test_incompatible_layout_and_token_inputs_are_rejected_before_forward(lab):
    reference = capture(lab)
    lab.runtime.model.config.num_attention_heads += 1
    with pytest.raises(ValueError, match="architectures"):
        lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    lab.runtime.model.config.num_attention_heads -= 1
    with (
        patch.object(lab.runtime.tokenizer, "model_input_names", ["input_ids"]),
        pytest.raises(ValueError, match="inputs differ"),
    ):
        lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))


def test_failed_capture_cleans_all_hooks_and_publishes_no_reference(lab):
    with (
        patch.object(lab.runtime.model, "forward", side_effect=RuntimeError("capture failed")),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        capture(lab)
    assert not lab.store.list()
    assert all(not block._forward_hooks for block in lab.runtime._blocks())


def test_evidence_limit_and_low_memory_reject_before_dequantization(lab):
    lab.store.MAX_BYTES = 1
    with (
        patch("circuit_lantern.services.checkpoints.logical_weight") as weights,
        pytest.raises(ValueError, match="evidence limit"),
    ):
        capture(lab)
    weights.assert_not_called()
    lab.store.MAX_BYTES = 128 * 1024**2
    with (
        patch(
            "circuit_lantern.services.checkpoints.hardware_snapshot",
            return_value={
                "ram_available_bytes": 1,
                "experiment_disk_free_bytes": 10**9,
            },
        ),
        pytest.raises(ValueError, match="RAM"),
    ):
        capture(lab)


def test_corrupt_reference_and_unsafe_identifiers_are_rejected(lab):
    reference = capture(lab)
    manifest, tensors = lab.store.load(reference["id"])
    assert manifest == reference
    assert tensors["residuals"].shape == (2, 16)
    with pytest.raises(ValueError, match="identifier"):
        lab.store.delete("../outside")


def test_blockwise_weight_metrics_agree_with_float64_ground_truth():
    left = torch.linspace(-2, 3, 600000).reshape(1000, 600)
    right = left.flip(0)
    result = weight_change(left, right)
    delta = left.double() - right.double()
    assert result["distance"] == pytest.approx(float(delta.norm()), rel=1e-12)
    assert result["rms_change"] == pytest.approx(float(delta.square().mean().sqrt()), rel=1e-12)
    zero = weight_change(torch.zeros(2, 2), torch.zeros(2, 2))
    assert zero["relative_distance"] is None and zero["cosine_similarity"] is None


def test_full_reference_library_is_rejected_before_teacher_capture(lab):
    lab.store.MAX_SNAPSHOTS = 0
    with patch.object(lab, "_capture") as forward, pytest.raises(ValueError, match="full"):
        capture(lab)
    forward.assert_not_called()


def test_interrupted_writes_and_unreadable_references_do_not_hide_valid_evidence(lab):
    good = capture(lab)
    broken_id = "b" * 32
    lab.store.path(broken_id).write_bytes(b"truncated")
    orphan = lab.store.directory / ".capture-interrupted"
    orphan.write_bytes(b"partial tensor data")
    reopened = CheckpointStore(lab.store.directory)
    assert not orphan.exists()
    report = reopened.library()
    assert [entry["id"] for entry in report["items"]] == [good["id"]]
    assert report["unreadable"] == [broken_id]
    reopened.delete(broken_id)
    assert not reopened.library()["unreadable"]


def test_comparison_library_returns_summaries_and_can_be_bounded(lab):
    reference = capture(lab)
    report = lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    summaries = lab.store.comparisons()["items"]
    assert summaries[0]["id"] == report["id"]
    assert set(summaries[0]) == {"id", "created_at", "reference", "selected"}
    assert set(summaries[0]["reference"]) == {"model_id", "revision"}
    lab.store.MAX_COMPARISONS = 1
    with patch.object(lab, "_capture") as forward, pytest.raises(ValueError, match="full"):
        lab.compare(CheckpointCompareRequest(reference_id=reference["id"]))
    forward.assert_not_called()
    lab.store.delete_comparison(report["id"])
    assert not lab.store.comparisons()["items"]


def test_checkpoint_endpoints_capture_compare_list_and_delete(lab, monkeypatch):
    from fastapi.testclient import TestClient

    from circuit_lantern.main import app

    monkeypatch.setattr(app.state, "checkpoints", lab, raising=False)
    client = TestClient(app)
    response = client.post(
        "/api/checkpoints/references",
        json={
            "name": "HTTP reference",
            "prompt": "the capital of France is",
            "matrices": ["transformer.h.0.attn.c_attn.weight"],
        },
    )
    assert response.status_code == 201
    identifier = response.json()["id"]
    assert client.get("/api/checkpoints/references").json()["items"][0]["id"] == identifier
    response = client.post("/api/checkpoints/compare", json={"reference_id": identifier})
    assert response.status_code == 201
    report_id = response.json()["id"]
    assert client.get("/api/checkpoints/comparisons").json()["items"][0]["id"] == report_id
    assert client.get("/api/checkpoints/comparisons/" + report_id).status_code == 200
    assert client.delete("/api/checkpoints/references/" + identifier).status_code == 204
    assert client.get("/api/checkpoints/comparisons/" + report_id).status_code == 200
    assert client.delete("/api/checkpoints/comparisons/" + report_id).status_code == 204
    assert (
        client.post("/api/checkpoints/compare", json={"reference_id": "invalid"}).status_code == 422
    )


def test_incomplete_metadata_and_tampered_tensors_are_isolated(lab):
    import json

    from safetensors.torch import save_file

    reference = capture(lab)
    manifest, tensors = lab.store.load(reference["id"])
    invalid_id = "c" * 32
    save_file(
        {"test": torch.zeros(1)},
        lab.store.path(invalid_id),
        metadata={"manifest": json.dumps({"id": invalid_id})},
    )
    assert lab.store.library()["unreadable"] == [invalid_id]
    tensors["weight_0"][0, 0] += 1
    save_file(tensors, lab.store.path(reference["id"]), metadata={"manifest": json.dumps(manifest)})
    with pytest.raises(ValueError, match="integrity"):
        lab.store.load(reference["id"])
