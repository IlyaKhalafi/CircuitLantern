import hashlib
import json
import math
from unittest.mock import patch

import pytest
import torch
from dictionary_fixtures import manifest

from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.store import DictionaryStore


@pytest.mark.parametrize("method", ["relu", "topk"])
def test_dictionary_reopens_after_store_restart_with_exact_weights_and_provenance(tmp_path, method):
    store = DictionaryStore(tmp_path)
    model = SparseAutoencoder(4, TrainingConfig(features=8, method=method, active_features=2))
    with torch.no_grad():
        model.input_scale.fill_(3.5)
    summary = store.save(manifest(), model)
    reopened = DictionaryStore(tmp_path)
    saved, restored = reopened.load_model(summary["id"])
    assert saved["model_id"] == "research/model"
    assert saved["revision"] == "pinned-revision"
    assert saved["training"] == model.config.description()
    assert saved["capture_point"] == "residual_post"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    assert reopened.list() == [summary]


def test_feature_annotations_do_not_change_the_dictionary_or_its_recorded_measurements(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    before = store.get(summary["id"])
    store.label_feature(summary["id"], 0, "Mary {x}", "Literal research notes")
    after = store.get(summary["id"])
    assert after["feature_labels"]["0"] == {"label": "Mary {x}", "notes": "Literal research notes"}
    assert {key: value for key, value in before.items() if key != "feature_labels"} == {
        key: value for key, value in after.items() if key != "feature_labels"
    }
    store.load_model(summary["id"])


def test_changed_weight_bytes_cannot_be_used_with_the_original_manifest(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    path = tmp_path / (summary["id"] + ".safetensors")
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="changed"):
        store.load_model(summary["id"])


def test_header_shapes_are_checked_before_allocating_a_dictionary_from_a_changed_manifest(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    with store.connect() as connection:
        row = connection.execute("SELECT manifest FROM dictionaries").fetchone()
        payload = json.loads(row["manifest"])
        payload["input_dim"] = 5
        encoded = json.dumps(payload)
        connection.execute(
            "UPDATE dictionaries SET manifest = ?, manifest_sha256 = ?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    with (
        patch("circuit_lantern.services.features.store.SparseAutoencoder") as constructor,
        pytest.raises(ValueError, match="shapes"),
    ):
        store.load_model(summary["id"])
    constructor.assert_not_called()


def test_dictionary_size_limit_is_enforced_before_writing_an_artifact(tmp_path):
    store = DictionaryStore(tmp_path, max_weight_bytes=1)
    with pytest.raises(ValueError, match="size limit"):
        store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    assert not list(tmp_path.glob("*.safetensors"))
    assert not store.list()


def test_invalid_measurements_cannot_publish_a_dictionary(tmp_path):
    store = DictionaryStore(tmp_path)
    payload = manifest()
    payload["validation_metrics"]["mse"] = math.nan
    with pytest.raises(ValueError):
        store.save(payload, SparseAutoencoder(4, TrainingConfig(features=8)))
    assert not list(tmp_path.glob("*.safetensors"))
    assert not store.list()


def test_invalid_decoder_norm_is_a_validation_failure(tmp_path):
    store = DictionaryStore(tmp_path)
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    with torch.no_grad():
        model.decoder.mul_(2)
    with pytest.raises(ValueError, match="unit norm"):
        store.save(manifest(), model)


@pytest.mark.parametrize("identifier", ["../weights", "a" * 31, "A" * 32, "g" * 32])
def test_dictionary_paths_cannot_escape_the_artifact_directory(tmp_path, identifier):
    with pytest.raises(ValueError, match="identifier"):
        DictionaryStore(tmp_path).get(identifier)


def test_identifier_collision_cannot_delete_an_existing_weight_file(tmp_path):
    store = DictionaryStore(tmp_path)
    first = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    original = (tmp_path / (first["id"] + ".safetensors")).read_bytes()
    with (
        patch("circuit_lantern.services.features.store.uuid.uuid4") as identifier,
        pytest.raises(ValueError, match="already uses"),
    ):
        identifier.return_value.hex = first["id"]
        store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    assert (tmp_path / (first["id"] + ".safetensors")).read_bytes() == original
    assert len(store.list()) == 1


def test_missing_training_provenance_cannot_publish_weights(tmp_path):
    payload = manifest()
    del payload["corpus"]
    store = DictionaryStore(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        store.save(payload, SparseAutoencoder(4, TrainingConfig(features=8)))
    assert not store.list()
    assert not list(tmp_path.glob("*.safetensors"))


def test_missing_config_fields_cannot_silently_use_new_defaults(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    with store.connect() as connection:
        payload = json.loads(
            connection.execute("SELECT manifest FROM dictionaries").fetchone()["manifest"]
        )
        del payload["training"]["seed"]
        encoded = json.dumps(payload)
        connection.execute(
            "UPDATE dictionaries SET manifest = ?, manifest_sha256 = ?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    with pytest.raises(ValueError, match="incomplete"):
        store.load_model(summary["id"])


def test_corrupted_scientific_metadata_is_detected_before_loading_weights(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    with store.connect() as connection:
        payload = json.loads(
            connection.execute("SELECT manifest FROM dictionaries").fetchone()["manifest"]
        )
        payload["training_mean"][0] = 42
        connection.execute("UPDATE dictionaries SET manifest = ?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="manifest has changed"):
        store.load_model(summary["id"])


def test_startup_recovers_interrupted_publication_and_deletion(tmp_path):
    store = DictionaryStore(tmp_path)
    summary = store.save(manifest(), SparseAutoencoder(4, TrainingConfig(features=8)))
    identifier = summary["id"]
    original = (tmp_path / (identifier + ".safetensors")).read_bytes()
    (tmp_path / (identifier + ".safetensors")).replace(
        tmp_path / (".trash-" + identifier + ".safetensors")
    )
    orphan = tmp_path / ("b" * 32 + ".safetensors")
    orphan.write_bytes(b"interrupted publication")
    partial = tmp_path / ("." + "c" * 32 + ".partial")
    partial.write_bytes(b"interrupted write")
    unrelated = tmp_path / "external.safetensors"
    unrelated.write_bytes(b"not a managed artifact")
    reopened = DictionaryStore(tmp_path)
    assert (tmp_path / (identifier + ".safetensors")).read_bytes() == original
    assert not orphan.exists() and not partial.exists()
    assert unrelated.exists()
    reopened.load_model(identifier)
    assert reopened.delete(identifier)
    assert not reopened.get(identifier)
    assert not (tmp_path / (identifier + ".safetensors")).exists()
    assert not reopened.delete(identifier)


def test_run_listing_reads_only_the_compact_summary(tmp_path):
    from circuit_lantern.services.features import store as module

    store = DictionaryStore(tmp_path)
    run = store.save_run(
        {
            "kind": "feature_circuit",
            "model_id": "source",
            "revision": "r",
            "prompt": "A short prompt",
            "complete": True,
            "dictionary_ids": [],
            "measurements": [{"value": "x" * 100}] * 10000,
        }
    )
    original = module.json.loads
    sizes = []

    def measured(value):
        sizes.append(len(value))
        return original(value)

    with patch.object(module.json, "loads", side_effect=measured):
        summaries = store.runs()
    assert summaries[0]["id"] == run["id"]
    assert max(sizes) < 1000
    assert len(store.get_run(run["id"])["measurements"]) == 10000
