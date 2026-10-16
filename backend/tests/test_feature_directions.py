import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import torch
from dictionary_fixtures import manifest

from circuit_lantern.main import app
from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.directions import decoder_cosines, dictionary_directions
from circuit_lantern.services.features.store import DictionaryStore


@pytest.fixture
def artifact(tmp_path):
    store = DictionaryStore(tmp_path)
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    with torch.no_grad():
        model.decoder.copy_(
            torch.tensor(
                [
                    [1, 1, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, -1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
        )
    record = manifest()
    record["feature_statistics"] = [{"index": index, "frequency": index / 10} for index in range(8)]
    saved = store.save(record, model)
    store.label_feature(saved["id"], 1, "Literal {direction}", "Research notes")
    return store, saved["id"]


def test_signed_cosines_use_normalized_columns_and_exclude_self_and_zero_directions():
    decoder = torch.tensor([[2.0, 4.0, -3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
    assert decoder_cosines(decoder, 0) == [(1, 1.0), (2, -1.0), (3, 0.0)]
    with pytest.raises(ValueError, match="nonzero norm"):
        decoder_cosines(decoder, 4)


def test_chunked_geometry_matches_independent_numpy_cosines(monkeypatch):
    rng = np.random.default_rng(73)
    values = rng.normal(size=(7, 31)).astype(np.float32)
    values *= np.linspace(0.1, 10, 31, dtype=np.float32)
    monkeypatch.setattr("circuit_lantern.services.features.directions._CHUNK_ELEMENTS", 14)
    result = dict(decoder_cosines(torch.from_numpy(values), 12))
    reference = values.astype(np.float64)
    for index, score in result.items():
        expected = np.dot(reference[:, 12], reference[:, index]) / (
            np.linalg.norm(reference[:, 12]) * np.linalg.norm(reference[:, index])
        )
        assert score == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("feature", [-1, 8, True])
def test_invalid_feature_numbers_are_rejected(artifact, feature):
    store, identifier = artifact
    with pytest.raises(ValueError, match="feature"):
        dictionary_directions(store, identifier, feature)


def test_rankings_have_stable_ties_recorded_frequencies_and_immutable_provenance(artifact):
    store, identifier = artifact
    path = store._weights_path(identifier)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    random_state = torch.random.get_rng_state().clone()
    result = dictionary_directions(store, identifier, 0, 4)
    assert [row["feature"] for row in result["highest"]] == [1, 3, 4, 5]
    assert [row["feature"] for row in result["lowest"]] == [2, 3, 4, 5]
    assert result["highest"][0] == {
        "feature": 1,
        "cosine": 1.0,
        "frequency": 0.1,
        "label": "Literal {direction}",
    }
    assert result["lowest"][0]["cosine"] == -1
    assert result["compared_features"] == 7
    assert result["weights_sha256"] == before
    assert result["manifest_sha256"] == store.get(identifier)["manifest_sha256"]
    assert result["capture_point"] == "residual_post"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    torch.testing.assert_close(torch.random.get_rng_state(), random_state, rtol=0, atol=0)


def test_memory_is_checked_before_loading_dictionary_tensors(artifact, monkeypatch):
    store, identifier = artifact
    monkeypatch.setattr(
        "circuit_lantern.services.features.directions.psutil.virtual_memory",
        lambda: SimpleNamespace(available=1),
    )
    with (
        patch.object(store, "load_model") as loading,
        pytest.raises(ValueError, match="Insufficient available RAM"),
    ):
        dictionary_directions(store, identifier, 0)
    loading.assert_not_called()


def test_changed_artifact_bytes_fail_integrity_checks(artifact):
    store, identifier = artifact
    path = store._weights_path(identifier)
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="changed"):
        dictionary_directions(store, identifier, 0)


@pytest.mark.anyio
async def test_directions_api_needs_no_runtime_and_validates_bounds(artifact, monkeypatch):
    store, identifier = artifact
    monkeypatch.setattr(app.state, "features", SimpleNamespace(store=store), raising=False)
    monkeypatch.setattr(app.state, "runtime", None, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/api/features/dictionaries/" + identifier + "/features/0/directions"
        response = await client.get(path + "?limit=2")
        assert response.status_code == 200, response.text
        assert len(response.json()["highest"]) == 2
        assert response.json()["lowest"][0]["cosine"] == -1
        for limit in ("0", "33"):
            assert (await client.get(path + "?limit=" + limit)).status_code == 422
        assert (await client.get(path.replace("/features/0/", "/features/-1/"))).status_code == 422
        assert (await client.get(path.replace(identifier, "a" * 32))).status_code == 404


@pytest.fixture
def anyio_backend():
    return "asyncio"
