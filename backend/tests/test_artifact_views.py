"""Immutable metadata views preserve evidence while repeated browsing avoids decoding."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from unittest.mock import Mock

import httpx
import pytest
import torch
from fastapi.encoders import jsonable_encoder

from circuit_lantern.main import app
from circuit_lantern.services.artifact_views import ArtifactManifestViews
from circuit_lantern.services.crosscoders.lab import CrosscoderLab
from circuit_lantern.services.tensor_artifacts import (
    TensorArtifactStore,
    canonical_json,
    digest_bytes,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def store(tmp_path):
    return TensorArtifactStore(tmp_path, 1024)


def saved(store):
    return store.save(
        "crosscoder",
        {
            "name": "Retained metadata",
            "features": 2,
            "banks": [
                {"input_dim": 2, "training_mean": [0.0, 0.0]},
                {"input_dim": 2, "training_mean": [0.0, 0.0]},
            ],
            "corpus": {
                "fingerprint": "fixed",
                "split_method": "document",
                "documents": [{"tokens": ["the", "capital"]}],
                "samples": [{"document": 0, "position": 1, "split": "train"}],
            },
            "feature_statistics": [
                {
                    "index": index,
                    "frequency": 0.5,
                    "decoder_cosine": 0.2,
                    "relative_decoder_norms": [0.5, 0.5],
                    "examples": [{"sample": 0, "activation": 1.0}],
                }
                for index in range(2)
            ],
        },
        {"values": torch.ones(2)},
        {"features": 2},
    )


def test_nested_views_are_immutable_and_mutable_reads_cannot_poison_them(store):
    manifest = saved(store)
    identifier = manifest["id"]
    view = store.get_readonly(identifier)
    assert jsonable_encoder(view) == manifest
    with pytest.raises(TypeError):
        view["name"] = "changed"
    with pytest.raises(TypeError):
        view["feature_statistics"][0]["examples"][0]["activation"] = 99
    with pytest.raises(AttributeError):
        view["corpus"]["documents"].append({})
    mutable = store.get(identifier)
    mutable["corpus"]["documents"][0]["tokens"][0] = "changed"
    assert store.get_readonly(identifier) is view
    assert view["corpus"]["documents"][0]["tokens"][0] == "the"
    assert store.get(identifier) == manifest


def test_warm_views_recheck_bytes_and_reject_corruption_with_an_unchanged_digest(
    store, monkeypatch
):
    manifest = saved(store)
    identifier = manifest["id"]
    spy = Mock(wraps=store._decode)
    monkeypatch.setattr(store, "_decode", spy)
    store.get_readonly(identifier)
    store.get_readonly(identifier)
    assert spy.call_count == 1
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE artifacts SET manifest = ? WHERE id = ?",
            ('{"kind":"crosscoder","name":"changed"}', identifier),
        )
    with pytest.raises(ValueError, match="checksum"):
        store.get_readonly(identifier)
    original = canonical_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with sqlite3.connect(store.database) as connection:
        connection.execute("UPDATE artifacts SET manifest = ? WHERE id = ?", (original, identifier))
    assert jsonable_encoder(store.get_readonly(identifier)) == manifest
    assert spy.call_count == 2


@pytest.mark.parametrize("reader", ["get", "get_readonly"])
def test_readers_reject_a_valid_digest_bound_to_the_wrong_identity(store, reader):
    manifest = saved(store)
    identifier = manifest["id"]
    changed = canonical_json({**manifest, "id": "f" * 32})
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE artifacts SET manifest = ?, manifest_sha256 = ? WHERE id = ?",
            (changed, digest_bytes(changed.encode()), identifier),
        )
    with pytest.raises(ValueError, match="metadata is invalid"):
        getattr(store, reader)(identifier)


def test_concurrent_browsing_decodes_once_and_returns_independent_immutable_access(
    store, monkeypatch
):
    identifier = saved(store)["id"]
    entered, resume = Event(), Event()
    original = store._decode

    def gated(*args):
        entered.set()
        assert resume.wait(5)
        return original(*args)

    spy = Mock(side_effect=gated)
    monkeypatch.setattr(store, "_decode", spy)
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(store.get_readonly, identifier)
        try:
            assert entered.wait(5)
            others = [executor.submit(store.get_readonly, identifier) for _ in range(3)]
        finally:
            resume.set()
        views = [first.result(timeout=5), *(value.result(timeout=5) for value in others)]
    assert spy.call_count == 1
    assert all(view is views[0] for view in views)


@pytest.mark.parametrize("max_bytes,max_entries", [(0, 8), (1024**2, 1)])
def test_cache_budgets_evict_or_skip_views_without_changing_results(
    store, monkeypatch, max_bytes, max_entries
):
    first, second = saved(store), saved(store)
    store._views = ArtifactManifestViews(max_bytes=max_bytes, max_entries=max_entries)
    spy = Mock(wraps=store._decode)
    monkeypatch.setattr(store, "_decode", spy)
    assert jsonable_encoder(store.get_readonly(first["id"])) == first
    assert jsonable_encoder(store.get_readonly(second["id"])) == second
    assert jsonable_encoder(store.get_readonly(first["id"])) == first
    assert spy.call_count == 3


@pytest.mark.parametrize("reader", ["get", "get_readonly"])
def test_ram_guard_runs_before_materializing_manifest_text(store, monkeypatch, reader):
    identifier = saved(store)["id"]
    original = store.connect
    statements = []

    @contextmanager
    def observed():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(store, "connect", observed)
    monkeypatch.setattr(
        "circuit_lantern.services.artifact_views.available_memory_bytes",
        lambda: 1,
    )
    with pytest.raises(ValueError, match="RAM"):
        getattr(store, reader)(identifier)
    assert any(statement.startswith("SELECT rowid, kind") for statement in statements)
    assert not any(statement.startswith("SELECT manifest,") for statement in statements)


def test_memory_pressure_evicts_a_warm_view_before_rejecting_a_new_read(store, monkeypatch):
    identifier = saved(store)["id"]
    spy = Mock(wraps=store._decode)
    monkeypatch.setattr(store, "_decode", spy)
    store.get_readonly(identifier)
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "circuit_lantern.services.artifact_views.available_memory_bytes",
            lambda: 1,
        )
        with pytest.raises(ValueError, match="RAM"):
            store.get_readonly(identifier)
    store.get_readonly(identifier)
    assert spy.call_count == 2


def test_deleted_or_wrong_kind_records_never_return_a_cached_view(store):
    identifier = saved(store)["id"]
    store.get_readonly(identifier)
    assert store.get_readonly(identifier, kind="capture") is None
    store.get_readonly(identifier)
    assert store.delete(identifier, kind="crosscoder")
    assert store.get_readonly(identifier) is None


@pytest.mark.anyio
async def test_api_serializes_views_and_reads_current_annotations_after_warming(store, monkeypatch):
    manifest = saved(store)
    identifier = manifest["id"]
    monkeypatch.setattr(app.state, "crosscoders", CrosscoderLab(None, store), raising=False)
    spy = Mock(wraps=store._decode)
    monkeypatch.setattr(store, "_decode", spy)
    path = "/api/crosscoders/" + identifier
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        overview = await client.get(path + "/overview")
        rows = await client.get(path + "/features")
        feature = await client.get(path + "/features/0")
        assert overview.status_code == rows.status_code == feature.status_code == 200
        assert overview.json()["manifest_sha256"] == manifest["manifest_sha256"]
        assert rows.json()["features"] == manifest["feature_statistics"]
        assert feature.json()["examples"][0]["tokens"] == ["the", "capital"]
        assert spy.call_count == 1
        store.label(identifier, 0, "New observation", "Current note")
        count_after_label = spy.call_count
        searched = await client.get(path + "/features?q=observation")
        assert searched.json()["total"] == 1
        assert searched.json()["features"][0]["label"] == "New observation"
        assert (await client.get(path + "/features/0")).json()["notes"] == "Current note"
        assert spy.call_count == count_after_label
        exported = await client.get(path + "/export")
        assert exported.status_code == 200
        assert exported.json()["artifact"] == manifest
        assert exported.json()["annotations"]["0"]["label"] == "New observation"
    app.state.crosscoders.downloads.close()


def test_astral_text_is_preserved_and_its_wider_read_budget_is_checked(store, monkeypatch):
    manifest = store.save(
        "crosscoder",
        {"name": "Non ASCII text", "features": 2, "text": "a" * 1000 + chr(0x1F3EE)},
        {"values": torch.ones(2)},
        {"features": 2},
    )
    assert jsonable_encoder(store.get_readonly(manifest["id"])) == manifest
    encoded_size = len(
        canonical_json(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ).encode()
    )
    monkeypatch.setattr(
        "circuit_lantern.services.artifact_views.available_memory_bytes",
        lambda: encoded_size * 5,
    )
    with pytest.raises(ValueError, match="RAM"):
        store.get_readonly(manifest["id"])


def test_a_deep_valid_manifest_does_not_exhaust_the_freezer_stack(store):
    nested = "leaf"
    for _ in range(450):
        nested = [nested]
    manifest = store.save(
        "crosscoder",
        {"name": "Nested metadata", "features": 2, "nested": nested},
        {"values": torch.ones(2)},
        {"features": 2},
    )
    view = store.get_readonly(manifest["id"])["nested"]
    for _ in range(450):
        assert isinstance(view, tuple) and len(view) == 1
        view = view[0]
    assert view == "leaf"


def test_oversized_metadata_is_rejected_before_copying_the_text(store, monkeypatch):
    identifier = saved(store)["id"]
    statements = []
    original = store.connect

    @contextmanager
    def observed():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(store, "connect", observed)
    monkeypatch.setattr("circuit_lantern.services.tensor_artifacts._MAX_METADATA_BYTES", 16)
    with pytest.raises(ValueError, match="checksum"):
        store.get_readonly(identifier)
    assert not any(statement.startswith("SELECT manifest,") for statement in statements)
