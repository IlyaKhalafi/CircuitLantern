"""Immutable metadata views preserve evidence while repeated browsing avoids decoding."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock
from unittest.mock import Mock

import httpx
import pytest
import torch
from fastapi.encoders import jsonable_encoder

from circuit_lantern.main import app
from circuit_lantern.services.artifact_views import (
    METADATA_JSON_RAM_MULTIPLIER,
    ArtifactManifestViews,
    MetadataMemoryBudget,
    MetadataMemoryError,
)
from circuit_lantern.services.crosscoders.lab import CrosscoderLab
from circuit_lantern.services.tensor_artifacts import (
    ArtifactBusyError,
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
    view = store.get_readonly(identifier)
    assert store.get_readonly(identifier, kind="capture") is None
    assert store.get_readonly(identifier) is view
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
    try:
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
    finally:
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


def test_a_warm_artifact_remains_available_while_another_artifact_is_decoding(store, monkeypatch):
    cold, warm = saved(store)["id"], saved(store)["id"]
    expected = store.get_readonly(warm)
    entered, resume = Event(), Event()
    original = store._decode

    def gated(*args):
        entered.set()
        assert resume.wait(5)
        return original(*args)

    monkeypatch.setattr(store, "_decode", gated)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store.get_readonly, cold)
        try:
            assert entered.wait(5)
            assert executor.submit(store.get_readonly, warm).result(timeout=1) is expected
        finally:
            resume.set()
        assert first.result(timeout=5)["id"] == cold
    assert not store._views._flights
    assert store._metadata_memory._reserved == 0


def test_deleting_during_a_cold_read_does_not_reinsert_a_stale_view(store, monkeypatch):
    identifier = saved(store)["id"]
    entered, resume = Event(), Event()
    original = store._decode

    def gated(*args):
        entered.set()
        assert resume.wait(5)
        return original(*args)

    monkeypatch.setattr(store, "_decode", gated)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reading = executor.submit(store.get_readonly, identifier)
        try:
            assert entered.wait(5)
            assert store.delete(identifier, kind="crosscoder")
        finally:
            resume.set()
        assert reading.result(timeout=5)["id"] == identifier
    assert not store._views._entries
    assert store.get_readonly(identifier) is None


def test_memory_reservations_are_shared_and_released_after_failure(monkeypatch):
    monkeypatch.setattr(
        "circuit_lantern.services.artifact_views.available_memory_bytes", lambda: 1000
    )
    budget = MetadataMemoryBudget()
    entered, resume = Event(), Event()

    def holding():
        with budget.reserve(500):
            entered.set()
            assert resume.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(holding)
        try:
            assert entered.wait(5)
            with pytest.raises(ValueError, match="RAM"), budget.reserve(300):
                pytest.fail("Overlapping allocations exceeded the shared budget")
        finally:
            resume.set()
        first.result(timeout=5)
    with pytest.raises(RuntimeError), budget.reserve(700):
        raise RuntimeError("Interrupted allocation")
    assert budget._reserved == 0
    with budget.reserve(700):
        assert budget._reserved == 700


def test_mutable_readers_do_not_decode_concurrently(store, monkeypatch):
    identifier = saved(store)["id"]
    entered, second_loaded, resume = Event(), Event(), Event()
    original_read, original_decode = store._read_manifest, store._decode
    counter_lock = Lock()
    reads = 0

    @contextmanager
    def observed(*args):
        nonlocal reads
        with original_read(*args) as loaded:
            with counter_lock:
                reads += 1
                if reads == 2:
                    second_loaded.set()
            yield loaded

    def gated(*args):
        entered.set()
        assert resume.wait(5)
        return original_decode(*args)

    spy = Mock(side_effect=gated)
    monkeypatch.setattr(store, "_read_manifest", observed)
    monkeypatch.setattr(store, "_decode", spy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store.get, identifier)
        try:
            assert entered.wait(5)
            second = executor.submit(store.get, identifier)
            assert second_loaded.wait(5)
            assert spy.call_count == 1
        finally:
            resume.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert spy.call_count == 2
    assert store._metadata_memory._reserved == 0


def test_sqlite_writer_contention_returns_busy_and_recovers(store):
    identifier = saved(store)["id"]
    with sqlite3.connect(store.database) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                reading = executor.submit(store.get_readonly, identifier)
                with pytest.raises(ArtifactBusyError, match="busy"):
                    reading.result(timeout=1)
        finally:
            connection.rollback()
    assert store._metadata_memory._reserved == 0
    assert store.get_readonly(identifier)["id"] == identifier


@pytest.mark.parametrize("reader", ["get", "get_readonly"])
def test_decode_failure_releases_all_transient_memory_reservations(store, monkeypatch, reader):
    identifier = saved(store)["id"]
    original = store._decode
    monkeypatch.setattr(store, "_decode", Mock(side_effect=ValueError("Invalid metadata")))
    with pytest.raises(ValueError, match="Invalid metadata"):
        getattr(store, reader)(identifier)
    assert store._metadata_memory._reserved == 0
    monkeypatch.setattr(store, "_decode", original)
    assert getattr(store, reader)(identifier)["id"] == identifier


@pytest.mark.anyio
async def test_export_keeps_its_memory_reservation_through_json_encoding(store, monkeypatch):
    from circuit_lantern.api import crosscoders

    manifest = saved(store)
    store.label(manifest["id"], 0, "Observation", "Current annotation")
    lab = CrosscoderLab(None, store)
    monkeypatch.setattr(app.state, "crosscoders", lab, raising=False)
    original = crosscoders.jsonable_encoder
    observed = []

    def checked(value):
        observed.append(store._metadata_memory._reserved)
        return original(value)

    monkeypatch.setattr(crosscoders, "jsonable_encoder", checked)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/crosscoders/" + manifest["id"] + "/export")
        assert response.status_code == 200
        assert response.json()["artifact"] == manifest
        assert len(observed) == 1 and observed[0] > 0
        assert store._metadata_memory._reserved == 0
    finally:
        lab.downloads.close()


def test_publication_checks_the_budget_needed_to_read_and_export_the_result(store, monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "circuit_lantern.services.artifact_views.available_memory_bytes", lambda: 64000
        )
        with pytest.raises(MetadataMemoryError, match="RAM"):
            store.save(
                "crosscoder",
                {"name": "Too large for current memory", "features": 2, "text": "x" * 1000},
                {"values": torch.ones(2)},
                {"features": 2},
            )
    assert store.list("crosscoder") == []
    assert not list(store.directory.glob("*.safetensors"))
    assert store._metadata_memory._reserved == 0
    manifest = saved(store)
    with store.read_metadata(manifest["id"], serializing=True) as value:
        assert value == manifest


@pytest.mark.anyio
async def test_annotation_volume_is_budgeted_before_fetch_and_returns_retryable_http_error(
    store, monkeypatch
):
    manifest = saved(store)
    identifier = manifest["id"]
    notes = "a" * 9999 + chr(0x1F3EE)
    store.label(identifier, 0, "Observation", notes)
    with store.connect() as connection:
        encoded_bytes = len(
            connection.execute("SELECT manifest FROM artifacts WHERE id = ?", (identifier,))
            .fetchone()[0]
            .encode()
        )
    statements = []
    original = store.connect

    @contextmanager
    def observed():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(store, "connect", observed)
    lab = CrosscoderLab(None, store)
    monkeypatch.setattr(app.state, "crosscoders", lab, raising=False)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    "circuit_lantern.services.artifact_views.available_memory_bytes",
                    lambda: (encoded_bytes * METADATA_JSON_RAM_MULTIPLIER + 64000) * 4 // 3,
                )
                response = await client.get("/api/crosscoders/" + identifier + "/export")
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "2"
            assert "RAM" in response.json()["detail"]
            assert any(statement.startswith("SELECT COALESCE(SUM(") for statement in statements)
            assert not any(
                statement.startswith("SELECT feature, label, notes, updated_at")
                for statement in statements
            )
            assert store._metadata_memory._reserved == 0
            recovered = await client.get("/api/crosscoders/" + identifier + "/export")
            assert recovered.status_code == 200
            assert recovered.json()["annotations"]["0"]["notes"] == notes
    finally:
        lab.downloads.close()


@pytest.mark.anyio
async def test_export_encoding_does_not_hold_the_cold_parse_lock(store, monkeypatch):
    from circuit_lantern.api import crosscoders

    identifier, other = saved(store)["id"], saved(store)["id"]
    entered, resume = Event(), Event()
    original = crosscoders.jsonable_encoder

    def gated(value):
        entered.set()
        assert resume.wait(5)
        return original(value)

    lab = CrosscoderLab(None, store)
    monkeypatch.setattr(app.state, "crosscoders", lab, raising=False)
    monkeypatch.setattr(crosscoders, "jsonable_encoder", gated)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            exporting = asyncio.create_task(
                client.get("/api/crosscoders/" + identifier + "/export")
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                value = await asyncio.wait_for(asyncio.to_thread(store.get_readonly, other), 1)
                assert value["id"] == other
            finally:
                resume.set()
                response = await exporting
            assert response.status_code == 200
            assert store._metadata_memory._reserved == 0
    finally:
        resume.set()
        lab.downloads.close()


@pytest.mark.anyio
async def test_export_reports_concurrent_deletion_instead_of_silently_dropping_notes(
    store, monkeypatch
):
    identifier = saved(store)["id"]
    store.label(identifier, 0, "Observation", "Retain this annotation")
    entered, resume = Event(), Event()
    original = store.read_annotations

    @contextmanager
    def gated(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        with original(*args, **kwargs) as value:
            yield value

    lab = CrosscoderLab(None, store)
    monkeypatch.setattr(app.state, "crosscoders", lab, raising=False)
    monkeypatch.setattr(store, "read_annotations", gated)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            exporting = asyncio.create_task(
                client.get("/api/crosscoders/" + identifier + "/export")
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert await asyncio.to_thread(store.delete, identifier)
            finally:
                resume.set()
                response = await exporting
            assert response.status_code == 404
            assert store._metadata_memory._reserved == 0
    finally:
        resume.set()
        lab.downloads.close()


@pytest.mark.parametrize("reader", ["list", "search_annotations"])
def test_lists_and_annotation_search_fail_quickly_under_sqlite_writer_contention(store, reader):
    identifier = saved(store)["id"]
    operation = (
        (lambda: store.list("crosscoder"))
        if reader == "list"
        else (lambda: store.search_annotations(identifier, "observation"))
    )
    with sqlite3.connect(store.database) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(ArtifactBusyError):
                executor.submit(operation).result(timeout=1)
        finally:
            connection.rollback()
    if reader == "list":
        assert operation()[0]["id"] == identifier
    else:
        assert operation() == set()


def test_tensor_limits_and_validation_memory_are_checked_before_finiteness(store, monkeypatch):
    finite = Mock(side_effect=AssertionError("Finiteness ran before the memory guard"))
    monkeypatch.setattr(torch, "isfinite", finite)
    with pytest.raises(ValueError, match="configured size limit"):
        store.save("crosscoder", {"name": "Too large"}, {"values": torch.ones(300)}, {})
    with monkeypatch.context() as patcher:
        patcher.setattr("circuit_lantern.services.artifact_views.available_memory_bytes", lambda: 1)
        with pytest.raises(MetadataMemoryError):
            store.save(
                "crosscoder", {"name": "No validation memory"}, {"values": torch.ones(2)}, {}
            )
    finite.assert_not_called()
    assert store._metadata_memory._reserved == 0
