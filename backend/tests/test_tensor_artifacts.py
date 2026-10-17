"""Durability and integrity checks for retained activation and crosscoder tensors."""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from filelock import FileLock

from circuit_lantern.services.tensor_artifacts import (
    ArtifactBusyError,
    ArtifactNotFoundError,
    TensorArtifactStore,
    canonical_json,
    digest_bytes,
)


def save(store, kind="capture"):
    manifest = {"name": "Recorded tensors", "features": 4, "source": {"revision": "pinned"}}
    values = {"activations": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    return store.save(kind, manifest, values, {"samples": 3, "features": 4}), values


def test_round_trip_metadata_checksums_and_scoped_lists(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, values = save(store)
    assert len(manifest["id"]) == 32
    assert manifest["source"]["revision"] == "pinned"
    assert store.get(manifest["id"], kind="crosscoder") is None
    loaded, actual = store.load(manifest["id"], kind="capture")
    assert loaded == manifest
    assert torch.equal(actual["activations"], values["activations"])
    assert store.list("crosscoder") == []
    assert store.list("capture")[0]["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["manifest_sha256"] == digest_bytes(
        canonical_json(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ).encode()
    )


def test_changed_tensor_bytes_are_rejected(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    path = tmp_path / (manifest["id"] + ".safetensors")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        store.load(manifest["id"], kind="capture")


@pytest.mark.parametrize("column", ["manifest", "summary"])
def test_changed_metadata_is_rejected(tmp_path, column):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            f"UPDATE artifacts SET {column} = ? WHERE id = ?",
            ('{"name":"changed"}', manifest["id"]),
        )
    if column == "manifest":
        with pytest.raises(ValueError, match="checksum"):
            store.get(manifest["id"])
    else:
        issue = store.list("capture")[0]
        assert issue["id"] == manifest["id"] and issue["integrity_error"]
        assert issue["name"] == manifest["id"]
    healthy, _ = save(store)
    assert healthy["id"] in {row["id"] for row in store.list("capture")}
    assert store.delete(manifest["id"], kind="capture")
    assert store.get(manifest["id"]) is None
    assert [row["id"] for row in store.list("capture")] == [healthy["id"]]


def test_nonfinite_and_oversized_tensors_never_publish(tmp_path):
    store = TensorArtifactStore(tmp_path, 8)
    for tensor in [torch.ones(3), torch.tensor([float("nan")]), torch.ones(1, dtype=torch.float64)]:
        with pytest.raises(ValueError):
            store.save("capture", {"name": "test"}, {"values": tensor}, {})
    assert store.list("capture") == []
    assert list(tmp_path.glob("*.safetensors")) == []
    assert list(tmp_path.glob("*.partial")) == []


def test_publication_failure_removes_weight_file(tmp_path, monkeypatch):
    store = TensorArtifactStore(tmp_path, 1024)
    original = store.connect

    def unavailable():
        raise sqlite3.OperationalError("write failed")

    monkeypatch.setattr(store, "connect", unavailable)
    with pytest.raises(sqlite3.OperationalError, match="write failed"):
        save(store)
    assert list(tmp_path.glob("*.safetensors")) == []
    monkeypatch.setattr(store, "connect", original)
    assert store.list("capture") == []


def test_failed_delete_restores_weights_and_metadata(tmp_path, monkeypatch):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, values = save(store)
    original = store.connect
    calls = 0

    def unavailable():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.OperationalError("delete failed")
        return original()

    monkeypatch.setattr(store, "connect", unavailable)
    with pytest.raises(sqlite3.OperationalError, match="delete failed"):
        store.delete(manifest["id"])
    monkeypatch.setattr(store, "connect", original)
    _, restored = store.load(manifest["id"], kind="capture")
    assert torch.equal(restored["activations"], values["activations"])


def test_startup_recovers_interrupted_deletion_and_unpublished_files(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, values = save(store)
    path = tmp_path / (manifest["id"] + ".safetensors")
    path.replace(tmp_path / (".trash-" + manifest["id"] + ".safetensors"))
    orphan = "f" * 32
    (tmp_path / (orphan + ".safetensors")).write_bytes(b"orphan")
    (tmp_path / ("." + orphan + ".partial")).write_bytes(b"partial")
    recovered = TensorArtifactStore(tmp_path, 1024)
    _, actual = recovered.load(manifest["id"], kind="capture")
    assert torch.equal(actual["activations"], values["activations"])
    assert not (tmp_path / (orphan + ".safetensors")).exists()
    assert (tmp_path / "recovery" / (orphan + ".safetensors")).read_bytes() == b"orphan"
    assert not (tmp_path / ("." + orphan + ".partial")).exists()


def test_annotations_are_separate_from_immutable_evidence_and_cascade_on_delete(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store, "crosscoder")
    note = store.label(manifest["id"], 2, "Candidate feature", "Test this context")
    assert note["feature"] == 2
    assert store.get(manifest["id"]) == manifest
    assert store.annotations(manifest["id"])[2]["label"] == "Candidate feature"
    with pytest.raises(ValueError, match="existing"):
        store.label(manifest["id"], 4, "", "")
    assert store.delete(manifest["id"])
    assert store.get(manifest["id"]) is None
    assert store.annotations(manifest["id"]) == {}
    assert not store.delete(manifest["id"])


def test_busy_store_returns_without_waiting(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    lock = FileLock(tmp_path / ".artifacts.lock")
    with lock, pytest.raises(ArtifactBusyError):
        save(store)
    assert store.list("capture") == []


def test_missing_and_invalid_identifiers_are_distinct(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    assert store.get("a" * 32) is None
    with pytest.raises(FileNotFoundError):
        store.load("a" * 32, kind="capture")
    with pytest.raises(ValueError, match="identifier"):
        store.get("../private")


def test_load_checks_available_memory_before_reading_tensor_bytes(tmp_path, monkeypatch):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)

    class Memory:
        available = 1

    monkeypatch.setattr("circuit_lantern.services.tensor_artifacts.psutil.virtual_memory", Memory)
    with pytest.raises(ValueError, match="RAM"):
        store.load(manifest["id"], kind="capture")


@pytest.mark.parametrize("operation", ["load", "open_tensors"])
def test_missing_physical_file_has_specific_not_found_error(tmp_path, operation):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    (tmp_path / (manifest["id"] + ".safetensors")).unlink()
    with pytest.raises(ArtifactNotFoundError, match="tensor file"):
        getattr(store, operation)(manifest["id"], kind="capture")
    assert store.get(manifest["id"]) == manifest
    assert store.delete(manifest["id"])


def test_stream_export_checks_integrity_and_survives_deletion(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    stream, length = store.open_tensors(manifest["id"], kind="capture")
    try:
        assert store.delete(manifest["id"])
        raw = stream.read()
        assert len(raw) == length
        assert digest_bytes(raw) == manifest["tensors_sha256"]
    finally:
        stream.close()


def test_filtered_annotations_and_unicode_search(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store, "crosscoder")
    store.label(manifest["id"], 1, "STRASSE", "Reference")
    store.label(manifest["id"], 2, "Other", "Compared")
    assert set(store.annotations(manifest["id"], [2])) == {2}
    assert store.annotations(manifest["id"], []) == {}
    assert store.search_annotations(manifest["id"], "Straße") == {1}
    assert store.search_annotations(manifest["id"], "compared") == {2}


def test_lost_database_preserves_retained_tensor_bytes_for_recovery(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    path = tmp_path / (manifest["id"] + ".safetensors")
    original = path.read_bytes()
    store.database.unlink()
    reopened = TensorArtifactStore(tmp_path, 1024)
    assert reopened.list("capture") == []
    recovered = tmp_path / "recovery" / path.name
    assert recovered.read_bytes() == original
    path.write_bytes(b"second unknown file")
    TensorArtifactStore(tmp_path, 1024)
    assert recovered.read_bytes() == original
    assert sorted(item.read_bytes() for item in recovered.parent.iterdir()) == sorted(
        [original, b"second unknown file"]
    )


def test_wrong_kind_cannot_delete_a_retained_artifact(tmp_path):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, values = save(store)
    assert not store.delete(manifest["id"], kind="crosscoder")
    _, actual = store.load(manifest["id"], kind="capture")
    assert torch.equal(actual["activations"], values["activations"])


def test_download_verification_allows_annotations_and_deletion(tmp_path, monkeypatch):
    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store, "crosscoder")
    original = hashlib.file_digest
    hashing = Event()
    resume = Event()

    def delayed_digest(stream, algorithm):
        hashing.set()
        assert resume.wait(5), "Concurrent annotation did not finish"
        return original(stream, algorithm)

    monkeypatch.setattr(
        "circuit_lantern.services.tensor_artifacts.hashlib.file_digest", delayed_digest
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        opening = executor.submit(store.open_tensors, manifest["id"], kind="crosscoder")
        try:
            assert hashing.wait(5), "Download verification did not start"
            annotation = store.label(manifest["id"], 2, "Retained context", "Still editable")
            assert annotation["label"] == "Retained context"
            assert store.delete(manifest["id"], kind="crosscoder")
        finally:
            resume.set()
        stream, length = opening.result(timeout=5)
        try:
            raw = stream.read()
            assert len(raw) == length
            assert digest_bytes(raw) == manifest["tensors_sha256"]
        finally:
            stream.close()


def test_cancelled_checksum_releases_the_pinned_file(tmp_path, monkeypatch):
    from circuit_lantern.services.prepared_downloads import DownloadCancelledError

    store = TensorArtifactStore(tmp_path, 1024)
    manifest, _ = save(store)
    opened = []
    original = type(tmp_path).open

    def tracking(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path.suffix == ".safetensors":
            opened.append(stream)
        return stream

    monkeypatch.setattr(type(tmp_path), "open", tracking)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(DownloadCancelledError):
        store.open_tensors(manifest["id"], kind="capture", cancelled=cancelled)
    assert opened and all(stream.closed for stream in opened)
    assert store.delete(manifest["id"])
