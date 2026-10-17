"""Atomic immutable tensor artifacts with independently verified metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch
from filelock import FileLock, Timeout
from safetensors import SafetensorError
from safetensors.torch import load, save_file

from circuit_lantern.services.artifact_views import (
    MAX_ARTIFACT_METADATA_BYTES as _MAX_METADATA_BYTES,
)
from circuit_lantern.services.artifact_views import (
    METADATA_DECODE_RAM_MULTIPLIER,
    METADATA_JSON_RAM_MULTIPLIER,
    METADATA_TEXT_RAM_MULTIPLIER,
    ArtifactManifestViews,
    ManifestSnapshot,
    MetadataMemoryBudget,
    checked_metadata_size,
)
from circuit_lantern.services.prepared_downloads import DownloadCancelledError
from circuit_lantern.services.resources import available_memory_bytes

_IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
_HEADER_ALLOWANCE = 64 * 1024


class ArtifactBusyError(Exception):
    pass


class ArtifactNotFoundError(FileNotFoundError):
    pass


def canonical_json(value: dict) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TensorArtifactStore:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.database = directory / "artifacts.sqlite3"
        self.file_lock = FileLock(directory / ".artifacts.lock")
        self._views = ArtifactManifestViews()
        self._metadata_memory = MetadataMemoryBudget()
        with self.file_lock:
            with self.connect() as connection:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS artifacts (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        summary_sha256 TEXT NOT NULL,
                        manifest TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS labels (
                        artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                        feature INTEGER NOT NULL,
                        label TEXT NOT NULL,
                        notes TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (artifact_id, feature)
                    );
                """)
                known = {row["id"] for row in connection.execute("SELECT id FROM artifacts")}
            for path in directory.iterdir():
                if re.fullmatch(r"\.[a-f0-9]{32}\.partial", path.name):
                    path.unlink(missing_ok=True)
                elif re.fullmatch(r"\.trash-[a-f0-9]{32}\.safetensors", path.name):
                    identifier = path.name[7:-12]
                    if identifier in known:
                        path.replace(self._path(identifier))
                    else:
                        path.unlink(missing_ok=True)
                elif (
                    path.suffix == ".safetensors"
                    and _IDENTIFIER.fullmatch(path.stem)
                    and path.stem not in known
                ):
                    recovery = directory / "recovery"
                    recovery.mkdir(exist_ok=True)
                    destination = recovery / path.name
                    if destination.exists():
                        destination = recovery / (path.stem + "-" + uuid.uuid4().hex + path.suffix)
                    path.replace(destination)
                    self._sync_directory()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=0.1)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", 0) & 255 in (
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            ):
                raise ArtifactBusyError(
                    "The saved tensor artifacts are busy. Try again shortly."
                ) from error
            raise
        finally:
            connection.close()

    @contextmanager
    def operation(self):
        try:
            self.file_lock.acquire(timeout=0)
        except Timeout as error:
            raise ArtifactBusyError(
                "The saved tensor artifacts are busy. Try again shortly."
            ) from error
        try:
            yield
        finally:
            self.file_lock.release()

    @contextmanager
    def _read_connection(self):
        with self.connect() as connection:
            connection.execute("BEGIN")
            yield connection

    def _path(self, identifier: str) -> Path:
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError("The tensor artifact identifier is invalid.")
        return self.directory / (identifier + ".safetensors")

    def _sync_directory(self):
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _decode(encoded: str, expected: str) -> dict:
        checked_metadata_size(encoded, expected)
        value = json.loads(encoded)
        if not isinstance(value, dict) or value.get("kind") not in ("capture", "crosscoder"):
            raise ValueError("The saved artifact metadata is invalid.")
        return value

    @contextmanager
    def _read_manifest(self, identifier: str, kind: str | None):
        self._path(identifier)
        with ExitStack() as reservations:
            loaded = None
            # Size and payload belong to one SQLite snapshot. Close it before decoding.
            with self._read_connection() as connection:
                metadata = connection.execute(
                    "SELECT rowid, kind FROM artifacts WHERE id = ?", (identifier,)
                ).fetchone()
                if metadata is not None:
                    loaded = ManifestSnapshot(metadata["kind"])
                    if kind is None or metadata["kind"] == kind:
                        with connection.blobopen(
                            "artifacts", "manifest", metadata["rowid"], readonly=True
                        ) as manifest:
                            size = len(manifest)
                        if size > _MAX_METADATA_BYTES:
                            raise ValueError("The saved artifact metadata checksum does not match.")
                        # Astral text can widen the whole Python string to four bytes.
                        reservations.enter_context(
                            self._metadata_memory.reserve(size * METADATA_TEXT_RAM_MULTIPLIER)
                        )
                        row = connection.execute(
                            "SELECT manifest, manifest_sha256 FROM artifacts WHERE id = ?",
                            (identifier,),
                        ).fetchone()
                        loaded = ManifestSnapshot(
                            metadata["kind"], row["manifest"], row["manifest_sha256"], size
                        )
            yield loaded

    @contextmanager
    def _decode_manifest(
        self, identifier: str, loaded: ManifestSnapshot, *, serializing=False, freezing=False
    ):
        multiplier = METADATA_JSON_RAM_MULTIPLIER if serializing else METADATA_DECODE_RAM_MULTIPLIER
        with (
            ExitStack() as parsing,
            self._metadata_memory.reserve(
                loaded.size * (multiplier - METADATA_TEXT_RAM_MULTIPLIER)
            ),
        ):
            parsing.enter_context(self._metadata_memory.decode_lock)
            payload = self._decode(loaded.encoded, loaded.digest)
            if payload.get("id") != identifier or payload["kind"] != loaded.kind:
                raise ValueError("The saved artifact metadata is invalid.")
            # Freeze under the cold-parse lock, but let exports encode independently.
            # Both paths retain their memory reservation through the caller's work.
            if not freezing:
                parsing.close()
            yield dict(payload, manifest_sha256=loaded.digest)

    @contextmanager
    def read_metadata(self, identifier: str, *, kind: str | None = None, serializing=False):
        with self._read_manifest(identifier, kind) as loaded:
            if loaded is None or (kind is not None and loaded.kind != kind):
                yield None
            else:
                with self._decode_manifest(identifier, loaded, serializing=serializing) as value:
                    yield value

    def get(self, identifier: str, *, kind: str | None = None) -> dict | None:
        with self.read_metadata(identifier, kind=kind) as value:
            return value

    def get_readonly(self, identifier: str, *, kind: str | None = None) -> Mapping | None:
        return self._views.read(
            identifier,
            lambda: self._read_manifest(identifier, kind),
            lambda loaded: self._decode_manifest(identifier, loaded, freezing=True),
            kind=kind,
        )

    def list(self, kind: str) -> list[dict]:
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT id, kind, created_at, summary, summary_sha256 FROM artifacts "
                "WHERE kind = ? ORDER BY created_at DESC, id",
                (kind,),
            ).fetchall()
        summaries = []
        for row in rows:
            try:
                summary = self._decode(row["summary"], row["summary_sha256"])
                if summary.get("id") != row["id"] or summary.get("kind") != row["kind"]:
                    raise ValueError("The saved artifact metadata is invalid.")
            except (ValueError, TypeError, AttributeError):
                summary = {
                    "id": row["id"],
                    "kind": row["kind"],
                    "created_at": row["created_at"],
                    "name": row["id"],
                    "integrity_error": "The saved artifact summary failed its integrity check. You can delete this record from retained data.",
                }
            summaries.append(summary)
        return summaries

    def save(
        self, kind: str, manifest: dict, tensors: dict[str, torch.Tensor], summary: dict
    ) -> dict:
        if (
            kind not in ("capture", "crosscoder")
            or not isinstance(manifest.get("name"), str)
            or not manifest["name"].strip()
        ):
            raise ValueError("A named capture or crosscoder artifact is required.")
        if not tensors or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type == "meta"
            or value.layout != torch.strided
            for value in tensors.values()
        ):
            raise ValueError("Saved artifacts require finite float32 tensors.")
        tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        if tensor_bytes > self.max_bytes:
            raise ValueError("The tensor artifact exceeds the configured size limit.")
        validation_bytes = max(
            (
                value.numel() * value.element_size() * 2
                for value in tensors.values()
                if value.device.type == "cpu"
            ),
            default=0,
        )
        with self._metadata_memory.reserve(validation_bytes):
            if any(not torch.isfinite(value).all() for value in tensors.values()):
                raise ValueError("Saved artifacts require finite float32 tensors.")
        with self.operation(), ExitStack() as reservations:
            identifier = uuid.uuid4().hex
            destination = self._path(identifier)
            temporary = self.directory / ("." + identifier + ".partial")
            if destination.exists():
                raise ValueError("An artifact already uses this identifier. Retry the save.")
            payload = dict(
                manifest,
                id=identifier,
                kind=kind,
                schema_version=1,
                created_at=datetime.now(UTC).isoformat(),
            )
            initial = canonical_json(payload)
            if len(initial.encode()) > _MAX_METADATA_BYTES:
                raise ValueError("The tensor artifact metadata exceeds the supported size limit.")
            copy_bytes = sum(
                value.numel() * value.element_size()
                for value in tensors.values()
                if value.device.type != "cpu" or not value.is_contiguous()
            )
            # Publish only when the same process can also browse and export the result.
            reservations.enter_context(
                self._metadata_memory.reserve(
                    copy_bytes + (len(initial.encode()) + 128) * METADATA_JSON_RAM_MULTIPLIER
                )
            )
            if tensor_bytes * 2 + len(initial.encode()) * 2 + _HEADER_ALLOWANCE > (
                shutil.disk_usage(self.directory).free * 0.9
            ):
                raise ValueError("Insufficient disk space to publish this tensor artifact.")
            published = False
            try:
                save_file(
                    {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
                    temporary,
                    metadata={"format": "circuit_lantern_" + kind + "_v1"},
                )
                if temporary.stat().st_size > self.max_bytes + _HEADER_ALLOWANCE:
                    raise ValueError("The tensor artifact exceeds the configured file size limit.")
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                payload["tensors_sha256"] = digest
                encoded = canonical_json(payload)
                if len(encoded.encode()) > _MAX_METADATA_BYTES:
                    raise ValueError(
                        "The tensor artifact metadata exceeds the supported size limit."
                    )
                digest = digest_bytes(encoded.encode())
                brief = dict(
                    summary,
                    id=identifier,
                    kind=kind,
                    name=payload["name"],
                    created_at=payload["created_at"],
                    tensors_sha256=payload["tensors_sha256"],
                    manifest_sha256=digest,
                )
                summary_text = canonical_json(brief)
                temporary.replace(destination)
                self._sync_directory()
                with self.connect() as connection:
                    connection.execute(
                        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            identifier,
                            kind,
                            payload["created_at"],
                            summary_text,
                            digest_bytes(summary_text.encode()),
                            encoded,
                            digest,
                        ),
                    )
                published = True
                return dict(payload, manifest_sha256=digest)
            finally:
                temporary.unlink(missing_ok=True)
                if not published:
                    destination.unlink(missing_ok=True)

    def load(self, identifier: str, *, kind: str) -> tuple[dict, dict[str, torch.Tensor]]:
        with self.operation():
            manifest = self.get(identifier, kind=kind)
            if manifest is None:
                raise ArtifactNotFoundError("The saved tensor artifact was not found.")
            path = self._path(identifier)
            try:
                with path.open("rb") as stream:
                    size = os.fstat(stream.fileno()).st_size
                    if size > self.max_bytes + _HEADER_ALLOWANCE:
                        raise ValueError(
                            "The saved artifact exceeds the configured file size limit."
                        )
                    if size * 4 > available_memory_bytes() * 0.75:
                        raise ValueError(
                            "Insufficient available RAM to load the saved tensor artifact."
                        )
                    data = stream.read(self.max_bytes + _HEADER_ALLOWANCE + 1)
            except FileNotFoundError as error:
                raise ArtifactNotFoundError("The saved tensor file was not found.") from error
            if len(data) != size or digest_bytes(data) != manifest["tensors_sha256"]:
                raise ValueError("The saved tensor artifact checksum does not match.")
            try:
                tensors = load(data)
            except SafetensorError as error:
                raise ValueError("The saved artifact is not a valid safetensors file.") from error
            if any(
                value.dtype != torch.float32 or not torch.isfinite(value).all()
                for value in tensors.values()
            ):
                raise ValueError("The saved artifact contains invalid tensor values.")
            return manifest, tensors

    def open_tensors(
        self,
        identifier: str,
        *,
        kind: str,
        cancelled: Callable[[], bool] | None = None,
    ):
        if cancelled and cancelled():
            raise DownloadCancelledError()
        with self.operation():
            manifest = self.get(identifier, kind=kind)
            if manifest is None:
                raise ArtifactNotFoundError("The saved tensor artifact was not found.")
            expected = manifest.get("tensors_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("The saved artifact metadata is invalid.")
            try:
                stream = self._path(identifier).open("rb")
            except FileNotFoundError as error:
                raise ArtifactNotFoundError("The saved tensor file was not found.") from error
        try:
            size = os.fstat(stream.fileno()).st_size
            if size > self.max_bytes + _HEADER_ALLOWANCE:
                raise ValueError("The saved artifact exceeds the configured file size limit.")
            if cancelled is None:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                digest = hashlib.sha256()
                while True:
                    if cancelled():
                        raise DownloadCancelledError()
                    chunk = stream.read(1024**2)
                    if not chunk:
                        break
                    digest.update(chunk)
                actual = digest.hexdigest()
            if actual != expected:
                raise ValueError("The saved tensor artifact checksum does not match.")
            stream.seek(0)
            return stream, size
        except BaseException:
            stream.close()
            raise

    def delete(self, identifier: str, *, kind: str | None = None) -> bool:
        with self.operation():
            path = self._path(identifier)
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT kind FROM artifacts WHERE id = ?", (identifier,)
                ).fetchone()
            if row is None or (kind is not None and row["kind"] != kind):
                return False
            trash = self.directory / (".trash-" + identifier + ".safetensors")
            moved = False
            try:
                if path.exists():
                    path.replace(trash)
                    moved = True
                    self._sync_directory()
                with self.connect() as connection:
                    connection.execute("DELETE FROM artifacts WHERE id = ?", (identifier,))
            except BaseException:
                if moved:
                    trash.replace(path)
                    self._sync_directory()
                raise
            trash.unlink(missing_ok=True)
            self._sync_directory()
            self._views.discard(identifier)
            return True

    @contextmanager
    def read_annotations(
        self,
        identifier: str,
        features: list[int] | None = None,
        *,
        serializing=False,
        require_artifact=False,
    ):
        self._path(identifier)
        if features == []:
            yield {}
            return
        suffix = ""
        arguments = [identifier]
        if features is not None:
            suffix = " AND feature IN (" + ",".join("?" for _ in features) + ")"
            arguments.extend(features)
        with ExitStack() as reservations:
            with self._read_connection() as connection:
                if (
                    require_artifact
                    and connection.execute(
                        "SELECT 1 FROM artifacts WHERE id = ?", (identifier,)
                    ).fetchone()
                    is None
                ):
                    raise ArtifactNotFoundError("The saved artifact was not found.")
                size, count = connection.execute(
                    "SELECT COALESCE(SUM(length(CAST(label AS BLOB)) + "
                    "length(CAST(notes AS BLOB))), 0), COUNT(*) "
                    "FROM labels WHERE artifact_id = ?" + suffix,
                    arguments,
                ).fetchone()
                # Text volume is independent of manifest size. Reserve before fetching any notes.
                reservations.enter_context(
                    self._metadata_memory.reserve(
                        size * (24 if serializing else 8) + count * (2048 if serializing else 1024)
                    )
                )
                rows = connection.execute(
                    "SELECT feature, label, notes, updated_at FROM labels WHERE artifact_id = ?"
                    + suffix,
                    arguments,
                ).fetchall()
            yield {row["feature"]: dict(row) for row in rows}

    def annotations(self, identifier: str, features: list[int] | None = None) -> dict[int, dict]:
        with self.read_annotations(identifier, features) as value:
            return value

    def search_annotations(self, identifier: str, query: str) -> set[int]:
        self._path(identifier)
        needle = query.casefold()
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT feature, label, notes FROM labels WHERE artifact_id = ?",
                (identifier,),
            )
            return {
                row["feature"]
                for row in rows
                if needle in row["label"].casefold() or needle in row["notes"].casefold()
            }

    def label(self, identifier: str, feature: int, label: str, notes: str) -> dict:
        with self.operation():
            manifest = self.get_readonly(identifier, kind="crosscoder")
            if manifest is None:
                raise ArtifactNotFoundError("The crosscoder was not found.")
            if type(feature) is not int or not 0 <= feature < manifest["features"]:
                raise ValueError("Choose an existing crosscoder feature.")
            if (
                not isinstance(label, str)
                or len(label) > 100
                or not isinstance(notes, str)
                or len(notes) > 10000
            ):
                raise ValueError("The feature label or notes exceed the supported length.")
            updated = datetime.now(UTC).isoformat()
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO labels VALUES (?, ?, ?, ?, ?) ON CONFLICT (artifact_id, feature) "
                    "DO UPDATE SET label = excluded.label, notes = excluded.notes, updated_at = excluded.updated_at",
                    (identifier, feature, label, notes, updated),
                )
            return {
                "artifact_id": identifier,
                "feature": feature,
                "label": label,
                "notes": notes,
                "updated_at": updated,
            }
