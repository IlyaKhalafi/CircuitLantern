"""Durable dictionary manifests and checksummed safetensors weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

import torch
from filelock import FileLock
from safetensors import SafetensorError
from safetensors.torch import load, save_file

from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.methods import method_record

_ID = re.compile(r"^[a-f0-9]{32}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_summary(result: dict) -> dict:
    keys = (
        "id",
        "kind",
        "model_id",
        "revision",
        "prompt",
        "created_at",
        "complete",
        "cancelled",
        "dictionary_ids",
    )
    return {key: result[key] for key in keys if key in result}


def artifact_operation(function):
    @wraps(function)
    def operation(self, *args, **kwargs):
        with self.file_lock:
            return function(self, *args, **kwargs)

    return operation


class DictionaryStore:
    def __init__(self, directory: Path, max_weight_bytes: int = 512 * 1024**2):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_weight_bytes = max_weight_bytes
        self.file_lock = FileLock(self.directory / ".artifacts.lock")

        self.database = directory / "dictionaries.sqlite3"
        self._initialize()

    @artifact_operation
    def _initialize(self):
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dictionaries (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    weights_sha256 TEXT NOT NULL,
                    manifest_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS feature_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    summary TEXT
                );
                CREATE TABLE IF NOT EXISTS feature_labels (
                    dictionary_id TEXT NOT NULL REFERENCES dictionaries(id) ON DELETE CASCADE,
                    feature INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    PRIMARY KEY (dictionary_id, feature)
                );
                """
            )

        with self.connect() as connection:
            run_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(feature_runs)")
            }
            if "summary" not in run_columns:
                connection.execute("ALTER TABLE feature_runs ADD COLUMN summary TEXT")
            for row in connection.execute(
                "SELECT id, result FROM feature_runs WHERE summary IS NULL"
            ).fetchall():
                summary = _run_summary(json.loads(row["result"]))
                connection.execute(
                    "UPDATE feature_runs SET summary = ? WHERE id = ?",
                    (json.dumps(summary), row["id"]),
                )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(dictionaries)")}
            if "manifest_sha256" not in columns:
                connection.execute("ALTER TABLE dictionaries ADD COLUMN manifest_sha256 TEXT")
            for row in connection.execute(
                "SELECT id, manifest FROM dictionaries WHERE manifest_sha256 IS NULL"
            ).fetchall():
                digest = hashlib.sha256(row["manifest"].encode()).hexdigest()
                connection.execute(
                    "UPDATE dictionaries SET manifest_sha256 = ? WHERE id = ?", (digest, row["id"])
                )

        # Writers and startup recovery share an interprocess lock. A second
        # server cannot sweep an artifact while another process is publishing it.
        with self.file_lock:
            with self.connect() as connection:
                known = {row["id"] for row in connection.execute("SELECT id FROM dictionaries")}
            for path in self.directory.iterdir():
                name = path.name
                if re.fullmatch(r"\.[a-f0-9]{32}\.partial", name):
                    path.unlink(missing_ok=True)
                elif re.fullmatch(r"\.trash-[a-f0-9]{32}\.safetensors", name):
                    identifier = name[7:-12]
                    if identifier in known:
                        path.replace(self._weights_path(identifier))
                    else:
                        path.unlink(missing_ok=True)
                elif (
                    path.suffix == ".safetensors"
                    and _ID.fullmatch(path.stem)
                    and path.stem not in known
                ):
                    path.unlink(missing_ok=True)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _weights_path(self, identifier: str) -> Path:
        if not _ID.fullmatch(identifier):
            raise ValueError("The dictionary identifier is invalid.")
        return self.directory / (identifier + ".safetensors")

    @artifact_operation
    def save(self, manifest: dict, model: SparseAutoencoder) -> dict:
        if any(
            not isinstance(manifest.get(key), str) or not manifest[key].strip()
            for key in ("name", "model_id", "revision")
        ):
            raise ValueError("A dictionary needs a name and a recorded checkpoint identity.")
        if not isinstance(manifest.get("layer"), int) or manifest["layer"] < 0:
            raise ValueError("A dictionary must identify its transformer layer.")
        corpus = manifest.get("corpus", {})
        training = manifest.get("training_summary", {})
        if (
            not _ID.fullmatch(manifest.get("execution_id", ""))
            or not isinstance(manifest.get("execution_config"), dict)
            or not manifest.get("execution_precision")
            or not re.fullmatch(r"[a-f0-9]{64}", corpus.get("fingerprint", ""))
            or not corpus.get("documents")
            or not corpus.get("samples")
            or not corpus.get("split_method")
            or any(
                not isinstance(corpus.get(key), bool)
                for key in ("special_tokens_excluded", "first_position_excluded")
            )
            or not 1 <= training.get("completed_steps", 0) <= model.config.steps
            or training.get("effective_batch_size", 0) < 1
            or training.get("tokens_seen")
            != training.get("completed_steps", 0) * training.get("effective_batch_size", 0)
        ):
            raise ValueError(
                "A dictionary requires execution, corpus, split and completed training provenance."
            )
        mean = manifest.get("training_mean")
        if (
            not isinstance(mean, list)
            or len(mean) != model.input_dim
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in mean)
        ):
            raise ValueError(
                "The recorded training mean must be finite and match the residual width."
            )
        state = model.state_dict()
        weight_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        if weight_bytes > self.max_weight_bytes:
            raise ValueError("The dictionary weights exceed the configured artifact size limit.")
        if any(tensor.dtype != torch.float32 for tensor in state.values()):
            raise ValueError("Dictionary artifacts require float32 weights.")
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Non finite dictionary weights cannot be saved.")
        if model.input_scale <= 0:
            raise ValueError("The dictionary activation scale must be positive.")
        if not torch.allclose(
            model.decoder.norm(dim=0),
            torch.ones(model.config.features, device=model.decoder.device),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("Dictionary decoder directions must have unit norm.")
        identifier = uuid.uuid4().hex
        destination = self._weights_path(identifier)
        temporary = self.directory / ("." + identifier + ".partial")
        if destination.exists():
            raise ValueError("A dictionary artifact already uses this identifier. Retry the save.")
        published = False
        payload = dict(
            manifest,
            id=identifier,
            schema_version=1,
            created_at=datetime.now(UTC).isoformat(),
            input_dim=model.input_dim,
            features=model.config.features,
            method=model.config.method,
            training=model.config.description(),
            capture_point="residual_post",
            method_record=method_record(model.config),
        )
        # JSON serialization happens before writing an artifact, so invalid
        # measurements cannot leave a published weight file behind.
        json.dumps(payload, allow_nan=False)
        try:
            save_file(
                {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
                temporary,
                metadata={"format": "circuit_lantern_sae_v1"},
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            payload["weights_sha256"] = _sha256(temporary)
            encoded = json.dumps(payload, allow_nan=False)
            summary = {
                key: payload.get(key)
                for key in (
                    "id",
                    "name",
                    "model_id",
                    "revision",
                    "execution_precision",
                    "layer",
                    "created_at",
                    "features",
                    "input_dim",
                    "method",
                    "train_metrics",
                    "validation_metrics",
                    "training_summary",
                )
            }
            for key in ("train_metrics", "validation_metrics"):
                summary[key] = {
                    name: value
                    for name, value in (summary[key] or {}).items()
                    if name != "feature_activation_counts"
                }
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO dictionaries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        payload["name"],
                        payload["model_id"],
                        payload["revision"],
                        payload["layer"],
                        payload["created_at"],
                        json.dumps(summary, allow_nan=False),
                        encoded,
                        payload["weights_sha256"],
                        hashlib.sha256(encoded.encode()).hexdigest(),
                    ),
                )
                temporary.replace(destination)
                published = True
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return summary
        except Exception:
            if published:
                destination.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def list(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT summary FROM dictionaries ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(row["summary"]) for row in rows]

    def get(self, identifier: str) -> dict | None:
        self._weights_path(identifier)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT name, manifest, manifest_sha256 FROM dictionaries WHERE id = ?",
                (identifier,),
            ).fetchone()
            if row is None:
                return None
            labels = connection.execute(
                "SELECT feature, label, notes FROM feature_labels WHERE dictionary_id = ?",
                (identifier,),
            ).fetchall()
        if hashlib.sha256(row["manifest"].encode()).hexdigest() != row["manifest_sha256"]:
            raise ValueError("The dictionary manifest has changed since it was recorded.")
        return dict(
            json.loads(row["manifest"]),
            name=row["name"],
            manifest_sha256=row["manifest_sha256"],
            feature_labels={
                str(label["feature"]): {"label": label["label"], "notes": label["notes"]}
                for label in labels
            },
        )

    @artifact_operation
    def load_model(self, identifier: str) -> tuple[dict, SparseAutoencoder]:
        manifest = self.get(identifier)
        if manifest is None:
            raise ValueError("The dictionary was not found.")
        path = self._weights_path(identifier)
        if not path.is_file() or path.stat().st_size > self.max_weight_bytes + 1024**2:
            raise ValueError("The dictionary weight file is missing or exceeds its size limit.")
        with path.open("rb") as stream:
            data = stream.read(self.max_weight_bytes + 1024**2 + 1)
        if len(data) > self.max_weight_bytes + 1024**2:
            raise ValueError("The dictionary weights exceed their size limit.")
        if hashlib.sha256(data).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("The dictionary weights have changed since they were recorded.")
        if manifest.get("schema_version") != 1:
            raise ValueError("This dictionary schema is not supported by the current runtime.")
        if set(manifest.get("training", {})) != {field.name for field in fields(TrainingConfig)}:
            raise ValueError("The dictionary training configuration is incomplete.")
        try:
            config = TrainingConfig(**manifest["training"])
        except TypeError as error:
            raise ValueError("The dictionary training configuration is invalid.") from error
        config.validate(manifest["input_dim"])
        if manifest["features"] != config.features or manifest["method"] != config.method:
            raise ValueError("The dictionary method and width do not match its training record.")
        width, features = manifest["input_dim"], config.features
        shapes = {
            "decoder": [width, features],
            "encoder": [features, width],
            "center": [width],
            "input_scale": [],
        }
        if config.method == "relu":
            shapes["encoder_bias"] = [features]
        try:
            state = load(data)
            if set(state) != set(shapes):
                raise ValueError("The dictionary weight names do not match its recorded method.")
            for name, shape in shapes.items():
                if state[name].dtype != torch.float32 or list(state[name].shape) != shape:
                    raise ValueError("The dictionary weight shapes do not match its manifest.")
        except SafetensorError as error:
            raise ValueError("The dictionary weight file is not valid safetensors.") from error
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("The dictionary weights contain non finite values.")
        if state["input_scale"] <= 0:
            raise ValueError("The dictionary activation scale must be positive.")
        if not torch.allclose(
            state["decoder"].norm(dim=0), torch.ones(features), rtol=1e-4, atol=1e-5
        ):
            raise ValueError("The dictionary decoder directions are no longer normalized.")
        mean = manifest.get("training_mean")
        if (
            not isinstance(mean, list)
            or len(mean) != width
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in mean)
        ):
            raise ValueError(
                "The recorded training mean must be finite and match the residual width."
            )
        model = SparseAutoencoder(width, config, initialize=False)
        model.load_state_dict(state, strict=True, assign=True)
        return manifest, model.eval()

    @artifact_operation
    def label_feature(self, identifier: str, feature: int, label: str, notes: str) -> dict:
        manifest = self.get(identifier)
        if manifest is None:
            raise ValueError("The dictionary was not found.")
        if not 0 <= feature < manifest["features"]:
            raise ValueError("The selected feature is outside this dictionary.")
        if len(label) > 100 or len(notes) > 10000:
            raise ValueError("The feature label or notes exceed their size limit.")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO feature_labels VALUES (?, ?, ?, ?) "
                "ON CONFLICT(dictionary_id, feature) DO UPDATE SET label = excluded.label, notes = excluded.notes",
                (identifier, feature, label, notes),
            )
        return {"feature": feature, "label": label, "notes": notes}

    def save_run(self, result: dict) -> dict:
        payload = dict(result, id=uuid.uuid4().hex, created_at=datetime.now(UTC).isoformat())
        encoded = json.dumps(payload, allow_nan=False)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO feature_runs VALUES (?, ?, ?, ?)",
                (
                    payload["id"],
                    payload["created_at"],
                    encoded,
                    json.dumps(_run_summary(payload), allow_nan=False),
                ),
            )
        return payload

    def runs(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT summary FROM feature_runs ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(row["summary"]) for row in rows]

    def get_run(self, identifier: str) -> dict | None:
        self._weights_path(identifier)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT result FROM feature_runs WHERE id = ?", (identifier,)
            ).fetchone()
        return json.loads(row["result"]) if row else None

    @artifact_operation
    def export_weights(self, identifier: str) -> Path:
        manifest = self.get(identifier)
        if manifest is None:
            raise ValueError("The dictionary was not found.")
        path = self._weights_path(identifier)
        if not path.is_file() or path.stat().st_size > self.max_weight_bytes + 1024**2:
            raise ValueError("The dictionary weight file is missing or exceeds its size limit.")
        if _sha256(path) != manifest["weights_sha256"]:
            raise ValueError("The dictionary weights have changed since they were recorded.")
        return path

    @artifact_operation
    def delete(self, identifier: str) -> bool:
        path = self._weights_path(identifier)
        trash = self.directory / (".trash-" + identifier + ".safetensors")
        renamed = False
        try:
            with self.connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM dictionaries WHERE id = ?", (identifier,)
                ).fetchone()
                if not exists:
                    return False
                if path.exists():
                    path.replace(trash)
                    renamed = True
                connection.execute("DELETE FROM dictionaries WHERE id = ?", (identifier,))
        except Exception:
            if renamed:
                trash.replace(path)
            raise
        trash.unlink(missing_ok=True)
        return True
