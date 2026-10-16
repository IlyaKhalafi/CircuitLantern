"""Atomic, bounded checkpoints of research evidence, independent of model residency."""

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
from filelock import FileLock
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file


def validate_reference(manifest, identifier):
    required = {
        "id": str,
        "name": str,
        "created_at": str,
        "model_id": str,
        "revision": str,
        "prompt": str,
        "tokens": list,
        "model_inputs": dict,
        "vocabulary_hash": str,
        "profile": dict,
        "matrices": list,
        "layers": list,
        "predictions": list,
        "tensor_hashes": dict,
        "execution_config": dict,
    }
    if not isinstance(manifest, dict) or any(
        type(manifest.get(key)) is not kind for key, kind in required.items()
    ):
        raise ValueError("Invalid reference metadata fields.")
    if manifest["id"] != identifier or not manifest["created_at"]:
        raise ValueError("Invalid reference identity.")
    datetime.fromisoformat(manifest["created_at"])
    profile, matrices, hashes = manifest["profile"], manifest["matrices"], manifest["tensor_hashes"]
    blocks, width = profile.get("blocks"), profile.get("hidden_size")
    if (
        not isinstance(blocks, list)
        or not 1 <= len(blocks) <= 4096
        or any(not isinstance(name, str) or not name for name in blocks)
        or type(width) is not int
        or width < 1
        or not isinstance(profile.get("architecture"), str)
    ):
        raise ValueError("Invalid reference block layout.")
    if not 1 <= len(matrices) <= 12:
        raise ValueError("Invalid reference matrix selection.")
    for index, item in enumerate(matrices):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or item.get("tensor") != f"weight_{index}"
            or not isinstance(item.get("shape"), list)
            or len(item["shape"]) != 2
            or any(type(n) is not int or n < 1 for n in item["shape"])
        ):
            raise ValueError("Invalid reference matrix metadata.")
    expected = {"residuals", "logits"} | {f"weight_{index}" for index in range(len(matrices))}
    if set(hashes) != expected or any(
        not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
        for value in hashes.values()
    ):
        raise ValueError("Invalid reference tensor hashes.")
    ids = manifest["model_inputs"].get("input_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(manifest["tokens"])
        or any(
            not isinstance(key, str)
            or not isinstance(values, list)
            or len(values) != len(ids)
            or any(type(value) is not int for value in values)
            for key, values in manifest["model_inputs"].items()
        )
    ):
        raise ValueError("Invalid reference model inputs.")
    if len(manifest["layers"]) != len(blocks):
        raise ValueError("Invalid reference layer measurements.")
    for index, layer in enumerate(manifest["layers"]):
        if (
            not isinstance(layer, dict)
            or layer.get("layer") != index
            or type(layer.get("entropy")) not in (int, float)
            or not math.isfinite(layer["entropy"])
        ):
            raise ValueError("Invalid reference layer measurement.")
        predictions = layer.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("Invalid reference layer predictions.")
        for prediction in predictions:
            if (
                not isinstance(prediction, dict)
                or not isinstance(prediction.get("token"), str)
                or type(prediction.get("token_id")) is not int
                or prediction["token_id"] < 0
                or type(prediction.get("probability")) not in (int, float)
                or not 0 <= prediction["probability"] <= 1
            ):
                raise ValueError("Invalid reference prediction.")


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class CheckpointStore:
    MAX_BYTES = 128 * 1024**2
    MAX_SNAPSHOTS = 32
    MAX_COMPARISONS = 128

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(directory / ".artifacts.lock")
        with self.lock:
            for path in directory.glob(".capture-*"):
                if path.is_file():
                    path.unlink()

    def path(self, identifier: str, suffix=".safetensors"):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("The checkpoint evidence identifier is invalid.")
        return self.directory / (identifier + suffix)

    def assert_room(self, comparison=False):
        suffix = "*.comparison.json" if comparison else "*.safetensors"
        maximum = self.MAX_COMPARISONS if comparison else self.MAX_SNAPSHOTS
        with self.lock:
            if len(list(self.directory.glob(suffix))) >= maximum:
                raise ValueError(
                    "The comparison library is full. Delete an older report first."
                    if comparison
                    else "The reference library is full. Delete an older capture before saving another."
                )

    def _listing(self, suffix, loader):
        items, unreadable = [], []
        with self.lock:
            for path in self.directory.glob(suffix):
                identifier = path.name.split(".")[0]
                if not re.fullmatch(r"[a-f0-9]{32}", identifier):
                    continue
                try:
                    item = loader(identifier)
                    if not isinstance(item, dict) or not isinstance(item.get("created_at"), str):
                        raise TypeError("Invalid saved evidence timestamp.")
                    datetime.fromisoformat(item["created_at"])
                    items.append(item)
                except (ValueError, KeyError, TypeError, OSError):
                    unreadable.append(identifier)
        return {
            "items": sorted(items, key=lambda item: item["created_at"], reverse=True),
            "unreadable": unreadable,
        }

    def library(self):
        return self._listing("*.safetensors", self.read)

    def list(self):
        return self.library()["items"]

    def read(self, identifier):
        with self.lock:
            path = self.path(identifier)
            if not path.is_file():
                raise ValueError("The reference capture was not found.")
            if path.stat().st_size > self.MAX_BYTES + 1024**2:
                raise ValueError("The reference capture exceeds the evidence size limit.")
            try:
                with safe_open(path, framework="pt", device="cpu") as stream:
                    manifest = json.loads((stream.metadata() or {})["manifest"])
                validate_reference(manifest, identifier)
                return manifest
            except (ValueError, KeyError, TypeError, SafetensorError) as error:
                raise ValueError(
                    "The reference capture is unreadable. Delete it from the reference library."
                ) from error

    def save(self, manifest, tensors):
        if sum(t.numel() * t.element_size() for t in tensors.values()) > self.MAX_BYTES:
            raise ValueError("The capture exceeds the evidence size limit. Select fewer matrices.")
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("Checkpoint evidence must contain finite float32 tensors.")
        identifier = uuid.uuid4().hex
        payload = dict(
            manifest,
            id=identifier,
            created_at=datetime.now(UTC).isoformat(),
            tensor_hashes={key: tensor_digest(t) for key, t in tensors.items()},
        )
        with self.lock:
            self.assert_room()
            self._atomic(
                self.path(identifier),
                lambda path: save_file(
                    tensors, path, metadata={"manifest": json.dumps(payload, allow_nan=False)}
                ),
            )
        return payload

    def load(self, identifier):
        with self.lock:
            manifest = self.read(identifier)
            tensors = load_file(self.path(identifier), device="cpu")
            if set(tensors) != set(manifest["tensor_hashes"]) or any(
                t.dtype != torch.float32
                or not torch.isfinite(t).all()
                or tensor_digest(t) != manifest["tensor_hashes"][key]
                for key, t in tensors.items()
            ):
                raise ValueError("The saved reference tensors failed their integrity check.")
            for item in manifest["matrices"]:
                if list(tensors[item["tensor"]].shape) != item["shape"]:
                    raise ValueError("The reference weight shape does not match its metadata.")
            if (
                list(tensors["residuals"].shape)
                != [len(manifest["layers"]), manifest["profile"]["hidden_size"]]
                or tensors["logits"].ndim != 1
            ):
                raise ValueError("The reference activation shape does not match its metadata.")
            return manifest, tensors

    def delete(self, identifier):
        with self.lock:
            path = self.path(identifier)
            if not path.exists():
                raise ValueError("The reference capture was not found.")
            path.unlink()

    def save_comparison(self, result):
        identifier = uuid.uuid4().hex
        payload = dict(result, id=identifier, created_at=datetime.now(UTC).isoformat())
        encoded = json.dumps(payload, allow_nan=False).encode()
        if len(encoded) > 4 * 1024**2:
            raise ValueError("The comparison exceeds the result size limit.")
        with self.lock:
            self.assert_room(comparison=True)
            self._atomic(
                self.path(identifier, ".comparison.json"),
                lambda path: Path(path).write_bytes(encoded),
            )
        return payload

    def comparisons(self):
        def summary(identifier):
            record = self.comparison(identifier)
            return {key: record[key] for key in ("id", "created_at", "reference", "selected")} | {
                "reference": {key: record["reference"][key] for key in ("model_id", "revision")},
                "selected": {key: record["selected"][key] for key in ("model_id", "revision")},
            }

        return self._listing("*.comparison.json", summary)

    def comparison(self, identifier):
        with self.lock:
            path = self.path(identifier, ".comparison.json")
            if not path.is_file():
                raise ValueError("The saved checkpoint comparison was not found.")
            if path.stat().st_size > 4 * 1024**2:
                raise ValueError("The saved checkpoint comparison exceeds its size limit.")
            return json.loads(path.read_text())

    def delete_comparison(self, identifier):
        with self.lock:
            path = self.path(identifier, ".comparison.json")
            if not path.exists():
                raise ValueError("The saved checkpoint comparison was not found.")
            path.unlink()

    def _atomic(self, destination, writer):
        handle, temporary = tempfile.mkstemp(prefix=".capture-", dir=self.directory)
        os.close(handle)
        try:
            writer(temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)
