"""Bounded decoder geometry from verified dictionary artifacts on the CPU."""

import math
from contextlib import contextmanager

import psutil
import torch
from filelock import Timeout

from circuit_lantern.services.features.evidence import measurement_record
from circuit_lantern.services.features.store import DictionaryStore

_CHUNK_ELEMENTS = 1024**2


class DictionaryBusyError(ValueError):
    pass


@contextmanager
def _available_artifacts(store):
    try:
        store.file_lock.acquire(timeout=0)
    except Timeout as error:
        raise DictionaryBusyError(
            "Dictionary artifacts are already in use. Try again when the current operation finishes."
        ) from error
    try:
        yield
    finally:
        store.file_lock.release()


@torch.inference_mode()
def decoder_cosines(decoder: torch.Tensor, feature: int) -> list[tuple[int, float]]:
    if decoder.device.type != "cpu" or decoder.ndim != 2 or not decoder.shape[0]:
        raise ValueError("Decoder comparisons require a nonempty matrix on the CPU.")
    if type(feature) is not int or not 0 <= feature < decoder.shape[1]:
        raise ValueError("The selected feature is outside this dictionary.")
    selected = decoder[:, feature].double()
    norm = selected.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("The selected decoder direction has no finite nonzero norm.")
    selected = selected / norm
    result = []
    width = max(1, _CHUNK_ELEMENTS // decoder.shape[0])
    for start in range(0, decoder.shape[1], width):
        block = decoder[:, start : start + width].double()
        if not torch.isfinite(block).all():
            raise ValueError("Decoder directions must contain finite values.")
        norms = block.norm(dim=0)
        valid = norms > 0
        cosines = (selected @ block) / norms.clamp_min(torch.finfo(torch.float64).tiny)
        for offset in valid.nonzero().flatten().tolist():
            index = start + offset
            if index != feature:
                result.append((index, min(1.0, max(-1.0, float(cosines[offset])))))
    return result


def dictionary_directions(store: DictionaryStore, identifier: str, feature: int, limit=8) -> dict:
    if type(feature) is not int or feature < 0:
        raise ValueError("Choose a nonnegative feature number.")
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Choose between one and 32 decoder directions.")
    with _available_artifacts(store):
        path = store._weights_path(identifier)
        with store.connect() as connection:
            row = connection.execute(
                "SELECT length(CAST(manifest AS BLOB)) AS bytes FROM dictionaries WHERE id = ?",
                (identifier,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError("The dictionary was not found.")
            labels = connection.execute(
                "SELECT COALESCE(SUM(length(CAST(label AS BLOB)) + "
                "length(CAST(notes AS BLOB))), 0) AS bytes, COUNT(*) AS count "
                "FROM feature_labels WHERE dictionary_id = ?",
                (identifier,),
            ).fetchone()
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValueError("The dictionary weight file is missing or unreadable.") from error
        if size > store.max_weight_bytes + 1024**2:
            raise ValueError("The dictionary weight file exceeds its size limit.")
        # Account for serialized bytes, deserialization, model construction,
        # Python metadata and the bounded float64 geometry workspace.
        required = 4 * size + 8 * (row["bytes"] + labels["bytes"] + 128 * labels["count"])
        required += 32 * 1024**2
        if required > psutil.virtual_memory().available * 0.75:
            raise ValueError("Insufficient available RAM to compare these decoder directions.")
        metadata = store.get(identifier)
        count = metadata.get("features") if metadata else None
        if type(count) is not int or not 1 <= count <= 65536:
            raise ValueError("The dictionary feature count is invalid.")
        if feature >= count:
            raise ValueError("The selected feature is outside this dictionary.")
        del metadata
        manifest, dictionary = store.load_model(identifier)
        scores = decoder_cosines(dictionary.decoder, feature)
        statistics = {row["index"]: row for row in manifest.get("feature_statistics", [])}

        def describe(item):
            index, cosine = item
            frequency = statistics.get(index, {}).get("frequency")
            if (
                type(frequency) not in (int, float)
                or not math.isfinite(frequency)
                or not 0 <= frequency <= 1
            ):
                frequency = None
            return {
                "feature": index,
                "cosine": cosine,
                "frequency": frequency,
                "label": manifest.get("feature_labels", {}).get(str(index), {}).get("label", ""),
            }

        return {
            **measurement_record("decoder-directions", manifest),
            "format": "circuit-lantern-decoder-directions-v2",
            "dictionary_id": identifier,
            "analysis_device": "cpu",
            "feature": feature,
            "compared_features": len(scores),
            "highest": [
                describe(item)
                for item in sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]
            ],
            "lowest": [
                describe(item)
                for item in sorted(scores, key=lambda item: (item[1], item[0]))[:limit]
            ],
            "convention": (
                "Signed cosine of decoder columns within one dictionary, accumulated in float64. "
                "The selected feature and zero directions are excluded. Ties use feature index. "
                "Frequency uses the recorded corpus. Inactive features are included. Geometry does "
                "not establish activation correlation, semantic equivalence or causal influence."
            ),
        }
