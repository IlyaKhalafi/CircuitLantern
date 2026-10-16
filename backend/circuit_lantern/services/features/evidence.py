"""Shared source identity for immutable feature measurement exports."""

from datetime import UTC, datetime


def measurement_record(kind: str, manifest: dict) -> dict:
    source = {
        key: manifest[key]
        for key in (
            "id",
            "model_id",
            "revision",
            "layer",
            "capture_point",
            "features",
            "input_dim",
            "method",
            "weights_sha256",
            "manifest_sha256",
            "training",
            "method_record",
        )
    }
    source["corpus_fingerprint"] = manifest["corpus"]["fingerprint"]
    source["training_execution"] = {
        "execution_id": manifest["execution_id"],
        "execution_precision": manifest["execution_precision"],
        "execution_config": manifest["execution_config"],
    }
    return {
        "format": "circuit-lantern-feature-" + kind + "-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "dictionary_source": source,
    }
