"""Shared input, layout and execution controls for saved and matching probes."""

import json


def _execution(result):
    config = result.get("execution_config")
    fields = ("precision", "device", "residual_dtype", "torch_version", "transformers_version")
    if (
        not isinstance(config, dict)
        or any(not isinstance(config.get(field), str) or not config[field] for field in fields)
        or result.get("execution_precision") != config["precision"]
        or result["device"] != config["device"]
        or type(config.get("context_length")) is not int
        or config["context_length"]
        < max(len(probe["token_ids"]) for probe in result["dataset"]["probes"])
    ):
        raise ValueError(
            "Execution settings are missing from a saved probe. Capture new probes with recorded precision and context settings."
        )
    return (
        result["device"],
        result["execution_precision"],
        json.dumps(config, sort_keys=True, allow_nan=False),
    )


def assert_probe_alignment(selected: dict, reference: dict):
    identity, other = selected["comparison_identity"], reference["comparison_identity"]
    if identity["layout"] != other["layout"]:
        raise ValueError(
            "The saved probes use different attention layouts. Compare checkpoints with matching model architecture and head coordinates."
        )
    if identity["vocabulary_mapping_sha256"] != other["vocabulary_mapping_sha256"]:
        raise ValueError(
            "The saved probes use different token mappings. Matching numeric IDs alone cannot establish aligned vocabulary rows."
        )
    if (
        selected["dataset"] != reference["dataset"]
        or selected["dataset_fingerprint"] != reference["dataset_fingerprint"]
    ):
        raise ValueError(
            "The saved probes use different inputs or sampling settings. Repeat the probe with the same dataset configuration."
        )
    if identity["decoded_probe_sha256"] != other["decoded_probe_sha256"]:
        raise ValueError(
            "The saved probe inputs decode differently. Compare captures with matching token mappings and decoded input text."
        )
    if _execution(selected) != _execution(reference):
        raise ValueError(
            "Probe execution settings differ. Comparison requires matching precision, quantization, device, context allowance and library versions."
        )
    scoring = (
        "implementation",
        "prefix_score",
        "other_scores",
        "numerics",
        "targets",
        "causal_test",
    )
    if any(
        selected["method_record"][field] != reference["method_record"][field] for field in scoring
    ):
        raise ValueError(
            "Probe measurement conventions differ. Capture new probes with matching scoring and intervention methods."
        )
