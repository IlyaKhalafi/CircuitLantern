"""Compare saved probes using only shared measured examples and head indices."""

import json
import math

from circuit_lantern.services.induction.comparison_validation import mean, validate_probe


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


def _difference(selected, reference):
    if selected is None or reference is None:
        return None
    difference = selected - reference
    if not math.isfinite(difference):
        raise ValueError("A probe comparison exceeds the supported numerical range.")
    return difference


def _source(evidence):
    result = evidence.result
    return {
        key: result.get(key)
        for key in (
            "model_id",
            "revision",
            "execution_id",
            "execution_precision",
            "execution_config",
            "device",
            "complete",
            "cancelled",
            "error",
            "requested_examples",
            "measured_examples",
            "requested_causal_heads",
            "completed_causal_heads",
            "dataset_fingerprint",
            "comparison_identity_sha256",
        )
    }


def _head_rows(selected, reference, shared):
    left_positions = {
        probe: index for index, probe in enumerate(selected.result["attention_probe_indices"])
    }
    right_positions = {
        probe: index for index, probe in enumerate(reference.result["attention_probe_indices"])
    }
    rows = []
    for coordinate in sorted(selected.heads.keys() | reference.heads.keys()):
        left, right = selected.heads.get(coordinate), reference.heads.get(coordinate)
        row = {
            "layer": coordinate[0],
            "head": coordinate[1],
            "pattern_paired_examples": len(shared) if left and right else 0,
        }
        for field in ("prefix", "duplicate", "previous"):
            left_value = (
                mean(left[field + "_by_probe"][left_positions[index]] for index in shared)
                if left
                else None
            )
            right_value = (
                mean(right[field + "_by_probe"][right_positions[index]] for index in shared)
                if right
                else None
            )
            row.update(
                {
                    "selected_" + field: left_value,
                    "reference_" + field: right_value,
                    field + "_difference": _difference(left_value, right_value),
                }
            )
        left_effects, right_effects = (
            selected.causal.get(coordinate, {}),
            reference.causal.get(coordinate, {}),
        )
        paired = sorted(left_effects.keys() & right_effects.keys())
        row.update(
            causal_paired_examples=len(paired),
            selected_causal_examples=len(left_effects),
            reference_causal_examples=len(right_effects),
        )
        for field, name in (("delta_log_probability", "causal_delta"), ("kl_divergence", "kl")):
            left_value = mean(left_effects[index][field] for index in paired) if paired else None
            right_value = mean(right_effects[index][field] for index in paired) if paired else None
            row.update(
                {
                    "selected_" + name: left_value,
                    "reference_" + name: right_value,
                    name + "_difference": _difference(left_value, right_value),
                }
            )
        row["causal_probe_indices"] = paired
        rows.append(row)
    return rows


def compare_probes(selected: dict, reference: dict) -> dict:
    left, right = validate_probe(selected), validate_probe(reference)
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
    if selected["method_record"] != reference["method_record"]:
        raise ValueError(
            "Probe measurement conventions differ. Capture new probes with matching scoring and intervention methods."
        )
    shared = sorted(left.probes.keys() & right.probes.keys())
    if not shared:
        raise ValueError(
            "The saved probes have no complete examples in common. Capture at least one shared probe before comparing measurements."
        )
    repeats = selected["dataset"]["config"]["repeats"]
    losses = []
    for repeat in range(repeats):
        a = mean(left.probes[index]["repeat_nll"][repeat] for index in shared)
        b = mean(right.probes[index]["repeat_nll"][repeat] for index in shared)
        losses.append(
            {"repeat": repeat + 1, "selected": a, "reference": b, "difference": _difference(a, b)}
        )
    heads = _head_rows(left, right, shared)
    return {
        "format": "circuit-lantern-induction-comparison-v1",
        "selected_source": _source(left),
        "reference_source": _source(right),
        "shared_probe_indices": shared,
        "shared_examples": len(shared),
        "selected_only_examples": sorted(left.probes.keys() - right.probes.keys()),
        "reference_only_examples": sorted(right.probes.keys() - left.probes.keys()),
        "shared_pattern_heads": sum(row["pattern_paired_examples"] > 0 for row in heads),
        "shared_causal_heads": sum(row["causal_paired_examples"] > 0 for row in heads),
        "first_loss_difference": losses[0]["difference"],
        "repeated_loss_difference": _difference(
            mean(row["selected"] for row in losses[1:]),
            mean(row["reference"] for row in losses[1:]),
        ),
        "loss_by_repeat": losses,
        "heads": heads,
        "convention": "Selected minus reference. Attention and loss means use only the shared measured probe indices. Causal differences for each head use only shared completed baseline and ablated pairs. Missing measurements remain null. Checkpoint identity may differ, while token mappings, decoded inputs, dataset, execution profile and attention layout must match. Head indices are aligned coordinates and do not establish the same learned function. Numerical rounding and zero ablation effects remain possible confounds.",
    }
