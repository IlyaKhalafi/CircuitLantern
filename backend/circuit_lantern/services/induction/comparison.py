"""Compare saved probes using only shared measured examples and head indices."""

import math

from circuit_lantern.services.induction.comparison_compatibility import assert_probe_alignment
from circuit_lantern.services.induction.comparison_validation import mean, validate_probe


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
    assert_probe_alignment(selected, reference)
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
