from copy import deepcopy
from statistics import mean, pstdev
from unittest.mock import patch

import pytest
from induction_fixtures import paired_captures

from circuit_lantern.services.induction.comparison import compare_probes
from circuit_lantern.services.induction.comparison_validation import INVALID_EVIDENCE
from circuit_lantern.services.induction.identity import identity_digest


@pytest.fixture
def captures(runtime):
    return paired_captures(runtime)


def strip_causal(result):
    result.update(causal_results=[], requested_causal_heads=0, completed_causal_heads=0)


def attention_aggregates(result):
    for row in result["heads"]:
        for field in ("prefix", "duplicate", "previous"):
            row[field + "_attention"] = mean(row[field + "_by_probe"])
        row["prefix_std"] = pstdev(row["prefix_by_probe"])


def loss_aggregates(result):
    for row in result["probe_metrics"]:
        row.update(first_repeat_nll=row["repeat_nll"][0], repeated_nll=mean(row["repeat_nll"][1:]))
    result.update(
        mean_first_repeat_nll=mean(row["first_repeat_nll"] for row in result["probe_metrics"]),
        mean_repeated_nll=mean(row["repeated_nll"] for row in result["probe_metrics"]),
    )


def effect_aggregate(row):
    row.update(
        completed_examples=len(row["examples"]),
        complete=len(row["examples"]) == row["requested_examples"],
        mean_delta_log_probability=mean(item["delta_log_probability"] for item in row["examples"]),
        mean_kl_divergence=mean(item["kl_divergence"] for item in row["examples"]),
    )


def test_real_checkpoint_changes_are_compared_without_forward_or_source_mutation(captures, runtime):
    selected, reference = captures
    original = deepcopy(captures)
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        result = compare_probes(selected, reference)
    assert captures == original
    assert result["shared_probe_indices"] == [0, 1, 2]
    assert result["shared_examples"] == 3
    assert result["shared_pattern_heads"] == result["shared_causal_heads"] == 4
    assert result["selected_source"]["model_id"] == "fine-checkpoint"
    assert result["reference_source"]["revision"] == "base-revision"
    assert result["first_loss_difference"] == pytest.approx(
        selected["mean_first_repeat_nll"] - reference["mean_first_repeat_nll"], abs=1e-12
    )
    assert any(abs(row["prefix_difference"]) > 1e-8 for row in result["heads"])
    for row in result["heads"]:
        left = next(
            item
            for item in selected["heads"]
            if (item["layer"], item["head"]) == (row["layer"], row["head"])
        )
        right = next(
            item
            for item in reference["heads"]
            if (item["layer"], item["head"]) == (row["layer"], row["head"])
        )
        assert row["prefix_difference"] == pytest.approx(
            left["prefix_attention"] - right["prefix_attention"], abs=1e-12
        )
        assert row["causal_paired_examples"] == 3


def test_partial_pattern_comparison_recomputes_both_means_on_the_common_examples(captures):
    selected, reference = captures
    for result in captures:
        strip_causal(result)
    left = next(row for row in selected["heads"] if (row["layer"], row["head"]) == (0, 0))
    right = next(row for row in reference["heads"] if (row["layer"], row["head"]) == (0, 0))
    left["prefix_by_probe"], right["prefix_by_probe"] = [0.2, 0.4, 0.8], [0.1, 0.3, 0.9]
    for result, losses in (
        (selected, [[4, 2], [8, 4], [100, 50]]),
        (reference, [[5, 3], [7, 6], [1000, 1000]]),
    ):
        for row, values in zip(result["probe_metrics"], losses, strict=True):
            row["repeat_nll"] = values
    selected.update(
        complete=False, cancelled=True, measured_examples=2, attention_probe_indices=[0, 1]
    )
    selected["probe_metrics"] = selected["probe_metrics"][:2]
    for row in selected["heads"]:
        for field in ("prefix", "duplicate", "previous"):
            row[field + "_by_probe"] = row[field + "_by_probe"][:2]
    for result in captures:
        attention_aggregates(result)
        loss_aggregates(result)
    comparison = compare_probes(selected, reference)
    row = next(row for row in comparison["heads"] if (row["layer"], row["head"]) == (0, 0))
    assert row["selected_prefix"] == pytest.approx(0.3)
    assert row["reference_prefix"] == pytest.approx(0.2)
    assert row["prefix_difference"] == pytest.approx(0.1)
    assert comparison["shared_probe_indices"] == [0, 1]
    assert comparison["reference_only_examples"] == [2]
    assert comparison["first_loss_difference"] == 0
    assert comparison["repeated_loss_difference"] == -1.5


def test_partial_causal_comparison_uses_only_common_completed_pairs(captures):
    selected, reference = captures
    left = selected["causal_results"][-1]
    right = next(
        row
        for row in reference["causal_results"]
        if (row["layer"], row["head"]) == (left["layer"], left["head"])
    )
    for row, deltas in ((left, [-0.2, -0.4, -1]), (right, [-0.1, -0.3, -2])):
        for example, delta in zip(row["examples"], deltas, strict=True):
            example.update(
                ablated_log_probability=example["baseline_log_probability"] + delta,
                delta_log_probability=delta,
            )
        effect_aggregate(row)
    left["examples"] = left["examples"][:2]
    effect_aggregate(left)
    selected.update(complete=False, cancelled=True, completed_causal_heads=3)
    comparison = compare_probes(selected, reference)
    row = next(
        row
        for row in comparison["heads"]
        if (row["layer"], row["head"]) == (left["layer"], left["head"])
    )
    assert comparison["shared_examples"] == 3
    assert row["causal_probe_indices"] == [0, 1]
    assert row["selected_causal_delta"] == pytest.approx(-0.3)
    assert row["reference_causal_delta"] == pytest.approx(-0.2)
    assert row["causal_delta_difference"] == pytest.approx(-0.1)
    assert row["selected_causal_examples"] == 2
    assert row["reference_causal_examples"] == 3


def test_unmeasured_layers_remain_null_and_do_not_change_loss_conditioning(captures):
    selected, reference = captures
    for result in captures:
        strip_causal(result)
    selected["layer_targets"] = selected["layer_targets"][:1]
    selected["heads"] = [row for row in selected["heads"] if row["layer"] == 0]
    result = compare_probes(selected, reference)
    assert result["shared_pattern_heads"] == 2
    missing = [row for row in result["heads"] if row["layer"] == 1]
    assert all(
        row["selected_prefix"] is None and row["prefix_difference"] is None for row in missing
    )
    assert all(row["reference_prefix"] is not None for row in missing)
    assert all(row["causal_delta_difference"] is None for row in result["heads"])


@pytest.mark.parametrize(
    "change,message",
    [
        ("mapping", "different token mappings"),
        ("decoder", "decode differently"),
        ("layout", "different attention layouts"),
        ("dataset", "different inputs"),
        ("profile", "execution settings differ"),
        ("method", "measurement conventions differ"),
    ],
)
def test_semantic_and_execution_mismatches_cannot_produce_differences(captures, change, message):
    selected, reference = captures
    if change == "mapping":
        selected["comparison_identity"]["vocabulary_mapping_sha256"] = "f" * 64
    elif change == "decoder":
        selected["comparison_identity"]["decoded_probe_sha256"][0] = "f" * 64
    elif change == "layout":
        selected["comparison_identity"]["layout"]["model_type"] = "llama"
    elif change == "dataset":
        selected["dataset"]["config"]["seed"] += 1
        selected["dataset_fingerprint"] = identity_digest(selected["dataset"])
    elif change == "profile":
        selected["execution_config"]["context_length"] = 128
    else:
        selected["method_record"]["numerics"] += " changed"
    selected["comparison_identity_sha256"] = identity_digest(selected["comparison_identity"])
    with pytest.raises(ValueError, match=message):
        compare_probes(selected, reference)


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "aggregate",
        "indices",
        "duplicate_head",
        "bad_target",
        "identity_digest",
        "legacy",
        "causal_baseline",
        "boolean_count",
        "boolean_query",
    ],
)
def test_inconsistent_or_legacy_evidence_is_rejected_before_comparison(captures, change):
    selected, reference = captures
    if change == "nan":
        selected["heads"][0]["prefix_by_probe"][0] = float("nan")
    elif change == "aggregate":
        selected["heads"][0]["prefix_attention"] += 0.1
    elif change == "indices":
        selected["attention_probe_indices"] = [0, 1, 1]
    elif change == "duplicate_head":
        selected["heads"][-1] = deepcopy(selected["heads"][0])
    elif change == "bad_target":
        selected["dataset"]["probes"][0]["target_ids"][1] = 0
        selected["dataset_fingerprint"] = identity_digest(selected["dataset"])
    elif change == "identity_digest":
        selected["comparison_identity_sha256"] = "f" * 64
    elif change == "legacy":
        selected["format"] = "circuit-lantern-induction-probe-v1"
    elif change == "causal_baseline":
        selected["causal_results"][0]["examples"][0]["baseline_log_probability"] -= 1
    elif change == "boolean_count":
        selected["completed_causal_heads"] = True
    else:
        selected["dataset"]["probes"][0]["first_queries"][0] = True
        selected["dataset_fingerprint"] = identity_digest(selected["dataset"])
    with pytest.raises(ValueError, match=INVALID_EVIDENCE):
        compare_probes(selected, reference)


def test_an_empty_cancelled_probe_cannot_be_compared_as_zero(captures):
    selected, reference = captures
    selected.update(
        complete=False,
        cancelled=True,
        measured_examples=0,
        attention_probe_indices=[],
        heads=[],
        probe_metrics=[],
        causal_results=[],
        completed_causal_heads=0,
    )
    with pytest.raises(ValueError, match="no complete examples in common"):
        compare_probes(selected, reference)
