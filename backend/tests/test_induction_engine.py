import json
from unittest.mock import patch

import pytest
import torch

from circuit_lantern.services.induction.engine import measure_induction
from circuit_lantern.services.induction.planning import probe_plan
from circuit_lantern.services.induction.probes import ProbeConfig, build_probe_batch
from circuit_lantern.services.interventions import attention_head_targets
from circuit_lantern.services.jobs import JobRecord


def probe(runtime, *, job=None, examples=2, causal_heads=1, start=0):
    batch = build_probe_batch(
        runtime.tokenizer.get_vocab().values(),
        runtime.tokenizer.all_special_ids,
        start,
        ProbeConfig(block_length=4, repeats=2, examples=examples, seed=19),
        context_limit=64,
    )
    targets = attention_head_targets(runtime._blocks())
    result = measure_induction(
        runtime.model,
        runtime.tokenizer,
        runtime._blocks(),
        runtime.metadata,
        runtime.device,
        batch,
        targets,
        causal_heads,
        job if job is not None else {},
        {
            "requested_forward_passes": examples * (1 + causal_heads)
            if type(causal_heads) is int
            else 1
        },
    )
    return result, batch


def assert_no_hooks(runtime):
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_engine_agrees_with_independent_full_sequence_attention_and_ablation(runtime):
    weights = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    result, batch = probe(runtime)
    assert result["complete"] and result["error"] is None
    assert result["measured_examples"] == 2
    assert result["requested_causal_heads"] == result["completed_causal_heads"] == 1
    assert result["forward_attempts"] == result["completed_forwards"] == 4
    assert result["dataset"] == batch.description()
    assert result["dataset_fingerprint"] == batch.fingerprint
    assert result["format"] == "circuit-lantern-induction-probe-v2"
    assert result["attention_probe_indices"] == [0, 1]
    assert len(result["comparison_identity"]["decoded_probe_sha256"]) == 2
    assert result["representative_probe"] == 0
    assert result["representative_query_position"] == 8
    assert result["logits_dtype"] == "float32"
    baseline = []
    per_head = {(layer, head): [] for layer in range(2) for head in range(2)}
    with torch.inference_mode():
        for index, example in enumerate(batch.probes):
            ids = torch.tensor([example.token_ids])
            output = runtime.model(ids, use_cache=False, output_attentions=True)
            probabilities = output.logits[0].double().log_softmax(-1)
            baseline.append(probabilities[list(example.repeated_queries)])
            for layer in range(2):
                for head in range(2):
                    attention = output.attentions[layer][0, head].double()
                    attention /= attention.sum(-1, keepdim=True)
                    scores = []
                    for query in example.repeated_queries:
                        keys = [
                            key
                            for key in range(1, query + 1)
                            if example.token_ids[key - 1] == example.token_ids[query]
                        ]
                        scores.append(float(attention[query, keys].sum()))
                    per_head[(layer, head)].append(sum(scores) / len(scores))
                    captured = next(
                        row
                        for row in result["heads"]
                        if (row["layer"], row["head"]) == (layer, head)
                    )
                    if index == 0:
                        assert captured["representative_attention"] == pytest.approx(
                            attention[-1].tolist(), abs=1e-12
                        )
            for repeat in range(2):
                positions = list(range(1 + repeat * 4, 1 + (repeat + 1) * 4))
                labels = torch.tensor(example.target_ids)[positions]
                expected = float(-probabilities[positions, labels].mean())
                assert result["probe_metrics"][index]["repeat_nll"][repeat] == pytest.approx(
                    expected
                )
        for row in result["heads"]:
            expected = sum(per_head[(row["layer"], row["head"])]) / 2
            assert row["prefix_attention"] == pytest.approx(expected, abs=1e-12)
            assert row["prefix_by_probe"] == pytest.approx(
                per_head[(row["layer"], row["head"])], abs=1e-12
            )
            assert len(row["duplicate_by_probe"]) == len(row["previous_by_probe"]) == 2
        candidate = result["causal_results"][0]
        projection = runtime._blocks()[candidate["layer"]].attn.c_proj
        start = candidate["head"] * 8

        def independently_zero_head(module, inputs):
            changed = inputs[0].clone()
            changed[..., start : start + 8] = 0
            return (changed, *inputs[1:])

        for index, example in enumerate(batch.probes):
            handle = projection.register_forward_pre_hook(independently_zero_head)
            try:
                changed = runtime.model(torch.tensor([example.token_ids]), use_cache=False).logits[
                    0
                ]
            finally:
                handle.remove()
            probabilities = changed.double().log_softmax(-1)[list(example.repeated_queries)]
            labels = torch.tensor(example.target_ids)[list(example.repeated_queries)]
            rows = torch.arange(len(labels))
            expected_delta = float(
                (probabilities[rows, labels] - baseline[index][rows, labels]).mean()
            )
            expected_kl = float(
                (baseline[index].exp() * (baseline[index] - probabilities)).sum(-1).mean()
            )
            captured = candidate["examples"][index]
            assert captured["delta_log_probability"] == pytest.approx(expected_delta, abs=1e-12)
            assert captured["kl_divergence"] == pytest.approx(expected_kl, abs=1e-12)
    assert all(
        torch.equal(weights[name], value) for name, value in runtime.model.state_dict().items()
    )
    assert_no_hooks(runtime)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "failed_forward,stage,measured,pairs", [(2, "patterns", 1, 0), (4, "causal", 2, 1)]
)
def test_failed_forward_retains_completed_examples_and_removes_hooks(
    runtime, failed_forward, stage, measured, pairs
):
    original = runtime.model.forward
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == failed_forward:
            raise RuntimeError("Induced forward failure")
        return original(*args, **kwargs)

    job = JobRecord({"status": "running", "progress": 0})
    with patch.object(runtime.model, "forward", side_effect=fail):
        result, _ = probe(runtime, job=job)
    assert not result["complete"]
    assert result["error"] == "Induced forward failure"
    assert result["failed_at"]["stage"] == stage
    assert result["measured_examples"] == measured
    assert result["attention_probe_indices"] == list(range(measured))
    assert all(len(row["prefix_by_probe"]) == measured for row in result["heads"])
    assert len(result["heads"]) == 4
    assert sum(row["completed_examples"] for row in result["causal_results"]) == pairs
    assert result["completed_causal_heads"] == 0
    job.finish(result, label="Induction probe", unit="measurements")
    assert job["status"] == "failed" and job["result"] is result
    assert_no_hooks(runtime)


@pytest.mark.parametrize("after,measured,pairs", [(1, 1, 0), (3, 2, 1)])
def test_cancellation_is_observed_between_forwards_and_retains_completed_pairs(
    runtime, after, measured, pairs
):
    class CancelAfter(JobRecord):
        done = 0

        def update(self, **changes):
            super().update(**changes)
            self.done += 1
            if self.done == after:
                self.request_cancel()

    job = CancelAfter({"status": "running", "progress": 0})
    result, _ = probe(runtime, job=job)
    assert result["forward_attempts"] == after
    assert not result["complete"] and result["error"] is None
    assert result["measured_examples"] == measured
    assert result["attention_probe_indices"] == list(range(measured))
    assert all(len(row["prefix_by_probe"]) == measured for row in result["heads"])
    assert sum(row["completed_examples"] for row in result["causal_results"]) == pairs
    job.finish(result, label="Induction probe", unit="measurements")
    assert job["status"] == "cancelled" and job["result"]["cancelled"]
    assert 0 < job["progress"] < 100
    assert_no_hooks(runtime)


def test_pattern_only_probe_without_start_token_needs_no_causal_forward(runtime):
    result, batch = probe(runtime, causal_heads=0, start=None)
    assert result["complete"]
    assert result["forward_attempts"] == 2
    assert result["causal_results"] == []
    assert result["dataset"]["start_token_id"] is None
    assert len(batch.probes[0].token_ids) == 8
    assert result["requested_causal_heads"] == result["completed_causal_heads"] == 0


def test_cancel_before_first_forward_returns_explicit_empty_evidence(runtime):
    job = {"cancel_requested": True}
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        result, _ = probe(runtime, job=job)
    assert not result["complete"]
    assert result["measured_examples"] == result["completed_forwards"] == 0
    assert result["heads"] == [] and result["causal_results"] == []
    assert result["representative_probe"] is None
    assert "mean_repeated_nll" not in result


def test_training_mode_is_rejected_before_forward(runtime):
    runtime.model.train()
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError, match="evaluation mode"),
    ):
        probe(runtime)


def test_malformed_attention_is_reported_without_a_fabricated_measurement(runtime):
    original = runtime.model.forward

    def missing(*args, **kwargs):
        output = original(*args, **kwargs)
        output.attentions = None
        return output

    with patch.object(runtime.model, "forward", side_effect=missing):
        result, _ = probe(runtime)
    assert not result["complete"]
    assert result["measured_examples"] == 0 and result["heads"] == []
    assert "attention patterns" in result["error"]
    assert_no_hooks(runtime)


def test_resource_plan_includes_full_vocabulary_cache_and_does_not_forward(runtime):
    targets = attention_head_targets(runtime._blocks())
    hardware = {"ram_available_bytes": 2**40, "experiment_disk_free_bytes": 2**40}
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        plan = probe_plan(
            runtime.model,
            ProbeConfig(block_length=4, repeats=3, examples=2),
            prefix_length=1,
            causal_heads=1,
            all_heads=targets,
            selected_heads=targets,
            device="cpu",
            hardware=hardware,
        )
        assert plan["compatible"] and plan["baseline_cache_bytes"] == 2 * 8 * 16 * 8
        assert plan["requested_forward_passes"] == 4
        assert plan["estimated_host_bytes"] > plan["baseline_cache_bytes"]
        assert plan["estimated_gpu_bytes"] == 0
        no_causal = probe_plan(
            runtime.model,
            ProbeConfig(block_length=4, repeats=3, examples=2),
            prefix_length=1,
            causal_heads=0,
            all_heads=targets,
            selected_heads=targets,
            device="cpu",
            hardware=hardware,
        )
        assert no_causal["baseline_cache_bytes"] == 0
        assert no_causal["requested_forward_passes"] == 2
        assert no_causal["estimated_host_bytes"] < plan["estimated_host_bytes"]


@pytest.mark.parametrize("constraint", ["ram_available_bytes", "experiment_disk_free_bytes", "gpu"])
def test_resource_plan_blocks_missing_headroom(runtime, constraint):
    targets = attention_head_targets(runtime._blocks())
    hardware = {"ram_available_bytes": 2**40, "experiment_disk_free_bytes": 2**40}
    if constraint != "gpu":
        hardware[constraint] = 1
    plan = probe_plan(
        runtime.model,
        ProbeConfig(block_length=4, repeats=2, examples=2),
        prefix_length=1,
        causal_heads=1,
        all_heads=targets,
        selected_heads=targets,
        device="cuda" if constraint == "gpu" else "cpu",
        hardware=hardware,
    )
    assert not plan["compatible"] and plan["blockers"]


@pytest.mark.parametrize("invalid", [-1, 13, True, "1"])
def test_engine_rejects_unbounded_or_untyped_causal_work_before_forward(runtime, invalid):
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError, match="zero to twelve"),
    ):
        probe(runtime, causal_heads=invalid)
