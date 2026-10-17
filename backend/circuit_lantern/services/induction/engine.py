"""Measure repeated token attention and signed effects of candidate head ablation."""

import inspect
import time
from datetime import UTC, datetime

import torch

from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.induction.identity import comparison_identity, identity_digest
from circuit_lantern.services.induction.patterns import score_repeated_attention
from circuit_lantern.services.induction.probes import ProbeBatch
from circuit_lantern.services.interventions import install_head_ablation
from circuit_lantern.services.scoring import log_probabilities

REFERENCES = [
    {
        "title": "In-context Learning and Induction Heads",
        "year": 2022,
        "url": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
    },
]


METHOD_RECORD = {
    "implementation": "circuit_lantern_repeated_token_probe_v1",
    "sampling": "Source blocks contain uniform distinct nonspecial vocabulary IDs, repeated without retokenization. Matching runs replay retained source blocks without resampling under the new tokenizer special token policy. This adapts the paper's frequency-filtered sampling.",
    "prefix_score": "Mean attention mass to causal keys whose preceding token equals the query token. Only queries after the first block are scored.",
    "other_scores": "Duplicate and previous token attention use the same repeated query positions. This differs from the paper's previous token evaluator on natural text.",
    "numerics": "Attention rows are validated against storage precision, normalized and accumulated in float64. The uniform baseline assumes full causal attention.",
    "targets": "Every target is the next token in the known periodic sequence, including the final prompt position. Losses use natural logarithms.",
    "causal_test": "Each selected query head is independently zeroed at every prompt position before its output projection. Delta is ablated minus baseline mean periodic target log probability. KL is baseline to ablated over the full output vocabulary, averaged over repeated queries.",
    "scope": "Attention pattern scores do not establish copying or causal importance. Ablations measure effects on this synthetic probe distribution and do not establish a mechanism for general in-context learning.",
}


def aggregate_heads(
    scores: list[torch.Tensor], targets: list[dict], representative: dict
) -> list[dict]:
    if not scores:
        return []
    values = torch.stack(scores)
    means, deviations = values.mean(0), values.std(0, unbiased=False)
    rows = []
    index = 0
    for item in targets:
        for head in range(item["heads"]):
            rows.append(
                {
                    "layer": item["layer"],
                    "head": head,
                    "prefix_attention": float(means[index, 0]),
                    "prefix_std": float(deviations[index, 0]),
                    "duplicate_attention": float(means[index, 1]),
                    "previous_attention": float(means[index, 2]),
                    "prefix_by_probe": values[:, index, 0].tolist(),
                    "duplicate_by_probe": values[:, index, 1].tolist(),
                    "previous_by_probe": values[:, index, 2].tolist(),
                    "representative_attention": representative[(item["layer"], head)],
                }
            )
            index += 1
    return sorted(rows, key=lambda row: (-row["prefix_attention"], row["layer"], row["head"]))


@torch.inference_mode()
def measure_induction(
    model,
    tokenizer,
    blocks,
    metadata: dict,
    device: str,
    batch: ProbeBatch,
    targets: list[dict],
    causal_heads: int,
    job,
    plan: dict,
    *,
    causal_candidates: list[dict] | None = None,
) -> dict:
    started = time.perf_counter()
    batch.config.validate()
    if type(causal_heads) is not int or not 0 <= causal_heads <= 12:
        raise ValueError("Choose zero to twelve causal candidate heads.")
    if model.training:
        raise ValueError("Repeated token probes require the model in evaluation mode.")
    requested = min(causal_heads, sum(item["heads"] for item in targets))
    if causal_candidates is not None:
        problem = "Choose distinct causal candidate heads within the selected attention layout."
        if not isinstance(causal_candidates, list) or any(
            not isinstance(row, dict) or "layer" not in row or "head" not in row
            for row in causal_candidates
        ):
            raise ValueError(problem)
        available = {item["layer"]: item["heads"] for item in targets}
        coordinates = [(row["layer"], row["head"]) for row in causal_candidates]
        if (
            len(coordinates) != requested
            or any(type(layer) is not int or type(head) is not int for layer, head in coordinates)
            or len(set(coordinates)) != len(coordinates)
            or any(
                layer not in available or not 0 <= head < available[layer]
                for layer, head in coordinates
            )
        ):
            raise ValueError(problem)
    identity = plan.get("comparison_identity") or comparison_identity(
        model, tokenizer, blocks, batch
    )
    result = {
        **execution_provenance(metadata),
        "format": "circuit-lantern-induction-probe-v2",
        "comparison_identity": identity,
        "comparison_identity_sha256": identity_digest(identity),
        "attention_probe_indices": [],
        "captured_at": datetime.now(UTC).isoformat(),
        "device": device,
        "dataset": batch.description(),
        "dataset_fingerprint": batch.fingerprint,
        "layer_targets": targets,
        "requested_examples": batch.config.examples,
        "measured_examples": 0,
        "requested_causal_heads": requested,
        "causal_candidate_policy": "explicit" if causal_candidates is not None else "prefix_ranked",
        "requested_causal_candidates": causal_candidates,
        "completed_causal_heads": 0,
        "representative_probe": None,
        "representative_query_position": len(batch.probes[0].token_ids) - 1,
        "representative_tokens": [tokenizer.decode([token]) for token in batch.probes[0].token_ids],
        "heads": [],
        "probe_metrics": [],
        "causal_results": [],
        "forward_attempts": 0,
        "completed_forwards": 0,
        "complete": False,
        "error": None,
        "failed_at": None,
        "references": REFERENCES,
        "method_record": dict(METHOD_RECORD),
    }
    scores, representative, baselines = [], {}, []
    arguments = inspect.signature(model.forward).parameters
    full_logits = {
        name: 0 for name in ("logits_to_keep", "num_logits_to_keep") if name in arguments
    }

    def forward(probe, *, attention):
        inputs = torch.tensor([probe.token_ids], dtype=torch.int64, device=device)
        result["forward_attempts"] += 1
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
            output_attentions=attention,
            output_hidden_states=False,
            return_dict=True,
            **full_logits,
        )
        result["completed_forwards"] += 1
        if output.logits.ndim != 3 or output.logits.shape[:2] != (1, len(probe.token_ids)):
            raise ValueError("The model did not return full sequence logits for this probe.")
        result["logits_dtype"] = str(output.logits.dtype).removeprefix("torch.")
        probabilities = log_probabilities(output.logits[0]).cpu()
        return output, probabilities

    def progress(message):
        job.update(
            progress=min(99, 100 * result["completed_forwards"] / plan["requested_forward_passes"]),
            message=message,
        )

    for index, probe in enumerate(batch.probes):
        if job.get("cancel_requested"):
            break
        try:
            output, probabilities = forward(probe, attention=True)
            if output.attentions is None or len(output.attentions) != len(blocks):
                raise ValueError("The model did not expose verified attention patterns.")
            example_scores, example_attention = [], {}
            for item in targets:
                pattern = output.attentions[item["layer"]]
                if pattern is None or pattern.ndim != 4 or pattern.shape[:2] != (1, item["heads"]):
                    raise ValueError(
                        "The attention head layout does not match its captured patterns."
                    )
                measured = score_repeated_attention(
                    pattern[0],
                    torch.tensor(probe.token_ids, dtype=torch.int64, device="cpu"),
                    probe.repeated_queries,
                )
                result["uniform_full_causal_baseline"] = measured["uniform_full_causal_baseline"]
                for row in measured["heads"]:
                    example_scores.append(
                        [
                            row["prefix_attention"],
                            row["duplicate_attention"],
                            row["previous_attention"],
                        ]
                    )
                    if not scores:
                        attention = pattern[0, row["head"], -1].detach().to("cpu", torch.float64)
                        example_attention[(item["layer"], row["head"])] = (
                            attention / attention.sum()
                        ).tolist()
            target_ids = torch.tensor(probe.target_ids, dtype=torch.int64, device="cpu")
            losses = []
            for repeat in range(batch.config.repeats):
                positions = torch.arange(
                    probe.prefix_length + repeat * probe.block_length,
                    probe.prefix_length + (repeat + 1) * probe.block_length,
                    device="cpu",
                )
                losses.append(float(-probabilities[positions, target_ids[positions]].mean()))
            # Finish tensor allocations before publishing a complete example.
            score_tensor = torch.tensor(example_scores, dtype=torch.float64, device="cpu")
            baseline = probabilities[list(probe.repeated_queries)].clone() if causal_heads else None
            scores.append(score_tensor)
            if baseline is not None:
                baselines.append(baseline)
            if len(scores) == 1:
                representative = example_attention
                result["representative_probe"] = index
            result["probe_metrics"].append(
                {
                    "probe": index,
                    "repeat_nll": losses,
                    "first_repeat_nll": losses[0],
                    "repeated_nll": sum(losses[1:]) / (len(losses) - 1),
                }
            )
            result["measured_examples"] = len(scores)
            del output, probabilities
            progress("Scoring repeated token attention")
        except Exception as error:  # noqa: BLE001 - retain complete examples after a failed capture
            result["error"] = str(error)[:1000] or type(error).__name__
            result["failed_at"] = {"stage": "patterns", "probe": index}
            break
    result["heads"] = aggregate_heads(scores, targets, representative)
    result["attention_probe_indices"] = list(range(len(scores)))
    metrics = result["probe_metrics"]
    if metrics:
        result["mean_first_repeat_nll"] = sum(row["first_repeat_nll"] for row in metrics) / len(
            metrics
        )
        result["mean_repeated_nll"] = sum(row["repeated_nll"] for row in metrics) / len(metrics)
    selected = result["heads"][: result["requested_causal_heads"]]
    if causal_candidates is not None and len(scores) == batch.config.examples:
        indexed = {(row["layer"], row["head"]): row for row in result["heads"]}
        selected = [indexed[(row["layer"], row["head"])] for row in causal_candidates]
    if len(scores) == batch.config.examples and not result["error"]:
        for candidate in selected:
            effects = []
            causal = {
                "layer": candidate["layer"],
                "head": candidate["head"],
                "requested_examples": batch.config.examples,
                "completed_examples": 0,
                "complete": False,
                "examples": effects,
            }
            for index, probe in enumerate(batch.probes):
                if job.get("cancel_requested"):
                    break
                handle = None
                try:
                    handle, description = install_head_ablation(
                        blocks, candidate["layer"], candidate["head"]
                    )
                    output, probabilities = forward(probe, attention=False)
                    ablated = probabilities[list(probe.repeated_queries)]
                    baseline = baselines[index]
                    if ablated.shape != baseline.shape:
                        raise ValueError("The ablated output vocabulary differs from its baseline.")
                    rows = torch.arange(len(probe.repeated_queries), device="cpu")
                    labels = torch.tensor(probe.target_ids, dtype=torch.int64, device="cpu")[
                        list(probe.repeated_queries)
                    ]
                    before = float(baseline[rows, labels].mean())
                    after = float(ablated[rows, labels].mean())
                    divergence = float((baseline.exp() * (baseline - ablated)).sum(-1).mean())
                    effects.append(
                        {
                            "probe": index,
                            "baseline_log_probability": before,
                            "ablated_log_probability": after,
                            "delta_log_probability": after - before,
                            "kl_divergence": max(0.0, divergence),
                        }
                    )
                    causal["projection"] = description["projection"]
                    del output, probabilities, ablated
                    progress("Testing repeated token candidate heads")
                except Exception as error:  # noqa: BLE001 - retain completed intervention pairs
                    result["error"] = str(error)[:1000] or type(error).__name__
                    result["failed_at"] = {
                        "stage": "causal",
                        "probe": index,
                        "layer": candidate["layer"],
                        "head": candidate["head"],
                    }
                    break
                finally:
                    if handle is not None:
                        handle.remove()
            if effects:
                causal.update(
                    completed_examples=len(effects),
                    complete=len(effects) == batch.config.examples,
                    mean_delta_log_probability=sum(row["delta_log_probability"] for row in effects)
                    / len(effects),
                    mean_kl_divergence=sum(row["kl_divergence"] for row in effects) / len(effects),
                )
                result["causal_results"].append(causal)
                result["completed_causal_heads"] += int(causal["complete"])
            if job.get("cancel_requested") or result["error"]:
                break
    result["complete"] = (
        len(scores) == batch.config.examples
        and result["completed_causal_heads"] == result["requested_causal_heads"]
        and not result["error"]
    )
    result["duration_ms"] = round((time.perf_counter() - started) * 1000)
    return result
