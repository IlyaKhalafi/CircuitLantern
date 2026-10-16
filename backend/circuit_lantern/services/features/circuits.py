"""Exact forward ablations over selected SAE features in the original model."""

import torch

from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.features.corpus import prompt_split
from circuit_lantern.services.features.inference import (
    eligible_positions,
    feature_intervention,
    forward,
    prediction,
    prompt_inputs,
)
from circuit_lantern.services.scoring import log_probabilities


def feature_ablation_graph(lab, payload: dict, job) -> dict:
    runtime = lab.runtime
    runtime._require_model()
    inputs, tokens = prompt_inputs(runtime, payload["prompt"])
    dictionaries = lab.load_dictionaries(payload["dictionary_ids"], len(tokens))
    if len({manifest["layer"] for manifest, _ in dictionaries}) != len(dictionaries):
        raise ValueError("Choose one dictionary per transformer layer.")
    dictionaries.sort(key=lambda item: item[0]["layer"])
    masks = {
        manifest["id"]: eligible_positions(manifest, inputs["input_ids"][0], runtime.tokenizer)
        for manifest, _ in dictionaries
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("The prompt has no eligible positions for a selected dictionary.")

    result = {
        **execution_provenance(runtime.metadata),
        "kind": "feature_circuit",
        "dictionary_ids": [manifest["id"] for manifest, _ in dictionaries],
        "dictionary_training": {
            manifest["id"]: manifest["training_summary"] for manifest, _ in dictionaries
        },
        "prompt_splits": {
            manifest["id"]: prompt_split(manifest, payload["prompt"])
            for manifest, _ in dictionaries
        },
        "eligible_positions": {
            identifier: mask.nonzero().flatten().tolist() for identifier, mask in masks.items()
        },
        "dictionary_weights": {
            manifest["id"]: manifest["weights_sha256"] for manifest, _ in dictionaries
        },
        "prompt": payload["prompt"],
        "tokens": tokens,
        "token_ids": inputs["input_ids"][0].tolist(),
        "features_per_layer": payload["features_per_layer"],
        "nodes": [],
        "edges": [],
        "measurements": [],
        "complete": False,
        "method": {
            "name": "Feature ablation graph",
            "intervention": "Remove one selected decoded feature contribution at eligible prompt positions while preserving the original SAE reconstruction error.",
            "selection": "Choose features by largest summed baseline activation over each dictionary's eligible prompt positions.",
            "edge_measurement": "Change in a later feature's mean normalized SAE coefficient across eligible prompt positions of the destination dictionary after the source ablation. All selected feature changes are retained.",
            "edge_position_scope": "destination_dictionary_eligible_positions",
            "output_measurement": "Change in the fixed target token log probability at the final prompt position.",
            "scope": "Total downstream intervention effects in the original language model. Edges can include indirect paths and are not direct attribution edges. This is not a cross layer transcoder replacement model or a replication of the linear circuit discovery algorithm.",
            "references": [
                "https://arxiv.org/abs/2403.19647",
                "https://transformer-circuits.pub/2025/attribution-graphs/methods.html",
            ],
        },
    }
    with torch.inference_mode():
        logits, baseline_features = forward(runtime, inputs, dictionaries)
        baseline = log_probabilities(logits[-1])
        target = payload.get("target_token_id")
        target = int(baseline.argmax()) if target is None else target
        if not 0 <= target < baseline.numel():
            raise ValueError("The target token ID is outside the loaded vocabulary.")
        result.update(
            target_token_id=target,
            target_token=runtime.tokenizer.decode([target]),
            baseline_predictions=prediction(runtime.tokenizer, baseline),
        )
        sources = []
        for manifest, dictionary in dictionaries:
            encoded = baseline_features[manifest["id"]][masks[manifest["id"]]]
            order = encoded.sum(dim=0).argsort(descending=True, stable=True)
            for feature in order[: payload["features_per_layer"]].tolist():
                mean = float(encoded[:, feature].mean())
                if mean <= 0:
                    continue
                node = {
                    "id": manifest["id"] + "." + str(feature),
                    "dictionary_id": manifest["id"],
                    "layer": manifest["layer"],
                    "feature": feature,
                    "mean_activation": mean,
                    "label": manifest.get("feature_labels", {})
                    .get(str(feature), {})
                    .get("label", ""),
                }
                result["nodes"].append(node)
                sources.append((node, dictionary))
        if not sources:
            raise ValueError("No selected feature is active at eligible positions.")
        total = len(sources)
        result["planned_measurements"] = total
        job.update(message="Measuring feature ablations", planned_measurements=total)
        try:
            for index, (node, dictionary) in enumerate(sources):
                if job.get("cancel_requested"):
                    break
                intervention = feature_intervention(
                    dictionary, node["feature"], 0, None, masks[node["dictionary_id"]]
                )
                logits, patched_features = forward(
                    runtime,
                    inputs,
                    dictionaries,
                    (node["layer"], intervention),
                )
                patched = log_probabilities(logits[-1])
                output_change = float(patched[target] - baseline[target])
                changes = []
                for destination in result["nodes"]:
                    if destination["layer"] <= node["layer"]:
                        continue
                    after = float(
                        patched_features[destination["dictionary_id"]][
                            masks[destination["dictionary_id"]], destination["feature"]
                        ].mean()
                    )
                    change = after - destination["mean_activation"]
                    edge = {
                        "source": node["id"],
                        "target": destination["id"],
                        "activation_change": change,
                    }
                    result["edges"].append(edge)
                    changes.append(dict(edge))
                result["edges"].append(
                    {
                        "source": node["id"],
                        "target": "output",
                        "log_probability_change": output_change,
                    }
                )
                result["measurements"].append(
                    {
                        "source": node["id"],
                        "target_log_probability_change": output_change,
                        "kl_baseline_to_patched": max(
                            0.0, float((baseline.exp() * (baseline - patched)).sum())
                        ),
                        "downstream_changes": changes,
                        "rounding": intervention.measurement,
                    }
                )
                job.update(
                    progress=100 * (index + 1) / max(1, total), completed_measurements=index + 1
                )
        except Exception as error:  # noqa: BLE001 - preserve successful measurements in a failed sweep
            result["error"] = str(error)[:1000]
        result["complete"] = len(result["measurements"]) == total
    result["cancelled"] = (
        bool(job.get("cancel_requested")) and not result["complete"] and not result.get("error")
    )
    return lab.store.save_run(result)
