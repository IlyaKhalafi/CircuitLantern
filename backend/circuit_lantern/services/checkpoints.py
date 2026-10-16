"""Sequential checkpoint comparisons with real weight and residual vector distances."""

import hashlib
import json
from contextlib import ExitStack

import torch

from circuit_lantern.services.checkpoint_store import CheckpointStore
from circuit_lantern.services.execution import describe_execution, execution_provenance
from circuit_lantern.services.features.inference import (
    model_operation,
    prompt_inputs,
    residual_hook,
)
from circuit_lantern.services.features.planning import forward_memory
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.weights import logical_weight, model_weight_inventory


def vector_change(selected, reference):
    left, right = selected.reshape(-1).double(), reference.reshape(-1).double()
    a, b = float(left.norm()), float(right.norm())
    delta = float((left - right).norm())
    return {
        "selected_norm": a,
        "reference_norm": b,
        "distance": delta,
        "relative_distance": delta / b if b else None,
        "cosine_similarity": max(-1.0, min(1.0, float(torch.dot(left, right)) / (a * b)))
        if a and b
        else None,
    }


def weight_change(selected, reference):
    # Accumulate in float64 with bounded work buffers rather than doubling whole matrices.
    a2 = b2 = delta2 = dot = maximum = 0.0
    for a, b in zip(
        selected.reshape(-1).split(262144), reference.reshape(-1).split(262144), strict=True
    ):
        a, b = a.double(), b.double()
        delta = a - b
        a2 += float(a.square().sum())
        b2 += float(b.square().sum())
        delta2 += float(delta.square().sum())
        dot += float((a * b).sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {
        "selected_norm": a2**0.5,
        "reference_norm": b2**0.5,
        "distance": delta2**0.5,
        "relative_distance": (delta2 / b2) ** 0.5 if b2 else None,
        "cosine_similarity": max(-1.0, min(1.0, dot / (a2 * b2) ** 0.5)) if a2 and b2 else None,
        "rms_change": (delta2 / selected.numel()) ** 0.5,
        "max_absolute_change": maximum,
    }


def predictions(logits, tokenizer, count=5):
    probabilities = logits.double().softmax(-1)
    values, indices = probabilities.topk(min(count, len(probabilities)))
    return [
        {
            "token_id": int(index),
            "token": tokenizer.decode([int(index)]),
            "probability": float(value),
        }
        for index, value in zip(indices, values, strict=True)
    ]


class CheckpointLab:
    def __init__(self, runtime, store: CheckpointStore):
        self.runtime, self.store = runtime, store

    def _profile(self):
        model, blocks = self.runtime.model, self.runtime._blocks()
        if blocks is None or not len(blocks):
            raise ValueError("This architecture does not expose comparable transformer blocks.")
        config = model.config
        return {
            "architecture": config.model_type,
            "blocks": [type(block).__name__ for block in blocks],
            "hidden_size": int(model.get_input_embeddings().weight.shape[1]),
            "vocab_size": int(model.get_output_embeddings().weight.shape[0]),
            "num_heads": getattr(config, "num_attention_heads", None),
            "num_kv_heads": getattr(config, "num_key_value_heads", None),
            "head_dim": getattr(config, "head_dim", None),
            "intermediate_size": getattr(config, "intermediate_size", None)
            or getattr(config, "n_inner", None),
            "parallel_residual": getattr(config, "use_parallel_residual", None),
            "tied_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        }

    def _validate(self, prompt, names):
        runtime = self.runtime
        runtime._require_model()
        if (
            not prompt.strip()
            or not 1 <= len(prompt) <= 8000
            or not 1 <= len(names) <= 12
            or len(set(names)) != len(names)
        ):
            raise ValueError("Provide a prompt and between one and twelve distinct matrices.")
        profile = self._profile()
        inputs, tokens = prompt_inputs(runtime, prompt)
        inventory = {item["name"]: item for item in model_weight_inventory(runtime.model)[0]}
        if any(name not in inventory for name in names):
            raise ValueError(
                "A reference matrix is missing from the loaded checkpoint. Compare models with matching weight names."
            )
        weights = sum(inventory[name]["elements"] * 4 for name in names)
        width = profile["hidden_size"]
        evidence = (
            weights
            + (
                len(profile["blocks"]) * width
                + runtime.model.get_output_embeddings().weight.shape[0]
            )
            * 4
        )
        if evidence > self.store.MAX_BYTES:
            raise ValueError(
                "The capture exceeds the 128 MiB evidence limit. Select fewer or smaller matrices."
            )
        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        hardware = hardware_snapshot(runtime.settings.data_dir)
        teacher = forward_memory(runtime.model, len(tokens))
        host = 3 * evidence + teacher["host_bytes"]
        workspace = (
            teacher["device_bytes"] + max(inventory[name]["elements"] for name in names) * 12
        )
        if runtime.device.startswith("cuda"):
            if workspace > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75:
                raise ValueError(
                    "Insufficient free GPU memory for checkpoint capture with the model resident."
                )
        else:
            host += workspace
        if host > hardware["ram_available_bytes"] * 0.75:
            raise ValueError("Insufficient available RAM for checkpoint capture and comparison.")
        if evidence * 1.25 > hardware["experiment_disk_free_bytes"]:
            raise ValueError("Insufficient disk space for checkpoint evidence.")
        return profile, inputs, tokens, inventory

    def _capture(self, name, prompt, names, prepared=None):
        runtime = self.runtime
        profile, inputs, tokens, inventory = prepared or self._validate(prompt, names)
        tensors = {}
        for index, matrix in enumerate(names):
            value, _ = logical_weight(runtime.model, matrix)
            tensors[f"weight_{index}"] = value.detach().cpu().clone().contiguous()
            del value
        layers, residuals = [], []
        norm, head = runtime._final_norm(), runtime.model.get_output_embeddings()
        with ExitStack() as stack:
            for index, block in enumerate(runtime._blocks()):

                def capture(hidden, index=index):
                    vector = hidden[:, -1:, :]
                    residuals.append(vector[0, 0].detach().float().cpu().clone())
                    projected = norm(vector) if norm is not None else vector
                    logits = head(projected)[0, 0].float().cpu()
                    if not torch.isfinite(logits).all():
                        raise ValueError("The diagnostic logit lens produced non finite logits.")
                    probabilities = logits.double().softmax(-1)
                    layers.append(
                        {
                            "layer": index,
                            "predictions": predictions(logits, runtime.tokenizer),
                            "entropy": float(
                                -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
                            ),
                        }
                    )

                stack.enter_context(residual_hook(block, capture))
            output = runtime.model(
                **inputs, use_cache=False, output_attentions=False, output_hidden_states=False
            )
            tensors["logits"] = output.logits[0, -1].detach().float().cpu().clone()
        if len(residuals) != len(profile["blocks"]):
            raise ValueError("The model did not execute each transformer block exactly once.")
        tensors["residuals"] = torch.stack(residuals)
        if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError("The checkpoint produced non finite evidence.")
        vocabulary = sorted(runtime.tokenizer.get_vocab().items())
        manifest = {
            "name": name,
            "prompt": prompt,
            "profile": profile,
            **execution_provenance(runtime.metadata),
            "execution_config": describe_execution(
                runtime.model,
                runtime.metadata.get("execution_precision", "float32"),
                runtime.device,
            ),
            "model_inputs": {key: value[0].cpu().tolist() for key, value in inputs.items()},
            "tokens": tokens,
            "vocabulary_hash": hashlib.sha256(
                json.dumps(vocabulary, ensure_ascii=False).encode()
            ).hexdigest(),
            "matrices": [
                dict(inventory[matrix], tensor=f"weight_{index}")
                for index, matrix in enumerate(names)
            ],
            "layers": layers,
            "predictions": predictions(tensors["logits"], runtime.tokenizer, 10),
            "method": "Raw block outputs at the final prompt token. Diagnostic per-layer predictions use the loaded final normalization and output projection. Final predictions use actual model logits. No generation or interventions.",
        }
        return manifest, tensors

    @model_operation
    def capture(self, request):
        self.runtime._require_model()
        self.runtime._verify_execution(request.expected_execution_id)
        if not request.name.strip():
            raise ValueError("Give the reference capture a name.")
        self.store.assert_room()
        manifest, tensors = self._capture(request.name.strip(), request.prompt, request.matrices)
        return self.store.save(manifest, tensors)

    @model_operation
    def compare(self, request):
        runtime = self.runtime
        runtime._require_model()
        runtime._verify_execution(request.expected_execution_id)
        self.store.assert_room(comparison=True)
        reference = self.store.read(request.reference_id)
        names = [item["name"] for item in reference["matrices"]]
        prepared = self._validate(reference["prompt"], names)
        if prepared[0] != reference["profile"]:
            raise ValueError(
                "The checkpoint architectures do not align. Compare versions with the same block layout, residual width and attention head configuration."
            )
        if any(
            prepared[3][item["name"]]["shape"] != item["shape"] for item in reference["matrices"]
        ):
            raise ValueError("The reference matrix shapes do not match the loaded checkpoint.")
        if {key: value[0].cpu().tolist() for key, value in prepared[1].items()} != reference[
            "model_inputs"
        ]:
            raise ValueError(
                "The prompt tokenization or model inputs differ. Checkpoint comparison requires identical prompt token IDs and inputs."
            )
        vocabulary_hash = hashlib.sha256(
            json.dumps(sorted(runtime.tokenizer.get_vocab().items()), ensure_ascii=False).encode()
        ).hexdigest()
        if vocabulary_hash != reference["vocabulary_hash"]:
            raise ValueError(
                "The vocabulary mappings differ. Prediction distributions cannot be aligned safely."
            )
        # Validate compatibility and memory before loading retained reference tensors.
        reference, original = self.store.load(request.reference_id)
        selected, current = self._capture(
            "Selected checkpoint", reference["prompt"], names, prepared
        )
        if (
            original["residuals"].shape != current["residuals"].shape
            or original["logits"].shape != current["logits"].shape
        ):
            raise ValueError("The captured residual or output vocabulary shapes do not align.")
        weights = []
        for item in reference["matrices"]:
            key = item["tensor"]
            weights.append(
                {
                    "name": item["name"],
                    "shape": item["shape"],
                    **weight_change(current[key], original[key]),
                }
            )
        layers = [
            {
                "layer": index,
                **vector_change(current["residuals"][index], original["residuals"][index]),
                "selected_prediction": selected["layers"][index]["predictions"][0],
                "reference_prediction": reference["layers"][index]["predictions"][0],
                "entropy_change": selected["layers"][index]["entropy"]
                - reference["layers"][index]["entropy"],
            }
            for index in range(len(selected["layers"]))
        ]
        base_logp, selected_logp = (
            original["logits"].double().log_softmax(-1),
            current["logits"].double().log_softmax(-1),
        )
        base_p, selected_p = base_logp.exp(), selected_logp.exp()
        warnings = []
        if selected["execution_config"] != reference["execution_config"]:
            warnings.append(
                "Execution settings differ. Observed changes combine checkpoint and execution format effects."
            )
        if (
            selected["model_id"] == reference["model_id"]
            and selected["revision"] == reference["revision"]
        ):
            warnings.append(
                "Both captures identify the same checkpoint. Use a different revision or finetuned repository to study checkpoint changes."
            )
        return self.store.save_comparison(
            {
                "format": "circuit-lantern-checkpoint-comparison-v1",
                "reference": reference,
                "selected": selected,
                "weights": weights,
                "layers": layers,
                "kl_reference_to_selected": max(
                    0.0, float((base_p * (base_logp - selected_logp)).sum())
                ),
                "kl_selected_to_reference": max(
                    0.0, float((selected_p * (selected_logp - base_logp)).sum())
                ),
                "total_variation": float((base_p - selected_p).abs().sum()) / 2,
                "predictions": [
                    {
                        "token_id": int(token_id),
                        "token": runtime.tokenizer.decode([int(token_id)]),
                        "reference_probability": float(base_p[token_id]),
                        "selected_probability": float(selected_p[token_id]),
                        "log_probability_change": float(
                            selected_logp[token_id] - base_logp[token_id]
                        ),
                    }
                    for token_id in torch.unique(
                        torch.cat(
                            (
                                base_p.topk(min(10, len(base_p))).indices,
                                selected_p.topk(min(10, len(selected_p))).indices,
                            )
                        )
                    ).tolist()
                ],
                "warnings": warnings,
                "convention": "Selected minus reference. Distances are actual aligned float32 weight and residual vector distances accumulated in float64. Quantized weights are dequantized. Residual coordinates and weight names are assumed aligned across checkpoints. Differences do not establish causal attribution. KL and total variation use the full actual final-token vocabulary distribution.",
            }
        )
