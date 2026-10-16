"""Measured feature interventions that retain the original reconstruction error."""

import gc
from contextlib import ExitStack, contextmanager
from functools import wraps

import torch

from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.features.corpus import prompt_split
from circuit_lantern.services.scoring import log_probabilities, target_score
from circuit_lantern.services.tokens import capture_token_limit


def model_operation(function):
    @wraps(function)
    def run(lab, *args, **kwargs):
        with lab.runtime.operation(), torch.inference_mode():
            try:
                return function(lab, *args, **kwargs)
            finally:
                gc.collect()
                if lab.runtime.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    return run


def eligible_positions(manifest: dict, token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    mask = torch.ones(len(token_ids), dtype=torch.bool, device="cpu")
    if manifest["corpus"]["special_tokens_excluded"]:
        mask &= ~torch.isin(
            token_ids.cpu(), torch.tensor(tokenizer.all_special_ids, dtype=torch.int64)
        )
    if manifest["corpus"]["first_position_excluded"] and len(mask):
        mask[0] = False
    return mask


def prompt_inputs(runtime, prompt: str) -> tuple[dict, list[str]]:
    encoded = runtime.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    length = encoded["input_ids"].shape[-1]
    limit = capture_token_limit(runtime.metadata, runtime.settings.max_prompt_tokens)
    if not 1 <= length <= limit:
        raise ValueError(
            f"The prompt must contain between 1 and {limit} tokens. It will not be silently truncated."
        )
    return {
        name: value.to(runtime.device)
        for name, value in encoded.items()
        if name in runtime.tokenizer.model_input_names
    }, [runtime.tokenizer.decode([int(token)]) for token in encoded["input_ids"][0]]


@contextmanager
def residual_hook(block, operation):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = operation(hidden)
        if changed is None:
            return None
        if not torch.isfinite(changed).all():
            raise ValueError("The intervention produced a non finite residual stream.")
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def forward(
    runtime, inputs: dict, dictionaries: list[tuple], intervention=None
) -> tuple[torch.Tensor, dict]:
    captured = {}
    with ExitStack() as stack:
        # An intervention hook is installed first so downstream capture observes it.
        if intervention is not None:
            layer, operation = intervention
            stack.enter_context(residual_hook(runtime._blocks()[layer], operation))
        for manifest, dictionary in dictionaries:

            def capture(hidden, manifest=manifest, dictionary=dictionary):
                features = dictionary.encode(hidden[0].float())
                if not torch.isfinite(features).all():
                    raise ValueError("The dictionary produced non finite feature activations.")
                captured[manifest["id"]] = features.cpu()

            stack.enter_context(residual_hook(runtime._blocks()[manifest["layer"]], capture))
        output = runtime.model(
            **inputs, use_cache=False, output_attentions=False, output_hidden_states=False
        )
        logits = output.logits[0].float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("The model produced non finite logits.")
    return logits, captured


def feature_intervention(
    dictionary,
    feature: int,
    multiplier: float,
    position: int | None,
    eligible: torch.Tensor | None = None,
):
    direction = dictionary.feature_direction(feature)

    def operation(hidden):
        original = hidden[0].float()
        activations = dictionary.encode(original)[:, feature]
        delta = (multiplier - 1) * activations[:, None] * direction
        if eligible is not None:
            delta = delta * eligible.to(original.device)[:, None]
        if position is not None:
            if position >= len(original):
                raise ValueError("The intervention position is outside the prompt.")
            mask = torch.zeros(len(original), device=original.device)
            mask[position] = 1
            delta = delta * mask[:, None]
        changed = hidden.clone()
        changed[0] = (original + delta).to(hidden.dtype)
        realized = changed[0].float() - original
        norm = float(delta.norm())
        operation.measurement = {
            "intended_delta_norm": norm,
            "realized_delta_norm": float(realized.norm()),
            "relative_rounding_error": float((realized - delta).norm()) / norm if norm else None,
            "residual_dtype": str(hidden.dtype).removeprefix("torch."),
        }
        return changed

    return operation


def prediction(tokenizer, log_probs: torch.Tensor) -> list[dict]:
    values, indices = log_probs.topk(min(10, len(log_probs)))
    return [
        {
            "token_id": int(index),
            "token": tokenizer.decode([int(index)]),
            "probability": float(value.exp()),
            "log_probability": float(value),
        }
        for value, index in zip(values, indices, strict=True)
    ]


@model_operation
def inspect_prompt(lab, identifier: str, request) -> dict:
    runtime = lab.runtime
    runtime._verify_execution(request.expected_execution_id)
    runtime._require_model()
    inputs, tokens = prompt_inputs(runtime, request.prompt)
    manifest, dictionary = lab.load_dictionaries([identifier], len(tokens))[0]
    if request.position is not None and request.position >= len(tokens):
        raise ValueError("The intervention position is outside the prompt.")
    eligible = eligible_positions(manifest, inputs["input_ids"][0], runtime.tokenizer)
    if request.feature is not None:
        dictionary.feature_direction(request.feature)
        if not eligible.any() or request.position is not None and not eligible[request.position]:
            raise ValueError(
                "The selected token position was excluded by the dictionary training policy."
            )
    baseline_logits, captured = forward(runtime, inputs, [(manifest, dictionary)])
    encoded = captured[identifier]
    rankings = (
        encoded[eligible]
        .sum(dim=0)
        .argsort(descending=True, stable=True)[: min(32, dictionary.config.features)]
    )
    selected = [int(index) for index in rankings if encoded[eligible, index].gt(0).any()]
    if request.feature is not None:
        dictionary.feature_direction(request.feature)
        if request.feature not in selected:
            selected.append(request.feature)
    baseline = log_probabilities(baseline_logits[-1])
    target = (
        request.target_token_id if request.target_token_id is not None else int(baseline.argmax())
    )
    if target >= baseline.numel():
        raise ValueError("The target token ID is outside the loaded vocabulary.")
    response = {
        **execution_provenance(runtime.metadata),
        "dictionary_id": identifier,
        "prompt": request.prompt,
        "tokens": tokens,
        "token_ids": inputs["input_ids"][0].tolist(),
        "features": [
            {
                "index": index,
                "activations": [
                    float(value) if valid else None
                    for value, valid in zip(encoded[:, index], eligible, strict=True)
                ],
            }
            for index in selected
        ],
        "active_features_per_token": [
            int(value) if valid else None
            for value, valid in zip(encoded.gt(0).sum(dim=-1), eligible, strict=True)
        ],
        "eligible_positions": eligible.nonzero().flatten().tolist(),
        "prompt_split": prompt_split(manifest, request.prompt),
        "prompt_split_scope": "Normalized document group matching. Shared phrases, prefixes and paraphrases are not detected.",
        "dictionary_training": manifest["training_summary"],
        "baseline_predictions": prediction(runtime.tokenizer, baseline),
        "activation_units": "Normalized SAE coefficients. Decoder directions convert changes back to original residual units.",
    }
    if request.feature is not None:
        operation = feature_intervention(
            dictionary, request.feature, request.multiplier, request.position, eligible
        )
        patched_logits, _ = forward(runtime, inputs, [], (manifest["layer"], operation))
        patched = log_probabilities(patched_logits[-1])
        response["intervention"] = {
            "feature": request.feature,
            "multiplier": request.multiplier,
            "position": request.position,
            "preserves_original_reconstruction_error": True,
            "rounding": operation.measurement,
            "target_token_id": target,
            "target_token": runtime.tokenizer.decode([target]),
            "baseline_target": target_score(baseline, target),
            "patched_target": target_score(patched, target),
            "target_log_probability_change": float(patched[target] - baseline[target]),
            "kl_baseline_to_patched": max(
                0.0, float((baseline.exp() * (baseline - patched)).sum())
            ),
            "predictions": prediction(runtime.tokenizer, patched),
        }
    return response


@model_operation
def evaluate_prompt(lab, identifier: str, request) -> dict:
    runtime = lab.runtime
    runtime._verify_execution(request.expected_execution_id)
    runtime._require_model()
    inputs, tokens = prompt_inputs(runtime, request.prompt)
    manifest, dictionary = lab.load_dictionaries([identifier], len(tokens))[0]
    baseline, _ = forward(runtime, inputs, [])
    eligible = eligible_positions(manifest, inputs["input_ids"][0], runtime.tokenizer).to(
        runtime.device
    )

    def reconstruction(hidden):
        reconstructed = dictionary(hidden.float())[0].to(hidden.dtype)
        return torch.where(eligible[None, :, None], reconstructed, hidden)

    mean = torch.tensor(manifest["training_mean"], device=runtime.device)

    def mean_ablation(hidden):
        return torch.where(eligible[None, :, None], mean.to(hidden.dtype).expand_as(hidden), hidden)

    reconstructed, _ = forward(runtime, inputs, [], (manifest["layer"], reconstruction))
    ablated, _ = forward(runtime, inputs, [], (manifest["layer"], mean_ablation))
    ids = inputs["input_ids"][0, 1:].cpu()

    def loss(logits):
        if not len(ids):
            return None
        probs = log_probabilities(logits[:-1])
        return float(-probs.gather(1, ids[:, None]).mean())

    original_loss, reconstructed_loss, ablated_loss = map(loss, (baseline, reconstructed, ablated))
    gap = ablated_loss - original_loss if original_loss is not None else None
    return {
        **execution_provenance(runtime.metadata),
        "dictionary_id": identifier,
        "prompt": request.prompt,
        "tokens": tokens,
        "scored_tokens": len(ids),
        "eligible_positions": eligible.nonzero().flatten().tolist(),
        "prompt_split": prompt_split(manifest, request.prompt),
        "prompt_split_scope": "Normalized document group matching. Shared phrases, prefixes and paraphrases are not detected.",
        "dictionary_training": manifest["training_summary"],
        "baseline_cross_entropy": original_loss,
        "reconstructed_cross_entropy": reconstructed_loss,
        "mean_ablated_cross_entropy": ablated_loss,
        "loss_recovered": (ablated_loss - reconstructed_loss) / gap
        if gap is not None and abs(gap) > 1e-12
        else None,
        "scope": "Next-token loss on this prompt after replacing one residual stream. Recovery is not clamped and is unavailable for a zero baseline gap or fewer than two tokens.",
    }
