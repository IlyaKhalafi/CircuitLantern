import time

import torch

from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.scoring import log_probabilities, target_score, target_token_id
from circuit_lantern.services.targets import single_target_id
from circuit_lantern.services.tokens import capture_token_limit

MIN_RECOVERY_GAP = 0.001
SMALL_RECOVERY_GAP = 0.05


def normalized_recovery(clean: float, corrupted: float, patched: float) -> float | None:
    gap = clean - corrupted
    return (patched - corrupted) / gap if gap > MIN_RECOVERY_GAP else None


def replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@torch.inference_mode()
def activation_patch_sweep(
    model,
    tokenizer,
    blocks,
    metadata: dict,
    device: str,
    clean_prompt: str,
    corrupted_prompt: str,
    target: str,
    position: int | list[int] | None,
    job: JobRecord | dict,
    max_tokens: int,
    layer_indices: list[int] | None = None,
    alternative: str | None = None,
) -> dict:
    """Restore clean block outputs in a corrupted run, one layer at a time."""
    started = time.perf_counter()
    clean = tokenizer(clean_prompt, return_tensors="pt", add_special_tokens=True)
    corrupted = tokenizer(corrupted_prompt, return_tensors="pt", add_special_tokens=True)
    length = clean.input_ids.shape[1]
    if not length or not corrupted.input_ids.shape[1]:
        raise ValueError("Both prompts must contain at least one token.")
    if length != corrupted.input_ids.shape[1]:
        raise ValueError(
            "Clean and corrupted prompts must tokenize to the same length. Adjust the wording so token positions align."
        )
    limit = capture_token_limit(metadata, max_tokens)
    if length > limit:
        raise ValueError(f"Causal captures are limited to {limit} prompt tokens.")
    positions = (
        position if isinstance(position, list) else [position] if position is not None else None
    )
    if positions is not None and (
        not positions or any(not -length <= item < length for item in positions)
    ):
        raise ValueError("Patch position is outside the tokenized prompt.")
    if blocks is None:
        raise ValueError("Activation patching is not supported for this architecture yet.")
    selected = list(range(len(blocks))) if layer_indices is None else sorted(layer_indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not 0 <= index < len(blocks) for index in selected)
    ):
        raise ValueError("Choose distinct transformer layers within the loaded model.")
    clean_ids = clean.input_ids[0].tolist()
    corrupted_ids = corrupted.input_ids[0].tolist()
    clean_tokens = [tokenizer.decode([i]) for i in clean_ids]
    corrupted_tokens = [tokenizer.decode([i]) for i in corrupted_ids]
    clean = {k: v.to(device) for k, v in clean.items()}
    corrupted = {k: v.to(device) for k, v in corrupted.items()}
    cache, handles = {}, []
    for index in selected:
        block = blocks[index]

        def capture(module, args, output, layer=index):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[layer] = (hidden if positions is None else hidden[:, positions]).detach().clone()

        handles.append(block.register_forward_hook(capture))
    try:
        clean_logits = model(**clean, use_cache=False).logits[0, -1].float().clone()
    finally:
        for handle in handles:
            handle.remove()
    clean_probs = log_probabilities(clean_logits)
    target_id = target_token_id(tokenizer, target, clean_probs)
    alternative_id = (
        single_target_id(tokenizer, alternative, clean_probs.numel()) if alternative else None
    )
    if alternative_id == target_id:
        raise ValueError("Target and alternative must identify different tokens.")

    def score(logits):
        result = target_score(log_probabilities(logits), target_id)
        if alternative_id is not None:
            result["logit_difference"] = float(
                logits.double()[target_id] - logits.double()[alternative_id]
            )
        return result

    corrupted_logits = model(**corrupted, use_cache=False).logits[0, -1].float().clone()
    clean_score = target_score(clean_probs, target_id)
    if alternative_id is not None:
        clean_score["logit_difference"] = float(
            clean_logits.double()[target_id] - clean_logits.double()[alternative_id]
        )
    corrupted_score = score(corrupted_logits)
    denominator = clean_score["log_probability"] - corrupted_score["log_probability"]
    recovery_defined = denominator > MIN_RECOVERY_GAP
    results = []
    failure = None
    failed_layer = None
    for step, index in enumerate(selected):
        block = blocks[index]
        if job.get("cancel_requested"):
            job.update(message="Cancellation requested. Publishing completed layers.")
            break

        def restore(module, args, output, layer=index):
            hidden = output[0] if isinstance(output, tuple) else output
            if positions is None:
                patched = cache[layer].clone()
            else:
                patched = hidden.clone()
                patched[:, positions] = cache[layer]
            return replace_hidden(output, patched)

        handle = None
        try:
            handle = block.register_forward_hook(restore)
            logits = model(**corrupted, use_cache=False).logits[0, -1].float().clone()
            patched = score(logits)
        except Exception as error:  # noqa: BLE001 - retain completed research on failed forwards
            failure = str(error)[:1000] or type(error).__name__
            failed_layer = index
            break
        finally:
            if handle is not None:
                handle.remove()
        delta = patched["log_probability"] - corrupted_score["log_probability"]
        results.append(
            {
                "layer": index,
                **patched,
                "delta_log_probability": delta,
                "recovery": normalized_recovery(
                    clean_score["log_probability"],
                    corrupted_score["log_probability"],
                    patched["log_probability"],
                ),
            }
        )
        job.update(
            progress=round((step + 1) / len(selected) * 100),
            message=f"Patched layer {step + 1} of {len(selected)}",
        )
    return {
        **execution_provenance(metadata),
        "device": device,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "token_ids": clean_ids,
        "corrupted_token_ids": corrupted_ids,
        "target_token": tokenizer.decode([target_id]),
        "requested_target": target,
        "target_token_id": target_id,
        "position": position,
        **(
            {
                "alternative_token_id": alternative_id,
                "alternative_token": tokenizer.decode([alternative_id]),
                "requested_alternative": alternative,
            }
            if alternative_id is not None
            else {}
        ),
        "clean": clean_score,
        "corrupted": corrupted_score,
        "layers": results,
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "recovery_defined": recovery_defined,
        "baseline_log_probability_gap": denominator,
        "recovery_minimum_gap": MIN_RECOVERY_GAP,
        "recovery_warning": "Small clean to corrupted score gap. Interpret normalized recovery cautiously."
        if recovery_defined and denominator < SMALL_RECOVERY_GAP
        else None,
        "total_layers": len(selected),
        "complete": len(results) == len(selected),
        "error": failure,
        "failed_layer": failed_layer,
        "method": "Clean residual output is restored in the corrupted forward pass at one transformer block per run. Recovery is normalized target log probability change. It may be negative or exceed one.",
    }
