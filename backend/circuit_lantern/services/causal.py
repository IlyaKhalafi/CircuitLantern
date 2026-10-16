import time

import torch

from circuit_lantern.services.jobs import JobRecord

MIN_RECOVERY_GAP = 0.001


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
    position: int | None,
    job: JobRecord | dict,
    max_tokens: int,
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
    limit = min(
        max_tokens,
        metadata["context_length"],
        metadata.get("execution_context_length", metadata["context_length"]),
    )
    if length > limit:
        raise ValueError(f"Causal captures are limited to {limit} prompt tokens.")
    if position is not None and not -length <= position < length:
        raise ValueError("Patch position is outside the tokenized prompt.")
    if blocks is None:
        raise ValueError("Activation patching is not supported for this architecture yet.")
    clean = {k: v.to(device) for k, v in clean.items()}
    corrupted = {k: v.to(device) for k, v in corrupted.items()}
    cache, handles = {}, []
    for index, block in enumerate(blocks):

        def capture(module, args, output, layer=index):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[layer] = hidden.detach().clone()

        handles.append(block.register_forward_hook(capture))
    try:
        clean_logits = model(**clean, use_cache=False).logits[0, -1].float().clone()
    finally:
        for handle in handles:
            handle.remove()
    if target:
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(target_ids) != 1:
            raise ValueError(
                "The target must encode to exactly one token. Include a leading space when needed, or leave it empty to use the clean top prediction."
            )
        target_id = target_ids[0]
    else:
        target_id = int(clean_logits.argmax())
    corrupted_logits = model(**corrupted, use_cache=False).logits[0, -1].float().clone()

    def score(logits):
        if not torch.isfinite(logits).all():
            raise ValueError(
                "The model produced non finite logits. Choose a different checkpoint or prompt."
            )
        log_probability = float(logits.log_softmax(-1)[target_id])
        return {
            "log_probability": log_probability,
            "probability": float(logits.softmax(-1)[target_id]),
        }

    clean_score, corrupted_score = score(clean_logits), score(corrupted_logits)
    denominator = clean_score["log_probability"] - corrupted_score["log_probability"]
    recovery_defined = denominator > MIN_RECOVERY_GAP
    results = []
    for index, block in enumerate(blocks):
        if job.get("cancel_requested"):
            job.update(message="Cancellation requested. Publishing completed layers.")
            break

        def restore(module, args, output, layer=index):
            hidden = output[0] if isinstance(output, tuple) else output
            if position is None:
                patched = cache[layer].clone()
            else:
                patched = hidden.clone()
                patched[:, position] = cache[layer][:, position]
            return replace_hidden(output, patched)

        handle = block.register_forward_hook(restore)
        try:
            logits = model(**corrupted, use_cache=False).logits[0, -1].float().clone()
        finally:
            handle.remove()
        patched = score(logits)
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
            progress=round((index + 1) / len(blocks) * 100),
            message=f"Patched layer {index + 1} of {len(blocks)}",
        )
    return {
        "model_id": metadata["id"],
        "execution_precision": metadata.get("execution_precision", "float32"),
        "execution_config": metadata.get("execution_config"),
        "device": device,
        "revision": metadata["revision"],
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "tokens": [tokenizer.decode([i]) for i in clean["input_ids"][0].tolist()],
        "corrupted_tokens": [tokenizer.decode([i]) for i in corrupted["input_ids"][0].tolist()],
        "target_token": tokenizer.decode([target_id]),
        "requested_target": target,
        "target_token_id": target_id,
        "position": position,
        "clean": clean_score,
        "corrupted": corrupted_score,
        "layers": results,
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "recovery_defined": recovery_defined,
        "baseline_log_probability_gap": denominator,
        "recovery_minimum_gap": MIN_RECOVERY_GAP,
        "recovery_warning": "Small clean to corrupted score gap. Interpret normalized recovery cautiously."
        if recovery_defined and denominator < 0.05
        else None,
        "total_layers": len(blocks),
        "method": "Clean residual output is restored in the corrupted forward pass at one transformer block per run. Recovery is normalized target log probability change. It may be negative or exceed one.",
    }
