import inspect
import time

import torch

from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.interventions import attention_head_targets, install_head_ablation
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.scoring import log_probabilities, target_score, target_token_id
from circuit_lantern.services.tokens import capture_token_limit

MAX_SWEEP_HEADS = 512


def select_head_targets(blocks, layers: list[int] | None) -> list[dict]:
    available = {item["layer"]: item for item in attention_head_targets(blocks)}
    selected = sorted(available) if layers is None else layers
    if not selected:
        raise ValueError("Choose at least one supported attention layer.")
    if len(set(selected)) != len(selected):
        raise ValueError("Choose each attention layer only once.")
    if any(layer not in available for layer in selected):
        raise ValueError("One of the selected layers has no verified attention head layout.")
    targets = [available[layer] for layer in sorted(selected)]
    if sum(item["heads"] for item in targets) > MAX_SWEEP_HEADS:
        raise ValueError(f"A sweep can test at most {MAX_SWEEP_HEADS} heads. Select fewer layers.")
    return targets


@torch.inference_mode()
def head_ablation_sweep(
    model,
    tokenizer,
    blocks,
    metadata: dict,
    device: str,
    prompt: str,
    target: str,
    layers: list[int] | None,
    job: JobRecord | dict,
    max_tokens: int,
) -> dict:
    """Zero each head independently and score the final prompt position."""
    started = time.perf_counter()
    if model.training:
        raise ValueError("Head sweeps require the model in evaluation mode.")
    targets = select_head_targets(blocks, layers)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    length = encoded.input_ids.shape[1]
    limit = capture_token_limit(metadata, max_tokens)
    if not length:
        raise ValueError("The prompt must contain at least one token.")
    if length > limit:
        raise ValueError(f"Head sweeps are limited to {limit} prompt tokens.")
    inputs = {name: tensor.to(device) for name, tensor in encoded.items()}

    last_token_options = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )

    def forward():
        return (
            model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **last_token_options,
            )
            .logits[0, -1]
            .clone()
        )

    baseline_logits = forward()
    logits_dtype = str(baseline_logits.dtype).removeprefix("torch.")
    baseline = log_probabilities(baseline_logits)
    del baseline_logits
    target_id = target_token_id(tokenizer, target, baseline)
    baseline_score = target_score(baseline, target_id)
    baseline_probabilities = baseline.exp()
    total = sum(item["heads"] for item in targets)
    results = []
    failure = None
    failed_head = None
    for item in targets:
        for head in range(item["heads"]):
            if job.get("cancel_requested"):
                break
            handle = None
            try:
                handle, description = install_head_ablation(blocks, item["layer"], head)
                ablated = log_probabilities(forward())
            except Exception as error:  # noqa: BLE001 - retain completed research on failed forwards
                failure = str(error)[:1000] or type(error).__name__
                failed_head = {"layer": item["layer"], "head": head}
                break
            finally:
                if handle is not None:
                    handle.remove()
            score = target_score(ablated, target_id)
            top_id = int(ablated.argmax())
            results.append(
                {
                    "layer": item["layer"],
                    "head": head,
                    "projection": description["projection"],
                    **score,
                    "delta_log_probability": score["log_probability"]
                    - baseline_score["log_probability"],
                    "kl_divergence": max(
                        0.0, float((baseline_probabilities * (baseline - ablated)).sum())
                    ),
                    "top_token_id": top_id,
                    "top_token": tokenizer.decode([top_id]),
                }
            )
            job.update(
                progress=round(len(results) / total * 100),
                message=f"Tested head {len(results)} of {total}",
            )
        if failure or job.get("cancel_requested"):
            break
    return {
        **execution_provenance(metadata),
        "device": device,
        "logits_dtype": logits_dtype,
        "last_token_logits_only": bool(last_token_options),
        "complete": len(results) == total,
        "error": failure,
        "failed_head": failed_head,
        "prompt": prompt,
        "tokens": [tokenizer.decode([i]) for i in inputs["input_ids"][0].tolist()],
        "token_ids": inputs["input_ids"][0].cpu().tolist(),
        "requested_target": target,
        "target_token_id": target_id,
        "target_token": tokenizer.decode([target_id]),
        "baseline": baseline_score,
        "baseline_top_token_id": int(baseline.argmax()),
        "layer_targets": targets,
        "heads": results,
        "total_heads": total,
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "scope": "all_prompt_tokens",
        "method": "Each attention head output is zeroed independently before the output projection at every prompt position. Scores describe the next token after the final prompt position. Delta is ablated minus baseline target log probability in nats. KL is baseline to ablated over the complete next token distribution. No tokens are generated. Query heads are tested independently even when key and value heads are shared. This is the total downstream effect of zero ablation, which can move activations outside their usual distribution. Scores describe the loaded execution precision and small changes may reflect rounding.",
    }
