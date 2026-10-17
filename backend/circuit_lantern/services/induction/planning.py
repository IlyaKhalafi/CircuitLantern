"""Forward, attention, baseline cache and result memory estimates."""

import psutil

from circuit_lantern.services.features.planning import forward_memory
from circuit_lantern.services.induction.probes import ProbeConfig
from circuit_lantern.services.resources import memory_budget


def output_vocabulary_size(model) -> int:
    output = model.get_output_embeddings()
    size = getattr(output, "out_features", None)
    return int(size if size is not None else output.weight.shape[0])


def probe_plan(
    model,
    config: ProbeConfig,
    *,
    prefix_length: int,
    causal_heads: int,
    all_heads: list[dict],
    selected_heads: list[dict],
    device: str,
    hardware: dict,
    retained_reference_bytes: int = 0,
) -> dict:
    config.validate()
    tokens = prefix_length + config.block_length * config.repeats
    repeated = config.block_length * (config.repeats - 1)
    vocabulary = output_vocabulary_size(model)
    count = sum(item["heads"] for item in selected_heads)
    causal = min(count, causal_heads)
    maximum_layer_heads = max(item["heads"] for item in selected_heads)
    teacher = forward_memory(model, tokens)
    attention = sum(item["heads"] for item in all_heads) * tokens**2 * 8
    cache = config.examples * repeated * vocabulary * 8 if causal else 0
    result = count * tokens * 40 + count * config.examples * 256 + config.examples * tokens * 128
    host = (
        teacher["host_bytes"]
        + retained_reference_bytes * 12
        + cache
        + maximum_layer_heads * tokens**2 * 32
        + int(model.get_input_embeddings().weight.shape[0]) * 256
        + result * 4
    )
    working = teacher["device_bytes"] + attention + tokens * vocabulary * 16
    gpu = working if device.startswith("cuda") else 0
    if not gpu:
        host += working
    disk = result * 3 + 1024**2
    blockers = []
    if host > hardware["ram_available_bytes"] * 0.75:
        blockers.append(
            "Insufficient available RAM for probe baselines and attention measurements."
        )
    if gpu > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75:
        blockers.append("Insufficient free GPU memory for this head probe with the model resident.")
    if disk > hardware["experiment_disk_free_bytes"] * 0.9:
        blockers.append("Insufficient disk space for the saved probe evidence.")
    return {
        "compatible": not blockers,
        "blockers": blockers,
        "estimated_host_bytes": host,
        "estimated_gpu_bytes": gpu,
        "estimated_disk_bytes": disk,
        "baseline_cache_bytes": cache,
        "retained_reference_bytes": retained_reference_bytes,
        "tokens_per_probe": tokens,
        "examples": config.examples,
        "selected_heads": count,
        "causal_heads": causal,
        "requested_forward_passes": config.examples * (1 + causal),
        "device": device,
        "estimate_scope": "Additional memory with the model resident. Full causal distributions are cached on CPU for reuse. Estimates retain 25 percent of available memory as headroom.",
    }


def ensure_saved_comparison_memory(source_bytes: int):
    memory = psutil.virtual_memory()
    available = memory_budget(memory.total, memory.available)["ram_available_bytes"]
    if source_bytes * 16 > available * 0.75:
        raise ValueError("Insufficient available RAM to compare these saved probes.")
