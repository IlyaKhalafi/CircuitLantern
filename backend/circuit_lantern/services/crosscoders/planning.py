"""Additional memory and storage estimates with source models left resident."""

import torch

from circuit_lantern.services.crosscoders.model import CrosscoderConfig


def capture_budget(
    width: int,
    samples: int,
    corpus_bytes: int,
    forward: dict,
    device: str,
    hardware: dict,
    limit: int,
) -> dict:
    tensors = width * samples * 4
    metadata = corpus_bytes + 16 * 1024
    host = tensors * 3 + metadata * 112
    gpu = forward["device_bytes"] if device.startswith("cuda") else 0
    if not gpu:
        host += forward["device_bytes"]
    disk = tensors * 2 + corpus_bytes * 3 + 1024**2
    return _budget(host, gpu, disk, tensors, hardware, limit, device)


def training_budget(
    widths: tuple[int, ...],
    samples: int,
    config: CrosscoderConfig,
    corpus_bytes: int,
    device: str,
    hardware: dict,
    limit: int,
) -> dict:
    config.validate(widths)
    parameters = 2 * sum(widths) * config.features + sum(widths) + config.features
    weights = 4 * (parameters + len(widths))
    batch = max(128, min(config.batch_size, samples))
    work = 32 * parameters + 40 * batch * config.features + 24 * batch * sum(widths)
    host = 16 * samples * sum(widths) + 12 * parameters + config.features * 4096 + corpus_bytes * 8
    # Include immutable metadata publication and later JSON export before starting training.
    metadata = config.features * 1024 + corpus_bytes * 2 + config.steps * 512 + 16 * 1024
    host = max(host, metadata * 112 + weights * 3)
    gpu = work if device == "cuda" else 0
    if device == "cpu":
        host += work
    disk = weights * 2 + config.features * 4096 + corpus_bytes * 3 + 2 * 1024**2
    result = _budget(host, gpu, disk, weights, hardware, limit, device)
    result.update(features=config.features, input_dims=list(widths), samples=samples)
    return result


def _budget(
    host: int, gpu: int, disk: int, tensors: int, hardware: dict, limit: int, device: str
) -> dict:
    blockers = []
    if tensors > limit:
        blockers.append(
            "The tensor artifact exceeds the server size limit. Reduce samples or expansion."
        )
    if host > hardware["ram_available_bytes"] * 0.75:
        blockers.append("Insufficient available RAM for the retained activation workflow.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        blockers.append("CUDA training is unavailable on the visible server devices.")
    elif gpu > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75:
        blockers.append("Insufficient free GPU memory for this operation with the model resident.")
    if disk > hardware["experiment_disk_free_bytes"] * 0.9:
        blockers.append("Insufficient disk space to publish the retained tensor artifacts.")
    return {
        "compatible": not blockers,
        "blockers": blockers,
        "estimated_host_bytes": host,
        "estimated_gpu_bytes": gpu,
        "estimated_disk_bytes": disk,
        "tensor_bytes": tensors,
        "device": device,
        "estimate_scope": "Additional memory with the current model left resident. Estimates keep 25 percent of available memory as headroom.",
    }
