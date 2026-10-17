"""Conservative additional memory estimates while the teacher stays resident."""

from circuit_lantern.services.features.autoencoder import TrainingConfig


def dictionary_plan(
    input_dim: int,
    config: TrainingConfig,
    samples: int,
    layers: int,
    device: str,
    hardware: dict,
    max_weight_bytes: int,
    corpus_bytes: int = 0,
) -> dict:
    config.validate(input_dim)
    parameters = 2 * input_dim * config.features + input_dim
    if config.method == "relu":
        parameters += config.features
    weights = 4 * (parameters + 1)
    batch = min(config.batch_size, samples)
    work_batch = max(batch, 256)  # Evaluation and feature ranking use fixed 256 row batches.
    training_work = (
        32 * parameters + 40 * work_batch * config.features + 16 * work_batch * input_dim
    )
    host = 24 * samples * input_dim + 12 * parameters + config.features * 4096 + corpus_bytes * 8
    gpu = training_work if device.startswith("cuda") else 0
    if not gpu:
        host += training_work
    disk = layers * (weights + config.features * 800 + 2 * 1024**2 + corpus_bytes * 2)
    blockers = []
    if weights > max_weight_bytes:
        blockers.append(
            "The dictionary exceeds the server artifact size limit. Reduce its expansion."
        )
    if host > hardware["ram_available_bytes"] * 0.75:
        blockers.append(
            "Insufficient available RAM for activation capture and dictionary training."
        )
    if gpu > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75:
        blockers.append(
            "Insufficient free GPU memory for dictionary training with the language model resident."
        )
    if disk > hardware["experiment_disk_free_bytes"] * 0.9:
        blockers.append("Insufficient disk space for the requested dictionary artifacts.")
    return {
        "compatible": not blockers,
        "blockers": blockers,
        "input_dim": input_dim,
        "features": config.features,
        "layers": layers,
        "estimated_host_bytes": host,
        "estimated_gpu_bytes": gpu,
        "estimated_disk_bytes": disk,
        "weight_bytes_per_dictionary": weights,
        "training_device": device,
        "estimate_scope": "Additional memory with the teacher resident. Layers train sequentially. Estimates retain 25 percent of available memory as headroom.",
    }


def forward_memory(model, token_count: int, feature_count: int = 0, *, logits: bool = True) -> dict:
    config = model.config
    width = int(
        getattr(config, "hidden_size", None) or model.get_input_embeddings().weight.shape[1]
    )
    vocab = int(model.get_output_embeddings().weight.shape[0]) if logits else 0
    intermediate = (
        getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None) or 4 * width
    )
    heads = getattr(config, "num_attention_heads", 1)
    teacher = (
        token_count * (vocab * 8 + intermediate * 12 + width * 16) + heads * token_count**2 * 8
    )
    features = token_count * feature_count * 16
    return {"device_bytes": teacher + features, "host_bytes": token_count * vocab * 32 + features}
