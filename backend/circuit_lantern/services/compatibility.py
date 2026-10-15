from circuit_lantern.schemas import CompatibilityRequest

GIB = 1024**3
PRECISION_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.6,
    "fp8": 1,
    "nvfp4": 0.6,
}


def assess_compatibility(metadata: dict, hardware: dict, request: CompatibilityRequest) -> dict:
    blockers, warnings = [], []
    quantization = metadata.get("quantization") or {}
    quant_method = str(quantization.get("quant_method", "")).lower()
    quant_text = str(quantization).lower()
    native_precision = metadata.get("dtype") or "float32"
    precision = request.precision
    if precision == "auto":
        precision = (
            "nvfp4" if "nvfp4" in quant_text else "fp8" if "fp8" in quant_text else native_precision
        )
    bytes_per_weight = PRECISION_BYTES.get(precision, 4)
    parameters = metadata.get("parameters") or 0
    if not parameters:
        blockers.append("Parameter count could not be determined. Memory fit cannot be verified.")
    weight_bytes = parameters * bytes_per_weight
    if quantization and metadata.get("weight_bytes"):
        weight_bytes = metadata["weight_bytes"]
    kv_bytes = (
        2
        * request.batch_size
        * request.context_length
        * metadata["num_layers"]
        * metadata["num_kv_heads"]
        * (metadata["hidden_size"] // max(metadata["num_heads"], 1))
        * 4
    )
    # Inspection uses float32 weights and captures residuals and eager attention.
    inspection_weights = parameters * 4 if not quantization else weight_bytes
    residual_bytes = (
        (metadata["num_layers"] + 1)
        * request.batch_size
        * request.context_length
        * metadata["hidden_size"]
        * 4
    )
    attention_bytes = (
        metadata["num_layers"] * metadata["num_heads"] * min(request.context_length, 128) ** 2 * 4
    )
    overhead_bytes = (
        max(256 * 1024**2, inspection_weights * 0.15) + residual_bytes + attention_bytes
    )
    required_bytes = inspection_weights + kv_bytes + overhead_bytes
    disk_bytes = metadata.get("weight_bytes") or weight_bytes
    available_bytes = hardware["ram_available_bytes"]
    cuda_gpus = hardware["gpus"]
    gpu = cuda_gpus[0] if cuda_gpus else None
    capability = gpu["compute_capability"] if gpu else 0
    if request.device == "cuda":
        available_bytes = gpu["free_bytes"] if gpu else 0
        if not hardware["cuda_available"]:
            blockers.append(
                "The Python runtime has no usable CUDA device. Install a CUDA enabled PyTorch build or use CPU."
            )
    if (precision == "nvfp4" or "nvfp4" in quant_text) and (
        request.device != "cuda" or capability < 10
    ):
        blockers.append(
            "Native NVFP4 requires a Blackwell GPU with compute capability 10.0 or newer. Hopper H100 and H200 cannot execute native NVFP4 kernels."
        )
    if precision == "fp8" and (request.device != "cuda" or capability < 9):
        blockers.append(
            "This FP8 execution path requires a Hopper or newer CUDA GPU with compute capability 9.0 or newer."
        )
    if request.device == "cuda" and precision == "bfloat16" and capability < 8:
        blockers.append("Native BF16 requires an Ampere or newer GPU in this runtime.")
    if quantization:
        blockers.append(
            f"This checkpoint uses {quant_method or 'an unrecognized quantization'}. Packed weight inspection is not supported by the current float32 analysis runtime. Select the original unquantized checkpoint."
        )
    elif request.precision in ("int4", "int8", "fp8", "nvfp4"):
        blockers.append(
            "On the fly quantization is not supported by the analysis runtime. Choose auto precision or an unquantized checkpoint."
        )
    if required_bytes > available_bytes * 0.85:
        blockers.append(
            "Insufficient available memory after reserving 15 percent for the operating system and other workloads."
        )
    if disk_bytes * 1.1 > hardware["disk_free_bytes"]:
        blockers.append("Insufficient free disk space for the checkpoint and download overhead.")
    if disk_bytes > hardware.get("max_model_bytes", 20 * GIB):
        blockers.append(
            "This checkpoint exceeds the configured download limit. Adjust LANTERN_MAX_MODEL_BYTES on the server."
        )
    if not metadata.get("has_safetensors"):
        blockers.append(
            "This repository has no safetensors checkpoint. Only safetensors weights are accepted."
        )
    if request.context_length > metadata["context_length"]:
        blockers.append(
            f"Requested context exceeds this model's {metadata['context_length']:,} token context window."
        )
    warnings.append(
        "Weight inspection uses float32. Memory estimates include capture overhead and a conservative KV cache allowance."
    )
    warnings.append(
        "Fit is an estimate for one device. Other running jobs can change available memory before loading."
    )
    if metadata.get("parameter_estimated"):
        warnings.append(
            "Parameter count is estimated from model configuration. Actual memory may differ."
        )
    if request.context_length > 128:
        warnings.append(
            "Interactive traces are limited to 128 prompt tokens to keep attention capture bounded."
        )
    return {
        "compatible": not blockers,
        "status": "ready" if not blockers else "blocked",
        "precision": precision,
        "runtime_precision": "float32",
        "device": request.device,
        "weight_bytes": int(weight_bytes),
        "analysis_weight_bytes": int(inspection_weights),
        "kv_cache_bytes": int(kv_bytes),
        "overhead_bytes": int(overhead_bytes),
        "required_bytes": int(required_bytes),
        "available_bytes": int(available_bytes * 0.85),
        "download_bytes": int(disk_bytes),
        "blockers": blockers,
        "warnings": warnings,
        "metadata": metadata,
    }
