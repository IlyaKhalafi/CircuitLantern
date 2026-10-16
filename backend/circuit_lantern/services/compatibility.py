from circuit_lantern.schemas import CompatibilityRequest
from circuit_lantern.services.execution import (
    FLOAT_DTYPES,
    PACKED_PRECISIONS,
    checkpoint_precision,
    execution_precision,
    native_eight_bit_error,
    native_four_bit_error,
)

GIB = 1024**3
PRECISION_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1.1,
    "int4": 0.625,
    "fp8": 1,
    "nvfp4": 0.625,
}


def assess_compatibility(
    metadata: dict, hardware: dict, request: CompatibilityRequest, cache: dict | None = None
) -> dict:
    blockers, warnings = [], []
    native = checkpoint_precision(metadata)
    precision = execution_precision(metadata, request.precision)
    parameters = metadata.get("parameters") or 0
    hidden = metadata.get("hidden_size") or 0
    layers = metadata.get("num_layers") or 0
    heads = metadata.get("num_heads") or 1
    kv_heads = metadata.get("num_kv_heads") or heads
    head_dim = metadata.get("head_dim") or hidden // max(heads, 1)
    if not parameters or not hidden or not layers:
        blockers.append(
            "The logical model dimensions could not be determined. Memory fit cannot be verified."
        )
    if metadata.get("architecture_supported") is False:
        blockers.append(
            "This architecture cannot be initialized by the installed Transformers runtime. "
            + (metadata.get("architecture_error") or "Choose a supported causal language model.")
        )
    if native and request.precision not in ("auto", native):
        blockers.append(
            "This checkpoint is already quantized. Select Auto or its existing execution format. Use the original checkpoint to choose another precision."
        )
    master_weight_error = native_eight_bit_error(metadata)
    if master_weight_error:
        blockers.append(master_weight_error)
    if precision in PACKED_PRECISIONS:
        supported = hardware.get("quantization", {}).get(request.device, {}).get(precision, {})
        if not supported.get("supported"):
            blockers.append(
                "The selected bitsandbytes quantization path is unavailable on this device. Install the quantization dependencies and a matching PyTorch runtime."
            )
        quantizable = metadata.get("quantizable_parameters")
        if quantizable is None:
            blockers.append(
                "The quantizable weight layout could not be verified for this checkpoint."
            )
            quantizable = 0
        quantizable = min(parameters, quantizable)
        weight_bytes = (parameters - quantizable) * 4 + quantizable * PRECISION_BYTES[precision]
        runtime_precision = (
            "NF4 / FP32 compute" if precision == "int4" else "LLM.int8 / FP32 residuals"
        )
        if native == "int4":
            kind = (metadata.get("quantization") or {}).get("bnb_4bit_quant_type", "fp4")
            runtime_precision = (
                kind.upper() if isinstance(kind, str) else "Unsupported 4-bit"
            ) + " / FP32 residuals"
        warnings.append(
            "Packed weights remain quantized for execution. Matrix analysis dequantizes one selected matrix to float32. Embeddings, output heads and skipped modules remain unquantized."
        )
        if precision == "int8":
            warnings.append(
                "LLM.int8 casts linear inputs to float16 internally. Residual states and matrix statistics remain float32."
            )
        if request.device == "cpu":
            warnings.append(
                "CPU quantization reduces resident weight storage. Fallback kernels can dequantize a matrix during a forward pass, so reduced memory does not guarantee faster inference."
            )
    else:
        weight_bytes = parameters * PRECISION_BYTES.get(precision, 4)
        runtime_precision = precision
        if precision not in FLOAT_DTYPES:
            method = (metadata.get("quantization") or {}).get("quant_method", precision)
            blockers.append(
                f"Packed weight inspection for {method} is not supported by the current runtime. Select a supported checkpoint format."
            )
    gpus = hardware.get("gpus") or []
    execution_gpu = hardware.get("execution_gpu") or (gpus[0] if gpus else None)
    capability = execution_gpu["compute_capability"] if execution_gpu else 0
    compute_error = native_four_bit_error(metadata, request.device, capability)
    if compute_error:
        blockers.append(compute_error)
    if request.device == "cuda" and not hardware["cuda_available"]:
        blockers.append(
            "The Python runtime has no usable CUDA device. Install a CUDA enabled PyTorch build or use CPU."
        )
    if precision == "nvfp4" and (request.device != "cuda" or capability < 10):
        blockers.append(
            "Native NVFP4 requires a Blackwell GPU with compute capability 10.0 or newer. Hopper H100 and H200 cannot execute native NVFP4 kernels."
        )
    if precision == "fp8" and (request.device != "cuda" or capability < 9):
        blockers.append(
            "This FP8 execution path requires a Hopper or newer CUDA GPU with compute capability 9.0 or newer."
        )
    if request.device == "cuda":
        if precision == "bfloat16" and capability < 8:
            blockers.append("Native BF16 requires an Ampere or newer GPU in this runtime.")
        if precision == "int8" and capability < 7.5:
            blockers.append("This CUDA LLM.int8 path requires a Turing or newer NVIDIA GPU.")
        if precision == "int4" and capability < 6:
            blockers.append(
                "CUDA NF4 and bitsandbytes FP4 require a Pascal or newer NVIDIA GPU. This format is different from native NVFP4."
            )
    element_bytes = PRECISION_BYTES.get(precision, 4) if precision in FLOAT_DTYPES else 4
    kv_bytes = (
        2
        * request.batch_size
        * request.context_length
        * layers
        * kv_heads
        * head_dim
        * element_bytes
    )
    residual_bytes = (layers + 1) * request.batch_size * request.context_length * hidden * 4
    capture_tokens = min(request.context_length, hardware.get("max_prompt_tokens", 128))
    attention_bytes = request.batch_size * layers * heads * capture_tokens**2 * 4
    config = metadata.get("config") or {}
    intermediate = config.get("intermediate_size") or config.get("n_inner") or hidden * 4
    activation_bytes = 3 * request.batch_size * capture_tokens * intermediate * 4
    logits_bytes = 5 * request.batch_size * capture_tokens * (metadata.get("vocab_size") or 0) * 4
    workspace_bytes = (
        (
            metadata.get("largest_matrix_elements")
            or hidden * max(hidden * 4, metadata.get("vocab_size") or 0)
        )
        * 4
        * (1 if precision == "float32" else 2)
    )
    overhead_bytes = (
        max(256 * 1024**2, weight_bytes * 0.15)
        + residual_bytes
        + attention_bytes
        + activation_bytes
        + logits_bytes
    )
    required_bytes = weight_bytes + kv_bytes + overhead_bytes + workspace_bytes
    native_bytes = PRECISION_BYTES.get(metadata.get("dtype"), 4)
    disk_bytes = (
        metadata.get("download_bytes") or metadata.get("weight_bytes") or parameters * native_bytes
    )
    download_bytes = cache["download_bytes"] if cache is not None else disk_bytes
    cached_bytes = cache["cached_bytes"] if cache is not None else 0
    resident = hardware.get("reclaimable_tensor_bytes") or {}
    # Host available RAM may already include reclaimable mmap checkpoint pages.
    # Only CUDA storage can be credited without counting those pages twice.
    reclaimable = 0
    available_bytes = hardware["ram_available_bytes"]
    if request.device == "cuda":
        reclaimable = resident.get("cuda:0", 0) if execution_gpu else 0
        available_bytes = (
            min(
                execution_gpu["free_bytes"] + reclaimable,
                execution_gpu.get("total_bytes", execution_gpu["free_bytes"] + reclaimable),
            )
            if execution_gpu
            else 0
        )
        staging_bytes = parameters * (4 if precision in PACKED_PRECISIONS else element_bytes)
        if staging_bytes * 1.15 > hardware["ram_available_bytes"] * 0.85:
            blockers.append(
                "Insufficient host RAM for the conservative checkpoint loading and analysis staging budget."
            )
    if required_bytes > available_bytes * 0.85:
        blockers.append(
            "Insufficient available memory after reserving 15 percent for the operating system and other workloads."
        )
    if download_bytes * 1.1 > hardware["disk_free_bytes"]:
        blockers.append("Insufficient free disk space for the checkpoint and download overhead.")
    if disk_bytes > hardware.get("max_model_bytes", 20 * GIB):
        blockers.append(
            "This checkpoint exceeds the configured download limit. Adjust LANTERN_MAX_MODEL_BYTES on the server."
        )
    if not metadata.get("has_safetensors"):
        blockers.append(
            "No supported Transformers safetensors checkpoint was found. A model.safetensors file or shard index is required."
        )
    context_limit = metadata.get("context_length")
    if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit < 1:
        blockers.append(
            "The model context window could not be determined. Context fit cannot be verified."
        )
    elif request.context_length > context_limit:
        blockers.append(
            f"Requested context exceeds this model's {metadata['context_length']:,} token context window."
        )
    warnings.append(
        "Matrix values use float32. Small full spectra use float64 arithmetic. The budget includes execution weights, logit and MLP buffers, a KV cache allowance, capture overhead and temporary matrix analysis space."
    )
    warnings.append(
        "Fit is an estimate for one visible execution device. Other running jobs can change available memory before loading."
    )
    if reclaimable:
        warnings.append(
            "The replacement budget includes only CUDA parameter and buffer storage owned by the active model. These tensors are released before loading the replacement. Other workloads are not credited."
        )
    if hardware.get("ram_scope") == "cgroup":
        warnings.append(
            "System memory reflects the process or container memory limit, current usage and reclaimable inactive file cache."
        )
    if metadata.get("parameter_estimated"):
        warnings.append(
            "Logical parameter count is derived from architecture configuration rather than packed tensor element counts. Checkpoint variants may differ."
        )
    if request.context_length > hardware.get("max_prompt_tokens", 128):
        warnings.append(
            f"Interactive captures are limited to {hardware.get('max_prompt_tokens', 128)} prompt tokens."
        )
    return {
        "compatible": not blockers,
        "status": "ready" if not blockers else "blocked",
        "precision": precision,
        "context_length": request.context_length,
        "runtime_precision": runtime_precision,
        "device": request.device,
        "weight_bytes": int(weight_bytes),
        "analysis_weight_bytes": int(weight_bytes),
        "analysis_workspace_bytes": int(workspace_bytes),
        "activation_bytes": int(activation_bytes),
        "logits_bytes": int(logits_bytes),
        "kv_cache_bytes": int(kv_bytes),
        "overhead_bytes": int(overhead_bytes),
        "required_bytes": int(required_bytes),
        "available_bytes": int(available_bytes * 0.85),
        "reclaimable_bytes": int(reclaimable),
        "source_bytes": int(disk_bytes),
        "cached_bytes": int(cached_bytes),
        "download_bytes": int(download_bytes),
        "blockers": blockers,
        "warnings": warnings,
        "metadata": metadata,
    }
