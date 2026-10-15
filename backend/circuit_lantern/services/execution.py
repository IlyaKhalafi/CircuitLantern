import importlib.metadata
from functools import lru_cache

import torch

FLOAT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
PACKED_PRECISIONS = {"int4", "int8"}


def checkpoint_precision(metadata: dict) -> str | None:
    config = metadata.get("quantization") or {}
    if not config:
        return None
    method = config.get("quant_method", "")
    method = str(getattr(method, "value", method)).lower()
    if method == "bitsandbytes":
        if config.get("load_in_4bit") or config.get("_load_in_4bit"):
            return "int4"
        if config.get("load_in_8bit") or config.get("_load_in_8bit"):
            return "int8"
    text = str(config).lower()
    if "nvfp4" in text:
        return "nvfp4"
    if "fp8" in text:
        return "fp8"
    return "unsupported"


def execution_precision(metadata: dict, requested: str) -> str:
    native = checkpoint_precision(metadata)
    return (native or "float32") if requested == "auto" else requested


def prepare_model_execution(model, precision: str, device: str):
    if precision == "int4" and device == "cpu":
        from bitsandbytes.nn import Linear4bit

        # Keep canonical NF4 storage for float32 execution and matrix inspection.
        # The AVX512 BF16 repacker mutates quantization state and rejects some shapes.
        for module in model.modules():
            if isinstance(module, Linear4bit):
                module.support_avx512bf16_for_cpu = False


@lru_cache(maxsize=1)
def quantization_capabilities() -> dict:
    result = {"version": None, "cpu": {}, "cuda": {}}
    try:
        result["version"] = importlib.metadata.version("bitsandbytes")
        from bitsandbytes import functional, nn
    except (ImportError, OSError, RuntimeError) as error:
        result["reason"] = "Install a compatible bitsandbytes build to use NF4 or 8-bit execution."
        result["detail"] = str(error)[:180]
        return result
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for precision in ("int4", "int8"):
            try:
                sample = torch.linspace(-1, 1, 128, device=device).reshape(4, 32)
                if precision == "int4":
                    packed, state = functional.quantize_4bit(
                        sample, quant_type="nf4", compress_statistics=True
                    )
                    restored = functional.dequantize_4bit(packed, state)
                else:
                    packed, scales, _ = functional.int8_vectorwise_quant(sample.half())
                    restored = functional.int8_vectorwise_dequant(packed, scales)
                if precision == "int4":
                    layer = nn.Linear4bit(
                        64, 64, bias=False, compute_dtype=torch.float32, quant_type="nf4"
                    )
                else:
                    layer = nn.Linear8bitLt(64, 64, bias=False, has_fp16_weights=False)
                layer.weight.data = torch.linspace(-1, 1, 4096).reshape(64, 64)
                layer = layer.to(device).eval()
                prepare_model_execution(layer, precision, device)
                with torch.inference_mode():
                    output = layer(torch.ones(1, 64, device=device))
                result[device][precision] = {
                    "supported": bool(
                        torch.isfinite(restored).all() and torch.isfinite(output).all()
                    )
                }
            except (
                OSError,
                RuntimeError,
                ValueError,
                NotImplementedError,
                AssertionError,
            ) as error:
                result[device][precision] = {"supported": False, "reason": str(error)[:180]}
    return result


def loading_options(metadata: dict, requested: str, device: str) -> tuple[dict, str]:
    precision = execution_precision(metadata, requested)
    native = checkpoint_precision(metadata)
    if native and requested not in ("auto", native):
        raise ValueError(
            "This checkpoint is already quantized. Select Auto or its existing execution format."
        )
    if precision not in FLOAT_DTYPES and precision not in PACKED_PRECISIONS:
        raise ValueError("This checkpoint format is not supported by the analysis runtime.")
    options = {"dtype": FLOAT_DTYPES.get(precision, torch.float32)}
    if precision in PACKED_PRECISIONS:
        from transformers import BitsAndBytesConfig

        if not quantization_capabilities().get(device, {}).get(precision, {}).get("supported"):
            raise ValueError(
                "The selected bitsandbytes execution path is unavailable on this device."
            )
        options["device_map"] = {"": device}
        if native is None:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=precision == "int4",
                load_in_8bit=precision == "int8",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32,
            )
    return options, precision


def describe_execution(model, precision: str, device: str) -> dict:
    config = getattr(model.config, "quantization_config", None)
    quantization = config.to_dict() if hasattr(config, "to_dict") else config
    return {
        "precision": precision,
        "device": device,
        "residual_dtype": str(model.dtype).removeprefix("torch."),
        "quantization": quantization,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes")
        if precision in PACKED_PRECISIONS
        else None,
    }
