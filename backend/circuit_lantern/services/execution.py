import importlib.metadata
import logging
import threading
from functools import lru_cache

import torch

FLOAT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
PACKED_PRECISIONS = {"int4", "int8"}


class _CastNoticeFilter(logging.Filter):
    """Keep the first expected activation cast notice for each input dtype."""

    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()
        self.lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg != "MatMul8bitLt: inputs will be cast from %s to float16 during quantization":
            return True
        notice = record.getMessage()
        with self.lock:
            if notice in self.seen:
                return False
            self.seen.add(notice)
            return True


@lru_cache(maxsize=1)
def _configure_quantization_logging():
    logging.getLogger("bitsandbytes.autograd._functions").addFilter(_CastNoticeFilter())


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


def native_eight_bit_error(metadata: dict) -> str | None:
    if checkpoint_precision(metadata) == "int8" and (metadata.get("quantization") or {}).get(
        "llm_int8_has_fp16_weight"
    ):
        return (
            "8-bit checkpoints with floating point master weights are not supported for packed "
            "weight inspection. Use an inference checkpoint with llm_int8_has_fp16_weight disabled."
        )
    return None


def four_bit_compute_dtype(metadata: dict) -> str | None:
    value = (metadata.get("quantization") or {}).get("bnb_4bit_compute_dtype")
    if value is None:
        return "float32"
    return next(
        (name for name, dtype in FLOAT_DTYPES.items() if value == name or value is dtype),
        None,
    )


def native_four_bit_error(
    metadata: dict, device: str, capability: float | None = None
) -> str | None:
    if checkpoint_precision(metadata) != "int4":
        return None
    config = metadata.get("quantization") or {}
    if config.get("bnb_4bit_quant_type", "fp4") not in ("nf4", "fp4"):
        return "This bitsandbytes 4-bit data type does not have a supported matrix adapter."
    compute = four_bit_compute_dtype(metadata)
    if compute is None:
        return "The checkpoint uses an unsupported bitsandbytes compute dtype. FP32, FP16 and BF16 compute are supported."
    if device == "cuda" and compute == "bfloat16" and capability is not None and capability < 8:
        return "The checkpoint uses BF16 compute in its bitsandbytes layers. Native BF16 compute requires an Ampere or newer CUDA GPU."
    return None


def execution_precision(metadata: dict, requested: str) -> str:
    native = checkpoint_precision(metadata)
    return (native or "float32") if requested == "auto" else requested


def prepare_model_execution(model, precision: str, device: str):
    if precision == "int4" and device == "cpu":
        from bitsandbytes.nn import Linear4bit

        # Keep canonical quantized storage for float32 residuals and matrix inspection.
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
    _configure_quantization_logging()
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
                        64,
                        64,
                        bias=False,
                        compute_dtype=torch.float32,
                        quant_type="nf4",
                        device="meta",
                    )
                    layer.weight = nn.Params4bit(
                        torch.linspace(-1, 1, 4096).reshape(64, 64),
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                        module=layer,
                    )
                else:
                    layer = nn.Linear8bitLt(
                        64, 64, bias=False, has_fp16_weights=False, device="meta"
                    )
                    layer.weight = nn.Int8Params(
                        torch.linspace(-1, 1, 4096).reshape(64, 64),
                        requires_grad=False,
                        has_fp16_weights=False,
                    )
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
    error = native_eight_bit_error(metadata)
    if error:
        raise ValueError(error)
    if native == "int4":
        capability = None
        if (
            device == "cuda"
            and four_bit_compute_dtype(metadata) == "bfloat16"
            and torch.cuda.is_available()
        ):
            capability = torch.cuda.get_device_capability(0)[0]
        error = native_four_bit_error(metadata, device, capability)
        if error:
            raise ValueError(error)
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


def execution_provenance(metadata: dict) -> dict:
    return {
        "model_id": metadata["id"],
        "revision": metadata["revision"],
        "execution_id": metadata.get("execution_id"),
        "execution_precision": metadata.get("execution_precision", "float32"),
        "execution_config": metadata.get("execution_config"),
    }
