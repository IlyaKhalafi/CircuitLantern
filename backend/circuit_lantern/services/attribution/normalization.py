"""Supported final-normalization maps with the observed denominator held fixed."""

import math

import torch

# These adapters describe the implemented module, not the checkpoint architecture.
# Unknown norms are left unavailable rather than inferred from a class-name suffix.
RMS_WEIGHT_MODES = {
    "transformers.models.llama.modeling_llama.LlamaRMSNorm": "weight",
    "transformers.models.mistral.modeling_mistral.MistralRMSNorm": "weight",
    "transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm": "weight",
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm": "weight",
    "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": "weight",
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm": "weight",
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm": "one_plus_weight",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm": "one_plus_weight",
    "transformers.models.gemma.modeling_gemma.GemmaRMSNorm": "one_plus_weight",
    "transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm": "one_plus_weight",
    "transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm": "one_plus_weight",
}


class UnsupportedAttribution(ValueError):
    """A diagnostic cannot be established without guessing model semantics."""


def qualified_name(module) -> str:
    return f"{type(module).__module__}.{type(module).__name__}"


def frozen_normalization(module, final: torch.Tensor, captured_dtype: torch.dtype) -> dict:
    width = final.numel()
    name = qualified_name(module) if module is not None else "none"
    center = False
    mode = "weight"
    kind = "identity"
    epsilon = 0.0
    if module is None or type(module) is torch.nn.Identity:
        gain, bias = torch.ones_like(final), torch.zeros_like(final)
    else:
        if type(module) is torch.nn.LayerNorm:
            if tuple(module.normalized_shape) != (width,):
                raise UnsupportedAttribution("unsupported_norm")
            kind, center = "layer_norm", True
        elif type(module) is torch.nn.RMSNorm:
            if tuple(module.normalized_shape) != (width,):
                raise UnsupportedAttribution("unsupported_norm")
            kind = "rms_norm"
        elif name in RMS_WEIGHT_MODES:
            kind, mode = "rms_norm", RMS_WEIGHT_MODES[name]
        else:
            raise UnsupportedAttribution("unsupported_norm")
        epsilon = getattr(module, "eps", getattr(module, "variance_epsilon", None))
        if epsilon is None and type(module) is torch.nn.RMSNorm:
            epsilon = torch.finfo(captured_dtype).eps
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
            raise UnsupportedAttribution("unsupported_norm")
        weight = getattr(module, "weight", None)
        gain = (
            weight.detach().to(device="cpu", dtype=torch.float64).clone()
            if weight is not None
            else torch.ones_like(final)
        )
        beta = getattr(module, "bias", None)
        bias = (
            beta.detach().to(device="cpu", dtype=torch.float64).clone()
            if beta is not None
            else torch.zeros_like(final)
        )
        if gain.shape != final.shape or bias.shape != final.shape:
            raise UnsupportedAttribution("unsupported_norm")
        if mode == "one_plus_weight":
            gain = gain + 1
    centered = final - final.mean() if center else final
    denominator = (
        float(torch.sqrt(centered.square().mean() + epsilon)) if kind != "identity" else 1.0
    )
    if not math.isfinite(denominator) or denominator <= 0:
        raise UnsupportedAttribution("non_finite")
    return {
        "kind": kind,
        "module": name,
        "epsilon": float(epsilon),
        "denominator": denominator,
        "statistic_dtype": "float64",
        "weight_mode": mode,
        "has_bias": getattr(module, "bias", None) is not None,
        "center": center,
        "gain": gain,
        "bias": bias,
    }


def normalize_components(vectors: torch.Tensor, specification: dict) -> torch.Tensor:
    if specification["center"]:
        vectors = vectors - vectors.mean(dim=-1, keepdim=True)
    return vectors * (specification["gain"] / specification["denominator"])
