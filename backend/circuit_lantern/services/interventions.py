"""Temporary interventions on supported attention output layouts."""

from dataclasses import dataclass

import torch
from transformers.pytorch_utils import Conv1D


@dataclass(frozen=True)
class AttentionOutput:
    projection: torch.nn.Module
    name: str
    heads: int
    head_dim: int


_LAYOUTS = {
    "GPT2Attention": ("attn", "c_proj"),
    "LlamaAttention": ("self_attn", "o_proj"),
    "Qwen2Attention": ("self_attn", "o_proj"),
    "GPTNeoXAttention": ("attention", "dense"),
}


def _attention_output(block) -> AttentionOutput:
    for class_name, (attention_name, projection_name) in _LAYOUTS.items():
        attention = getattr(block, attention_name, None)
        if attention is None or type(attention).__name__ != class_name:
            continue
        if getattr(attention, "pruned_heads", None):
            raise ValueError("Head ablation is unavailable for pruned attention layouts.")
        projection = getattr(attention, projection_name, None)
        config = getattr(attention, "config", None)
        heads = getattr(attention, "num_heads", None) or getattr(config, "num_attention_heads", 0)
        head_dim = getattr(attention, "head_dim", None) or getattr(attention, "head_size", 0)
        width = getattr(projection, "in_features", None)
        if isinstance(projection, Conv1D):
            width = projection.weight.shape[0]
        if not heads or not head_dim or width != heads * head_dim:
            raise ValueError("The attention output layout could not be verified for head ablation.")
        return AttentionOutput(projection, f"{attention_name}.{projection_name}", heads, head_dim)
    raise ValueError("Head ablation is unavailable for this attention architecture.")


def attention_head_targets(blocks) -> list[dict]:
    targets = []
    if blocks is not None:
        for layer, block in enumerate(blocks):
            try:
                output = _attention_output(block)
            except ValueError:
                continue
            targets.append({"layer": layer, "heads": output.heads, "head_dim": output.head_dim})
    return targets


def install_head_ablation(blocks, layer: int, head: int):
    if blocks is None or not 0 <= layer < len(blocks):
        raise ValueError("The selected attention layer is outside this model.")
    output = _attention_output(blocks[layer])
    if not 0 <= head < output.heads:
        raise ValueError(f"Choose an attention head between 0 and {output.heads - 1}.")

    def zero_head(module, args, kwargs):
        key = next((name for name in ("input", "x", "hidden_states") if name in kwargs), None)
        value = args[0] if args else kwargs.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or value.shape[-1] != output.heads * output.head_dim
        ):
            raise ValueError(
                "The captured attention output does not match the verified head layout."
            )
        changed = value.clone()
        changed[..., head * output.head_dim : (head + 1) * output.head_dim] = 0
        if args:
            return (changed, *args[1:]), kwargs
        return args, {**kwargs, key: changed}

    handle = output.projection.register_forward_pre_hook(zero_head, with_kwargs=True)
    return handle, {
        "layer": layer,
        "head": head,
        "heads": output.heads,
        "head_dim": output.head_dim,
        "projection": output.name,
        "scope": "all_prompt_and_generated_tokens",
        "method": "Zero the selected concatenated head output before the attention output projection.",
    }
