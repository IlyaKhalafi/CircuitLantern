"""Portable activation measurements without retaining captured model tensors."""

import torch

from circuit_lantern.services.attention import attention_summary
from circuit_lantern.services.geometry import ResidualGeometry


@torch.inference_mode()
def summarize_trace(
    outputs,
    tokenizer,
    final_norm,
    unembed,
    *,
    raw_final: list[torch.Tensor],
    bypassed_hidden: list[torch.Tensor],
    ablate_layer: int | None,
    attention_query_position: int = -1,
) -> dict:
    """Materialize prompt diagnostics so their tensors can be released before generation."""
    if not torch.isfinite(outputs.logits).all():
        raise ValueError("The model produced non finite logits. Try BF16 or FP32 execution.")
    layers = []
    geometry = ResidualGeometry()
    for index, hidden in enumerate(outputs.hidden_states):
        # Some Transformers recorders run inside forward before our bypass hook.
        # Keep their final normalized state, but replace a bypassed block capture.
        is_final = index == len(outputs.hidden_states) - 1
        if bypassed_hidden and index == ablate_layer + 1 and not is_final:
            hidden = bypassed_hidden[0]
        # The final hidden state is already normalized by Transformers.
        projected = (
            final_norm(hidden)
            if final_norm is not None and index < len(outputs.hidden_states) - 1
            else hidden
        )
        logits = unembed(projected).float()
        if not torch.isfinite(logits).all():
            raise ValueError(
                "The logit lens produced non finite values. Try BF16 or FP32 execution."
            )
        probabilities = torch.softmax(logits[0, -1], dim=-1)
        values, indices = probabilities.topk(min(5, probabilities.numel()))
        raw_hidden = raw_final[0] if is_final and raw_final else hidden
        vector = raw_hidden[0].float()
        label = "Embedding" if index == 0 else f"Layer {index - 1}"
        update = geometry.observe(vector, label)
        layers.append(
            {
                "layer": index - 1,
                "label": label,
                **update,
                "capture_point": "residual_post"
                if not is_final or raw_final
                else "final_normalized",
                "entropy": float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()),
                "predictions": [
                    {
                        "token": tokenizer.decode([int(i)]),
                        "probability": float(v),
                        "token_id": int(i),
                    }
                    for i, v in zip(indices, values)
                ],
                "token_predictions": [
                    tokenizer.decode([i]) for i in logits[0].argmax(-1).cpu().tolist()
                ],
            }
        )
    attention = []
    if outputs.attentions is not None:
        for index, tensor in enumerate(outputs.attentions):
            if tensor is None:
                continue
            attention.append(
                {"layer": index, **attention_summary(tensor, attention_query_position)}
            )
    final_probabilities = torch.softmax(outputs.logits[0, -1].float(), -1)
    top_values, top_indices = final_probabilities.topk(min(10, final_probabilities.numel()))
    return {
        "layers": layers,
        "attention": attention,
        "geometry": geometry.finish(),
        "predictions": [
            {
                "token": tokenizer.decode([int(i)]),
                "probability": float(v),
                "token_id": int(i),
                "logit": float(outputs.logits[0, -1, i]),
            }
            for i, v in zip(top_indices, top_values)
        ],
        "method": f"Logit lens {'with the model final normalization' if final_norm is not None else 'without final normalization for this architecture'}. Intermediate predictions are diagnostic projections, not calibrated probabilities.",
        "final_state_normalized": not bool(raw_final),
    }
