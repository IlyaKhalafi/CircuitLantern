import torch


@torch.inference_mode()
def attention_summary(tensor: torch.Tensor, query_position: int = -1) -> dict:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or any(size == 0 for size in tensor.shape):
        raise ValueError(
            "Attention capture requires one example with nonempty head and token axes."
        )
    if query_position < -1 or query_position >= tensor.shape[2]:
        raise ValueError(
            "Choose an attention query position inside the prompt or use -1 for its final token."
        )
    query = tensor.shape[2] - 1 if query_position == -1 else query_position
    values = tensor[0].float()
    if not torch.isfinite(values).all():
        raise ValueError("The captured attention contains non finite values.")
    entropy = -(values * values.clamp_min(1e-30).log()).sum(-1)
    return {
        "pattern": values.mean(0).cpu().tolist(),
        "head_entropy": entropy.mean(-1).cpu().tolist(),
        "heads": values.shape[0],
        "query_position": query,
        "query_by_head": values[:, query, :].cpu().tolist(),
        "query_head_entropy": entropy[:, query].cpu().tolist(),
    }
