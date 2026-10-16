import torch


@torch.inference_mode()
def attention_summary(tensor: torch.Tensor) -> dict:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or any(size == 0 for size in tensor.shape):
        raise ValueError(
            "Attention capture requires one example with nonempty head and token axes."
        )
    values = tensor[0].float()
    if not torch.isfinite(values).all():
        raise ValueError("The captured attention contains non finite values.")
    entropy = -(values * values.clamp_min(1e-30).log()).sum(-1)
    return {
        "pattern": values.mean(0).cpu().tolist(),
        "head_entropy": entropy.mean(-1).cpu().tolist(),
        "heads": values.shape[0],
        "query_position": values.shape[1] - 1,
        "query_by_head": values[:, -1, :].cpu().tolist(),
        "query_head_entropy": entropy[:, -1].cpu().tolist(),
    }
