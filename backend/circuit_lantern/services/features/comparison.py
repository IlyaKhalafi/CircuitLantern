"""Descriptive comparisons of two baseline SAE activation streams."""

import math

import torch


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(left.norm() * right.norm())
    if not denominator or not math.isfinite(denominator):
        return None
    value = float(left @ right) / denominator
    return min(1.0, max(-1.0, value))


@torch.inference_mode()
def feature_pair_statistics(activations: torch.Tensor, directions: torch.Tensor) -> dict:
    if (
        activations.ndim != 2
        or activations.shape[1] != 2
        or directions.ndim != 2
        or directions.shape[1] != 2
        or not directions.shape[0]
        or not torch.isfinite(activations).all()
        or not torch.isfinite(directions).all()
    ):
        raise ValueError("Feature comparisons require finite two-column matrices.")
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    decoder = directions.detach().to(device="cpu", dtype=torch.float64)
    count = len(values)
    active = values > 0
    intersection = int(active.all(dim=1).sum())
    union = int(active.any(dim=1).sum())
    left, right = values.unbind(dim=1)
    return {
        "eligible_position_count": count,
        "active_counts": [int(value) for value in active.sum(dim=0)],
        "coactive_count": intersection,
        "active_union_count": union,
        "activation_overlap": intersection / union if union else None,
        "activation_cosine": _cosine(left, right) if count else None,
        "pearson_correlation": _cosine(left - left.mean(), right - right.mean())
        if count > 1
        else None,
        "decoder_cosine": _cosine(*decoder.unbind(dim=1)),
        "scope": (
            "Baseline normalized SAE coefficients across eligible positions in this prompt. "
            "Active means coefficient greater than zero. Overlap is active intersection "
            "divided by active union and is unavailable for an empty union. Pearson "
            "correlation is unavailable for fewer than two positions or constant activity. "
            "Cosines are unavailable for zero norm vectors. All reductions use float64. "
            "These descriptive statistics do not establish semantic equivalence, causal "
            "influence or statistical significance."
        ),
    }
