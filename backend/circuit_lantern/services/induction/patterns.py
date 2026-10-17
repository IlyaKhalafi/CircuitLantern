"""Attention evidence for repeated sequences without assigning causal head types.

The prefix mask follows the A B ... A to B definition in
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
A matching key has a preceding token equal to the current query token.
"""

from collections.abc import Sequence

import torch


def repeated_attention_masks(token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    if (
        not isinstance(token_ids, torch.Tensor)
        or token_ids.device.type != "cpu"
        or token_ids.dtype != torch.int64
        or token_ids.ndim != 1
        or len(token_ids) < 2
        or (token_ids < 0).any()
    ):
        raise ValueError(
            "Pattern measurements require at least two nonnegative int64 CPU token IDs."
        )
    count = len(token_ids)
    positions = torch.arange(count, dtype=torch.int64, device="cpu")
    prefix = torch.zeros((count, count), dtype=torch.bool, device="cpu")
    prefix[:, 1:] = token_ids[:, None] == token_ids[:-1][None, :]
    return {
        "prefix": prefix & (positions[None, :] <= positions[:, None]),
        "duplicate": (token_ids[:, None] == token_ids[None, :])
        & (positions[None, :] < positions[:, None]),
        "previous": positions[None, :] == positions[:, None] - 1,
    }


@torch.inference_mode()
def score_repeated_attention(
    attention: torch.Tensor,
    token_ids: torch.Tensor,
    query_positions: Sequence[int],
) -> dict:
    masks = repeated_attention_masks(token_ids)
    count = len(token_ids)
    queries = tuple(query_positions)
    if (
        not queries
        or any(type(position) is not int or not 0 <= position < count for position in queries)
        or len(set(queries)) != len(queries)
    ):
        raise ValueError("Choose distinct query positions inside the repeated prompt.")
    if not masks["prefix"][list(queries)].any(-1).all():
        raise ValueError("Each measured query must have an earlier matching prefix.")
    if (
        not isinstance(attention, torch.Tensor)
        or attention.device.type == "meta"
        or attention.ndim != 3
        or attention.shape[0] < 1
        or attention.shape[1:] != (count, count)
        or attention.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        raise ValueError("Attention must contain heads by query by key probabilities.")
    values = attention.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention probabilities must be finite and nonnegative.")
    if values.triu(diagonal=1).count_nonzero():
        raise ValueError("Repeated token probes require causal attention without future keys.")
    totals = values.sum(-1, keepdim=True)
    tolerance = max(1e-5, 2 * torch.finfo(attention.dtype).eps)
    if not torch.allclose(totals, torch.ones_like(totals), atol=tolerance, rtol=0):
        raise ValueError(
            "Each captured attention row must sum to one within its storage precision."
        )
    # Normalize only after validating the dtype-specific rounding bound.
    values = values / totals
    measured = values[:, list(queries)]
    scores = {
        name: (measured * mask[list(queries)]).sum(-1).mean(-1).tolist()
        for name, mask in masks.items()
    }
    uniform = (
        masks["prefix"][list(queries)].sum(-1).double()
        / (torch.tensor(queries, dtype=torch.float64, device="cpu") + 1)
    ).mean()
    return {
        "query_positions": list(queries),
        "queries": len(queries),
        "uniform_full_causal_baseline": float(uniform),
        "accumulation": "Validated probabilities normalized per row and accumulated in float64.",
        "scope": "Attention pattern evidence on repeated prompts. Prefix matching alone does not establish copying or causal importance.",
        "heads": [
            {
                "head": head,
                "prefix_attention": scores["prefix"][head],
                "duplicate_attention": scores["duplicate"][head],
                "previous_attention": scores["previous"][head],
            }
            for head in range(attention.shape[0])
        ],
    }
