"""Portable token predictions from full vocabulary diagnostic distributions."""

import torch

from circuit_lantern.limits import MAX_TRACKED_TARGETS
from circuit_lantern.services.scoring import log_probabilities

MAX_LOGIT_ELEMENTS = 1_048_576


def _summarize_chunk(logits: torch.Tensor, target_ids: tuple[int, ...]) -> dict:
    log_probs = log_probabilities(logits)
    probabilities = log_probs.exp()
    ids = logits.argmax(-1)
    top = probabilities.gather(-1, ids[:, None]).flatten()
    entropy = torch.special.entr(probabilities).sum(-1)
    return {
        "token_prediction_ids": ids.cpu().tolist(),
        "token_prediction_probabilities": top.cpu().tolist(),
        "token_entropies": entropy.cpu().tolist(),
        **(
            {
                "token_target_log_probabilities": log_probs[:, target_ids].cpu().tolist(),
                # Reduce one target at a time to bound the vocabulary workspace.
                "token_target_ranks": torch.stack(
                    [(logits > logits[:, target_id, None]).sum(-1) + 1 for target_id in target_ids],
                    dim=-1,
                )
                .cpu()
                .tolist(),
            }
            if target_ids
            else {}
        ),
    }


@torch.inference_mode()
def summarize_token_logits(logits: torch.Tensor, target_ids: tuple[int, ...] = ()) -> dict:
    """Keep full vocabulary statistics while bounding float64 workspace by token chunks."""
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]:
        raise ValueError("Token predictions require a nonempty token by vocabulary matrix.")
    if (
        len(target_ids) > MAX_TRACKED_TARGETS
        or len(set(target_ids)) != len(target_ids)
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < logits.shape[1]
            for token_id in target_ids
        )
    ):
        raise ValueError(
            "Tracked target IDs must be distinct vocabulary IDs within the capture limit."
        )
    summary = {
        "token_prediction_ids": [],
        "token_prediction_probabilities": [],
        "token_entropies": [],
        **({"token_target_log_probabilities": [], "token_target_ranks": []} if target_ids else {}),
    }
    step = max(1, MAX_LOGIT_ELEMENTS // logits.shape[1])
    for start in range(0, logits.shape[0], step):
        # Chunk tensors leave scope before the next chunk is materialized.
        chunk = _summarize_chunk(logits[start : start + step], target_ids)
        for key, values in chunk.items():
            summary[key].extend(values)
    return summary
