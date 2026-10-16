"""Portable token predictions from full vocabulary diagnostic distributions."""

import torch

from circuit_lantern.services.scoring import log_probabilities

MAX_LOGIT_ELEMENTS = 1_048_576


def _summarize_chunk(logits: torch.Tensor) -> tuple[list[int], list[float], list[float]]:
    log_probs = log_probabilities(logits)
    probabilities = log_probs.exp()
    ids = logits.argmax(-1)
    top = probabilities.gather(-1, ids[:, None]).flatten()
    entropy = torch.special.entr(probabilities).sum(-1)
    return ids.cpu().tolist(), top.cpu().tolist(), entropy.cpu().tolist()


@torch.inference_mode()
def summarize_token_logits(logits: torch.Tensor) -> dict:
    """Keep full vocabulary statistics while bounding float64 workspace by token chunks."""
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]:
        raise ValueError("Token predictions require a nonempty token by vocabulary matrix.")
    ids, probabilities, entropies = [], [], []
    step = max(1, MAX_LOGIT_ELEMENTS // logits.shape[1])
    for start in range(0, logits.shape[0], step):
        # Chunk tensors leave scope before the next chunk is materialized.
        chunk_ids, chunk_probabilities, chunk_entropies = _summarize_chunk(
            logits[start : start + step]
        )
        ids.extend(chunk_ids)
        probabilities.extend(chunk_probabilities)
        entropies.extend(chunk_entropies)
    return {
        "token_prediction_ids": ids,
        "token_prediction_probabilities": probabilities,
        "token_entropies": entropies,
    }
