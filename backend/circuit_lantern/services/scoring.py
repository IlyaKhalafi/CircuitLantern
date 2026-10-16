import torch


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise ValueError(
            "The model produced non finite logits. Choose a different checkpoint or prompt."
        )
    return logits.double().log_softmax(-1)


def target_token_id(tokenizer, target: str, baseline: torch.Tensor) -> int:
    if target:
        ids = tokenizer.encode(target, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                "The target must encode to exactly one token. Include a leading space when needed, or leave it empty to use the baseline top prediction."
            )
        target_id = ids[0]
        if not 0 <= target_id < baseline.numel():
            raise ValueError("The target token is outside this model vocabulary.")
        return target_id
    return int(baseline.argmax())


def target_score(log_probs: torch.Tensor, target_id: int) -> dict:
    return {
        "log_probability": float(log_probs[target_id]),
        "probability": float(log_probs[target_id].exp()),
    }
