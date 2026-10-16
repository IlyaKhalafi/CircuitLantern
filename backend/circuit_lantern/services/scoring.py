import torch

from circuit_lantern.services.targets import single_target_id


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise ValueError(
            "The model produced non finite logits. Choose a different checkpoint or prompt."
        )
    return logits.double().log_softmax(-1)


def target_token_id(tokenizer, target: str, baseline: torch.Tensor) -> int:
    return (
        single_target_id(tokenizer, target, baseline.numel()) if target else int(baseline.argmax())
    )


def target_score(log_probs: torch.Tensor, target_id: int) -> dict:
    return {
        "log_probability": float(log_probs[target_id]),
        "probability": float(log_probs[target_id].exp()),
    }
