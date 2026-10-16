import torch

from circuit_lantern.limits import MAX_TARGET_CHARACTERS, MAX_TRACKED_TARGETS


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise ValueError(
            "The model produced non finite logits. Choose a different checkpoint or prompt."
        )
    return logits.double().log_softmax(-1)


def single_target_id(tokenizer, target: str, vocabulary_size: int) -> int:
    ids = tokenizer.encode(target, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            "The target must encode to exactly one token. Include a leading space when needed."
        )
    target_id = ids[0]
    if not 0 <= target_id < vocabulary_size:
        raise ValueError("The target token is outside this model vocabulary.")
    return target_id


def target_token_id(tokenizer, target: str, baseline: torch.Tensor) -> int:
    return (
        single_target_id(tokenizer, target, baseline.numel()) if target else int(baseline.argmax())
    )


def resolve_tracked_targets(tokenizer, targets: list[str], vocabulary_size: int) -> list[dict]:
    if len(targets) > MAX_TRACKED_TARGETS:
        raise ValueError(f"Track at most {MAX_TRACKED_TARGETS} target tokens per capture.")
    resolved = []
    seen = set()
    for text in targets:
        if not isinstance(text, str) or not 1 <= len(text) <= MAX_TARGET_CHARACTERS:
            raise ValueError(
                f"Each tracked target must contain between 1 and {MAX_TARGET_CHARACTERS} characters."
            )
        token_id = single_target_id(tokenizer, text, vocabulary_size)
        if token_id in seen:
            raise ValueError("Tracked targets must resolve to distinct vocabulary IDs.")
        seen.add(token_id)
        resolved.append(
            {"requested_text": text, "token_id": token_id, "token": tokenizer.decode([token_id])}
        )
    return resolved


def target_score(log_probs: torch.Tensor, target_id: int) -> dict:
    return {
        "log_probability": float(log_probs[target_id]),
        "probability": float(log_probs[target_id].exp()),
    }
