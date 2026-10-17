"""Vocabulary resolution shared by tracked captures and read-only previews."""

from circuit_lantern.limits import (
    MAX_TARGET_CHARACTERS,
    MAX_TARGET_PREVIEW_PIECES,
    MAX_TRACKED_TARGETS,
)
from circuit_lantern.services.tokens import special_token_ids


def _target_problem(ids: list[int], vocabulary_size: int) -> str | None:
    if len(ids) != 1:
        return "The target must encode to exactly one token. Include a leading space when needed."
    if not 0 <= ids[0] < vocabulary_size:
        return "The target token is outside this model vocabulary."
    return None


def single_target_id(tokenizer, target: str, vocabulary_size: int) -> int:
    ids = tokenizer.encode(target, add_special_tokens=False)
    problem = _target_problem(ids, vocabulary_size)
    if problem:
        raise ValueError(problem)
    return ids[0]


def _inspect_targets(
    tokenizer, targets: list[str], vocabulary_size: int, *, allow_empty: bool
) -> list[dict]:
    if len(targets) > MAX_TRACKED_TARGETS:
        raise ValueError(f"Track at most {MAX_TRACKED_TARGETS} target tokens per capture.")
    for text in targets:
        if (
            not isinstance(text, str)
            or not (0 if allow_empty else 1) <= len(text) <= MAX_TARGET_CHARACTERS
        ):
            raise ValueError(
                f"Each tracked target must contain between {0 if allow_empty else 1} and {MAX_TARGET_CHARACTERS} characters."
            )
    special = special_token_ids(tokenizer)
    previews = []
    seen = {}
    for index, text in enumerate(targets):
        ids = tokenizer.encode(text, add_special_tokens=False) if text else []
        problem = _target_problem(ids, vocabulary_size) if text else None
        duplicate = None
        if text and problem is None:
            duplicate = seen.get(ids[0])
            if duplicate is not None:
                problem = (
                    "Tracked targets must resolve to distinct vocabulary IDs. "
                    f"This ID is already used by target {duplicate + 1}."
                )
            else:
                seen[ids[0]] = index
        shown = ids[:MAX_TARGET_PREVIEW_PIECES]
        previews.append(
            {
                "requested_text": text,
                "token_count": len(ids),
                "token_ids": shown,
                "tokens": [tokenizer.decode([token]) for token in shown],
                "pieces": tokenizer.convert_ids_to_tokens(shown),
                "special_tokens": [token in special for token in shown],
                "truncated": len(shown) != len(ids),
                "skipped": not bool(text),
                "valid": problem is None,
                "error": problem,
                "duplicate_of": duplicate,
            }
        )
    return previews


def preview_target_tokens(tokenizer, targets: list[str], vocabulary_size: int) -> list[dict]:
    return _inspect_targets(tokenizer, targets, vocabulary_size, allow_empty=True)


def resolve_tracked_targets(tokenizer, targets: list[str], vocabulary_size: int) -> list[dict]:
    previews = _inspect_targets(tokenizer, targets, vocabulary_size, allow_empty=False)
    for target in previews:
        if not target["valid"]:
            raise ValueError(target["error"])
    return [
        {
            "requested_text": target["requested_text"],
            "token_id": target["token_ids"][0],
            "token": target["tokens"][0],
        }
        for target in previews
    ]
