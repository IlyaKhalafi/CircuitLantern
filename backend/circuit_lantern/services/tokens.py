"""Token previews and the shared bounds for interactive captures."""

from circuit_lantern.services.execution import execution_provenance

MAX_PREVIEW_TOKENS = 256


def capture_token_limit(metadata: dict, max_prompt_tokens: int, reserved_tokens: int = 0) -> int:
    context = min(
        metadata["context_length"],
        metadata.get("execution_context_length", metadata["context_length"]),
    )
    return max(0, min(max_prompt_tokens, context - reserved_tokens))


def preview_tokens(
    tokenizer, metadata: dict, prompts: list[str], max_prompt_tokens: int, reserved_tokens: int = 0
) -> dict:
    limit = capture_token_limit(metadata, max_prompt_tokens, reserved_tokens)
    special = set(tokenizer.all_special_ids)
    previews = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        shown = ids[:MAX_PREVIEW_TOKENS]
        previews.append(
            {
                "token_count": len(ids),
                "token_ids": shown,
                "tokens": [tokenizer.decode([token]) for token in shown],
                "pieces": tokenizer.convert_ids_to_tokens(shown),
                "special_tokens": [token in special for token in shown],
                "truncated": len(shown) < len(ids),
                "fits": 0 < len(ids) <= limit,
            }
        )
    return {
        **execution_provenance(metadata),
        "token_limit": limit,
        "reserved_tokens": reserved_tokens,
        "prompts": previews,
    }
