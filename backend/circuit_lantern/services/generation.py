def decode_generation(tokenizer, prompt_ids: list[int], sequence_ids: list[int]) -> dict:
    """Decode in context so word boundaries and subword joins remain intact."""
    options = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    decoded_prompt = tokenizer.decode(prompt_ids, **options)
    generated_text = tokenizer.decode(sequence_ids, **options)
    generated_ids = sequence_ids[len(prompt_ids) :]
    return {
        "decoded_prompt": decoded_prompt,
        "generated_text": generated_text,
        "continuation_text": generated_text[len(decoded_prompt) :]
        if generated_text.startswith(decoded_prompt)
        else None,
        "continuation": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_token_ids": generated_ids,
        "generated_tokens": [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in generated_ids
        ],
        "generated_pieces": tokenizer.convert_ids_to_tokens(generated_ids),
    }


def generation_settings(tokenizer, max_new_tokens: int) -> dict:
    """Keep generation bounded even when a checkpoint requests extra outputs."""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "token_healing": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "output_logits": False,
    }
