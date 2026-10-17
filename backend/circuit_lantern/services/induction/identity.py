"""Mapping and input evidence required for index based checkpoint comparisons."""

import hashlib
import json


def identity_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def comparison_identity(model, tokenizer, blocks, batch, *, vocabulary: dict | None = None) -> dict:
    from circuit_lantern.services.induction.planning import output_vocabulary_size
    from circuit_lantern.services.interventions import attention_head_targets

    input_rows, width = map(int, model.get_input_embeddings().weight.shape)
    output_rows = output_vocabulary_size(model)
    maximum = max(input_rows, output_rows)
    vocabulary = tokenizer.get_vocab() if vocabulary is None else vocabulary
    if any(
        not isinstance(token, str) or type(index) is not int or index < 0
        for token, index in vocabulary.items()
    ):
        raise ValueError(
            "The tokenizer vocabulary must map token strings to nonnegative integer IDs."
        )
    mapping = sorted((index, token) for token, index in vocabulary.items() if index < maximum)
    digest = hashlib.sha256(b"circuit-lantern-token-mapping-v1\0")
    for index, token in mapping:
        # Length framed JSON records avoid ambiguity without retaining another full JSON copy.
        encoded = json.dumps([index, token], ensure_ascii=True, separators=(",", ":")).encode(
            "ascii"
        )
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    texts = [
        identity_digest(tokenizer.decode(list(probe.token_ids), skip_special_tokens=False))
        for probe in batch.probes
    ]
    layout = {
        "model_type": str(model.config.model_type),
        "layers": len(blocks),
        "hidden_size": width,
        "input_vocabulary_rows": input_rows,
        "output_vocabulary_rows": output_rows,
        "key_value_heads": int(
            getattr(model.config, "num_key_value_heads", None) or model.config.num_attention_heads
        ),
        "attention_heads": attention_head_targets(blocks),
    }
    return {
        "format": "circuit-lantern-probe-identity-v1",
        "vocabulary_mapping_sha256": digest.hexdigest(),
        "mapped_token_entries": len(mapping),
        "decoded_probe_sha256": texts,
        "layout": layout,
        "scope": "Vocabulary token names and IDs within active input or output rows, decoded full probe inputs and attention layout. Head comparisons align indices and do not establish functional identity.",
    }
