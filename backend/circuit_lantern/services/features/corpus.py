"""Prompt-separated training data from exact transformer block outputs."""

import hashlib
import json
import random
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import torch

from circuit_lantern.services.tokens import capture_token_limit


class CaptureCancelled(Exception):
    """Stop before publishing an incomplete activation dataset."""


@dataclass
class PreparedCorpus:
    documents: list[dict]
    samples: list[dict]
    model_inputs: list[dict[str, list[int]]]
    fingerprint: str
    duplicate_documents: int
    special_tokens_excluded: bool
    first_position_excluded: bool
    token_limit: int

    @property
    def train_indices(self) -> list[int]:
        return [index for index, sample in enumerate(self.samples) if sample["split"] == "train"]

    @property
    def validation_indices(self) -> list[int]:
        return [
            index for index, sample in enumerate(self.samples) if sample["split"] == "validation"
        ]

    def description(self) -> dict:
        return {
            "documents": self.documents,
            "samples": self.samples,
            "fingerprint": self.fingerprint,
            "duplicate_documents": self.duplicate_documents,
            "special_tokens_excluded": self.special_tokens_excluded,
            "first_position_excluded": self.first_position_excluded,
            "token_limit": self.token_limit,
            "train_document_groups": len(
                {
                    document["split_group"]
                    for document in self.documents
                    if document["split"] == "train"
                }
            ),
            "validation_document_groups": len(
                {
                    document["split_group"]
                    for document in self.documents
                    if document["split"] == "validation"
                }
            ),
            "split_method": "Seeded split of whole document groups before token sampling. Unicode compatibility normalization, whitespace collapse and case folding group normalized text duplicates only for splitting. Paraphrases and overlapping templates are not deduplicated. Original text and tokenization are preserved.",
        }


def document_group(text: str) -> str:
    return hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", text).split()).casefold().encode()
    ).hexdigest()


def prompt_split(manifest: dict, prompt: str) -> str:
    group = document_group(prompt)
    matches = [
        document["split"]
        for document in manifest["corpus"]["documents"]
        if document.get("split_group") == group
    ]
    return matches[0] if matches else "unseen"


def prepare_corpus(
    tokenizer,
    metadata: dict,
    documents: list[str],
    *,
    seed: int,
    max_samples: int,
    max_tokens: int,
    exclude_special_tokens: bool = True,
    exclude_first_position: bool = True,
) -> PreparedCorpus:
    if not 3 <= max_samples <= 32768:
        raise ValueError("Choose between 3 and 32768 captured tokens per dictionary.")
    if not 4 <= len(documents) <= 256 or any(
        not isinstance(text, str) or not text.strip() or len(text) > 8000 for text in documents
    ):
        raise ValueError("Provide between 4 and 256 nonempty documents of at most 8000 characters.")
    distinct = list(dict.fromkeys(documents))
    if len(distinct) < 4:
        raise ValueError("At least four distinct documents are needed for a held out split.")
    token_limit = capture_token_limit(metadata, max_tokens, 0)
    if token_limit < 1:
        raise ValueError("The loaded context allowance has no room for activation capture.")
    rng = random.Random(seed)
    group_ids = [document_group(text) for text in distinct]
    order = list(dict.fromkeys(group_ids))
    if len(order) < 4:
        raise ValueError("At least four document groups are needed after grouping near duplicates.")
    rng.shuffle(order)
    validation_groups = set(order[: max(1, round(len(order) * 0.2))])
    special = set(tokenizer.all_special_ids) if exclude_special_tokens else set()
    encoded = []
    descriptions = []
    candidates = {"train": [], "validation": []}
    for index, text in enumerate(distinct):
        inputs = tokenizer(text, add_special_tokens=True)
        ids = inputs["input_ids"]
        captured_ids = ids[:token_limit]
        captured_inputs = {
            name: values[:token_limit]
            for name, values in inputs.items()
            if name in tokenizer.model_input_names
        }
        encoded.append(captured_inputs)
        split = "validation" if group_ids[index] in validation_groups else "train"
        descriptions.append(
            {
                "index": index,
                "text": text,
                "split": split,
                "split_group": group_ids[index],
                "tokens": [tokenizer.decode([value]) for value in captured_ids],
                "token_ids": captured_ids,
                "model_inputs": captured_inputs,
                "original_tokens": len(ids),
                "truncated": len(ids) > token_limit,
            }
        )
        candidates[split].extend(
            (index, position)
            for position, token in enumerate(captured_ids)
            if token not in special and not (exclude_first_position and position == 0)
        )
    if len(candidates["train"]) < 2 or not candidates["validation"]:
        raise ValueError("The corpus needs at least two training tokens and one held out token.")
    validation_count = min(len(candidates["validation"]), max(1, round(max_samples * 0.2)))
    train_count = min(len(candidates["train"]), max_samples - validation_count)
    validation_count = min(len(candidates["validation"]), max_samples - train_count)
    selected = {
        split: rng.sample(candidates[split], min(count, len(candidates[split])))
        for split, count in (("train", train_count), ("validation", validation_count))
    }
    samples = sorted(
        (
            {"document": document, "position": position, "split": split}
            for split, positions in selected.items()
            for document, position in positions
        ),
        key=lambda item: (item["document"], item["position"]),
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "documents": distinct,
                "samples": samples,
                "model_inputs": encoded,
                "exclude_special_tokens": exclude_special_tokens,
                "exclude_first_position": exclude_first_position,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return PreparedCorpus(
        documents=descriptions,
        samples=samples,
        model_inputs=encoded,
        fingerprint=fingerprint,
        duplicate_documents=len(documents) - len(distinct),
        special_tokens_excluded=exclude_special_tokens,
        first_position_excluded=exclude_first_position,
        token_limit=token_limit,
    )


def capture_activations(
    model,
    blocks,
    corpus: PreparedCorpus,
    layer: int,
    input_dim: int,
    *,
    device: str,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> torch.Tensor:
    if blocks is None or not 0 <= layer < len(blocks):
        raise ValueError("Choose a transformer block that exists in the loaded model.")
    backbone = model.base_model
    if backbone is model:
        raise ValueError("This architecture has no separate backbone for activation collection.")
    activations = torch.empty(len(corpus.samples), input_dim, dtype=torch.float32, device="cpu")
    positions_by_document = {}
    for index, sample in enumerate(corpus.samples):
        positions_by_document.setdefault(sample["document"], []).append((index, sample["position"]))
    for completed, (document, positions) in enumerate(positions_by_document.items(), start=1):
        if cancelled and cancelled():
            raise CaptureCancelled()
        indices = torch.tensor([position for _, position in positions], device=device)
        captured = []

        def capture(module, args, output, *, indices=indices, captured=captured):
            hidden = output[0] if isinstance(output, tuple) else output
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[0] != 1
                or hidden.shape[-1] != input_dim
            ):
                raise ValueError("The residual capture shape does not match the model layout.")
            captured.append(hidden[0].index_select(0, indices).detach().float().cpu())

        handle = blocks[layer].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                inputs = {
                    name: torch.tensor([values], device=device)
                    for name, values in corpus.model_inputs[document].items()
                }
                backbone(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            if len(captured) != 1 or not torch.isfinite(captured[0]).all():
                raise ValueError("The model did not produce one finite residual capture.")
            rows = torch.tensor([index for index, _ in positions], device="cpu")
            activations[rows] = captured[0]
        finally:
            handle.remove()
        if progress:
            progress(completed, len(positions_by_document))
    return activations
