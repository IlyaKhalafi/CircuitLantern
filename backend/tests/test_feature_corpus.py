from unittest.mock import patch

import pytest
import torch
from model_fixtures import assert_no_forward_hooks

from circuit_lantern.services.features.corpus import (
    CaptureCancelled,
    capture_activations,
    prepare_corpus,
)

DOCUMENTS = [
    "The capital of France is the capital of France",
    "The capital of Poland is the capital of Poland",
    "France is the capital of France",
    "Poland is the capital of Poland",
    "the capital of France is France",
    "the capital of Poland is Poland",
]


def corpus(runtime, **options):
    return prepare_corpus(
        runtime.tokenizer,
        runtime.metadata,
        options.pop("documents", DOCUMENTS),
        seed=options.pop("seed", 7),
        max_samples=options.pop("max_samples", 64),
        max_tokens=options.pop("max_tokens", 16),
        **options,
    )


def test_corpus_uses_whole_distinct_documents_for_held_out_data(runtime):
    prepared = corpus(runtime, documents=DOCUMENTS + [DOCUMENTS[0]])
    assert prepared.duplicate_documents == 1
    train = {item["document"] for item in prepared.samples if item["split"] == "train"}
    validation = {item["document"] for item in prepared.samples if item["split"] == "validation"}
    assert train and validation
    assert train.isdisjoint(validation)
    assert len(prepared.train_indices) + len(prepared.validation_indices) == len(prepared.samples)
    assert prepared.description()["fingerprint"] == corpus(runtime).fingerprint


def test_seeded_sampling_is_reproducible_and_preserves_global_rng(runtime):
    before = torch.get_rng_state().clone()
    first = corpus(runtime, max_samples=12)
    second = corpus(runtime, max_samples=12)
    other = corpus(runtime, seed=123, max_samples=12)
    assert first.description() == second.description()
    assert first.fingerprint != other.fingerprint
    assert torch.equal(before, torch.get_rng_state())
    assert len(first.samples) == 12


def test_special_token_exclusion_and_document_truncation_are_recorded(runtime):
    prepared = corpus(
        runtime,
        documents=["<eos> " + text for text in DOCUMENTS],
        max_tokens=6,
    )
    assert prepared.token_limit == 6
    assert all(document["truncated"] for document in prepared.documents)
    assert all(
        document["token_ids"][0] == runtime.tokenizer.eos_token_id
        for document in prepared.documents
    )
    assert all(sample["position"] > 0 for sample in prepared.samples)
    assert prepared.special_tokens_excluded
    included = corpus(
        runtime,
        documents=["<eos> " + text for text in DOCUMENTS],
        max_tokens=6,
        exclude_special_tokens=False,
        exclude_first_position=False,
    )
    assert any(sample["position"] == 0 for sample in included.samples)


def test_collection_matches_exact_raw_block_states_and_preserves_model_weights(runtime):
    prepared = corpus(runtime)
    before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.named_parameters()
    }
    before_rng = torch.get_rng_state().clone()
    actual = capture_activations(runtime.model, runtime._blocks(), prepared, 1, 16, device="cpu")
    assert not actual.is_inference()
    expected = {}
    for index, inputs in enumerate(prepared.model_inputs):
        captured = []
        handle = runtime._blocks()[1].register_forward_hook(
            lambda module, args, output, captured=captured: captured.append(
                (output[0] if isinstance(output, tuple) else output).detach().clone()
            )
        )
        try:
            with torch.inference_mode():
                runtime.model.base_model(
                    **{name: torch.tensor([values]) for name, values in inputs.items()},
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
        finally:
            handle.remove()
        expected[index] = captured[0][0]
    for row, sample in zip(actual, prepared.samples, strict=True):
        torch.testing.assert_close(row, expected[sample["document"]][sample["position"]])
    assert torch.equal(before_rng, torch.get_rng_state())
    for name, parameter in runtime.model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert_no_forward_hooks(runtime._blocks())


def test_failed_collection_does_not_leave_a_model_hook(runtime):
    prepared = corpus(runtime)
    with (
        patch.object(
            runtime.model.base_model, "forward", side_effect=RuntimeError("capture failed")
        ),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        capture_activations(runtime.model, runtime._blocks(), prepared, 0, 16, device="cpu")
    assert_no_forward_hooks([runtime._blocks()[0]])


def test_cancelled_collection_does_not_publish_partial_activations(runtime):
    with pytest.raises(CaptureCancelled):
        capture_activations(
            runtime.model,
            runtime._blocks(),
            corpus(runtime),
            0,
            16,
            device="cpu",
            cancelled=lambda: True,
        )
    assert_no_forward_hooks([runtime._blocks()[0]])


@pytest.mark.parametrize("documents", [DOCUMENTS[:3], ["the"] * 4, [""] * 4, ["x" * 8001] * 4])
def test_invalid_corpus_cannot_be_used_as_a_held_out_evaluation(runtime, documents):
    with pytest.raises(ValueError):
        corpus(runtime, documents=documents)


def test_unknown_only_validation_documents_do_not_create_fabricated_samples(runtime):
    with pytest.raises(ValueError, match="tokens"):
        corpus(runtime, documents=["unknownword " * count for count in range(1, 5)])


def test_near_duplicate_groups_stay_in_one_split_without_rewriting_research_text(runtime):
    variants = [DOCUMENTS[0].upper(), "  " + DOCUMENTS[0] + "  "]
    prepared = corpus(runtime, documents=DOCUMENTS + variants)
    related = [prepared.documents[index] for index in [0, 6, 7]]
    assert len({document["split"] for document in related}) == 1
    assert len({document["split_group"] for document in related}) == 1
    assert prepared.documents[6]["text"] == variants[0]
    assert prepared.documents[7]["text"] == variants[1]
    assert prepared.duplicate_documents == 0


def test_first_position_policy_is_explicit_and_can_be_overridden(runtime):
    excluded = corpus(runtime)
    assert excluded.first_position_excluded
    assert all(sample["position"] > 0 for sample in excluded.samples)
    included = corpus(runtime, exclude_first_position=False)
    assert not included.first_position_excluded
    assert any(sample["position"] == 0 for sample in included.samples)
    assert included.fingerprint != excluded.fingerprint


def test_validation_never_reuses_an_exact_training_prefix(runtime):
    prepared = corpus(runtime, max_samples=32768)

    def prefix(sample):
        inputs = prepared.model_inputs[sample["document"]]
        return tuple(
            (key, tuple(values[: sample["position"] + 1])) for key, values in sorted(inputs.items())
        )

    train = {prefix(item) for item in prepared.samples if item["split"] == "train"}
    validation = {prefix(item) for item in prepared.samples if item["split"] == "validation"}
    assert train.isdisjoint(validation)
    assert prepared.shared_prefixes_excluded > 0
    assert prepared.description()["shared_prefixes_excluded"] == prepared.shared_prefixes_excluded


def test_added_special_tokens_are_excluded_consistently_from_capture_and_interventions(runtime):
    from tokenizers import AddedToken

    from circuit_lantern.services.features.inference import eligible_positions

    identifier = runtime.tokenizer.convert_tokens_to_ids("France")
    runtime.tokenizer.add_tokens([AddedToken("France", special=True)], special_tokens=True)
    assert identifier not in runtime.tokenizer.all_special_ids
    prepared = corpus(runtime)
    assert all(
        prepared.documents[sample["document"]]["token_ids"][sample["position"]] != identifier
        for sample in prepared.samples
    )
    mask = eligible_positions(
        {"corpus": {**prepared.description(), "first_position_excluded": False}},
        torch.tensor([3, identifier, 4]),
        runtime.tokenizer,
    )
    assert mask.tolist() == [True, False, True]
    assert identifier in prepared.description()["excluded_special_token_ids"]
    legacy = {**prepared.description(), "first_position_excluded": False}
    legacy.pop("excluded_special_token_ids")
    assert eligible_positions(
        {"corpus": legacy}, torch.tensor([3, identifier, 4]), runtime.tokenizer
    ).tolist() == [True, True, True]
    runtime.tokenizer.add_special_tokens({"additional_special_tokens": ["capital"]})
    assert eligible_positions(
        {"corpus": {**prepared.description(), "first_position_excluded": False}},
        torch.tensor([3, identifier, 4]),
        runtime.tokenizer,
    ).tolist() == [True, False, True]
    included = corpus(runtime, exclude_special_tokens=False)
    assert any(
        included.documents[sample["document"]]["token_ids"][sample["position"]] == identifier
        for sample in included.samples
    )
