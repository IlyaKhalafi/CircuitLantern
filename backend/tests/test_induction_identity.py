from unittest.mock import patch

import pytest
import torch

from circuit_lantern.induction_schemas import InductionProbeRequest
from circuit_lantern.services.induction.identity import comparison_identity, identity_digest
from circuit_lantern.services.induction.lab import InductionLab
from circuit_lantern.services.induction.probes import ProbeConfig, build_probe_batch


def batch(runtime):
    return build_probe_batch(
        runtime.tokenizer.get_vocab().values(),
        runtime.tokenizer.all_special_ids,
        0,
        ProbeConfig(block_length=4, repeats=2, examples=2, seed=19),
        context_limit=64,
    )


def identity(runtime, probes=None, **kwargs):
    return comparison_identity(
        runtime.model, runtime.tokenizer, runtime._blocks(), probes or batch(runtime), **kwargs
    )


def test_identity_ignores_vocabulary_iteration_order_and_weight_values(runtime):
    probes = batch(runtime)
    original = identity(runtime, probes)
    vocabulary = dict(reversed(list(runtime.tokenizer.get_vocab().items())))
    before = torch.get_rng_state()
    with patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")):
        reordered = identity(runtime, probes, vocabulary=vocabulary)
        with torch.no_grad():
            runtime.model.transformer.h[0].attn.c_attn.weight.add_(0.02)
        changed_weights = identity(runtime, probes)
    assert original == reordered == changed_weights
    assert torch.equal(before, torch.get_rng_state())
    assert original["layout"] == {
        "model_type": "gpt2",
        "layers": 2,
        "hidden_size": 16,
        "input_vocabulary_rows": 16,
        "output_vocabulary_rows": 16,
        "key_value_heads": 2,
        "attention_heads": [
            {"layer": 0, "heads": 2, "head_dim": 8},
            {"layer": 1, "heads": 2, "head_dim": 8},
        ],
    }


def test_permuted_token_names_are_detected_despite_identical_numeric_probe_fingerprints(runtime):
    original_batch = batch(runtime)
    original = identity(runtime, original_batch)
    vocabulary = runtime.tokenizer.get_vocab()
    vocabulary["capital"], vocabulary["the"] = vocabulary["the"], vocabulary["capital"]
    with patch.object(runtime.tokenizer, "get_vocab", return_value=vocabulary):
        permuted_batch = batch(runtime)
        permuted = identity(runtime, permuted_batch)
    assert original_batch.fingerprint == permuted_batch.fingerprint
    assert original_batch.probes == permuted_batch.probes
    assert original["vocabulary_mapping_sha256"] != permuted["vocabulary_mapping_sha256"]
    assert identity_digest(original) != identity_digest(permuted)


def test_decoded_input_changes_are_detected_when_token_mapping_is_unchanged(runtime):
    probes = batch(runtime)
    original = identity(runtime, probes)
    decode = runtime.tokenizer.decode
    with patch.object(
        runtime.tokenizer,
        "decode",
        side_effect=lambda *args, **kwargs: decode(*args, **kwargs) + "!",
    ):
        changed = identity(runtime, probes)
    assert changed["vocabulary_mapping_sha256"] == original["vocabulary_mapping_sha256"]
    assert changed["decoded_probe_sha256"] != original["decoded_probe_sha256"]
    assert len(changed["decoded_probe_sha256"]) == probes.config.examples


def test_unused_tokens_outside_both_embedding_domains_do_not_change_mapping_evidence(runtime):
    original = identity(runtime)
    vocabulary = runtime.tokenizer.get_vocab()
    vocabulary["unused_added_token"] = 2000
    changed = identity(runtime, vocabulary=vocabulary)
    assert original == changed
    vocabulary["alias_for_input_row"] = 3
    aliased = identity(runtime, vocabulary=vocabulary)
    assert original["vocabulary_mapping_sha256"] != aliased["vocabulary_mapping_sha256"]


@pytest.mark.parametrize("invalid", [-1, True, "3"])
def test_invalid_vocabulary_ids_are_not_silently_hashed(runtime, invalid):
    vocabulary = runtime.tokenizer.get_vocab()
    vocabulary["bad"] = invalid
    with pytest.raises(ValueError, match="nonnegative integer IDs"):
        identity(runtime, vocabulary=vocabulary)


def test_layout_identity_distinguishes_architectures_with_equal_tensor_dimensions(runtime):
    original = identity(runtime)
    runtime.model.config.model_type = "different-architecture"
    changed = identity(runtime)
    assert original["vocabulary_mapping_sha256"] == changed["vocabulary_mapping_sha256"]
    assert original["layout"] != changed["layout"]
    assert identity_digest(original) != identity_digest(changed)


def test_worker_rejects_a_changed_token_mapping_before_any_teacher_forward(runtime, monkeypatch):
    runtime.metadata.update(execution_id="a" * 32, execution_context_length=64)
    monkeypatch.setattr(
        "circuit_lantern.services.induction.lab.hardware_snapshot",
        lambda _: {"ram_available_bytes": 2**40, "experiment_disk_free_bytes": 2**40},
    )
    lab = InductionLab(runtime)
    request = InductionProbeRequest(block_length=4, repeats=2, examples=2, causal_heads=1)
    planned = lab.plan(request)
    payload = request.model_dump()
    payload.update(
        expected_probe_fingerprint=planned["dataset_fingerprint"],
        expected_comparison_identity=planned["comparison_identity_sha256"],
    )
    vocabulary = runtime.tokenizer.get_vocab()
    vocabulary["capital"], vocabulary["the"] = vocabulary["the"], vocabulary["capital"]
    with (
        patch.object(runtime.tokenizer, "get_vocab", return_value=vocabulary),
        patch.object(runtime.model, "forward", side_effect=AssertionError("Unexpected forward")),
        pytest.raises(ValueError, match="token mapping or layout changed"),
    ):
        lab._measure(payload, {})
    assert runtime.jobs == {}
