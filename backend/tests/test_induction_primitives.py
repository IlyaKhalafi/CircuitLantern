"""Independent position oracles and input preservation for repeated token probes."""

import pytest
import torch

from circuit_lantern.services.induction.patterns import (
    repeated_attention_masks,
    score_repeated_attention,
)
from circuit_lantern.services.induction.probes import ProbeConfig, build_probe_batch


def four_heads():
    tokens = torch.tensor([0, 11, 12, 13, 11], dtype=torch.int64)
    patterns = torch.eye(5, dtype=torch.float64).repeat(4, 1, 1)
    patterns[0, 4] = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64)
    patterns[1, 4] = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float64)
    for query in range(1, 5):
        patterns[2, query] = 0
        patterns[2, query, query - 1] = 1
    patterns[3] = torch.ones(5, 5, dtype=torch.float64).tril() / torch.arange(1, 6)[:, None]
    return tokens, patterns


def test_prefix_duplicate_and_previous_keys_are_different_algorithms():
    tokens, _ = four_heads()
    actual = repeated_attention_masks(tokens)
    assert actual["prefix"][4].nonzero().flatten().tolist() == [2]
    assert actual["duplicate"][4].nonzero().flatten().tolist() == [1]
    assert actual["previous"][4].nonzero().flatten().tolist() == [3]
    assert actual["prefix"].sum() == 1
    assert actual["prefix"].triu(diagonal=1).sum() == 0


def test_all_earlier_matching_prefixes_are_included():
    patterns = repeated_attention_masks(torch.tensor([0, 11, 12, 11, 12, 11, 12]))
    assert patterns["prefix"][5].nonzero().flatten().tolist() == [2, 4]
    assert patterns["prefix"][6].nonzero().flatten().tolist() == [3, 5]


def test_head_scores_distinguish_exact_behaviors_and_uniform_attention():
    tokens, attention = four_heads()
    original = attention.clone()
    measured = score_repeated_attention(attention, tokens, [4])
    rows = measured["heads"]
    assert [row["prefix_attention"] for row in rows] == pytest.approx([1, 0, 0, 0.2])
    assert [row["duplicate_attention"] for row in rows] == pytest.approx([0, 1, 0, 0.2])
    assert [row["previous_attention"] for row in rows] == pytest.approx([0, 0, 1, 0.2])
    assert measured["uniform_full_causal_baseline"] == 0.2
    assert measured["queries"] == 1
    assert torch.equal(original, attention)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_storage_rounding_is_validated_then_normalized_without_mutation(dtype):
    tokens = torch.tensor([0, 11, 12, 11, 12, 11, 12])
    attention = (torch.ones(7, 7, dtype=torch.float64).tril() / torch.arange(1, 8)[:, None]).to(
        dtype
    )[None]
    original = attention.clone()
    measured = score_repeated_attention(attention, tokens, [3, 4, 5, 6])
    assert measured["heads"][0]["prefix_attention"] == pytest.approx(
        measured["uniform_full_causal_baseline"], abs=1e-12
    )
    assert torch.equal(original, attention)


@pytest.mark.parametrize("corruption", ["negative", "nan", "inf", "future", "unnormalized"])
def test_invalid_attention_is_never_reported_as_a_head_measurement(corruption):
    tokens, attention = four_heads()
    if corruption == "negative":
        attention[0, 0, 0] = -1
    elif corruption == "nan":
        attention[0, 0, 0] = float("nan")
    elif corruption == "inf":
        attention[0, 0, 0] = float("inf")
    elif corruption == "future":
        attention[0, 1] = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64)
    else:
        attention[0, 4] *= 0.5
    with pytest.raises(ValueError):
        score_repeated_attention(attention, tokens, [4])


@pytest.mark.parametrize("positions", [[], [1], [4, 4], [-1], [5], [True], [4.0]])
def test_only_distinct_valid_repeated_queries_are_measured(positions):
    tokens, attention = four_heads()
    with pytest.raises(ValueError):
        score_repeated_attention(attention, tokens, positions)


def test_mask_and_score_shapes_and_dtypes_are_validated():
    tokens, attention = four_heads()
    for invalid in [tokens.float(), tokens[None], torch.tensor([-1, 2]), tokens.to("meta")]:
        with pytest.raises(ValueError):
            repeated_attention_masks(invalid)
    for invalid in [
        attention[None],
        attention[:, :, :4],
        attention[:0],
        attention.long(),
        attention.to("meta"),
    ]:
        with pytest.raises(ValueError):
            score_repeated_attention(invalid, tokens, [4])


def test_local_generator_inputs_and_json_descriptions_are_reproducible():
    config = ProbeConfig(block_length=4, repeats=3, examples=3, seed=17)
    before = torch.get_rng_state().clone()
    batch = build_probe_batch(range(30), [0, 1, 29], 0, config, context_limit=13)
    reordered = build_probe_batch(reversed(range(30)), [29, 1, 0], 0, config, context_limit=13)
    assert batch.description() == reordered.description()
    assert batch.fingerprint == reordered.fingerprint
    assert batch.vocabulary_pool_size == 27
    assert torch.equal(before, torch.get_rng_state())
    for probe in batch.probes:
        assert len(probe.token_ids) == 13
        assert probe.token_ids[0] == 0
        assert len(set(probe.token_ids[1:5])) == 4
        assert not set(probe.token_ids[1:]) & {0, 1, 29}
        assert probe.token_ids[1:5] == probe.token_ids[5:9] == probe.token_ids[9:13]
        assert probe.target_ids[:-1] == probe.token_ids[1:]
        assert probe.target_ids[-1] == probe.token_ids[1]
        assert probe.first_queries == (1, 2, 3, 4)
        assert probe.repeated_queries == tuple(range(5, 13))
    description = batch.description()
    description["probes"][0]["token_ids"][0] = 999
    assert batch.probes[0].token_ids[0] == 0
    assert batch.fingerprint == reordered.fingerprint


def test_start_token_is_optional_and_target_positions_remain_aligned():
    batch = build_probe_batch(
        range(20), [0], None, ProbeConfig(block_length=4, repeats=2, examples=1), context_limit=8
    )
    probe = batch.probes[0]
    assert len(probe.token_ids) == 8
    assert probe.first_queries == (0, 1, 2, 3)
    assert probe.repeated_queries == (4, 5, 6, 7)
    assert probe.target_ids[:-1] == probe.token_ids[1:]
    assert probe.target_ids[-1] == probe.token_ids[0]
    assert 0 not in probe.token_ids
    assert repeated_attention_masks(torch.tensor(probe.token_ids))["prefix"][
        4
    ].nonzero().flatten().tolist() == [1]


@pytest.mark.parametrize(
    "change",
    [
        {"block_length": 3},
        {"block_length": 65},
        {"repeats": 1},
        {"repeats": 5},
        {"examples": 0},
        {"examples": 33},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_probe_configuration_is_rejected_before_sampling(change):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        build_probe_batch(range(100), [0], 0, ProbeConfig(**change), context_limit=1024)
    assert torch.equal(before, torch.get_rng_state())


def test_context_and_vocabulary_bounds_are_checked_before_sampling():
    config = ProbeConfig(block_length=4, repeats=2, examples=1)
    for vocabulary, specials, start, context in [
        (range(20), [0], 0, 8),
        (range(20), [0], 0, True),
        (range(5), [0, 1], 0, 20),
        (range(20), [0], 99, 20),
        ([True, 2, 3, 4, 5], [0], None, 20),
        ([0, 1, 2, 3, 2**63], [], None, 20),
    ]:
        with pytest.raises(ValueError):
            build_probe_batch(vocabulary, specials, start, config, context_limit=context)


def test_cpu_planning_and_scoring_ignore_ambient_tensor_device():
    tokens, attention = four_heads()
    with torch.device("meta"):
        batch = build_probe_batch(
            range(20), [0], 0, ProbeConfig(block_length=4, examples=1), context_limit=17
        )
        measured = score_repeated_attention(attention, tokens, [4])
    assert len(batch.probes[0].token_ids) == 17
    assert measured["heads"][0]["prefix_attention"] == 1
