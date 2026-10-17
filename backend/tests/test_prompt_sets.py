import json
from unittest.mock import patch

import pytest
from model_fixtures import assert_no_forward_hooks
from pydantic import ValidationError

from circuit_lantern.prompt_set_schemas import PromptSetRequest
from circuit_lantern.services.causal import activation_patch_sweep
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.prompt_set_statistics import summarize_pairs
from circuit_lantern.services.prompt_sets import measure_set, prepare_set


@pytest.fixture
def session(runtime):
    runtime.metadata["execution_id"] = "a" * 32
    return runtime


def request_for(runtime, **overrides):
    pair = {
        "label": "France",
        "clean_prompt": "the capital of France is",
        "corrupted_prompt": "the capital of Poland is",
        "target": "France",
        "alternative": "Poland",
    }
    return PromptSetRequest.model_validate(
        {
            "expected_execution_id": runtime.metadata["execution_id"],
            "pairs": [
                pair,
                dict(
                    pair,
                    label="Poland",
                    clean_prompt=pair["corrupted_prompt"],
                    corrupted_prompt=pair["clean_prompt"],
                ),
            ],
            **overrides,
        }
    )


def job():
    return JobRecord({"status": "running", "progress": 0})


def test_contrastive_last_layer_restores_each_clean_score_and_keeps_pair_provenance(session):
    request = request_for(session)
    result = measure_set(session, request, job())
    assert result["complete"] and result["completed_pairs"] == 2
    assert result["execution_id"] == session.metadata["execution_id"]
    assert result["requested_pairs"] == [pair.model_dump() for pair in request.pairs]
    for pair in result["pairs"]:
        capture = pair["result"]
        assert capture["layers"][-1]["logit_difference"] == pytest.approx(
            capture["clean"]["logit_difference"], abs=1e-6
        )
        assert capture["target_token_id"] != capture["alternative_token_id"]
        assert capture["layers"][-1]["metric_delta"] == pytest.approx(
            pair["baseline_gap"], abs=1e-6
        )
        assert pair["positions"] == [4]
    json.dumps(result, allow_nan=False)
    assert_no_forward_hooks(session._blocks())


def test_all_position_patching_restores_every_selected_layer(session):
    result = measure_set(session, request_for(session, position_mode="all", layers=[1, 0]), job())
    assert result["selected_layers"] == [0, 1]
    eligible = sum(pair["baseline_gap"] > 0.001 for pair in result["pairs"])
    assert result["recovery_pairs"] == eligible
    if eligible:
        assert all(layer["recovery"] == pytest.approx(1, abs=1e-5) for layer in result["layers"])
    for pair in result["pairs"]:
        for layer in pair["result"]["layers"]:
            assert layer["metric_delta"] == pytest.approx(pair["baseline_gap"], abs=1e-6)


def test_validated_source_tokens_and_rng_are_retained(session):
    request = request_for(session)
    plan, _ = prepare_set(session, request)
    measured = measure_set(session, request, job())
    for expected, pair in zip(plan["pairs"], measured["pairs"], strict=True):
        assert pair["result"]["target_token_id"] == expected["target_token_id"]
        assert pair["result"]["alternative_token_id"] == expected["alternative_token_id"]
    assert measured["statistics"]["generator"] == "PCG64"
    assert measured["statistics"]["numpy_version"]
    assert (
        prepare_set(session, request_for(session, layers=[1, 0]))[0]["input_fingerprint"]
        == prepare_set(session, request_for(session, layers=[0, 1]))[0]["input_fingerprint"]
    )


@pytest.mark.parametrize("metric", ["log_probability", "logit_difference"])
def test_empty_partial_cohort_has_consistent_finite_fields(session, metric):
    task = job()
    task.request_cancel()
    result = measure_set(session, request_for(session, metric=metric), task)
    assert result["completed_pairs"] == 0 and result["layers"] == []
    assert "clean_target_preferred_pairs" in result
    json.dumps(result, allow_nan=False)


def test_full_restoration_recovery_interval_for_a_positive_gap_cohort(session):
    request = request_for(session, position_mode="all")
    pair = request.pairs[0].model_copy(deep=True)
    pair.clean_prompt = "the capital of France"
    pair.corrupted_prompt = "the capital of Poland"
    request.pairs = [pair]
    initial = measure_set(session, request, job())
    gap = initial["pairs"][0]["baseline_gap"]
    assert abs(gap) > 0.001
    if gap < 0:
        pair.target, pair.alternative = pair.alternative, pair.target
    request.pairs = [pair.model_copy(deep=True) for _ in range(3)]
    result = measure_set(session, request, job())
    assert result["recovery_pairs"] == 3
    for layer in result["layers"]:
        assert layer["recovery"] == pytest.approx(1, abs=1e-5)
        assert layer["recovery_interval"] == pytest.approx([1, 1], abs=1e-5)


def test_projected_embedding_models_use_residual_width_for_cache_planning(session):
    base = prepare_set(session, request_for(session))[0]["cache_bytes"]
    session.model.config.hidden_size *= 2
    assert prepare_set(session, request_for(session))[0]["cache_bytes"] == base * 2


def test_changed_provenance_is_rejected_and_original_set_identity_preserved(session):
    from circuit_lantern.services.prompt_sets import activation_patch_sweep as actual

    calls = 0

    def change_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = actual(*args, **kwargs)
        if calls == 1:
            session.metadata["execution_id"] = "b" * 32
        return result

    with patch(
        "circuit_lantern.services.prompt_sets.activation_patch_sweep",
        side_effect=change_after_first,
    ):
        result = measure_set(session, request_for(session), job())
    assert result["execution_id"] == "a" * 32
    assert result["completed_pairs"] == 1 and result["failed_pair"] == 1


def test_different_scored_token_ids_never_enter_an_aggregate(session):
    from circuit_lantern.services.prompt_sets import activation_patch_sweep as actual

    calls = 0

    def different_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = actual(*args, **kwargs)
        if calls == 2:
            result["target_token_id"] = result["alternative_token_id"]
        return result

    with patch(
        "circuit_lantern.services.prompt_sets.activation_patch_sweep", side_effect=different_second
    ):
        result = measure_set(session, request_for(session), job())
    assert result["failed_pair"] == 1 and result["completed_pairs"] == 1
    assert len(result["pairs"]) == 1


def test_changed_position_mode_uses_each_pairs_actual_changed_token_ids(session):
    result = measure_set(session, request_for(session, position_mode="changed", layers=[0]), job())
    assert [pair["positions"] for pair in result["pairs"]] == [[3], [3]]
    assert all(pair["result"]["total_layers"] == 1 for pair in result["pairs"])


def test_single_pair_log_probability_matches_existing_sweep(session):
    request = request_for(session, metric="log_probability")
    request.pairs = request.pairs[:1]
    pair = request.pairs[0]
    existing = activation_patch_sweep(
        session.model,
        session.tokenizer,
        session._blocks(),
        session.metadata,
        session.device,
        pair.clean_prompt,
        pair.corrupted_prompt,
        pair.target,
        -1,
        {},
        session.settings.max_prompt_tokens,
    )
    result = measure_set(session, request, job())
    for old, new in zip(existing["layers"], result["pairs"][0]["result"]["layers"], strict=True):
        assert new["metric_delta"] == pytest.approx(old["delta_log_probability"])
    assert result["layers"][0]["mean_interval"] is None
    assert result["clean_target_preferred_pairs"] is None


def test_plan_reports_every_invalid_pair_without_any_forward(session):
    request = request_for(session)
    request.pairs[0].target = "France Poland"
    request.pairs[1].corrupted_prompt = "the"
    with patch.object(session.model, "forward", wraps=session.model.forward) as forward:
        plan, _ = prepare_set(session, request)
        assert not plan["compatible"]
        assert [error["index"] for error in plan["pair_errors"]] == [0, 1]
        with pytest.raises(ValueError, match="invalid prompt pairs"):
            measure_set(session, request, job())
        forward.assert_not_called()
    assert_no_forward_hooks(session._blocks())


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"alternative": "France"}, "different tokens"),
        ({"alternative": ""}, "alternative token"),
        ({"alternative": "France Poland"}, "exactly one token"),
    ],
)
def test_contrastive_validation_rejects_ambiguous_alternatives(session, change, reason):
    request = request_for(session)
    for key, value in change.items():
        setattr(request.pairs[0], key, value)
    plan, _ = prepare_set(session, request)
    assert reason in plan["pair_errors"][0]["message"]


def test_identical_pairs_are_retained_but_never_normalized(session):
    request = request_for(session)
    for pair in request.pairs:
        pair.corrupted_prompt = pair.clean_prompt
    result = measure_set(session, request, job())
    assert result["complete"] and result["recovery_pairs"] == 0
    assert all(
        layer["mean_delta"] == pytest.approx(0, abs=1e-7) and layer["recovery"] is None
        for layer in result["layers"]
    )
    request.position_mode = "changed"
    assert "no changed token" in prepare_set(session, request)[0]["pair_errors"][0]["message"]


def test_stale_sessions_and_invalid_layers_are_rejected_before_inference(session):
    request = request_for(session, expected_execution_id="b" * 32)
    with pytest.raises(ValueError, match="changed"):
        prepare_set(session, request)
    for layers in ([2], [0, 0]):
        with pytest.raises(ValueError, match="distinct transformer layers"):
            prepare_set(session, request_for(session, layers=layers))


def test_cancellation_retains_partial_pair_but_aggregates_one_balanced_cohort(session):
    task = job()
    forward = session.model.forward
    calls = 0

    def stop_mid_pair(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = forward(*args, **kwargs)
        if calls == 7:
            task.request_cancel()
        return value

    with patch.object(session.model, "forward", side_effect=stop_mid_pair):
        result = measure_set(session, request_for(session), task)
    task.finish(result)
    assert task["status"] == "cancelled"
    assert result["completed_pairs"] == 1 and len(result["pairs"]) == 2
    assert len(result["pairs"][1]["result"]["layers"]) == 1
    for summary, measured in zip(
        result["layers"], result["pairs"][0]["result"]["layers"], strict=True
    ):
        assert summary["mean_delta"] == measured["metric_delta"]
        assert summary["mean_interval"] is None
    assert_no_forward_hooks(session._blocks())


def test_forward_failure_keeps_prior_pairs_and_the_failed_pair_identity(session):
    forward = session.model.forward
    calls = 0

    def fail_second_pair(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError("test forward failure")
        return forward(*args, **kwargs)

    task = job()
    with patch.object(session.model, "forward", side_effect=fail_second_pair):
        result = measure_set(session, request_for(session), task)
    task.finish(result)
    assert task["status"] == "failed"
    assert result["failed_pair"] == 1 and result["completed_pairs"] == 1
    assert result["pairs"][1]["result"]["failed_layer"] == 1
    assert_no_forward_hooks(session._blocks())


def test_plans_enforce_context_forward_and_resident_memory_budgets(session):
    session.metadata["execution_context_length"] = 4
    assert "4 token" in prepare_set(session, request_for(session))[0]["pair_errors"][0]["message"]
    session.metadata["execution_context_length"] = 64
    with patch("circuit_lantern.services.prompt_sets.MAX_SET_FORWARDS", 7):
        assert "forwards" in prepare_set(session, request_for(session))[0]["blockers"][0]
    with patch(
        "circuit_lantern.services.prompt_sets.hardware_snapshot",
        return_value={"ram_available_bytes": 0, "experiment_disk_free_bytes": 0},
    ):
        plan, _ = prepare_set(session, request_for(session))
        assert not plan["compatible"] and len(plan["blockers"]) == 2


@pytest.mark.parametrize(
    "overrides", [{"layers": [True]}, {"seed": -1}, {"pairs": []}, {"seed": 1.5}]
)
def test_request_bounds_reject_coercions_and_empty_experiments(session, overrides):
    with pytest.raises(ValidationError):
        request_for(session, **overrides)


def test_bootstrap_is_reproducible_and_recovery_uses_ratio_of_sums():
    pairs = [
        {
            "baseline_gap": gap,
            "clean_score": 1,
            "result": {"complete": True, "layers": [{"metric_delta": delta}]},
        }
        for gap, delta in [(1, 1), (9, 0), (0, 4)]
    ]
    result = summarize_pairs(pairs, [3], 42)
    assert result == summarize_pairs(pairs, [3], 42)
    layer = result["layers"][0]
    assert layer["recovery"] == pytest.approx(0.1)
    assert result["recovery_pairs"] == 2 and result["completed_pairs"] == 3
    assert layer["mean_delta"] == pytest.approx(5 / 3)
    assert layer["positive_fraction"] == pytest.approx(2 / 3)
    assert layer["near_zero_fraction"] == pytest.approx(1 / 3)
    assert layer["mean_interval"][0] <= layer["mean_delta"] <= layer["mean_interval"][1]
    for pair in pairs:
        pair["result"]["layers"][0]["metric_delta"] = 0.5
    assert summarize_pairs(pairs, [3], 42)["layers"][0]["mean_interval"] == [0.5, 0.5]


def test_raw_direction_and_normalized_restoration_have_distinct_cohorts():
    pairs = [
        {
            "baseline_gap": gap,
            "clean_score": gap,
            "result": {"complete": True, "layers": [{"metric_delta": gap}]},
        }
        for gap in (2, -2)
    ]
    result = summarize_pairs(pairs, [0], 0)
    assert result["layers"][0]["mean_delta"] == 0
    assert result["layers"][0]["negative_fraction"] == 0.5
    assert result["layers"][0]["recovery"] == 1
    assert result["recovery_pairs"] == 1
