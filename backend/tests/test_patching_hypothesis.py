"""Source identity and tokens are enforced by the server before patching forwards."""

from unittest.mock import patch

import pytest

from circuit_lantern.prompt_set_schemas import PromptSetRequest
from circuit_lantern.services.execution import describe_execution
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.prompt_sets import measure_set, prepare_set
from circuit_lantern.services.store import ExperimentStore


@pytest.fixture
def hypothesis(runtime):
    runtime.metadata.update(
        execution_id="a" * 32,
        execution_precision="float32",
        execution_config=describe_execution(runtime.model, "float32", "cpu"),
    )
    trace = runtime.trace(
        "the capital of France is", 0, False, None, tracked_tokens=["France", "Poland"]
    )
    recorded = trace["logit_attribution"]
    block = next(
        row for row in recorded["components"] if row["kind"] == "block" and row["layer"] == 1
    )
    score = recorded["checks"]["actual_logits"][0] - recorded["checks"]["actual_logits"][1]
    reconstructed = sum(row["values"][0] - row["values"][1] for row in recorded["components"])
    source = {
        "trace_id": trace["trace_id"],
        "attribution_schema_version": 1,
        "attribution_value": block["values"][0] - block["values"][1],
        "attribution_score": score,
        "attribution_error": abs(reconstructed - score),
        "model_id": trace["model_id"],
        "revision": trace["revision"],
        "device": trace["device"],
        "execution_id": trace["execution_id"],
        "execution_precision": trace["execution_precision"],
        "execution_config": trace["execution_config"],
        "clean_prompt": trace["prompt"],
        "token_ids": trace["token_ids"],
        "position": len(trace["token_ids"]) - 1,
        "layer": 1,
        "target_token_id": 6,
        "alternative_token_id": 9,
    }
    request = {
        "expected_execution_id": runtime.metadata["execution_id"],
        "pairs": [
            {
                "clean_prompt": trace["prompt"],
                "corrupted_prompt": "the capital of Poland is",
                "target": "France",
                "alternative": "Poland",
            }
        ],
        "layers": [1],
        "hypothesis_source": source,
    }
    return trace, request


def test_hypothesis_and_source_trace_survive_offline_notebook_roundtrip(
    runtime, hypothesis, tmp_path
):
    trace, payload = hypothesis
    request = PromptSetRequest.model_validate(payload)
    plan, _ = prepare_set(runtime, request)
    assert plan["compatible"]
    result = measure_set(runtime, request, JobRecord({"status": "running", "progress": 0}))
    assert result["complete"]
    assert result["hypothesis_source"] == request.hypothesis_source.model_dump()
    notebook = ExperimentStore(tmp_path / "notebook.sqlite3")
    a = notebook.create("Source {literal}", "", trace)
    b = notebook.create_artifact("Patching {literal}", "", "causal_batch", result)
    runtime.unload()
    assert notebook.get(a["id"])["trace"] == trace
    assert notebook.get(b["id"])["result"] == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "different-model"),
        ("revision", "different-revision"),
        ("device", "cuda"),
        ("execution_precision", "bfloat16"),
        ("execution_config", {}),
    ],
)
def test_other_source_profiles_are_rejected_before_any_forward(runtime, hypothesis, field, value):
    _, payload = hypothesis
    payload["hypothesis_source"][field] = value
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("must not forward")),
        pytest.raises(ValueError, match="source does not match"),
    ):
        prepare_set(runtime, PromptSetRequest.model_validate(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_ids", [1, 2, 3, 4, 5]),
        ("position", 0),
        ("target_token_id", 5),
        ("alternative_token_id", 5),
    ],
)
def test_source_token_mismatch_blocks_the_plan_and_run(runtime, hypothesis, field, value):
    _, payload = hypothesis
    payload["hypothesis_source"][field] = value
    request = PromptSetRequest.model_validate(payload)
    with patch.object(runtime.model, "forward", side_effect=AssertionError("must not forward")):
        plan, _ = prepare_set(runtime, request)
        assert not plan["compatible"]
        assert "token IDs differ" in plan["pair_errors"][0]["message"]
        with pytest.raises(ValueError, match="token IDs differ"):
            measure_set(runtime, request, JobRecord({"status": "running", "progress": 0}))


@pytest.mark.parametrize("change", ["metric", "position_mode", "layer", "clean_prompt"])
def test_changing_the_hypothesis_requires_an_independent_experiment(runtime, hypothesis, change):
    _, payload = hypothesis
    if change == "metric":
        payload["metric"] = "log_probability"
    elif change == "position_mode":
        payload["position_mode"] = "all"
    elif change == "layer":
        payload["layers"] = [0]
    else:
        payload["pairs"][0]["clean_prompt"] = "the capital of Poland is"
    with (
        patch.object(runtime.model, "forward", side_effect=AssertionError("must not forward")),
        pytest.raises(ValueError, match="draft differs"),
    ):
        prepare_set(runtime, PromptSetRequest.model_validate(payload))


def test_reloaded_same_checkpoint_profile_can_test_the_original_hypothesis(runtime, hypothesis):
    _, payload = hypothesis
    runtime.metadata["execution_id"] = "b" * 32
    payload["expected_execution_id"] = runtime.metadata["execution_id"]
    plan, _ = prepare_set(runtime, PromptSetRequest.model_validate(payload))
    assert plan["compatible"]
