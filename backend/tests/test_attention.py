import math

import pytest
import torch

from circuit_lantern.services.attention import attention_summary


def test_head_routing_preserves_individual_heads_and_entropy_axes():
    weights = torch.tensor(
        [
            [
                [[1, 0, 0], [0.2, 0.8, 0], [0.1, 0.2, 0.7]],
                [[1, 0, 0], [0.9, 0.1, 0], [0.5, 0.5, 0]],
            ]
        ]
    )
    result = attention_summary(weights)
    assert result["query_position"] == 2
    assert result["heads"] == 2
    assert result["query_by_head"][0] == pytest.approx([0.1, 0.2, 0.7])
    assert result["query_by_head"][1] == pytest.approx([0.5, 0.5, 0])
    assert result["pattern"][-1] == pytest.approx([0.3, 0.35, 0.35])
    assert result["query_head_entropy"][1] == pytest.approx(math.log(2))
    assert result["head_entropy"][1] < result["query_head_entropy"][1]


def test_head_routing_matches_the_real_captured_model_attention(runtime):
    with torch.inference_mode():
        outputs = runtime.model(
            **runtime.tokenizer("the capital of France is", return_tensors="pt"),
            output_attentions=True,
            use_cache=False,
        )
    trace = runtime.trace("the capital of France is", 1, True, None)
    for index, captured in enumerate(trace["attention"]):
        expected = outputs.attentions[index][0, :, -1, :]
        assert torch.tensor(captured["query_by_head"]) == pytest.approx(expected)
        assert captured["query_position"] == len(trace["tokens"]) - 1
        assert all(sum(row) == pytest.approx(1) for row in captured["query_by_head"])


@pytest.mark.parametrize(
    "weights", [torch.ones(2, 2), torch.ones(2, 2, 2, 2), torch.ones(1, 0, 2, 2)]
)
def test_invalid_capture_shapes_are_rejected(weights):
    with pytest.raises(ValueError, match="one example"):
        attention_summary(weights)


def test_nonfinite_attention_is_rejected():
    with pytest.raises(ValueError, match="non finite"):
        attention_summary(torch.full((1, 1, 1, 1), float("nan")))


@pytest.mark.parametrize("query", [0, 2, 4])
def test_selected_query_matches_real_routing_without_changing_prompt_measurements(runtime, query):
    prompt = "the capital of France is"
    with torch.inference_mode():
        outputs = runtime.model(
            **runtime.tokenizer(prompt, return_tensors="pt"),
            output_attentions=True,
            use_cache=False,
        )
    baseline = runtime.trace(prompt, 0, True, None)
    trace = runtime.trace(prompt, 0, True, None, attention_query_position=query)
    assert trace["attention_query_position"] == query
    assert trace["layers"] == baseline["layers"]
    assert trace["predictions"] == baseline["predictions"]
    assert trace["geometry"] == baseline["geometry"]
    for index, captured in enumerate(trace["attention"]):
        expected = outputs.attentions[index][0, :, query, :]
        assert torch.tensor(captured["query_by_head"]) == pytest.approx(expected)
        assert captured["query_position"] == query
        assert captured["pattern"] == baseline["attention"][index]["pattern"]
        assert captured["head_entropy"] == baseline["attention"][index]["head_entropy"]
        if query == 0:
            assert all(row[0] == pytest.approx(1) for row in captured["query_by_head"])
            assert all(sum(row[1:]) == pytest.approx(0) for row in captured["query_by_head"])
            assert captured["query_head_entropy"] == pytest.approx([0] * captured["heads"])


@pytest.mark.parametrize("query", [-2, 5, 999])
def test_invalid_query_is_rejected_before_model_execution(runtime, monkeypatch, query):
    def unexpected_forward(*args, **kwargs):
        raise AssertionError("An invalid query must not execute the model")

    monkeypatch.setattr(runtime.model, "forward", unexpected_forward)
    with pytest.raises(ValueError, match="query position inside the prompt"):
        runtime.trace("the capital of France is", 0, True, None, attention_query_position=query)


@pytest.mark.parametrize("query", [-2, 3])
def test_attention_summary_rejects_positions_outside_the_query_axis(query):
    with pytest.raises(ValueError, match="query position inside the prompt"):
        attention_summary(torch.ones(1, 2, 3, 3), query)


def test_routing_query_is_unused_without_attention_capture(runtime):
    trace = runtime.trace("the capital", 0, False, None, attention_query_position=999)
    assert trace["attention"] == []
    assert trace["attention_query_position"] is None


@pytest.mark.parametrize("query", [-2, 131072])
def test_trace_schema_bounds_attention_query_positions(query):
    from pydantic import ValidationError

    from circuit_lantern.schemas import TraceRequest

    with pytest.raises(ValidationError):
        TraceRequest(prompt="the capital", attention_query_position=query)


def test_query_position_is_not_claimed_when_the_model_returns_no_attention(runtime, monkeypatch):
    original = runtime.model.forward

    def without_attention(*args, **kwargs):
        output = original(*args, **kwargs)
        output.attentions = None
        return output

    monkeypatch.setattr(runtime.model, "forward", without_attention)
    trace = runtime.trace("the capital", 0, True, None, attention_query_position=1)
    assert trace["capture_attention"] is True
    assert trace["attention"] == []
    assert trace["attention_query_position"] is None


def test_selected_query_uses_the_same_special_token_positions_as_the_preview(runtime):
    from tokenizers.processors import TemplateProcessing

    runtime.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="<pad> $A", special_tokens=[("<pad>", 0)]
    )
    preview = runtime.tokenize(["the capital"], 0)["prompts"][0]
    trace = runtime.trace("the capital", 0, True, None, attention_query_position=1)
    assert preview["special_tokens"] == [True, False, False]
    assert preview["token_ids"] == trace["token_ids"] == [0, 3, 4]
    assert preview["tokens"][1] == trace["tokens"][1] == "the"
    assert trace["attention_query_position"] == 1
    for layer in trace["attention"]:
        assert layer["query_position"] == 1
        assert all(row[2] == 0 for row in layer["query_by_head"])
