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
