import json
import math
from unittest.mock import patch

import pytest
import torch

from circuit_lantern.services import token_logits
from circuit_lantern.services.token_logits import summarize_token_logits


def test_known_token_distributions_preserve_ids_probabilities_and_entropy():
    result = summarize_token_logits(
        torch.tensor([[math.log(2), 0.0], [1000.0, 1000.0], [-1000.0, 1000.0]], dtype=torch.float64)
    )
    assert result["token_prediction_ids"] == [0, 0, 1]
    assert result["token_prediction_probabilities"] == pytest.approx([2 / 3, 0.5, 1])
    assert result["token_entropies"] == pytest.approx(
        [-(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)), math.log(2), 0]
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chunked_statistics_match_full_vocabulary_distributions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn((17, 13), generator=generator).to(device) * 37
    probabilities = logits.double().softmax(-1)
    expected_probabilities, expected_ids = probabilities.max(-1)
    expected_entropy = torch.special.entr(probabilities).sum(-1)
    sizes = []
    score = token_logits.log_probabilities

    def observe(values):
        sizes.append(values.numel())
        return score(values)

    with (
        patch.object(token_logits, "MAX_LOGIT_ELEMENTS", 39),
        patch.object(token_logits, "log_probabilities", side_effect=observe),
    ):
        result = summarize_token_logits(logits)
    assert len(sizes) > 1 and max(sizes) <= 39
    assert result["token_prediction_ids"] == expected_ids.cpu().tolist()
    assert result["token_prediction_probabilities"] == pytest.approx(
        expected_probabilities.cpu().tolist(), rel=1e-12, abs=1e-15
    )
    assert result["token_entropies"] == pytest.approx(
        expected_entropy.cpu().tolist(), rel=1e-12, abs=1e-15
    )


def test_large_finite_logits_do_not_produce_false_confidence_or_undefined_entropy():
    largest = torch.finfo(torch.float32).max
    result = summarize_token_logits(
        torch.tensor([[largest, largest], [largest, -largest], [-largest, -largest]])
    )
    assert result["token_prediction_ids"] == [0, 0, 0]
    assert result["token_prediction_probabilities"] == pytest.approx([0.5, 1, 0.5])
    assert result["token_entropies"] == pytest.approx([math.log(2), 0, math.log(2)])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape", [(3,), (0, 3), (3, 0), (1, 2, 3)])
def test_nonempty_token_vocabulary_shape_is_required(shape):
    with pytest.raises(ValueError, match="nonempty token by vocabulary"):
        summarize_token_logits(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_later_chunk_rejects_the_whole_summary(value):
    logits = torch.tensor([[1.0, 0], [value, 0]])
    with (
        patch.object(token_logits, "MAX_LOGIT_ELEMENTS", 2),
        pytest.raises(ValueError, match="non finite logits"),
    ):
        summarize_token_logits(logits)


def test_every_captured_position_matches_the_actual_logit_lens(runtime):
    prompt = "the capital of France is"
    inputs = runtime.tokenizer(prompt, return_tensors="pt")
    expected = []
    with torch.inference_mode():
        outputs = runtime.model(**inputs, use_cache=False, output_hidden_states=True)
        for index, hidden in enumerate(outputs.hidden_states):
            projected = (
                runtime.model.transformer.ln_f(hidden)
                if index < len(outputs.hidden_states) - 1
                else hidden
            )
            logits = runtime.model.lm_head(projected).float()[0]
            probabilities = logits.double().softmax(-1)
            values, ids = probabilities.max(-1)
            expected.append(
                {
                    "ids": ids.tolist(),
                    "probabilities": values.tolist(),
                    "entropy": torch.special.entr(probabilities).sum(-1).tolist(),
                }
            )
    result = runtime.trace(prompt, 0, False, None)
    for layer, reference in zip(result["layers"], expected, strict=True):
        assert layer["token_prediction_ids"] == reference["ids"]
        assert layer["token_prediction_probabilities"] == pytest.approx(
            reference["probabilities"], rel=1e-12, abs=1e-15
        )
        assert layer["token_entropies"] == pytest.approx(reference["entropy"], rel=1e-12, abs=1e-15)
        assert layer["token_predictions"] == [
            runtime.tokenizer.decode([token_id]) for token_id in reference["ids"]
        ]
        assert len(layer["token_entropies"]) == len(result["token_ids"])
    assert "float64" in result["token_prediction_method"]


@pytest.mark.parametrize("layer, head", [(0, None), (None, {"layer": 0, "head": 0})])
def test_intervention_capture_keeps_position_summaries_portable_and_aligned(runtime, layer, head):
    result = runtime.trace("the capital", 0, True, layer, head)
    for capture in result["layers"]:
        assert len(capture["token_prediction_ids"]) == len(result["tokens"])
        assert capture["token_predictions"] == [
            runtime.tokenizer.decode([token_id]) for token_id in capture["token_prediction_ids"]
        ]
        assert all(0 <= value <= 1 for value in capture["token_prediction_probabilities"])
        assert all(math.isfinite(value) and value >= 0 for value in capture["token_entropies"])
    json.dumps(result, allow_nan=False)
    assert all(not block._forward_hooks for block in runtime._blocks())
