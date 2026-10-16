import pytest
import torch

from circuit_lantern.services.scoring import log_probabilities, target_score, target_token_id


def test_large_logits_keep_finite_target_log_probability():
    values = log_probabilities(torch.tensor([1e20, -1e20]))
    assert torch.isfinite(values).all()
    score = target_score(values, 1)
    assert score["probability"] == 0
    assert score["log_probability"] < -1e20


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_are_rejected(value):
    with pytest.raises(ValueError, match="non finite logits"):
        log_probabilities(torch.tensor([value, 0]))


def test_target_outside_model_vocabulary_is_actionable(runtime):
    with pytest.raises(ValueError, match="outside this model vocabulary"):
        target_token_id(runtime.tokenizer, "France", torch.zeros(3))
