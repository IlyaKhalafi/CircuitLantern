import json

import numpy as np
import pytest
import torch

from circuit_lantern.services.features.comparison import feature_pair_statistics


def test_statistics_match_independent_numpy_reductions():
    rng = np.random.default_rng(92)
    activations = rng.lognormal(size=(23, 2)).astype(np.float32)
    activations[::3, 0] = 0
    activations[::4, 1] = 0
    decoder = rng.normal(size=(11, 2)).astype(np.float32)
    result = feature_pair_statistics(torch.from_numpy(activations), torch.from_numpy(decoder))
    values, directions = activations.astype(np.float64), decoder.astype(np.float64)

    def cosine(matrix):
        return np.dot(matrix[:, 0], matrix[:, 1]) / (
            np.linalg.norm(matrix[:, 0]) * np.linalg.norm(matrix[:, 1])
        )

    assert result["eligible_position_count"] == 23
    assert result["active_counts"] == (activations > 0).sum(axis=0).tolist()
    active = activations > 0
    assert result["coactive_count"] == int(active.all(axis=1).sum())
    assert result["activation_overlap"] == active.all(axis=1).sum() / active.any(axis=1).sum()
    assert result["activation_cosine"] == pytest.approx(cosine(values), abs=1e-14)
    assert result["pearson_correlation"] == pytest.approx(np.corrcoef(values.T)[0, 1], abs=1e-14)
    assert result["decoder_cosine"] == pytest.approx(cosine(directions), abs=1e-14)
    json.dumps(result, allow_nan=False)


def test_decoder_geometry_and_activation_correlation_are_distinct():
    aligned = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    disjoint = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = feature_pair_statistics(disjoint, aligned)
    assert result["decoder_cosine"] == 1
    assert result["pearson_correlation"] == pytest.approx(-1)
    assert result["activation_cosine"] == 0
    assert result["activation_overlap"] == 0
    opposed = aligned * torch.tensor([1.0, -1.0])
    identical = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    result = feature_pair_statistics(identical, opposed)
    assert result["decoder_cosine"] == -1
    assert result["pearson_correlation"] == pytest.approx(1)
    assert result["activation_overlap"] == 1


@pytest.mark.parametrize("count", [0, 1, 4])
def test_empty_and_inactive_samples_report_undefined_statistics(count):
    result = feature_pair_statistics(torch.zeros(count, 2), torch.eye(2))
    assert result["eligible_position_count"] == count
    assert result["active_counts"] == [0, 0]
    assert result["coactive_count"] == 0
    assert result["pearson_correlation"] is None
    assert result["activation_cosine"] is None
    assert result["activation_overlap"] is None
    assert result["decoder_cosine"] == 0
    json.dumps(result, allow_nan=False)


def test_constant_activity_and_single_position_have_no_pearson_correlation():
    values = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    result = feature_pair_statistics(values, torch.eye(2))
    assert result["pearson_correlation"] is None
    assert result["activation_cosine"] > 0
    single = feature_pair_statistics(values[:1], torch.zeros(2, 2))
    assert single["pearson_correlation"] is None
    assert single["activation_cosine"] == 1
    assert single["activation_overlap"] == 1
    assert single["decoder_cosine"] is None


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_nonzero_scales_do_not_become_zero_variance(scale):
    values = torch.tensor([[0.0, 2.0], [1.0, 4.0], [2.0, 1.0], [3.0, 0.0]])
    reference = feature_pair_statistics(values, torch.eye(2))
    scaled = feature_pair_statistics(values * scale, torch.eye(2))
    for key in ("pearson_correlation", "activation_cosine", "activation_overlap"):
        assert scaled[key] == pytest.approx(reference[key], abs=1e-7)


@pytest.mark.parametrize(
    "values,directions",
    [
        (torch.ones(2, 1), torch.eye(2)),
        (torch.ones(2, 2), torch.ones(0, 2)),
        (torch.ones(2, 2), torch.ones(2, 3)),
        (torch.tensor([[float("nan"), 1.0]]), torch.eye(2)),
        (torch.ones(2, 2), torch.tensor([[1.0, float("inf")]])),
    ],
)
def test_malformed_or_nonfinite_pair_inputs_are_rejected(values, directions):
    with pytest.raises(ValueError, match="finite two-column"):
        feature_pair_statistics(values, directions)
