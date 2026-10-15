import pytest
import torch

from circuit_lantern.services.metrics import matrix_metrics


def test_known_diagonal_matrix():
    result = matrix_metrics(torch.diag(torch.tensor([3.0, 2.0, 1.0])))
    assert result["singular_values"] == [3, 2, 1]
    assert result["spectral_radius"] == 3
    assert result["frobenius_norm"] == pytest.approx(14**0.5)
    assert result["stable_rank"] == pytest.approx(14 / 9)
    assert result["condition_number"] == 3
    assert result["cumulative_energy"][-1] == 1


def test_rectangular_matrix_has_no_spectral_radius():
    result = matrix_metrics(torch.ones(3, 4))
    assert result["spectral_radius"] is None
    assert result["shape"] == [3, 4]


def test_zero_matrix_stays_finite():
    result = matrix_metrics(torch.zeros(4, 4))
    assert result["stable_rank"] == 0
    assert result["condition_number"] is None
    assert result["sparsity"] == 1


def test_large_spectrum_is_explicitly_labeled_as_subsampled():
    result = matrix_metrics(torch.eye(800))
    assert result["spectrum_exact"] is False
    assert result["spectrum_stride"] == 2


def test_nonfinite_weights_rejected():
    with pytest.raises(ValueError, match="non finite"):
        matrix_metrics(torch.tensor([[float("nan")]]))
