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


def test_large_spectrum_uses_full_matrix_and_preserves_total_energy():
    initial_rng = torch.get_rng_state()
    result = matrix_metrics(torch.eye(800))
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert result["spectrum_exact"] is False
    assert result["spectrum_shape"] == [800, 800]
    assert result["spectral_norm"] == pytest.approx(1, rel=1e-5)
    assert result["stable_rank"] == pytest.approx(800, rel=1e-5)
    assert result["condition_number"] is None
    assert result["cumulative_energy"][-1] == pytest.approx(64 / 800, rel=1e-5)


def test_nonfinite_weights_rejected():
    with pytest.raises(ValueError, match="non finite"):
        matrix_metrics(torch.tensor([[float("nan")]]))


@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
def test_spectral_diagnostics_are_scale_invariant(scale):
    result = matrix_metrics(torch.diag(torch.tensor([3.0, 2.0, 1.0])) * scale)
    assert result["condition_number"] == pytest.approx(3, rel=1e-6)
    assert result["stable_rank"] == pytest.approx(14 / 9, rel=1e-6)
    assert result["cumulative_energy"][-1] == pytest.approx(1, rel=1e-6)
    assert result["numerical_rank"] == 3
    assert result["effective_rank"] == pytest.approx(2.749459273997205, rel=1e-6)
    assert result["nuclear_norm"] == pytest.approx(6 * scale, rel=1e-6)


def test_operator_norm_is_distinct_from_entrywise_l1():
    result = matrix_metrics(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]))
    assert result["l1_norm"] == 10
    assert result["operator_1_norm"] == 6
    assert result["infinity_norm"] == 7


@pytest.mark.parametrize("value", [0.0, 1e-20, 1e20])
def test_constant_matrices_have_zero_variance_and_valid_histograms(value):
    result = matrix_metrics(torch.full((3, 4), value))
    assert result["std"] == 0
    assert sum(bin["count"] for bin in result["histogram"]) == 12
    assert result["numerical_rank"] == (0 if value == 0 else 1)
    assert result["condition_number"] is None
    assert result["effective_rank"] == (None if value == 0 else pytest.approx(1))


def test_large_tiny_spectra_keep_the_full_frobenius_energy_denominator():
    result = matrix_metrics(torch.eye(800) * 1e-20)
    assert result["stable_rank"] == pytest.approx(800, rel=1e-5)
    assert result["cumulative_energy"][-1] == pytest.approx(64 / 800, rel=1e-5)
    assert result["nuclear_norm"] is None
    assert result["numerical_rank"] is None
