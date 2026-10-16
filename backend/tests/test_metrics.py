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


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
def test_narrow_matrix_spectra_are_complete_and_retain_rank_measures(transpose, scale):
    matrix = torch.zeros(1000, 3)
    matrix[0, 0], matrix[400, 1], matrix[999, 2] = 3, 2, 1
    matrix *= scale
    if transpose:
        matrix = matrix.T
    result = matrix_metrics(matrix)
    assert result["spectrum_exact"]
    assert result["spectrum_precision"] == "float64"
    assert result["spectral_radius"] is None
    assert result["singular_values"] == pytest.approx([3 * scale, 2 * scale, scale], rel=1e-6)
    assert result["condition_number"] == pytest.approx(3, rel=1e-6)
    assert result["numerical_rank"] == 3
    assert result["effective_rank"] == pytest.approx(2.749459273997205, rel=1e-6)
    assert result["nuclear_norm"] == pytest.approx(6 * scale, rel=1e-6)
    assert result["cumulative_energy"][-1] == pytest.approx(1)


def test_narrow_zero_matrix_has_a_complete_spectrum_and_no_condition_number():
    result = matrix_metrics(torch.zeros(1000, 3))
    assert result["spectrum_exact"]
    assert result["numerical_rank"] == 0
    assert result["nuclear_norm"] == 0
    assert result["condition_number"] is result["effective_rank"] is None
    assert result["cumulative_energy"] == [0, 0, 0]


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_narrow_exact_spectrum_respects_the_workspace_budget(monkeypatch, threads):
    from circuit_lantern.services import metrics

    monkeypatch.setattr(metrics, "EXACT_THIN_SVD_MAX_ELEMENTS", 2000)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        result = matrix_metrics(torch.ones(1000, 3))
    finally:
        torch.set_num_threads(previous_threads)
    assert not result["spectrum_exact"]
    assert result["condition_number"] is result["numerical_rank"] is None
    # Float32 randomized QR rounding varies across BLAS kernels and thread counts.
    assert result["spectral_norm"] == pytest.approx(3000**0.5, rel=1e-4)
    assert result["spectrum_method"] == (
        "Full matrix randomized SVD with 3 components and 4 power iterations"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_narrow_full_svd_covers_all_sixty_four_components(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not installed")
    matrix = torch.zeros(1200, 64, device=device)
    matrix[:64] = torch.diag(torch.arange(1, 65, device=device))
    result = matrix_metrics(matrix)
    assert result["spectrum_exact"]
    assert result["numerical_rank"] == 64
    assert len(result["singular_values"]) == 64
    assert result["condition_number"] == pytest.approx(64)
    assert result["nuclear_norm"] == pytest.approx(2080)
    assert result["cumulative_energy"][-1] == pytest.approx(1)
