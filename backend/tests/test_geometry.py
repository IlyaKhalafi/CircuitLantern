import pytest
import torch

from circuit_lantern.services.geometry import ResidualGeometry


def test_linear_cka_is_invariant_to_rotation_translation_and_uniform_scale():
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    geometry = ResidualGeometry()
    geometry.observe(values, "Embedding")
    geometry.observe(3 * values @ rotation + torch.tensor([20.0, -3.0]), "Layer 0")
    result = geometry.finish()
    assert result["cka"][0] == pytest.approx([1, 1])
    assert result["cka"][1] == pytest.approx([1, 1])


def test_orthogonal_centered_sample_directions_have_zero_linear_cka():
    geometry = ResidualGeometry()
    geometry.observe(torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]), "A")
    geometry.observe(torch.tensor([[1.0], [1.0], [-1.0], [-1.0]]), "B")
    result = geometry.finish()
    assert result["cka"][0] == pytest.approx([1, 0])
    assert result["cka"][1] == pytest.approx([0, 1])


def test_degenerate_layers_and_zero_token_vectors_are_undefined():
    geometry = ResidualGeometry()
    geometry.observe(torch.ones(4, 2), "Constant")
    geometry.observe(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), "Varied")
    result = geometry.finish()
    assert result["undefined_layers"] == [0]
    assert result["cka"][0] == [None, None]
    assert result["cka"][1][1] == pytest.approx(1)
    assert result["token_cosine"][0] == pytest.approx([1, -1, None, 0], nan_ok=True)
    assert result["token_cosine"][2] == [None] * 4


@pytest.mark.parametrize("tokens", [1, 2])
def test_insufficient_samples_do_not_produce_misleading_cka(tokens):
    geometry = ResidualGeometry()
    geometry.observe(torch.arange(tokens * 4).reshape(tokens, 4), "Embedding")
    assert geometry.finish()["cka"] == [[None]]


def test_residual_updates_and_sampling_preserve_full_prompt_statistics():
    values = torch.arange(20).reshape(10, 2).float()
    geometry = ResidualGeometry(max_positions=4)
    assert geometry.observe(values, "Embedding")["update_norm"] is None
    unchanged = geometry.observe(values.clone(), "Bypassed")
    assert unchanged["update_norm"] == 0
    assert unchanged["relative_update"] == 0
    changed = geometry.observe(values + 2, "Updated")
    assert changed["update_norm"] == pytest.approx(8**0.5)
    assert changed["update_rms"] == 2
    assert geometry.finish()["sampled_positions"] == [0, 3, 6, 9]


def test_nonfinite_hidden_states_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        ResidualGeometry().observe(torch.tensor([[float("nan")]]), "Embedding")


def test_small_constant_layers_and_extreme_finite_scales_stay_well_defined():
    constant = ResidualGeometry()
    constant.observe(torch.full((3, 2), 0.1), "Constant")
    assert constant.finish()["cka"] == [[None]]
    large = torch.tensor([[1e30, 0], [-1e30, 0], [0, 1e30]])
    geometry = ResidualGeometry()
    result = geometry.observe(large, "Large")
    assert result["rms"] == pytest.approx((large.double().square().mean().sqrt()).item())
    geometry.observe(large * 0.5, "Smaller")
    assert geometry.finish()["cka"][0] == pytest.approx([1, 1], abs=1e-6)


def test_residual_cosine_is_scale_invariant_and_zero_vectors_are_undefined():
    geometry = ResidualGeometry()
    values = torch.tensor([[1e-30, 0], [0, 1e-30], [-1e-30, 0]])
    geometry.observe(values, "Tiny")
    assert geometry.observe(values * 2, "Larger")["cosine_similarity"] == pytest.approx(1)
    assert geometry.observe(torch.zeros_like(values), "Zero")["cosine_similarity"] is None
