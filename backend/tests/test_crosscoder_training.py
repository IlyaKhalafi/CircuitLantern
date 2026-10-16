"""Numerical oracles for shared features, bank units and joint optimization."""

from dataclasses import replace

import pytest
import torch

from circuit_lantern.services.crosscoders.model import CrosscoderConfig, SparseCrosscoder
from circuit_lantern.services.crosscoders.training import train_crosscoder, validate_banks


def paired_data():
    generator = torch.Generator().manual_seed(13)
    latent = torch.relu(torch.randn(96, 2, generator=generator))
    banks = [latent @ torch.randn(2, width, generator=generator) for width in (3, 5)]
    return [value[:72] for value in banks], [value[72:] for value in banks]


def test_sparsity_uses_sum_of_bank_norms_and_handles_zero_bank_gradients():
    model = SparseCrosscoder((2, 1), CrosscoderConfig(features=3))
    with torch.no_grad():
        model.decoders[0].copy_(torch.tensor([[3.0, 0, 3], [0, 0, 0]]))
        model.decoders[1].copy_(torch.tensor([[0.0, 4, 4]]))
    features = torch.tensor([[2.0, 1, 3]], requires_grad=True)
    loss = model.sparsity_loss(features)
    assert loss.item() == 31
    concatenated = torch.linalg.vector_norm(torch.cat(list(model.decoders)), dim=0)
    assert (features * concatenated).sum().item() == 25
    loss.backward()
    assert all(torch.isfinite(decoder.grad).all() for decoder in model.decoders)
    torch.testing.assert_close(features.grad, torch.tensor([[3.0, 4, 7]]))


def test_joint_initialization_and_local_seed_do_not_change_global_rng():
    state = torch.random.get_rng_state()
    model = SparseCrosscoder((3, 5), CrosscoderConfig(features=8, seed=11))
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(
        model.decoder_norms().square().sum(dim=0),
        torch.ones(8, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    other = SparseCrosscoder((3, 5), model.config)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(a, b)


def test_model_units_and_positive_feature_scaling_preserve_reconstruction_and_penalty():
    train, _ = paired_data()
    model = SparseCrosscoder((3, 5), CrosscoderConfig(features=8, seed=7))
    with torch.no_grad():
        model.input_scales.copy_(torch.tensor([2.0, 5.0]))
        model.centers[0].fill_(0.25)
        model.centers[1].fill_(-0.4)
        model.encoder_bias.fill_(0.3)
    values = [value[:4].clone() for value in train]
    before = [value.clone() for value in values]
    outputs, features = model(values)
    expected_features = torch.relu(
        model.encoder_bias
        + sum(
            (value / model.input_scales[bank] - model.centers[bank]) @ model.encoders[bank].T
            for bank, value in enumerate(values)
        )
    )
    torch.testing.assert_close(features, expected_features)
    for bank in range(2):
        expected = (features @ model.decoders[bank].T + model.centers[bank]) * model.input_scales[
            bank
        ]
        torch.testing.assert_close(outputs[bank], expected)
        torch.testing.assert_close(
            model.feature_direction(bank, 2), model.decoders[bank][:, 2] * model.input_scales[bank]
        )
    penalty = model.sparsity_loss(features).detach()
    factors = torch.linspace(0.25, 3, 8)
    with torch.no_grad():
        for encoder, decoder in zip(model.encoders, model.decoders, strict=True):
            encoder.div_(factors[:, None])
            decoder.mul_(factors[None, :])
        model.encoder_bias.div_(factors)
    changed, scaled_features = model(values)
    for a, b in zip(outputs, changed, strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(penalty, model.sparsity_loss(scaled_features))
    assert all(torch.equal(a, b) for a, b in zip(values, before, strict=True))


def test_geometry_keeps_tiny_opposed_directions_and_zero_banks():
    model = SparseCrosscoder((2, 2), CrosscoderConfig(features=4))
    with torch.no_grad():
        model.decoders[0].copy_(torch.tensor([[1.0, 0, 1e-30, 0], [0, 0, 0, 0]]))
        model.decoders[1].copy_(torch.tensor([[1.0, 2, -1e-30, 0], [0, 0, 0, 0]]))
    rows = model.geometry()
    assert rows[0]["decoder_cosine"] == pytest.approx(1)
    assert rows[1]["decoder_cosine"] is None
    assert rows[1]["relative_decoder_norms"] == [0, 1]
    assert rows[2]["decoder_cosine"] == pytest.approx(-1)
    assert rows[2]["decoder_norms"][0] > 0
    assert rows[2]["relative_decoder_norms"] == [0.5, 0.5]
    assert rows[3]["relative_decoder_norms"] is None
    assert rows[3]["decoder_cosine"] is None
    different = SparseCrosscoder((2, 3), CrosscoderConfig(features=4))
    assert all(row["decoder_cosine"] is None for row in different.geometry())


def test_model_explicit_float32_and_metadata_only_loading():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = SparseCrosscoder((2, 3), CrosscoderConfig(features=4))
    finally:
        torch.set_default_dtype(original)
    assert all(value.dtype == torch.float32 for value in model.state_dict().values())
    unloaded = SparseCrosscoder(model.widths, model.config, initialize=False)
    assert all(value.device.type == "meta" for value in unloaded.state_dict().values())
    unloaded.load_state_dict(model.state_dict(), assign=True)
    inputs = [torch.ones(2, width) for width in model.widths]
    for a, b in zip(model(inputs)[0], unloaded(inputs)[0], strict=True):
        assert torch.equal(a, b)


def test_training_learns_without_changing_inputs_or_global_rng():
    train, validation = paired_data()
    originals = [value.clone() for value in [*train, *validation]]
    config = CrosscoderConfig(
        features=10, steps=80, batch_size=24, learning_rate=0.01, l1_coefficient=0.01, seed=5
    )
    state = torch.random.get_rng_state()
    result = train_crosscoder(train, validation, config)
    assert torch.equal(state, torch.random.get_rng_state())
    assert result.completed_steps == 80
    assert result.tokens_seen == 80 * 24
    assert not result.cancelled
    assert result.history[-1]["validation_normalized_l2_error"] < (
        result.history[0]["validation_normalized_l2_error"] * 0.75
    )
    assert result.validation_metrics["samples"] == 24
    assert len(result.validation_metrics["banks"]) == 2
    for actual, original in zip([*train, *validation], originals, strict=True):
        assert torch.equal(actual, original)


def test_preprocessing_and_optimization_never_fit_held_out_activations():
    train, validation = paired_data()
    config = CrosscoderConfig(features=8, steps=3, batch_size=24, seed=9)
    first = train_crosscoder(train, validation, config)
    second = train_crosscoder(train, [value * 100 for value in validation], config)
    for a, b in zip(
        first.model.state_dict().values(), second.model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    expected = torch.tensor(
        [
            torch.linalg.vector_norm(value, dtype=torch.float64) / len(value) ** 0.5
            for value in train
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(first.model.input_scales, expected)


def test_constant_bank_remains_available_with_undefined_variance_metrics():
    train, validation = paired_data()
    train[0].zero_()
    validation[0].zero_()
    result = train_crosscoder(train, validation, CrosscoderConfig(features=8, steps=2))
    assert result.model.input_scales[0] == 1
    assert result.validation_metrics["banks"][0]["normalized_mse"] is None
    assert result.validation_metrics["banks"][0]["explained_variance"] is None


def test_cancellation_keeps_exact_completed_step_evidence():
    train, validation = paired_data()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 3

    result = train_crosscoder(
        train,
        validation,
        CrosscoderConfig(features=8, steps=100, batch_size=8),
        cancelled=cancelled,
    )
    assert result.cancelled and result.completed_steps == 3
    assert result.tokens_seen == 24
    assert result.history[-1]["step"] == 3
    assert result.history[-1]["tokens_seen"] == 24


@pytest.mark.parametrize(
    "change",
    [
        {"features": True},
        {"steps": 0},
        {"batch_size": True},
        {"seed": -1},
        {"learning_rate": 0},
        {"l1_coefficient": float("nan")},
        {"l1_coefficient": "0.1"},
    ],
)
def test_invalid_training_configurations_are_rejected(change):
    with pytest.raises(ValueError):
        replace(CrosscoderConfig(features=8), **change).validate((3, 5))


def test_invalid_bank_alignment_and_nonfinite_values_are_rejected():
    train, validation = paired_data()
    with pytest.raises(ValueError, match="matching sample"):
        validate_banks([train[0][:-1], train[1]], validation)
    with pytest.raises(ValueError, match="float32"):
        validate_banks([train[0].double(), train[1]], validation)
    validation[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_banks(train, validation)


def test_unrepresentable_training_scale_is_rejected_before_optimization():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    first = torch.zeros(4, 1)
    first[0] = tiny
    train = [first, torch.arange(4, dtype=torch.float32).reshape(-1, 1)]
    with pytest.raises(ValueError, match="float32 range"):
        train_crosscoder(train, [value[:1] for value in train], CrosscoderConfig(features=2))
