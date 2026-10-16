import dataclasses
import math

import pytest
import torch

from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.training import evaluate_dictionary, train_dictionary


def synthetic_activations():
    generator = torch.Generator().manual_seed(123)
    directions = torch.linalg.qr(torch.randn(8, 8, generator=generator))[0]
    values = torch.rand(640, 8, generator=generator)
    values *= torch.rand(640, 8, generator=generator).gt(0.8)
    return values @ directions.T


@pytest.mark.parametrize("method", ["relu", "topk"])
def test_training_learns_sparse_data_and_generalizes_to_held_out_vectors(method):
    data = synthetic_activations()
    result = train_dictionary(
        data[:512],
        data[512:],
        TrainingConfig(
            features=32,
            method=method,
            active_features=2,
            steps=160,
            batch_size=64,
            learning_rate=0.01,
            l1_coefficient=0.003,
            seed=42,
            dead_after_tokens=128,
        ),
    )
    assert result.validation_metrics["explained_variance"] > 0.8
    assert result.validation_metrics["mse"] < result.history[0]["validation_mse"] / 4
    assert result.completed_steps == 160
    assert result.tokens_seen == 160 * 64
    assert not result.cancelled
    torch.testing.assert_close(result.model.decoder.norm(dim=0), torch.ones(32))
    if method == "topk":
        assert result.validation_metrics["mean_active_features"] <= 2
        assert any(row["auxiliary_loss"] > 0 for row in result.history)
        assert result.auxiliary_steps > 0


def test_training_is_reproducible_without_consuming_the_language_model_rng():
    data = synthetic_activations()
    config = TrainingConfig(features=16, steps=12, batch_size=32, seed=19)
    before = torch.get_rng_state().clone()
    first = train_dictionary(data[:512], data[512:], config)
    second = train_dictionary(data[:512], data[512:], config)
    assert torch.equal(before, torch.get_rng_state())
    assert first.history == second.history
    for key, tensor in first.model.state_dict().items():
        torch.testing.assert_close(tensor, second.model.state_dict()[key], rtol=0, atol=0)


def test_validation_values_cannot_change_training_weights_or_normalization():
    data = synthetic_activations()
    config = TrainingConfig(features=16, steps=12, batch_size=32, seed=19)
    first = train_dictionary(data[:512], data[512:], config)
    second = train_dictionary(data[:512], data[512:] + 50, config)
    for key, tensor in first.model.state_dict().items():
        torch.testing.assert_close(tensor, second.model.state_dict()[key], rtol=0, atol=0)
    assert first.validation_metrics != second.validation_metrics


def test_topk_is_nonnegative_and_can_have_fewer_than_k_active_features():
    model = SparseAutoencoder(4, TrainingConfig(features=8, method="topk", active_features=3))
    pre = torch.tensor([[-8.0, -7, -6, -5, -4, -3, -2, -1], [1, 2, 3, 4, 5, 6, 7, 8]])
    features = model.activate(pre)
    assert torch.equal(features[0], torch.zeros(8))
    assert torch.equal(features[1], torch.tensor([0.0, 0, 0, 0, 0, 6, 7, 8]))
    assert model.encoder_bias is None


def test_feature_direction_is_in_original_model_units_and_preserves_other_components():
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    with torch.no_grad():
        model.input_scale.fill_(7.5)
        model.center.fill_(0.25)
    value = torch.tensor([[0.5, -0.2, 0.4, 1.0]])
    features = model.encode(value)
    altered = features.clone()
    altered[:, 2] = 3
    delta = (altered[:, 2] - features[:, 2]).unsqueeze(-1) * model.feature_direction(2)
    torch.testing.assert_close(model.decode(altered) - model.decode(features), delta)
    with pytest.raises(ValueError, match="outside"):
        model.feature_direction(8)


def test_cancellation_keeps_measured_steps_and_does_not_claim_completion():
    data = synthetic_activations()
    last_step = 0

    def progress(row):
        nonlocal last_step
        last_step = row["step"]

    result = train_dictionary(
        data[:512],
        data[512:],
        TrainingConfig(features=16, steps=24, batch_size=32),
        progress=progress,
        cancelled=lambda: last_step >= 4,
    )
    assert result.cancelled
    assert result.completed_steps == 4
    assert result.tokens_seen == 128
    assert result.history[-1]["step"] == 4


def test_zero_variance_evaluation_is_unavailable_without_clamping_negative_explained_variance():
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    values = torch.ones(3, 4)
    metrics = evaluate_dictionary(model, values, values[0])
    assert metrics["normalized_mse"] is None
    assert metrics["explained_variance"] is None
    with torch.no_grad():
        model.center.fill_(10)
        model.encoder.zero_()
    metrics = evaluate_dictionary(model, values, torch.zeros(4))
    assert metrics["explained_variance"] < 0


@pytest.mark.parametrize(
    "changes",
    [
        {"features": 2},
        {"method": "unknown"},
        {"steps": 0},
        {"batch_size": 0},
        {"learning_rate": math.nan},
        {"learning_rate": 0},
        {"l1_coefficient": -1},
        {"method": "topk", "active_features": 99},
        {"auxiliary_coefficient": math.inf},
        {"dead_after_tokens": 0},
        {"seed": -1},
    ],
)
def test_invalid_training_options_are_rejected_before_allocation(changes):
    with pytest.raises(ValueError):
        SparseAutoencoder(4, dataclasses.replace(TrainingConfig(features=8), **changes))


@pytest.mark.parametrize("invalid", ["empty", "nan", "constant", "wrong_dtype", "wrong_width"])
def test_invalid_activation_data_is_rejected(invalid):
    data = synthetic_activations()
    train, validation = data[:512], data[512:]
    if invalid == "empty":
        train = train[:0]
    elif invalid == "nan":
        train = train.clone()
        train[0, 0] = math.nan
    elif invalid == "constant":
        train = torch.ones_like(train)
    elif invalid == "wrong_dtype":
        train = train.double()
    else:
        validation = validation[:, :2]
    with pytest.raises(ValueError):
        train_dictionary(train, validation, TrainingConfig(features=16, steps=2))


def test_full_batches_keep_step_and_token_exposure_counts_consistent():
    data = synthetic_activations()
    result = train_dictionary(
        data[:10],
        data[512:],
        TrainingConfig(features=16, steps=3, batch_size=6),
    )
    assert result.tokens_seen == 18


def test_decoder_projection_removes_radial_gradient_components():
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    gradient = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    model.decoder.grad = gradient.clone()
    model.project_decoder_gradient()
    torch.testing.assert_close(
        (model.decoder.grad * model.decoder).sum(dim=0),
        torch.zeros(8),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize(
    "config",
    [
        TrainingConfig(features=8.0),
        TrainingConfig(features=8, steps=True),
        TrainingConfig(features=8, learning_rate=True),
    ],
)
def test_saved_configuration_rejects_non_numeric_types(config):
    with pytest.raises(ValueError):
        SparseAutoencoder(4, config)
