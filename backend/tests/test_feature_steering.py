import pytest
import torch
from pydantic import ValidationError

from circuit_lantern.feature_schemas import FeatureInspectionRequest
from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.inference import feature_intervention


@pytest.fixture
def dictionary():
    model = SparseAutoencoder(2, TrainingConfig(features=2, active_features=1))
    with torch.no_grad():
        model.decoder.copy_(torch.tensor([[1.0, 0.6], [0.0, 0.8]]))
        model.encoder.zero_()
        model.encoder_bias.copy_(torch.tensor([1.0, 0.0]))
        model.center.copy_(torch.tensor([0.25, -0.125]))
        model.input_scale.fill_(2)
    return model


@pytest.mark.parametrize("coefficient", [-3.0, 0.0, 2.5])
@pytest.mark.parametrize("position", [None, 2])
@torch.inference_mode()
def test_fixed_coefficient_preserves_original_error_and_untouched_positions(
    dictionary, coefficient, position
):
    hidden = torch.tensor([[[2.0, 4.0], [-1.0, 6.0], [7.0, 5.0]]])
    original = hidden.clone()
    encoded = dictionary.encode(hidden[0])
    error = hidden[0] - dictionary.decode(encoded)
    selected = torch.tensor([True, False, True])
    if position is not None:
        selected &= torch.arange(3) == position
    modified = encoded.clone()
    modified[selected, 0] = coefficient
    expected = dictionary.decode(modified) + error
    operation = feature_intervention(
        dictionary, 0, 0, position, torch.tensor([True, False, True]), coefficient=coefficient
    )
    changed = operation(hidden)
    torch.testing.assert_close(changed[0], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(changed[0][~selected], original[0][~selected], atol=0, rtol=0)
    torch.testing.assert_close(hidden, original, atol=0, rtol=0)
    assert operation.measurement["intended_delta_norm"] > 0


@torch.inference_mode()
def test_fixed_value_can_activate_a_zero_coefficient_without_changing_other_components(dictionary):
    hidden = torch.tensor([[[2.0, 4.0], [7.0, 5.0]]])
    assert not dictionary.encode(hidden[0])[:, 1].any()
    scaled = feature_intervention(dictionary, 1, 2, None)(hidden)
    changed = feature_intervention(dictionary, 1, 0, None, coefficient=2.5)(hidden)
    torch.testing.assert_close(scaled, hidden, atol=0, rtol=0)
    expected = hidden + 2.5 * dictionary.feature_direction(1)
    torch.testing.assert_close(changed, expected)
    # The fixed decomposition does not claim an inverse of the nonorthogonal encoder.
    assert not dictionary.encode(changed[0])[:, 1].any()


@torch.inference_mode()
def test_zero_fixed_coefficient_equals_feature_ablation(dictionary):
    hidden = torch.tensor([[[2.0, 4.0], [7.0, 5.0]]])
    scaled = feature_intervention(dictionary, 0, 0, None)(hidden)
    fixed = feature_intervention(dictionary, 0, 0, None, coefficient=0)(hidden)
    torch.testing.assert_close(scaled, fixed, atol=0, rtol=0)


@pytest.mark.parametrize("coefficient", [-2.5, 0, 3])
def test_set_request_round_trip_keeps_an_unambiguous_mode(coefficient):
    request = FeatureInspectionRequest(
        prompt="Example", feature=0, intervention_mode="set", coefficient=coefficient
    )
    assert FeatureInspectionRequest.model_validate(request.model_dump()) == request
    assert FeatureInspectionRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "changes",
    [
        {"coefficient": 2},
        {"intervention_mode": "set"},
        {"intervention_mode": "set", "coefficient": 2},
        {"feature": 0, "intervention_mode": "set", "coefficient": 2, "multiplier": 2},
        *[
            {"feature": 0, "intervention_mode": "set", "coefficient": value}
            for value in (True, "2", float("nan"), float("inf"), -10001, 10001)
        ],
    ],
)
def test_invalid_or_ambiguous_coefficient_requests_fail_validation(changes):
    with pytest.raises(ValidationError):
        FeatureInspectionRequest(prompt="Example", **changes)


@torch.inference_mode()
def test_rounding_norm_does_not_underflow_for_a_tiny_representable_update(dictionary):
    hidden = torch.tensor([[[2.0, 4.0], [7.0, 5.0]]])
    operation = feature_intervention(dictionary, 1, 0, None, coefficient=1e-30)
    changed = operation(hidden)
    torch.testing.assert_close(changed, hidden, atol=0, rtol=0)
    assert operation.measurement["intended_delta_norm"] > 0
    assert operation.measurement["realized_delta_norm"] == 0
    assert operation.measurement["relative_rounding_error"] == pytest.approx(1)
