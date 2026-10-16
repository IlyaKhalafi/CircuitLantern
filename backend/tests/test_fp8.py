import math

import pytest
import torch

from circuit_lantern.services.fp8 import (
    ADAPTER_VERSION,
    BLOCK_SIZES,
    configuration_error,
    execution_probe,
    logical_fp8_weight,
    prepare_model,
)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("rows", [70, 128])
def test_logical_fp8_matrix_uses_each_block_scale_and_preserves_stored_codes(block, rows):
    columns = 256
    values = (torch.arange(rows * columns).reshape(rows, columns) % 16 - 8).float()
    weight = values.to(torch.float8_e4m3fn)
    scales = (
        torch.arange(
            1, math.ceil(rows / block[0]) * (columns // block[1]) + 1, dtype=torch.float32
        ).reshape(math.ceil(rows / block[0]), columns // block[1])
        / 8
    )
    codes = weight.view(torch.uint8).clone()
    original_scales = scales.clone()
    expected = torch.empty_like(values)
    for row in range(rows):
        for column in range(columns):
            expected[row, column] = (
                values[row, column] * scales[row // block[0], column // block[1]]
            )
    actual = logical_fp8_weight(weight, scales, block)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(weight.view(torch.uint8), codes)
    torch.testing.assert_close(scales, original_scales, rtol=0, atol=0)


def test_zero_scale_blocks_decode_to_exact_zero_even_with_nan_codes():
    weight = torch.ones(70, 128).to(torch.float8_e4m3fn)
    weight[:64, :64] = float("nan")
    scales = torch.ones(2, 2)
    scales[0, 0] = 0
    actual = logical_fp8_weight(weight, scales, (64, 64))
    expected = torch.ones(70, 128)
    expected[:64, :64] = 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_scales_are_rejected_before_an_analysis(value):
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scales = torch.tensor([[value]], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        logical_fp8_weight(weight, scales, (128, 128))


def test_nonfinite_codes_in_a_nonzero_weight_block_are_rejected():
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non finite codes"):
        logical_fp8_weight(weight, torch.ones(1, 1), (128, 128))


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "compressed-tensors"},
        {"quant_method": "fp8", "fmt": "e5m2"},
        {"quant_method": "fp8", "activation_scheme": "static"},
        {"quant_method": "fp8", "weight_block_size": [128.0, 128]},
        {"quant_method": "fp8", "weight_block_size": [128, 32]},
        {"quant_method": "fp8", "modules_to_not_convert": "lm_head"},
        {"quant_method": "fp8", "scale_fmt": "ue8m0"},
    ],
)
def test_incompatible_fp8_formats_do_not_claim_adapter_support(config):
    assert configuration_error(config)


@pytest.fixture
def native_fp8():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9:
        pytest.skip("A Hopper or newer CUDA GPU is required.")
    return pytest.importorskip("transformers.integrations.finegrained_fp8").FP8Linear


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_native_fp8_zero_groups_and_head_ablation_preserve_the_linear_operator(
    native_fp8, block, dtype
):
    module = native_fp8(256, 70, bias=True, block_size=block, device="cuda:0").eval()
    values = (torch.arange(70 * 256, device="cuda:0").reshape(70, 256) % 16 - 8).float()
    module.weight.copy_(values.to(torch.float8_e4m3fn))
    module.weight_scale_inv.fill_(0.125)
    module.bias = torch.nn.Parameter(torch.linspace(-1, 1, 70, device="cuda:0", dtype=dtype))
    inputs = torch.ones(2, 3, 256, device="cuda:0", dtype=dtype)
    original = module(inputs)
    codes = module.weight.view(torch.uint8).clone()
    scales = module.weight_scale_inv.clone()
    assert prepare_model(module) == 1
    torch.testing.assert_close(module(inputs), original, rtol=0, atol=0)
    inputs[0, 1, : block[1]] = 0
    inputs[1, 2] = 0
    expected = torch.nn.functional.linear(inputs.float(), values * 0.125, module.bias.float()).to(
        dtype
    )
    actual = module(inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert torch.equal(module.weight.view(torch.uint8), codes)
    torch.testing.assert_close(module.weight_scale_inv, scales, rtol=0, atol=0)


@torch.inference_mode()
def test_native_zero_weight_blocks_do_not_mutate_checkpoint_storage(native_fp8):
    module = native_fp8(128, 128, block_size=(64, 64), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight[:64, :64] = float("nan")
    module.weight_scale_inv.fill_(1)
    module.weight_scale_inv[0, 0] = 0
    before = module.weight.view(torch.uint8).clone()
    prepare_model(module)
    inputs = torch.ones(3, 128, device="cuda:0")
    inputs[-1] = 0
    expected_weight = torch.ones(128, 128, device="cuda:0")
    expected_weight[:64, :64] = 0
    torch.testing.assert_close(module(inputs), inputs @ expected_weight.T, rtol=1e-4, atol=1e-4)
    assert torch.equal(module.weight.view(torch.uint8), before)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_execution_probe_verifies_real_kernels(native_fp8, block):
    result = execution_probe(block)
    assert result["supported"], result
    assert result["adapter_version"] == ADAPTER_VERSION
    assert result["block_size"] == list(block)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_standard_fp8_configuration_and_default_settings_are_supported(block):
    assert configuration_error({"quant_method": "fp8"}) is None
    assert (
        configuration_error(
            {
                "quant_method": "fp8",
                "weight_block_size": list(block),
                "modules_to_not_convert": ["lm_head"],
            }
        )
        is None
    )


@pytest.mark.parametrize("block", [(0, 128), (128,), (128, 32)])
def test_unsupported_decoder_layouts_fail_before_arithmetic(block):
    with pytest.raises(ValueError, match="unsupported layout"):
        logical_fp8_weight(torch.ones(128, 128).to(torch.float8_e4m3fn), torch.ones(1, 1), block)


@torch.inference_mode()
def test_inventory_and_metrics_use_the_logical_fp8_operator(native_fp8):
    from circuit_lantern.services.metrics import matrix_metrics
    from circuit_lantern.services.weights import (
        logical_weight,
        model_weight_inventory,
        resident_tensor_bytes,
    )

    module = native_fp8(128, 70, block_size=(64, 64), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight_scale_inv.copy_(torch.tensor([[1, 2], [3, 4]], device="cuda:0"))
    model = torch.nn.Sequential(module, torch.nn.LayerNorm(70, device="cuda:0"))
    prepare_model(model)
    inventory, parameters = model_weight_inventory(model)
    assert [item["name"] for item in inventory] == ["0.weight"]
    assert parameters == 70 * 128 + 2 * 70
    assert inventory[0]["quantized"]
    assert inventory[0]["dtype"] == "fp8_e4m3"
    assert inventory[0]["shape"] == [70, 128]
    expected = torch.ones(70, 128, device="cuda:0")
    expected[:64, 64:] = 2
    expected[64:, :64] = 3
    expected[64:, 64:] = 4
    weight, description = logical_weight(model, "0.weight")
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    actual = matrix_metrics(weight, 1_000_000)
    reference = matrix_metrics(expected, 1_000_000)
    for key in ["mean", "std", "frobenius_norm", "spectral_norm"]:
        assert actual[key] == pytest.approx(reference[key])
    assert description["quantized"]
    assert resident_tensor_bytes(model)["cuda:0"] == 70 * 128 + 4 * 4 + 2 * 70 * 4
    with pytest.raises(ValueError, match="auxiliary data"):
        logical_weight(model, "0.weight_scale_inv")


def test_unrecognized_float8_storage_cannot_be_mistaken_for_dequantized_weights():
    from circuit_lantern.services.weights import logical_weight

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(4, 4).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="dequantization adapter"):
        logical_weight(model, "weight")


def test_fp8_decode_reports_float32_overflow_instead_of_returning_invalid_statistics():
    weight = torch.full((128, 128), 448).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="finite float32 analysis range"):
        logical_fp8_weight(weight, torch.tensor([[1e38]]), (128, 128))


@pytest.mark.parametrize(
    "kind", ["noncontiguous", "scale_dtype", "scale_device", "columns", "rank"]
)
def test_invalid_matrix_storage_returns_a_clear_layout_error(kind):
    weight = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scales = torch.ones(1, 1)
    if kind == "noncontiguous":
        weight = weight.T
    elif kind == "scale_dtype":
        scales = scales.to(torch.bfloat16)
    elif kind == "scale_device":
        scales = torch.ones(1, 1, device="meta")
    elif kind == "columns":
        weight = torch.ones(128, 70).to(torch.float8_e4m3fn)
    else:
        weight = weight.flatten()
    with pytest.raises(ValueError, match="unsupported layout"):
        logical_fp8_weight(weight, scales, (128, 128))


@torch.inference_mode()
def test_repreparing_fp8_after_new_parameter_values_updates_zero_block_handling(native_fp8):
    module = native_fp8(128, 128, block_size=(64, 64), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight_scale_inv.fill_(1)
    prepare_model(module)
    module.weight[:64, :64] = float("nan")
    module.weight_scale_inv[0, 0] = 0
    codes = module.weight.view(torch.uint8).clone()
    assert prepare_model(module) == 1
    values = torch.ones(1, 128, device="cuda:0")
    expected = torch.ones(1, 128, device="cuda:0") * 128
    expected[:, :64] = 64
    torch.testing.assert_close(module(values), expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(module.weight.view(torch.uint8), codes)


@torch.inference_mode()
def test_accelerate_dispatch_hooks_remain_active_after_fp8_preparation(native_fp8):
    from accelerate.hooks import ModelHook, add_hook_to_module

    calls = []

    class CaptureHook(ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            calls.append("before")
            return args, kwargs

        def post_forward(self, module, output):
            calls.append("after")
            return output

    module = native_fp8(128, 128, block_size=(128, 128), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight_scale_inv.fill_(1)
    add_hook_to_module(module, CaptureHook())
    wrapper = module.forward
    prepare_model(module)
    inputs = torch.zeros(1, 128, device="cuda:0")
    torch.testing.assert_close(module(inputs), torch.zeros_like(inputs), rtol=0, atol=0)
    prepare_model(module)
    torch.testing.assert_close(module(inputs), torch.zeros_like(inputs), rtol=0, atol=0)
    assert module.forward is wrapper
    assert calls == ["before", "after", "before", "after"]


def test_failed_fp8_probe_is_not_cached_and_a_successful_probe_is_cached():
    from unittest.mock import patch

    from circuit_lantern.services import fp8

    class BackendError(Exception):
        pass

    fp8._verified_probe.cache_clear()
    try:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch.object(
                fp8,
                "_probe_kernel",
                side_effect=[
                    BackendError("temporary resource error"),
                    {"triton_version": "verified-test-runtime"},
                ],
            ) as kernel,
        ):
            first = execution_probe()
            assert not first["supported"]
            assert "temporary resource error" in first["reason"]
            assert fp8._verified_probe.cache_info().currsize == 0
            recovered = execution_probe()
            assert recovered["supported"]
            recovered["supported"] = False
            assert execution_probe()["supported"]
            assert kernel.call_count == 2
    finally:
        fp8._verified_probe.cache_clear()


def test_fp8_capability_probe_does_not_change_random_generator_states(native_fp8):
    from circuit_lantern.services import fp8

    fp8._verified_probe.cache_clear()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(0).clone()
    assert execution_probe()["supported"]
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_state)


@pytest.mark.parametrize("magnitude", [1e-45, 1e-42, 1e-38, 1e-30])
@torch.inference_mode()
def test_subnormal_activation_groups_do_not_create_nonfinite_outputs(native_fp8, magnitude):
    module = native_fp8(128, 128, block_size=(128, 128), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight_scale_inv.fill_(1)
    prepare_model(module)
    output = module(torch.full((1, 128), magnitude, device="cuda:0"))
    assert torch.isfinite(output).all()
    if magnitude >= 1e-38:
        torch.testing.assert_close(
            output, torch.full_like(output, 128 * magnitude), rtol=1e-3, atol=0
        )


@torch.inference_mode()
def test_empty_linear_inputs_retain_their_output_shape(native_fp8):
    module = native_fp8(128, 70, block_size=(128, 128), device="cuda:0").eval()
    module.weight.fill_(1)
    module.weight_scale_inv.fill_(1)
    prepare_model(module)
    result = module(torch.empty(0, 128, device="cuda:0"))
    assert result.shape == (0, 70)
    assert result.dtype == torch.float32


def test_device_probe_errors_are_reported_without_crashing_a_hardware_check():
    from unittest.mock import patch

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", side_effect=RuntimeError("device unavailable")),
    ):
        report = execution_probe([128, 128])
    assert not report["supported"]
    assert report["reason"] == "device unavailable"
    assert report["block_size"] == [128, 128]
