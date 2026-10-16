import logging
from unittest.mock import patch

import pytest

from circuit_lantern.services.execution import _CastNoticeFilter


def notice(dtype):
    return logging.makeLogRecord(
        {
            "msg": "MatMul8bitLt: inputs will be cast from %s to float16 during quantization",
            "args": (dtype,),
        }
    )


def test_expected_cast_notices_are_emitted_once_per_input_dtype():
    log_filter = _CastNoticeFilter()
    assert log_filter.filter(notice("torch.float32"))
    assert not log_filter.filter(notice("torch.float32"))
    assert log_filter.filter(notice("torch.bfloat16"))
    assert not log_filter.filter(notice("torch.bfloat16"))


def test_repeated_unrelated_backend_warnings_remain_visible():
    log_filter = _CastNoticeFilter()
    warning = logging.makeLogRecord({"msg": "A kernel returned non finite output", "args": ()})
    assert log_filter.filter(warning)
    assert log_filter.filter(warning)


def test_loading_rejects_native_bf16_compute_on_a_pre_ampere_gpu():
    from circuit_lantern.services.execution import loading_options

    metadata = {
        "quantization": {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    }
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        pytest.raises(ValueError, match="Ampere"),
    ):
        loading_options(metadata, "auto", "cuda")


def test_invalid_native_compute_dtype_is_rejected_before_kernel_setup():
    from circuit_lantern.services.execution import loading_options

    with pytest.raises(ValueError, match="unsupported bitsandbytes compute dtype"):
        loading_options(
            {
                "quantization": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "fp4",
                    "bnb_4bit_compute_dtype": "float64",
                }
            },
            "auto",
            "cpu",
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", ["auto", "int8"])
def test_training_int8_config_is_rejected_before_kernel_setup(device, precision):
    from transformers import BitsAndBytesConfig

    from circuit_lantern.services.execution import loading_options

    config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_has_fp16_weight=True)
    with (
        patch("circuit_lantern.services.execution.quantization_capabilities") as capabilities,
        pytest.raises(ValueError, match="master weights"),
    ):
        loading_options({"quantization": config.to_dict()}, precision, device)
    capabilities.assert_not_called()
