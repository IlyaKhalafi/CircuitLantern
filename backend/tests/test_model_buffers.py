import pytest
import torch

from circuit_lantern.services.architecture import model_buffer_layout


@pytest.mark.parametrize(
    ("dtype", "element_bytes"),
    [
        (torch.bool, 1),
        (torch.int64, 8),
        (torch.float16, 4),
        (torch.bfloat16, 4),
        (torch.float32, 4),
        (torch.float64, 8),
        (torch.complex64, 8),
    ],
)
def test_model_buffers_have_a_conservative_float32_floor_without_changing_storage(
    dtype, element_bytes
):
    model = torch.nn.Module()
    with torch.device("meta"):
        value = torch.empty(17, dtype=dtype)
    model.register_buffer("persistent", value, persistent=True)
    model.register_buffer("runtime_alias", value, persistent=False)
    result = model_buffer_layout(model)
    assert result["model_buffer_bytes"] == 2 * 17 * element_bytes
    assert result["largest_model_buffer_bytes"] == 17 * element_bytes
    assert model.persistent is value and model.runtime_alias is value
    assert value.dtype == dtype and value.device.type == "meta"


def test_model_without_buffers_has_no_buffer_reservation():
    assert model_buffer_layout(torch.nn.Module()) == {
        "model_buffer_bytes": 0,
        "largest_model_buffer_bytes": 0,
    }
