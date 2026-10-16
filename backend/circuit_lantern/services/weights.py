import math
from itertools import chain

import torch


def weight_description(name: str, parameter: torch.Tensor) -> dict:
    shape = list(parameter.shape)
    storage_format = str(parameter.dtype).removeprefix("torch.")
    if type(parameter).__name__ == "Params4bit":
        state = getattr(parameter, "quant_state", None)
        if state is None:
            raise ValueError("The 4-bit matrix has no quantization state.")
        shape = list(state.shape)
        storage_format = state.quant_type
    elif type(parameter).__name__ == "Int8Params" and parameter.dtype == torch.int8:
        storage_format = "int8"
    return {
        "name": name,
        "shape": shape,
        "elements": math.prod(shape),
        "dtype": storage_format,
        "storage_dtype": str(parameter.dtype).removeprefix("torch."),
        "storage_shape": list(parameter.shape),
        "storage_bytes": parameter.numel() * parameter.element_size(),
        "quantized": storage_format in ("nf4", "fp4", "int8"),
    }


def logical_weight(model, name: str) -> tuple[torch.Tensor, dict]:
    parameter = dict(model.named_parameters()).get(name)
    if parameter is None:
        raise ValueError("Weight matrix was not found in the loaded model.")
    description = weight_description(name, parameter)
    if len(description["shape"]) != 2:
        raise ValueError("Select a two dimensional weight matrix.")
    cls_name = type(parameter).__name__
    if cls_name == "Params4bit":
        from bitsandbytes import functional

        value = functional.dequantize_4bit(parameter.data, parameter.quant_state)
    elif cls_name == "Int8Params" and parameter.dtype == torch.int8:
        from bitsandbytes import functional

        module = model.get_submodule(name.rsplit(".", 1)[0])
        scales = getattr(parameter, "SCB", None)
        if scales is None:
            scales = getattr(getattr(module, "state", None), "SCB", None)
        if scales is None:
            raise ValueError("The 8-bit matrix is missing its scaling factors.")
        value = functional.int8_vectorwise_dequant(parameter.data, scales)
    else:
        if not parameter.is_floating_point():
            raise ValueError(
                "This packed matrix format does not have a supported dequantization adapter."
            )
        value = parameter.detach()
    return value.float(), description


def model_weight_inventory(model) -> tuple[list[dict], int]:
    descriptions = [
        weight_description(name, parameter) for name, parameter in model.named_parameters()
    ]
    return [item for item in descriptions if len(item["shape"]) == 2], sum(
        item["elements"] for item in descriptions
    )


def resident_tensor_bytes(model) -> dict[str, int]:
    """Count unique parameter and buffer storage by device, including packed storage."""
    if model is None:
        return {}
    seen = set()
    totals = {}
    for tensor in chain(model.parameters(), model.buffers()):
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        device = str(tensor.device)
        key = (device, storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        totals[device] = totals.get(device, 0) + storage.nbytes()
    return totals
