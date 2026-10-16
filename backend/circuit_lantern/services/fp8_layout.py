"""Plan FP8 parameter storage with FP32 residuals using installed conversion rules.

Model buffers are outside this parameter budget. Matrix specifications stay internal.
"""

import math
from collections import Counter

import torch
from transformers.quantizers.base import HfQuantizer

from circuit_lantern.services.fp8 import configuration_error


def describe_fp8_layout(model, quantization: dict | None = None) -> dict:
    prequantized = quantization is not None
    config = dict(quantization) if prequantized else {"quant_method": "fp8"}
    result = {"supported": False, "prequantized": prequantized}
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        return dict(result, reason="FP8 storage planning requires a model on the meta device.")
    error = configuration_error(config)
    if error:
        return dict(result, reason=error)
    try:
        skipped = HfQuantizer.get_modules_to_not_convert(
            model,
            config.get("modules_to_not_convert"),
            getattr(model, "_keep_in_fp32_modules", None),
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as error:
        return dict(
            result,
            reason="The installed Transformers FP8 conversion rules could not be verified.",
            detail=str(error)[:250],
        )
    operators = []
    for name, module in model.named_modules(remove_duplicate=False):
        # Match replace_with_fp8_linear, including its substring skip semantics.
        if isinstance(module, torch.nn.Linear) and not any(key in name for key in skipped):
            operators.append((name, module))
    if not operators:
        return dict(result, reason="The model contains no supported FP8 linear operators.")
    aliases = Counter(
        id(parameter) for _, parameter in model.named_parameters(remove_duplicate=False)
    )
    if any(
        aliases[id(parameter)] != 1
        for _, module in operators
        for parameter in module.parameters(recurse=False)
    ):
        return dict(
            result,
            reason="The FP8 conversion would change shared parameter storage that cannot be verified.",
        )
    if prequantized:
        block = tuple(config.get("weight_block_size", (128, 128)))
    else:
        block = tuple(
            next(
                (
                    size
                    for size in (128, 64)
                    if all(
                        module.weight.shape[axis] % size == 0
                        and module.weight.shape[axis] // size > 1
                        for _, module in operators
                    )
                ),
                0,
            )
            for axis in (0, 1)
        )
        if not all(block):
            return dict(
                result,
                reason=(
                    "The installed FP8 quantizer requires aligned matrix dimensions "
                    "with at least two blocks on each axis."
                ),
            )
    matrices = []
    for name, module in operators:
        rows, columns = module.weight.shape
        if not rows or not columns or columns % block[1]:
            return dict(
                result,
                reason="Some FP8 input dimensions are not divisible by the checkpoint block size.",
            )
        prefix = name + "." if name else ""
        matrices.append(
            {
                "name": prefix + "weight",
                "shape": [rows, columns],
                "scale_name": prefix + "weight_scale_inv",
                "scale_shape": [math.ceil(rows / block[0]), columns // block[1]],
            }
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    quantizable = sum(math.prod(matrix["shape"]) for matrix in matrices)
    scale_bytes = sum(math.prod(matrix["scale_shape"]) * 4 for matrix in matrices)
    return dict(
        result,
        supported=True,
        config=dict(
            config,
            activation_scheme="dynamic",
            weight_block_size=list(block),
        ),
        block_size=list(block),
        modules_to_not_convert=sorted(set(skipped)),
        quantizable_parameters=quantizable,
        unquantized_parameters=parameters - quantizable,
        scale_bytes=scale_bytes,
        residual_dtype="float32",
        resident_weight_bytes=(parameters - quantizable) * 4 + quantizable + scale_bytes,
        largest_quantized_matrix_elements=max(math.prod(matrix["shape"]) for matrix in matrices),
        matrix_count=len(matrices),
        matrices=matrices,
    )
