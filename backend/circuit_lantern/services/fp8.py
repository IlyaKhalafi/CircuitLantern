"""Block scaled FP8 execution and logical matrix decoding."""

import importlib.metadata
import math
from functools import lru_cache
from types import MethodType

import torch

ADAPTER_VERSION = "finegrained_fp8_zero_safe_v1"
BLOCK_SIZES = ((64, 64), (64, 128), (128, 64), (128, 128))


def configuration_error(config: dict) -> str | None:
    if config.get("quant_method") != "fp8":
        return "Only Transformers fine grained FP8 checkpoints have a supported FP8 matrix adapter."
    if config.get("activation_scheme", "dynamic") != "dynamic":
        return "This FP8 adapter requires dynamic activation scaling."
    if config.get("fmt", "e4m3") != "e4m3":
        return "This FP8 adapter requires E4M3 weights."
    block = config.get("weight_block_size", (128, 128))
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in block)
        or tuple(block) not in BLOCK_SIZES
    ):
        return "This FP8 adapter supports 64 or 128 elements along each weight block dimension."
    skipped = config.get("modules_to_not_convert")
    if skipped is not None and (
        not isinstance(skipped, (list, tuple))
        or any(not isinstance(name, str) or not name for name in skipped)
    ):
        return "FP8 skipped modules must be a list of module names."
    if set(config) - {
        "quant_method",
        "activation_scheme",
        "fmt",
        "weight_block_size",
        "modules_to_not_convert",
    }:
        return "This FP8 configuration contains options that the matrix adapter cannot verify."
    return None


def is_fp8_linear(module) -> bool:
    return (
        type(module).__name__ == "FP8Linear"
        and type(module).__module__ == "transformers.integrations.finegrained_fp8"
    )


def _row_scales(weight: torch.Tensor, scales: torch.Tensor, block: tuple[int, int]):
    error = configuration_error({"quant_method": "fp8", "weight_block_size": block})
    if error or weight.ndim != 2 or not all(weight.shape):
        raise ValueError("The FP8 weight and its block scaling factors have an unsupported layout.")
    rows, columns = weight.shape
    block_rows, block_columns = block
    if (
        weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or columns % block_columns
        or tuple(scales.shape) != (math.ceil(rows / block_rows), columns // block_columns)
        or scales.dtype != torch.float32
        or scales.device != weight.device
    ):
        raise ValueError("The FP8 weight and its block scaling factors have an unsupported layout.")
    if not torch.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("FP8 weight scaling factors must be finite and nonnegative.")
    expanded = scales.detach().repeat_interleave(block_rows, dim=0)[:rows]
    codes = weight.detach().view(torch.uint8).reshape(rows, columns // block_columns, block_columns)
    if (((codes & 127) == 127) & expanded.ne(0).unsqueeze(-1)).any():
        raise ValueError("The FP8 matrix contains non finite codes with nonzero scaling factors.")
    return expanded


@torch.inference_mode()
def logical_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """Decode one operator without expanding scales to a second full weight matrix."""
    row_scales = _row_scales(weight, scales, block)
    value = weight.detach().float().reshape(*row_scales.shape, block[1])
    value.mul_(row_scales.unsqueeze(-1))
    value.masked_fill_(row_scales.eq(0).unsqueeze(-1), 0)
    if not torch.isfinite(value).all():
        raise ValueError("The FP8 matrix exceeds the finite float32 analysis range.")
    return value.reshape(weight.shape)


def _forward(module, input: torch.Tensor) -> torch.Tensor:
    from transformers.integrations.finegrained_fp8 import (
        act_quant,
        w8a8_block_fp8_matmul_triton,
    )

    with torch.cuda.device(input.device):
        qinput, scales = act_quant(input.contiguous(), module.block_size[1])
        # E4M3 has no finite result for 0 / 0. A zero group represents exact zero.
        qinput.view(torch.uint8).reshape(*scales.shape, module.block_size[1]).masked_fill_(
            scales.eq(0).unsqueeze(-1), 0
        )
        weight = module.weight
        if module._lantern_fp8_zero_weight_blocks:
            row_scales = module.weight_scale_inv.detach().repeat_interleave(
                module.block_size[0], dim=0
            )[: module.out_features]
            codes = weight.detach().view(torch.uint8).clone()
            codes.reshape(*row_scales.shape, module.block_size[1]).masked_fill_(
                row_scales.eq(0).unsqueeze(-1), 0
            )
            weight = codes.view(torch.float8_e4m3fn)
        output = w8a8_block_fp8_matmul_triton(
            qinput,
            weight,
            scales,
            module.weight_scale_inv,
            module.block_size,
            output_dtype=input.dtype,
        )
    torch.cuda.synchronize(input.device)
    if module.bias is not None:
        output = output + module.bias
    return output.to(dtype=input.dtype)


def prepare_model(model) -> int:
    """Install after final parameter loading or movement, leaving stored values untouched."""
    count = 0
    for module in model.modules():
        if not is_fp8_linear(module):
            continue
        config = {
            "quant_method": "fp8",
            "activation_scheme": module.activation_scheme,
            "weight_block_size": module.block_size,
        }
        error = configuration_error(config)
        if error:
            raise ValueError(error)
        row_scales = _row_scales(module.weight, module.weight_scale_inv, module.block_size)
        module._lantern_fp8_zero_weight_blocks = bool(row_scales.eq(0).any())
        if hasattr(module, "_hf_hook"):
            if not callable(getattr(module, "_old_forward", None)):
                raise ValueError("The FP8 module has an unsupported dispatch hook.")
            module._old_forward = MethodType(_forward, module)
        else:
            module.forward = MethodType(_forward, module)
        count += 1
    if not count:
        raise ValueError("The checkpoint contains no supported FP8 linear operators.")
    return count


@torch.inference_mode()
def _probe_kernel(block: tuple[int, int]) -> dict:
    from transformers.integrations.finegrained_fp8 import FP8Linear

    with torch.device("meta"):
        module = FP8Linear(256, 128, block_size=block, device="meta").eval()
    with torch.cuda.device(0):
        module.to_empty(device="cuda:0")
        values = (torch.arange(128 * 256, device="cuda:0").reshape(128, 256) % 16 - 8).float()
        module.weight.copy_(values.to(torch.float8_e4m3fn))
        module.weight_scale_inv.fill_(0.125)
        prepare_model(module)
        inputs = torch.ones(3, 256, device="cuda:0", dtype=torch.float32)
        inputs[1, : block[1]] = 0
        inputs[2] = 0
        actual = module(inputs)
        expected = torch.nn.functional.linear(inputs, values * 0.125)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        return {"triton_version": importlib.metadata.version("triton")}


@lru_cache(maxsize=16)
def _verified_probe(block: tuple[int, int]) -> dict:
    return _probe_kernel(block)


def execution_probe(block: tuple[int, int] | list[int] = (128, 128)) -> dict:
    """Probe logical CUDA device zero, caching success while allowing failed probes to recover."""
    result = {
        "supported": False,
        "block_size": list(block) if isinstance(block, (list, tuple)) else [],
        "adapter_version": ADAPTER_VERSION,
    }
    if configuration_error({"quant_method": "fp8", "weight_block_size": block}):
        return dict(result, reason="This FP8 block size has not been verified.")
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 9:
            return dict(
                result, reason="This FP8 execution path requires a Hopper or newer CUDA GPU."
            )
        result.update(_verified_probe(tuple(block)), supported=True)
    except Exception as error:  # noqa: BLE001 - kernel backends have independent exception types
        result["reason"] = str(error)[:250]
    return result
