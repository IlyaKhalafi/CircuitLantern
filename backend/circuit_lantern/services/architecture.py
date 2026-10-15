from functools import lru_cache

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D


def describe_model_layout(model, skipped_modules: list[str] | None = None) -> dict:
    parameters = list(model.parameters())
    output = model.get_output_embeddings()
    inputs = model.get_input_embeddings()
    excluded = {id(module.weight) for module in (output, inputs) if module is not None}
    skipped = (skipped_modules or []) + list(getattr(model, "_keep_in_fp32_modules", None) or [])
    quantizable = {}
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, Conv1D)) or id(module.weight) in excluded:
            continue
        if not any(
            part == name or name.endswith("." + part) or ("." + part + ".") in ("." + name + ".")
            for part in skipped
        ):
            quantizable[id(module.weight)] = module.weight.numel()
    return {
        "parameters": sum(p.numel() for p in parameters),
        "quantizable_parameters": sum(quantizable.values()),
        "largest_matrix_elements": max((p.numel() for p in parameters if p.ndim == 2), default=0),
        "tied_embeddings": output is not None
        and inputs is not None
        and output.weight is inputs.weight,
    }


@lru_cache(maxsize=64)
def architecture_layout(config_directory: str) -> dict:
    config = AutoConfig.from_pretrained(
        config_directory, local_files_only=True, trust_remote_code=False
    )
    # The device context is local to this thread. No global module monkeypatching.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=False, attn_implementation="eager"
        )
    model.tie_weights()
    quantization = getattr(config, "quantization_config", None) or {}
    return describe_model_layout(model, quantization.get("llm_int8_skip_modules"))
