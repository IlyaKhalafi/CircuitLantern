import weakref
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize("capture_attention", [False, True])
@pytest.mark.parametrize(
    ("layer", "head"),
    [(None, None), (0, None), (None, {"layer": 0, "head": 0})],
)
def test_prompt_capture_tensors_are_released_before_continuation(
    runtime, layer, head, capture_attention
):
    references = []
    forward = runtime.model.forward

    def capture_forward(*args, **kwargs):
        outputs = forward(*args, **kwargs)
        tensors = [outputs.logits, *outputs.hidden_states, *(outputs.attentions or ())]
        references.extend(weakref.ref(tensor) for tensor in tensors if tensor is not None)
        return outputs

    def generate(**kwargs):
        assert references
        assert all(reference() is None for reference in references)
        ids = kwargs["input_ids"]
        eos = torch.full((1, 1), runtime.tokenizer.eos_token_id, device=ids.device, dtype=ids.dtype)
        return torch.cat((ids, eos), dim=1)

    handle = runtime._blocks()[-1].register_forward_hook(
        lambda module, args, output: references.append(
            weakref.ref(output[0] if isinstance(output, tuple) else output)
        )
    )
    try:
        with (
            patch.object(runtime.model, "forward", new=capture_forward),
            patch.object(runtime.model, "generate", new=generate),
        ):
            result = runtime.trace("the capital", 1, capture_attention, layer, head)
    finally:
        handle.remove()
    assert result["generated_token_ids"] == [runtime.tokenizer.eos_token_id]
    assert result["predictions"] and result["layers"]
    assert bool(result["attention"]) == capture_attention
    assert all(not block._forward_hooks for block in runtime._blocks())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_block_bypass_stays_active_without_retaining_generation_residuals(runtime):
    block = runtime._blocks()[0]

    def generate(**kwargs):
        references = []

        def observe_input(module, args):
            references.append(weakref.ref(args[0]))

        def observe_output(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            assert hidden is references[-1]()

        before = block.register_forward_pre_hook(observe_input)
        after = block.register_forward_hook(observe_output)
        try:
            inputs = {
                name: kwargs[name]
                for name in ("input_ids", "attention_mask", "token_type_ids")
                if name in kwargs
            }
            outputs = runtime.model(
                **inputs, use_cache=False, output_hidden_states=False, output_attentions=False
            )
            del outputs
        finally:
            before.remove()
            after.remove()
        assert references and all(reference() is None for reference in references)
        ids = kwargs["input_ids"]
        eos = torch.full((1, 1), runtime.tokenizer.eos_token_id, device=ids.device, dtype=ids.dtype)
        return torch.cat((ids, eos), dim=1)

    with patch.object(runtime.model, "generate", new=generate):
        result = runtime.trace("the capital", 1, True, 0)
    assert result["ablate_layer"] == 0
    assert result["generated_token_ids"] == [runtime.tokenizer.eos_token_id]
    assert all(not block._forward_hooks for block in runtime._blocks())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())
