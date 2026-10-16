from unittest.mock import patch

import pytest
import torch
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from circuit_lantern.schemas import TraceRequest
from circuit_lantern.services.interventions import (
    _attention_output,
    attention_head_targets,
    install_head_ablation,
)


@pytest.fixture(params=["gpt2", "llama", "qwen2", "gpt_neox"])
def attention_model(runtime, request):
    if request.param != "gpt2":
        config_type, model_type = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "gpt_neox": (GPTNeoXConfig, GPTNeoXForCausalLM),
        }[request.param]
        config = config_type(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rotary_pct=0.5,
            attn_implementation="eager",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            runtime.model = model_type(config).eval()
        runtime.tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return runtime


@torch.inference_mode()
def test_zero_ablation_changes_only_one_head_input_and_applies_during_generation(attention_model):
    runtime = attention_model
    blocks = runtime._blocks()
    targets = attention_head_targets(blocks)
    assert len(targets) == 2
    target = targets[0]
    head = target["heads"] - 1  # Exercises query heads beyond the KV head count in GQA.
    adapter = _attention_output(blocks[0])
    inputs = runtime.tokenizer("the capital of", return_tensors="pt")
    baseline = runtime.model(**inputs, output_attentions=True)
    weights = {name: tensor.clone() for name, tensor in runtime.model.named_parameters()}
    before, after = [], []
    handles = [
        adapter.projection.register_forward_pre_hook(
            lambda module, args: before.append(args[0].clone())
        )
    ]
    handle, description = install_head_ablation(blocks, 0, head)
    handles.append(handle)
    handles.append(
        adapter.projection.register_forward_pre_hook(
            lambda module, args: after.append(args[0].clone())
        )
    )
    try:
        changed = runtime.model(**inputs, output_attentions=True)
        generated = runtime.model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=1,
            use_cache=True,
        )
        assert generated.shape[1] == inputs.input_ids.shape[1] + 3
        assert len(after) == 4
        assert any(value.shape[1] == 1 for value in after)
        for original, ablated in zip(before, after):
            expected = original.clone()
            expected[..., head * target["head_dim"] : (head + 1) * target["head_dim"]] = 0
            torch.testing.assert_close(ablated, expected, rtol=0, atol=0)
        assert not torch.allclose(changed.logits, baseline.logits, atol=1e-7, rtol=0)
        torch.testing.assert_close(changed.attentions[0], baseline.attentions[0], rtol=0, atol=0)
        assert description["scope"] == "all_prompt_and_generated_tokens"
    finally:
        for installed in handles:
            installed.remove()
    torch.testing.assert_close(runtime.model(**inputs).logits, baseline.logits, rtol=0, atol=0)
    if runtime.model.config.model_type == "gpt2":
        mask = torch.ones(len(blocks), target["heads"])
        mask[0, head] = 0
        reference = runtime.model(**inputs, head_mask=mask)
        torch.testing.assert_close(changed.logits, reference.logits, rtol=1e-5, atol=1e-6)
    for name, tensor in runtime.model.named_parameters():
        torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)


def test_head_trace_exports_the_intervention_and_cleans_up_failed_generation(attention_model):
    runtime = attention_model
    result = runtime.trace("the capital", 2, True, None, {"layer": 0, "head": 0})
    assert result["ablate_layer"] is None
    assert result["ablate_head"]["head"] == 0
    assert result["ablate_head"]["layer"] == 0
    assert result["ablate_head"]["head_dim"] > 0
    assert len(result["attention"]) == 2
    projection = _attention_output(runtime._blocks()[0]).projection
    assert not projection._forward_pre_hooks
    with (
        patch.object(runtime.model, "generate", side_effect=RuntimeError("generation failed")),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        runtime.trace("the capital", 2, True, None, {"layer": 0, "head": 0})
    assert not projection._forward_pre_hooks
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_invalid_or_ambiguous_head_targets_fail_before_installing_hooks(runtime):
    for target in ({"layer": 2, "head": 0}, {"layer": 0, "head": 2}):
        with pytest.raises(ValueError):
            runtime.trace("the capital", 1, True, None, target)
    with pytest.raises(ValueError, match="either a block bypass"):
        runtime.trace("the capital", 1, True, 0, {"layer": 0, "head": 0})
    with pytest.raises(ValueError, match="either a block bypass"):
        TraceRequest(prompt="the capital", ablate_layer=0, ablate_head={"layer": 0, "head": 0})
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())
    assert all(not block._forward_hooks for block in runtime._blocks())


def test_unknown_and_pruned_attention_layouts_are_not_offered(runtime):
    assert attention_head_targets([torch.nn.Identity()]) == []
    with pytest.raises(ValueError, match="unavailable"):
        install_head_ablation([torch.nn.Identity()], 0, 0)
    runtime._blocks()[0].attn.prune_heads([0])
    assert [target["layer"] for target in attention_head_targets(runtime._blocks())] == [1]
    with pytest.raises(ValueError, match="pruned"):
        install_head_ablation(runtime._blocks(), 0, 0)


def test_head_sweep_supports_verified_architectures_and_query_heads(attention_model):
    from circuit_lantern.services.head_sweep import head_ablation_sweep

    runtime = attention_model
    result = head_ablation_sweep(
        runtime.model,
        runtime.tokenizer,
        runtime._blocks(),
        runtime.metadata,
        "cpu",
        prompt="the capital",
        target="",
        layers=[0],
        job={},
        max_tokens=64,
    )
    expected_heads = attention_head_targets(runtime._blocks())[0]["heads"]
    assert len(result["heads"]) == expected_heads
    assert result["heads"][-1]["head"] == expected_heads - 1
    assert all(row["kl_divergence"] >= 0 for row in result["heads"])
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())
