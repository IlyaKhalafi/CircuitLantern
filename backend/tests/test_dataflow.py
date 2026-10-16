import pytest
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

from circuit_lantern.services.dataflow import block_dataflow
from circuit_lantern.services.weights import model_weight_inventory


def connections(flow):
    return {(edge["source"], edge["target"]) for edge in flow["edges"]}


def test_gpt2_fused_attention_has_real_branches_and_two_residual_additions(runtime):
    flow = runtime.graph()["dataflow"][0]
    assert flow["supported"] and not flow["parallel_residual"]
    edges = connections(flow)
    assert {
        ("qkv", "q"),
        ("qkv", "k"),
        ("qkv", "v"),
        ("q", "scores"),
        ("k", "scores"),
        ("scores", "mix"),
        ("v", "mix"),
        ("input", "add-attn"),
        ("add-attn", "norm-mlp"),
        ("add-attn", "output"),
    } <= edges
    assert ("v", "scores") not in edges
    inventory = {item["name"] for item in runtime.graph()["matrices"]}
    assert all(not node["weight"] or node["weight"] in inventory for node in flow["nodes"])


def test_gated_mlp_branches_multiply_before_the_down_projection():
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=20,
        )
    )
    flow = block_dataflow(model, model.model.layers, model_weight_inventory(model)[0])[0]
    edges = connections(flow)
    assert {
        ("norm-mlp", "up"),
        ("norm-mlp", "gate"),
        ("gate", "act"),
        ("act", "multiply"),
        ("up", "multiply"),
        ("multiply", "down"),
    } <= edges
    assert ("up", "act") not in edges


@pytest.mark.parametrize("parallel", [True, False])
def test_neox_parallel_configuration_controls_mlp_input_and_residuals(parallel):
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=20,
            use_parallel_residual=parallel,
        )
    )
    flow = block_dataflow(model, model.gpt_neox.layers, model_weight_inventory(model)[0])[0]
    assert flow["parallel_residual"] is parallel
    edges = connections(flow)
    assert (("input", "norm-mlp") in edges) is parallel
    assert (("add-attn", "norm-mlp") in edges) is not parallel
    if parallel:
        branch = next(
            edge
            for edge in flow["edges"]
            if edge["source"] == "input" and edge["target"] == "norm-mlp"
        )
        assert branch["branch"] and not branch["residual"]


@pytest.mark.parametrize("family", ["mistral", "qwen2", "qwen3"])
def test_other_supported_rotary_blocks_have_connected_weight_paths(family):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        family,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=20,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    flow = block_dataflow(model, model.model.layers, model_weight_inventory(model)[0])[0]
    assert flow["supported"]
    assert ("v", "mix") in connections(flow)
    assert ("v", "scores") not in connections(flow)
    assert all(
        any(edge["target"] == node["id"] for edge in flow["edges"])
        for node in flow["nodes"]
        if node["id"] != "input"
    )
    scores = next(node for node in flow["nodes"] if node["id"] == "scores")
    assert "RoPE" in scores["detail"]
    if family == "qwen3":
        assert "Q/K RMSNorm" in scores["detail"]


def test_missing_weight_or_unknown_block_does_not_invent_connections(runtime):
    import torch

    missing = block_dataflow(runtime.model, runtime._blocks(), [])
    assert all(not item["supported"] for item in missing)
    unknown = block_dataflow(runtime.model, [torch.nn.Identity()], [])
    assert unknown[0] == {"layer": 0, "supported": False, "nodes": [], "edges": []}
