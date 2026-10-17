"""Connections follow forward source, never module registration order or names."""

import pytest
import torch
from torch import nn
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from circuit_lantern.services.dataflow import block_dataflow
from circuit_lantern.services.source_dataflow import forward_tree
from circuit_lantern.services.weights import model_weight_inventory


class MisleadingBlock(nn.Module):
    def __init__(self, parallel=False):
        super().__init__()
        self.down_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 8)
        self.parallel = parallel

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.down_proj(hidden_states)
        if self.parallel:
            hidden_states = hidden_states + self.up_proj(residual)
        else:
            hidden_states = self.up_proj(hidden_states)
        return residual + hidden_states


class OpaqueMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        if x.sum() > 0:
            return self.a(x)
        return self.b(x)


class OpaqueBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = OpaqueMixer()

    def forward(self, x):
        return x + self.mixer(x)


class RepeatedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)

    def forward(self, x):
        x = self.shared(x)
        x += self.shared(x)
        return x


def diagram(block):
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    weights, _ = model_weight_inventory(model)
    return block_dataflow(model, model.blocks, weights)[0], weights


def assert_graph(flow, weights):
    assert flow["supported"]
    ids = {node["id"] for node in flow["nodes"]}
    seen = set()
    for node in flow["nodes"]:
        parents = [edge["source"] for edge in flow["edges"] if edge["target"] == node["id"]]
        assert set(parents) <= seen
        seen.add(node["id"])
    assert all(edge["source"] in ids and edge["target"] in ids for edge in flow["edges"])
    inventory = {item["name"] for item in weights}
    assert all(not node["weight"] or node["weight"] in inventory for node in flow["nodes"])
    assert flow["mapped_matrices"] + len(flow["unmapped_matrices"]) == flow["total_matrices"]


@pytest.mark.parametrize("parallel", [False, True])
def test_connections_follow_source_and_configuration_despite_misleading_names(parallel):
    block = MisleadingBlock(parallel)
    flow, weights = diagram(block)
    assert_graph(flow, weights)
    down, up = [node for node in flow["nodes"] if node["weight"]]
    assert down["weight"].endswith("down_proj.weight")
    assert up["weight"].endswith("up_proj.weight")
    input_edge = next(edge for edge in flow["edges"] if edge["target"] == up["id"])
    assert input_edge["source"] == (flow["nodes"][0]["id"] if parallel else down["id"])
    assert any(edge["residual"] for edge in flow["edges"])


def test_dynamic_mixer_stays_opaque_with_exact_inventory_and_no_fake_internal_edges():
    flow, weights = diagram(OpaqueBlock())
    assert_graph(flow, weights)
    assert flow["mapped_matrices"] == 0
    assert len(flow["unmapped_matrices"]) == 2
    assert len(flow["opaque_modules"]) == 1
    opaque = next(node for node in flow["nodes"] if node.get("opaque"))
    assert set(opaque["matrices"]) == {item["name"] for item in weights}
    assert not any(node["weight"] for node in flow["nodes"])


def test_shared_module_has_distinct_call_sites_and_one_logical_weight():
    flow, weights = diagram(RepeatedBlock())
    assert_graph(flow, weights)
    assert len([node for node in flow["nodes"] if node["weight"]]) == 2
    assert flow["mapped_matrices"] == flow["total_matrices"] == 1


def test_qwen35_hybrid_graphs_cover_real_weights_and_distinguish_two_mixers():
    config = Qwen3_5TextConfig(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
    )
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    weights, _ = model_weight_inventory(model)
    before = forward_tree.cache_info()
    flows = block_dataflow(model, model.model.layers, weights)
    for flow in flows:
        assert_graph(flow, weights)
        assert flow["mapped_matrices"] == flow["total_matrices"]
        assert flow["opaque_modules"] == []
        assert len([edge for edge in flow["edges"] if edge["residual"]]) == 2
    assert [flow["mixer_type"] for flow in flows] == ["linear_attention"] * 3 + ["full_attention"]
    linear_labels = {node["label"] for node in flows[0]["nodes"]}
    full_labels = {node["label"] for node in flows[3]["nodes"]}
    assert "Gated DeltaNet recurrence" in linear_labels
    assert "Attention probabilities" not in linear_labels
    assert {
        "Query and gate projection",
        "Multiply output gate",
        "Attention probabilities",
    } <= full_labels
    assert forward_tree.cache_info().hits > before.hits


def test_discovery_never_runs_model_or_tensor_operations(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("Dataflow discovery executed a tensor operation")

    block = MisleadingBlock()
    monkeypatch.setattr(nn.Module, "_call_impl", lambda *a, **k: pytest.fail("Model executed"))
    monkeypatch.setattr(block, "forward", lambda *a, **k: pytest.fail("Forward executed"))
    with NoTensorOps():
        flow, weights = diagram(block)
    assert_graph(flow, weights)


def test_missing_source_does_not_claim_connections(monkeypatch):
    monkeypatch.setattr("circuit_lantern.services.source_dataflow.forward_tree", lambda cls: None)
    flow, _ = diagram(MisleadingBlock())
    assert flow == {"layer": 0, "supported": False, "nodes": [], "edges": []}


@pytest.mark.parametrize(
    "family",
    [
        "gemma",
        "gemma2",
        "gemma3_text",
        "phi",
        "olmo",
        "olmo2",
        "starcoder2",
        "granite",
        "bloom",
        "gptj",
    ],
)
def test_other_architectures_discover_module_connections_without_new_adapters(family):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        family,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=20,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    try:
        blocks = model.get_submodule("model.layers")
    except AttributeError:
        blocks = model.get_submodule("transformer.h")
    weights, _ = model_weight_inventory(model)
    flow = block_dataflow(model, blocks, weights)[0]
    assert_graph(flow, weights)
    assert flow["provenance"] == "source"
    if family != "bloom":
        assert any(edge["residual"] for edge in flow["edges"])
    else:
        # BLOOM performs both residual additions inside opaque child modules.
        assert flow["opaque_modules"] and flow["edges"]


class TupleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        a, b = x, self.a(x)
        a, b = b, a
        _unused = self.b(a)
        return b + a


class OptionalBranchBlock(MisleadingBlock):
    def forward(self, x, position_embeddings=None):
        if position_embeddings is None:
            return self.up_proj(x)
        return self.down_proj(x)


class TrainingBranchBlock(MisleadingBlock):
    def forward(self, x):
        if self.training:
            return self.up_proj(x)
        return self.down_proj(x)


class CacheReassignmentBlock(MisleadingBlock):
    def forward(self, x, cache_params=None):
        cache_params = x
        if cache_params is None:
            return self.up_proj(x)
        return self.down_proj(x)


class TwoInputMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, first, second):
        return self.a(first) + self.b(second)


class TwoInputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.mixer = TwoInputMixer()

    def forward(self, x):
        return self.mixer(self.a(x), x)


def test_tuple_assignments_and_swaps_preserve_ports_and_discard_unused_paths():
    flow, weights = diagram(TupleBlock())
    assert_graph(flow, weights)
    projection = next(node for node in flow["nodes"] if node["weight"])
    assert projection["weight"].endswith(".a.weight")
    assert flow["mapped_matrices"] == 1
    assert len(flow["unmapped_matrices"]) == 1
    assert not any(node["label"] == "Module output" for node in flow["nodes"])
    add = next(node for node in flow["nodes"] if node["label"] == "Add")
    assert {edge["source"] for edge in flow["edges"] if edge["target"] == add["id"]} == {
        flow["nodes"][0]["id"],
        projection["id"],
    }
    assert all(
        any(edge["source"] == node["id"] for edge in flow["edges"]) for node in flow["nodes"][:-1]
    )


def test_optional_inputs_are_not_assumed_absent_and_reassigned_cache_is_unknown():
    for block in [OptionalBranchBlock(), CacheReassignmentBlock()]:
        flow, _ = diagram(block)
        assert not flow["supported"]


def test_inference_diagram_pins_eval_mode_without_mutating_module():
    block = TrainingBranchBlock()
    assert block.training
    flow, weights = diagram(block)
    assert_graph(flow, weights)
    assert next(node["weight"] for node in flow["nodes"] if node["weight"]).endswith(
        ".down_proj.weight"
    )
    assert block.training


def test_multiple_tensor_arguments_keep_child_opaque_instead_of_rebinding_first_input():
    flow, weights = diagram(TwoInputBlock())
    assert_graph(flow, weights)
    assert flow["mapped_matrices"] == 1
    assert len(flow["unmapped_matrices"]) == 2
    opaque = next(node for node in flow["nodes"] if node.get("opaque"))
    assert len([edge for edge in flow["edges"] if edge["target"] == opaque["id"]]) == 2


def test_optional_discovery_failure_keeps_inventory_accessible(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("Unavailable source")

    monkeypatch.setattr("circuit_lantern.services.source_dataflow.forward_tree", broken)
    flow, weights = diagram(MisleadingBlock())
    assert not flow["supported"]
    assert len(weights) == 2
