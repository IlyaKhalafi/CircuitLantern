"""Exercise current hybrid checkpoints through the actual loader and tracer."""

import pytest
import torch
from model_fixtures import assert_no_forward_hooks, checkpoint_metadata
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5Tokenizer

from circuit_lantern.services.architecture import architecture_layout
from circuit_lantern.services.tracing import attention_layer_indices


@pytest.mark.parametrize(
    ("kinds", "count", "expected"),
    [
        (["linear_attention", "linear_attention", "linear_attention", "full_attention"], 1, [3]),
        (["conv", "full_attention", "conv", "full_attention"], 2, [1, 3]),
        (["mamba", "attention", "mamba", "sliding_attention"], 2, [1, 3]),
        (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            4,
            [0, 1, 2, 3],
        ),
        (None, 4, [0, 1, 2, 3]),
        (None, 0, []),
    ],
)
def test_attention_outputs_keep_original_hybrid_layer_indices(kinds, count, expected):
    assert attention_layer_indices(kinds, 4, count) == expected


def test_unverifiable_compressed_attention_is_never_assigned_an_invented_layer():
    with pytest.raises(ValueError, match="Turn off attention capture"):
        attention_layer_indices(["unknown"] * 4, 4, 2)


@pytest.fixture
def hybrid_checkpoint(tmp_path):
    text = Qwen3_5TextConfig(
        vocab_size=257,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
        bos_token_id=0,
        eos_token_id=1,
    )
    config = Qwen3_5Config(text_config=text.to_dict())
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    model.save_pretrained(tmp_path)
    # Official checkpoints use the outer multimodal configuration. The causal
    # loader must select its text backbone without allocating a vision tower.
    config.save_pretrained(tmp_path)
    tokenizer = Qwen3_5Tokenizer(
        vocab={
            "<|endoftext|>": 0,
            **{piece: i + 1 for i, piece in enumerate(sorted(ByteLevel.alphabet()))},
        },
        merges=[],
    )
    tokenizer.save_pretrained(tmp_path)
    return model, tokenizer, tmp_path


def test_qwen35_outer_config_has_a_real_causal_layout(hybrid_checkpoint):
    model, _, directory = hybrid_checkpoint
    layout = architecture_layout(str(directory))
    assert layout["parameters"] == sum(p.numel() for p in model.parameters())
    assert layout["context_length"] == 128
    assert layout["quantizable_parameters"] > 0


def test_qwen35_checkpoint_loads_and_keeps_hybrid_attention_layer_identity(
    runtime, hybrid_checkpoint, monkeypatch
):
    model, _, directory = hybrid_checkpoint
    metadata = dict(checkpoint_metadata(model, directory), context_length=128)
    monkeypatch.setattr(
        "circuit_lantern.services.runtime.snapshot_download", lambda *args, **kwargs: str(directory)
    )
    monkeypatch.setattr(runtime, "_assess_fit", lambda *args: {"compatible": True})
    job = {"status": "queued"}
    runtime._load(job, metadata, "cpu", "float32", 64)
    assert job["status"] == "ready", job["message"]
    assert runtime.model.config.model_type == "qwen3_5_text"
    assert len(runtime._blocks()) == 4
    trace = runtime.trace("the capital of France", 1, True, None)
    assert len(trace["layers"]) == 5
    assert [row["layer"] for row in trace["attention"]] == [3]
    assert len(trace["generated_token_ids"]) == 1
    assert trace["predictions"] and all(row["predictions"] for row in trace["layers"])
    assert len(runtime.graph()["matrices"]) > 0
    assert_no_forward_hooks(runtime._blocks())
    assert all(torch.isfinite(p).all() for p in runtime.model.parameters())
