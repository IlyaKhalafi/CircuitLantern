import json
import math
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from circuit_lantern.services.fp8_layout import describe_fp8_layout


def meta_llama(*, hidden=256, heads=4, kv_heads=2, vocab=512, tied=True, bias=False, layers=1):
    with torch.device("meta"):
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=vocab,
                hidden_size=hidden,
                intermediate_size=hidden * 2,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
                tie_word_embeddings=tied,
                attention_bias=bias,
                attn_implementation="eager",
            )
        )
    model.tie_weights()
    return model


@pytest.mark.parametrize(
    ("hidden", "heads", "kv_heads", "block"),
    [(256, 4, 2, [64, 128]), (576, 9, 3, [64, 64]), (896, 14, 2, [64, 128])],
)
def test_fp8_plan_accounts_for_real_linear_weights_scales_and_fp32_remainder(
    hidden, heads, kv_heads, block
):
    model = meta_llama(hidden=hidden, heads=heads, kv_heads=kv_heads)
    before = {name: id(parameter) for name, parameter in model.named_parameters()}
    result = describe_fp8_layout(model)
    assert result["supported"], result
    assert result["block_size"] == block
    linears = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module is not model.get_output_embeddings()
    ]
    assert {matrix["name"] for matrix in result["matrices"]} == {
        name + ".weight" for name, _ in linears
    }
    quantizable = sum(module.weight.numel() for _, module in linears)
    remainder = sum(parameter.numel() for parameter in model.parameters()) - quantizable
    scales = sum(
        math.ceil(module.out_features / block[0]) * math.ceil(module.in_features / block[1]) * 4
        for _, module in linears
    )
    assert result["quantizable_parameters"] == quantizable
    assert result["unquantized_parameters"] == remainder
    assert result["scale_bytes"] == scales
    assert result["resident_weight_bytes"] == quantizable + remainder * 4 + scales
    assert result["resident_weight_bytes"] > sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == before
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    json.dumps(result)


def test_native_fp8_uses_checkpoint_blocks_and_keeps_partial_output_blocks():
    model = meta_llama(vocab=70, tied=False)
    result = describe_fp8_layout(
        model,
        {
            "quant_method": "fp8",
            "weight_block_size": [128, 64],
            "modules_to_not_convert": [],
        },
    )
    assert result["supported"], result
    assert result["prequantized"]
    assert result["block_size"] == [128, 64]
    head = next(matrix for matrix in result["matrices"] if matrix["name"] == "lm_head.weight")
    assert head["shape"] == [70, 256]
    assert head["scale_shape"] == [1, 4]


def test_explicit_empty_skips_cannot_silently_break_tied_output_storage():
    result = describe_fp8_layout(
        meta_llama(),
        {"quant_method": "fp8", "modules_to_not_convert": []},
    )
    assert not result["supported"]
    assert "shared parameter storage" in result["reason"]


def test_skip_names_follow_transformers_substring_matching():
    model = meta_llama()
    result = describe_fp8_layout(
        model,
        {
            "quant_method": "fp8",
            "modules_to_not_convert": ["lm_head", "self_attn.k", "down_proj"],
        },
    )
    assert result["supported"], result
    names = {matrix["name"] for matrix in result["matrices"]}
    assert "model.layers.0.self_attn.k_proj.weight" not in names
    assert "model.layers.0.mlp.down_proj.weight" not in names
    assert "model.layers.0.self_attn.q_proj.weight" in names


def test_float_model_with_unaligned_input_dimensions_does_not_claim_fp8_support():
    result = describe_fp8_layout(meta_llama(hidden=160, heads=5, kv_heads=1))
    assert not result["supported"]
    assert "aligned matrix dimensions" in result["reason"]


def test_native_checkpoint_with_unaligned_input_dimensions_is_rejected():
    result = describe_fp8_layout(
        meta_llama(hidden=576, heads=9, kv_heads=3),
        {"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    assert not result["supported"]
    assert "input dimensions" in result["reason"]


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "compressed-tensors"},
        {"quant_method": "fp8", "activation_scheme": "static"},
        {"quant_method": "fp8", "weight_block_size": [32, 128]},
        {"quant_method": "fp8", "modules_to_not_convert": "lm_head"},
    ],
)
def test_unverified_configuration_cannot_generate_an_fp8_storage_estimate(config):
    result = describe_fp8_layout(meta_llama(), config)
    assert not result["supported"]
    assert "resident_weight_bytes" not in result


def test_materialized_weights_are_not_copied_for_layout_discovery():
    model = torch.nn.Linear(128, 128)
    with patch(
        "circuit_lantern.services.fp8_layout.HfQuantizer.get_modules_to_not_convert"
    ) as skipped:
        result = describe_fp8_layout(model)
    assert not result["supported"]
    assert "meta device" in result["reason"]
    skipped.assert_not_called()


def test_optional_conversion_discovery_failure_does_not_invalidate_other_model_formats():
    with patch(
        "circuit_lantern.services.fp8_layout.HfQuantizer.get_modules_to_not_convert",
        side_effect=OSError("An optional integration could not be imported."),
    ):
        result = describe_fp8_layout(meta_llama())
    assert not result["supported"]
    assert "conversion rules" in result["reason"]


def test_shared_bias_storage_cannot_be_undercounted_during_linear_replacement():
    model = meta_llama(bias=True)
    attention = model.model.layers[0].self_attn
    attention.v_proj.bias = attention.k_proj.bias
    result = describe_fp8_layout(
        model, {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}
    )
    assert not result["supported"]
    assert "shared parameter storage" in result["reason"]


def test_model_fp32_skip_rules_are_preserved():
    model = meta_llama()
    model._keep_in_fp32_modules = ["q_proj"]
    result = describe_fp8_layout(model)
    assert result["supported"], result
    assert all(
        matrix["name"] != "model.layers.0.self_attn.q_proj.weight" for matrix in result["matrices"]
    )


@pytest.mark.parametrize(
    ("tied", "vocab", "native"),
    [
        (True, 512, None),
        (True, 512, {"quant_method": "fp8", "modules_to_not_convert": ["lm_head", "self_attn.k"]}),
        (False, 70, {"quant_method": "fp8", "modules_to_not_convert": []}),
    ],
)
def test_fp8_storage_and_operator_names_match_the_installed_quantizer(tied, vocab, native):
    pytest.importorskip("triton")
    pytest.importorskip("transformers.integrations.finegrained_fp8")
    from transformers import FineGrainedFP8Config
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    from circuit_lantern.services.fp8 import is_fp8_linear

    model = meta_llama(tied=tied, vocab=vocab, bias=True)
    result = describe_fp8_layout(model, native)
    assert result["supported"], result
    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(**result["config"]), pre_quantized=native is not None
    )
    with torch.device("meta"):
        quantizer.preprocess_model(
            model, device_map={"": "cuda:0"}, keep_in_fp32_modules=model._keep_in_fp32_modules
        )
    assert {
        name + ".weight" for name, module in model.named_modules() if is_fp8_linear(module)
    } == {matrix["name"] for matrix in result["matrices"]}
    assert result["resident_weight_bytes"] == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    for matrix in result["matrices"]:
        module = model.get_submodule(matrix["name"].rsplit(".", 1)[0])
        assert list(module.weight_scale_inv.shape) == matrix["scale_shape"]
    assert all(parameter.device.type == "meta" for parameter in model.parameters())


def test_gpt2_conv1d_layers_are_not_mistaken_for_convertible_fp8_operators():
    from transformers import GPT2Config, GPT2LMHeadModel

    with torch.device("meta"):
        model = GPT2LMHeadModel(
            GPT2Config(n_embd=128, n_layer=1, n_head=2, n_positions=32, vocab_size=256)
        )
    model.tie_weights()
    result = describe_fp8_layout(model)
    assert not result["supported"]
    assert "no supported FP8 linear" in result["reason"]


def test_cached_architecture_exposes_the_fp8_plan_without_changing_logical_layout(tmp_path):
    from circuit_lantern.services.architecture import architecture_layout, describe_model_layout

    model = meta_llama()
    model.config.save_pretrained(tmp_path)
    result = architecture_layout(str(tmp_path))
    expected = describe_model_layout(model)
    assert {key: result[key] for key in expected} == expected
    assert result["fp8_layout"] == describe_fp8_layout(model)


def test_optional_fp8_discovery_error_preserves_the_standard_architecture_layout(tmp_path):
    from circuit_lantern.services.architecture import architecture_layout

    model = meta_llama()
    model.config.save_pretrained(tmp_path)
    with patch(
        "circuit_lantern.services.fp8_layout.HfQuantizer.get_modules_to_not_convert",
        side_effect=ImportError("Optional integration is unavailable."),
    ):
        result = architecture_layout(str(tmp_path))
    assert result["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert not result["fp8_layout"]["supported"]


@pytest.mark.parametrize(("hidden", "heads", "kv_heads"), [(256, 4, 2), (576, 9, 3), (896, 14, 2)])
def test_planned_on_the_fly_blocks_survive_real_transformers_parameter_quantization(
    hidden, heads, kv_heads
):
    pytest.importorskip("triton")
    pytest.importorskip("transformers.integrations.finegrained_fp8")
    from transformers import FineGrainedFP8Config
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    model = meta_llama(hidden=hidden, heads=heads, kv_heads=kv_heads)
    result = describe_fp8_layout(model)
    assert result["supported"], result
    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(**result["config"]), pre_quantized=False
    )
    with torch.device("meta"):
        quantizer.preprocess_model(
            model, device_map={"": "cuda:0"}, keep_in_fp32_modules=model._keep_in_fp32_modules
        )
    for matrix in result["matrices"]:
        values = torch.ones(matrix["shape"], dtype=torch.float32)
        quantizer.create_quantized_param(model, values, matrix["name"], torch.device("cpu"))
        module = model.get_submodule(matrix["name"].rsplit(".", 1)[0])
        assert list(module.weight_scale_inv.shape) == matrix["scale_shape"]
        assert module.weight.dtype == torch.float8_e4m3fn
        assert torch.isfinite(module.weight_scale_inv).all()


def test_generated_scales_cannot_have_a_single_block_dimension():
    result = describe_fp8_layout(meta_llama(hidden=128, heads=4, kv_heads=2))
    assert not result["supported"]
    assert "at least two blocks" in result["reason"]


def test_shared_module_with_a_skipped_first_path_cannot_hide_a_converted_alias():
    model = meta_llama(layers=2)
    model.model.layers[1].self_attn.q_proj = model.model.layers[0].self_attn.q_proj
    result = describe_fp8_layout(
        model,
        {
            "quant_method": "fp8",
            "modules_to_not_convert": ["lm_head", "model.layers.0.self_attn.q_proj"],
        },
    )
    assert not result["supported"]
    assert "shared parameter storage" in result["reason"]
