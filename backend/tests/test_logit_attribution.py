"""Compare additive diagnostics with actual model forwards, not reconstructed fixtures."""

import weakref
from unittest.mock import patch

import pytest
import torch
from model_fixtures import assert_no_forward_hooks
from transformers import (
    AutoModelForCausalLM,
    GPTNeoXConfig,
    LlamaConfig,
    Qwen3_5TextConfig,
)

from circuit_lantern.services.attribution import capture as capture_module
from circuit_lantern.services.attribution.normalization import (
    frozen_normalization,
    normalize_components,
)
from circuit_lantern.services.attribution.projection import target_records


def attribution(runtime, **kwargs):
    return runtime.trace("the capital of France is", 0, False, None, **kwargs)["logit_attribution"]


def assert_closes(result):
    assert result["status"] == "available", result
    values = torch.tensor([row["values"] for row in result["components"]], dtype=torch.float64)
    actual = torch.tensor(result["checks"]["actual_logits"], dtype=torch.float64)
    assert torch.allclose(values.sum(0), actual, atol=1e-5, rtol=1e-5)
    assert result["checks"]["status"] == "ok"
    assert result["checks"]["residual_maximum_error"] < 1e-12
    assert result["compute_dtype"] == "float64"


def test_default_predictions_close_without_another_model_forward(runtime):
    with patch.object(runtime.model, "forward", wraps=runtime.model.forward) as forward:
        trace = runtime.trace("the capital", 0, False, None)
    assert forward.call_count == 1
    result = trace["logit_attribution"]
    assert_closes(result)
    assert [target["token_id"] for target in result["targets"]] == [
        p["token_id"] for p in trace["predictions"][:2]
    ]
    assert result["selection_mode"] == "top_predictions"
    assert result["position"] == len(trace["token_ids"]) - 1
    assert result["token_id"] == trace["token_ids"][-1]
    assert result["normalization"]["kind"] == "layer_norm"
    assert result["normalization"]["statistic_dtype"] == "float64"
    assert result["retained_vectors"] is False


def test_all_tracked_answers_and_actual_predictions_are_retained_once(runtime):
    result = attribution(runtime, tracked_tokens=["France", "Poland", "<pad>"])
    assert_closes(result)
    assert [target["token_id"] for target in result["targets"][:3]] == [6, 9, 0]
    assert len(result["targets"]) <= 5
    assert len({target["token_id"] for target in result["targets"]}) == len(result["targets"])
    assert sorted(
        target["prediction_rank"] for target in result["targets"] if target["prediction_rank"]
    ) == [1, 2]
    assert all(target["patchable"] for target in result["targets"])


def test_equal_logits_break_prediction_ties_by_vocabulary_id(runtime):
    selected, mode = target_records(torch.zeros(16), runtime.tokenizer, [])
    assert [target["token_id"] for target in selected] == [0, 1]
    assert mode == "top_predictions"


@pytest.mark.parametrize("family", ["gpt_neox", "llama", "qwen35"])
def test_actual_parallel_and_rmsnorm_models_reconstruct_their_logits(runtime, family):
    torch.manual_seed(17)
    if family == "gpt_neox":
        config = GPTNeoXConfig(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=64,
            use_parallel_residual=True,
        )
    elif family == "llama":
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    else:
        config = Qwen3_5TextConfig(
            vocab_size=16,
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
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        )
    runtime.model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    result = attribution(runtime, tracked_tokens=["France", "Poland"])
    assert_closes(result)
    assert [row["layer"] for row in result["components"] if row["kind"] == "block"] == list(
        range(config.num_hidden_layers)
    )
    if family == "qwen35":
        assert result["normalization"]["weight_mode"] == "one_plus_weight"
        assert result["normalization"]["kind"] == "rms_norm"
    elif family == "llama":
        assert result["normalization"]["weight_mode"] == "weight"
    assert_no_forward_hooks(runtime.model.modules())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_layernorm_centers_each_component_and_counts_bias_once():
    norm = torch.nn.LayerNorm(3).double()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
        norm.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    vectors = torch.tensor([[10.0, 0.0, 2.0], [-5.0, 7.0, 7.0]], dtype=torch.float64)
    final = vectors.sum(0)
    spec = frozen_normalization(norm, final, torch.float64)
    normalized = normalize_components(vectors, spec)
    assert torch.allclose(normalized.sum(0) + spec["bias"], norm(final))
    assert spec["has_bias"] is True
    assert not torch.allclose(vectors * spec["gain"] / spec["denominator"], normalized)


def test_nonzero_norm_and_readout_biases_are_each_counted_once(runtime):
    with torch.no_grad():
        runtime.model.transformer.ln_f.bias.fill_(0.4)
        runtime.model.transformer.ln_f.weight.fill_(1.3)
    runtime.model.lm_head.bias = torch.nn.Parameter(torch.arange(16, dtype=torch.float32) / 7)
    result = attribution(runtime, tracked_tokens=["France", "Poland"])
    assert_closes(result)
    norm_row = next(row for row in result["components"] if row["kind"] == "norm_bias")
    head_row = next(row for row in result["components"] if row["kind"] == "head_bias")
    ids = [target["token_id"] for target in result["targets"]]
    assert norm_row["values"] == pytest.approx(
        (
            runtime.model.lm_head.weight[ids].double()
            @ torch.full((16,), 0.4, dtype=torch.float32).double()
        ).tolist()
    )
    assert head_row["values"] == pytest.approx(runtime.model.lm_head.bias[ids].tolist())


def test_identity_final_norm_has_an_explicit_identity_map(runtime):
    runtime.model.transformer.ln_f = torch.nn.Identity()
    result = attribution(runtime)
    assert_closes(result)
    assert result["normalization"]["kind"] == "identity"


def test_unknown_norm_is_unavailable_without_breaking_the_trace(runtime):
    class CustomNorm(torch.nn.LayerNorm):
        pass

    runtime.model.transformer.ln_f = CustomNorm(16)
    result = attribution(runtime)
    assert result["status"] == "unavailable"
    assert result["reason"] == "unsupported_norm"


@pytest.mark.parametrize(
    "field,value",
    [("final_logit_softcapping", 30.0), ("logit_scale", 2.0), ("logits_scaling", 0.0)],
)
def test_declared_output_transforms_are_not_treated_as_linear(runtime, field, value):
    setattr(runtime.model.config, field, value)
    result = attribution(runtime)
    assert result["status"] == "unavailable"
    assert result["reason"] == "output_transform"


def test_undeclared_output_transform_is_detected_against_actual_logits(runtime):
    forward = runtime.model.forward

    def scaled(*args, **kwargs):
        output = forward(*args, **kwargs)
        output.logits = output.logits * 2
        return output

    with patch.object(runtime.model, "forward", new=scaled):
        result = attribution(runtime)
    assert result["status"] == "unavailable"
    assert result["reason"] == "readout_mismatch"
    assert max(result["checks"]["head_absolute_errors"]) > 1e-5


def test_interblock_transform_is_not_silently_charged_to_a_block(runtime):
    def shift(module, args, kwargs):
        kwargs["hidden_states"] = args[0] + 0.1
        return (), kwargs

    handle = runtime._blocks()[1].register_forward_pre_hook(shift, with_kwargs=True)
    try:
        result = attribution(runtime)
    finally:
        handle.remove()
    assert result["status"] == "unavailable"
    assert result["reason"] == "residual_discontinuity"


def test_capture_budget_is_bounded_without_breaking_ordinary_tracing(runtime, monkeypatch):
    monkeypatch.setattr(capture_module, "MAX_CAPTURE_BYTES", 1)
    result = attribution(runtime)
    assert result["reason"] == "capture_budget"
    assert_no_forward_hooks(runtime.model.modules())


def test_bypassed_block_contribution_is_zero_in_the_captured_intervened_run(runtime):
    trace = runtime.trace("the capital of France is", 0, False, 0)
    result = trace["logit_attribution"]
    assert_closes(result)
    first = next(
        row for row in result["components"] if row["kind"] == "block" and row["layer"] == 0
    )
    assert first["values"] == [0.0] * len(result["targets"])


def test_capture_tensors_and_hooks_are_released_before_generation(runtime):
    references = []
    original = capture_module.last_vector

    def record(value):
        result = original(value)
        references.append(weakref.ref(result))
        return result

    def generate(**kwargs):
        assert references and all(reference() is None for reference in references)
        assert_no_forward_hooks(runtime.model.modules())
        assert all(not module._forward_pre_hooks for module in runtime.model.modules())
        return kwargs["input_ids"]

    with (
        patch.object(capture_module, "last_vector", new=record),
        patch.object(runtime.model, "generate", new=generate),
    ):
        runtime.trace("the capital", 1, False, None)
    assert references and all(reference() is None for reference in references)


def test_forward_failure_removes_every_attribution_hook(runtime):
    original = runtime.model.forward

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure")

    with (
        patch.object(runtime.model, "forward", new=fail),
        pytest.raises(RuntimeError, match="injected"),
    ):
        attribution(runtime)
    assert_no_forward_hooks(runtime.model.modules())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_low_precision_rounding_is_exposed(runtime, dtype):
    runtime.model.to(dtype=dtype)
    result = attribution(runtime, tracked_tokens=["France", "Poland"])
    assert result["status"] == "available", result
    assert result["captured_dtype"] == str(dtype).removeprefix("torch.")
    checks = result["checks"]
    assert any(error > 0 for error in checks["absolute_errors"])
    assert all(
        error <= allowance
        for error, allowance in zip(checks["absolute_errors"], checks["rounding_allowances"])
    )
    assert checks["actual_logits"] != checks["reconstructed_logits"]


def test_nested_decoder_output_transform_is_unavailable(runtime):
    from types import SimpleNamespace

    runtime.model.config.text_config = SimpleNamespace(final_logit_softcapping=20.0)
    result = attribution(runtime)
    assert result["status"] == "unavailable"
    assert result["reason"] == "output_transform"


def test_output_rounding_dtype_is_included_in_native_epsilon(runtime):
    original = runtime.model.forward

    def rounded(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits = output.logits.to(torch.bfloat16)
        return output

    with patch.object(runtime.model, "forward", new=rounded):
        result = attribution(runtime)
    assert result["status"] == "available"
    assert result["captured_dtype"] == "float32"
    assert result["checks"]["output_dtype"] == "bfloat16"
    assert result["checks"]["native_epsilon"] == torch.finfo(torch.bfloat16).eps
    assert any(error > 0 for error in result["checks"]["absolute_errors"])


def test_projection_failure_preserves_ordinary_trace_and_removes_hooks(runtime):
    with patch(
        "circuit_lantern.services.attribution.projection.decompose",
        side_effect=RuntimeError("diagnostic failed"),
    ):
        trace = runtime.trace("the capital", 0, False, None)
    assert trace["predictions"] and trace["layers"]
    assert trace["logit_attribution"]["reason"] == "diagnostic_error"
    assert_no_forward_hooks(runtime.model.modules())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_partial_installation_failure_does_not_break_tracing_or_leak_hooks(runtime):
    with patch.object(
        runtime.model.lm_head,
        "register_forward_pre_hook",
        side_effect=RuntimeError("unsupported hook"),
    ):
        trace = runtime.trace("the capital", 0, False, None)
    assert trace["logit_attribution"]["reason"] == "diagnostic_error"
    assert trace["predictions"]
    assert_no_forward_hooks(runtime.model.modules())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_norm_lookup_failure_leaves_no_capture_or_intervention_hooks(runtime):
    with (
        patch.object(runtime, "_final_norm", side_effect=RuntimeError("bad lookup")),
        pytest.raises(RuntimeError, match="bad lookup"),
    ):
        runtime.trace("the capital", 0, False, 0)
    assert_no_forward_hooks(runtime.model.modules())
    assert all(not module._forward_pre_hooks for module in runtime.model.modules())


def test_diagnostic_setup_failure_can_leave_the_original_trace_usable(runtime):
    original = runtime._final_norm
    calls = 0

    def lookup():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("only diagnostic lookup failed")
        return original()

    with patch.object(runtime, "_final_norm", new=lookup):
        trace = runtime.trace("the capital", 0, False, None)
    assert trace["logit_attribution"]["reason"] == "diagnostic_error"
    assert trace["predictions"]
    assert_no_forward_hooks(runtime.model.modules())


def test_rounding_allowance_uses_observed_absolute_products_for_cancellation():
    from circuit_lantern.services.attribution.projection import tolerances

    actual = torch.tensor([0.001], dtype=torch.float64)
    magnitude = torch.tensor([1000.0], dtype=torch.float64)
    strict, allowance = tolerances(actual, magnitude, torch.finfo(torch.bfloat16).eps)
    assert float(strict[0]) == pytest.approx(2e-5)
    assert float(allowance[0]) == pytest.approx(4 * torch.finfo(torch.bfloat16).eps * 1001)


def test_shared_blocks_are_explicitly_unavailable(runtime):
    capture = capture_module.AttributionCapture(
        [runtime._blocks()[0]] * 2,
        runtime._final_norm(),
        runtime.model.lm_head,
        runtime.model.config,
    )
    capture.install()
    assert capture.reason == "reused_blocks"
    assert capture.handles == []


def test_non_roundtripping_candidates_keep_their_ids_but_cannot_prepare_patching(runtime):
    with patch.object(runtime.tokenizer, "encode", return_value=[14]):
        selected, _ = target_records(torch.zeros(16), runtime.tokenizer, [])
    assert [target["token_id"] for target in selected] == [0, 1]
    assert not any(target["patchable"] for target in selected)


def test_trailing_padded_prompt_is_not_mislabeled_as_a_real_final_token(runtime):
    original = runtime.tokenizer

    class PaddedTokenizer:
        def __getattr__(self, name):
            return getattr(original, name)

        def __call__(self, *args, **kwargs):
            result = original(*args, **kwargs)
            result["attention_mask"][0, -1] = 0
            return result

    runtime.tokenizer = PaddedTokenizer()
    result = attribution(runtime)
    assert result["status"] == "unavailable"
    assert result["reason"] == "padded_prompt"
