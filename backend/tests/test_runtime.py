from unittest.mock import patch

import pytest
from model_fixtures import assert_no_forward_hooks


def test_trace_last_residual_is_before_final_normalization(runtime):
    captured = []
    handle = runtime.model.transformer.h[-1].register_forward_hook(
        lambda m, a, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().clone())
    )
    trace = runtime.trace("the capital of France is", 1, True, None)
    handle.remove()
    expected = float(captured[0][0, -1].norm())
    assert trace["layers"][-1]["norm"] == pytest.approx(expected)
    assert trace["final_state_normalized"] is False
    assert len(trace["attention"]) == 2


def test_block_bypass_preserves_input_residual_and_hooks_are_removed(runtime):
    trace = runtime.trace("the capital", 1, False, 1)
    assert trace["layers"][1]["norm"] == pytest.approx(trace["layers"][2]["norm"])
    assert trace["layers"][2]["update_norm"] == 0
    assert trace["layers"][2]["relative_update"] == 0
    assert_no_forward_hooks(runtime.model.transformer.h)


def test_hook_cleanup_even_when_generation_fails(runtime):
    with (
        patch.object(runtime.model, "generate", side_effect=RuntimeError("generation failed")),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        runtime.trace("the capital", 1, False, 0)
    assert_no_forward_hooks(runtime.model.transformer.h)


def test_failed_download_preserves_active_model(runtime):
    active_model = runtime.model
    job = {"status": "queued"}
    with patch(
        "circuit_lantern.services.runtime.snapshot_download", side_effect=OSError("network down")
    ):
        runtime._load(job, {"id": "missing", "revision": "main"}, "cpu")
    assert runtime.model is active_model
    assert job["status"] == "failed"


def test_invalid_interventions_are_rejected(runtime):
    with pytest.raises(ValueError, match="cannot be bypassed"):
        runtime.trace("the capital", 1, False, -1)
    with pytest.raises(ValueError, match="between 0 and 64"):
        runtime.trace("the capital", -1, False, None)


def test_model_context_reserves_generated_tokens(runtime):
    with pytest.raises(ValueError, match="too long"):
        runtime.trace("the " * 60, 8, False, None)


def test_invalid_execution_options_preserve_the_active_model(runtime):
    active_model = runtime.model
    job = {"status": "queued"}
    with patch("circuit_lantern.services.runtime.snapshot_download", return_value="unused"):
        runtime._load(job, {"id": "replacement", "revision": "main"}, "cpu", "nvfp4")
    assert job["status"] == "failed"
    assert runtime.model is active_model
    assert runtime.state()["loaded"]
    assert "previous model was released" not in job["message"]


def test_nonfinite_logits_fail_cleanly_and_remove_capture_hooks(runtime):
    handle = runtime.model.get_output_embeddings().register_forward_hook(
        lambda module, args, output: output * float("nan")
    )
    try:
        with pytest.raises(ValueError, match="non finite logits"):
            runtime.trace("the capital", 1, True, None)
    finally:
        handle.remove()
    assert_no_forward_hooks(runtime._blocks())


def test_loaded_context_allowance_bounds_prompt_and_generation_together(runtime):
    runtime.metadata["execution_context_length"] = 16
    with pytest.raises(ValueError, match="at most 8 tokens"):
        runtime.trace("the " * 9, 8, False, None)
    trace = runtime.trace("the " * 9, 4, False, None)
    assert len(trace["tokens"]) == 9
    assert len(trace["layers"]) == 3


def test_generation_must_leave_room_for_a_prompt_token(runtime):
    runtime.metadata["execution_context_length"] = 16
    with pytest.raises(ValueError, match="generation length exceeds"):
        runtime.trace("the", 16, False, None)


def test_native_compute_mismatch_is_rejected_before_download_or_model_release(runtime):
    active_model = runtime.model
    job = {"status": "queued"}
    metadata = dict(
        runtime.metadata,
        quantization={
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("circuit_lantern.services.runtime.snapshot_download") as download,
    ):
        runtime._load(job, metadata, "cuda", "auto")
    assert job["status"] == "failed"
    assert "Ampere" in job["message"]
    assert runtime.model is active_model
    download.assert_not_called()


def test_training_int8_checkpoint_preserves_active_model_without_downloading(runtime):
    from transformers import BitsAndBytesConfig

    active_model = runtime.model
    active_tokenizer = runtime.tokenizer
    active_metadata = runtime.metadata
    config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_has_fp16_weight=True)
    metadata = dict(active_metadata, quantization=config.to_dict())
    job = {"status": "queued"}
    with patch("circuit_lantern.services.runtime.snapshot_download") as download:
        runtime._load(job, metadata, "cpu")
    download.assert_not_called()
    assert job["status"] == "failed"
    assert "master weights" in job["message"]
    assert runtime.model is active_model
    assert runtime.tokenizer is active_tokenizer
    assert runtime.metadata is active_metadata
    assert runtime.state()["loaded"]
