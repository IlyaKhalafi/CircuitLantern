import json
from unittest.mock import patch

import pytest
import torch
from model_fixtures import tiny_gpt2

from circuit_lantern.config import Settings
from circuit_lantern.services.architecture import architecture_layout, describe_model_layout
from circuit_lantern.services.causal import activation_patch_sweep
from circuit_lantern.services.execution import checkpoint_precision, quantization_capabilities
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.runtime import ModelRuntime


@pytest.fixture(params=["cpu", "cuda"])
def execution_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this environment.")
    return request.param


@pytest.fixture(params=["float16", "bfloat16", "int4", "int8"])
def precision_runtime(request, tmp_path, execution_device):
    if request.param in ("int4", "int8"):
        pytest.importorskip("bitsandbytes")
        if (
            not quantization_capabilities()[execution_device]
            .get(request.param, {})
            .get("supported")
        ):
            pytest.skip("The installed CPU quantization kernel is unavailable.")
    model, tokenizer = tiny_gpt2()
    checkpoint = tmp_path / "source"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    metadata = {
        "id": "local-precision-test",
        "revision": "local-revision",
        "context_length": 64,
        **describe_model_layout(model),
    }
    instance = ModelRuntime(Settings(data_dir=tmp_path))
    job = JobRecord({"status": "queued", "progress": 0})
    with patch("circuit_lantern.services.runtime.snapshot_download", return_value=str(checkpoint)):
        instance._load(job, metadata, execution_device, request.param, 32)
    assert job["status"] == "ready", job.snapshot()
    yield instance, metadata, request.param
    instance.executor.shutdown(wait=True)


def test_execution_precision_propagates_through_real_traces(precision_runtime):
    runtime, _, precision = precision_runtime
    trace = runtime.trace("the capital of France is", 1, True, None)
    assert trace["execution_precision"] == precision
    assert trace["execution_config"]["precision"] == precision
    assert trace["execution_config"]["device"] == runtime.device
    assert trace["execution_config"]["context_length"] == runtime.state()["context_length"] == 32
    assert bool(trace["execution_config"]["quantization"]) == (precision in ("int4", "int8"))
    assert len(trace["layers"]) == 3
    assert len(trace["geometry"]["cka"]) == 3
    assert len(trace["geometry"]["sampled_positions"]) == len(trace["tokens"])
    assert len(trace["attention"]) == 2
    assert len(trace["attention"][0]["query_by_head"]) == 2
    assert len(trace["attention"][0]["query_by_head"][0]) == len(trace["tokens"])
    assert trace["layers"][-1]["norm"] > 0
    assert all(0 <= p["probability"] <= 1 for p in trace["predictions"])


def test_weight_inventory_counts_logical_parameters(precision_runtime):
    runtime, original, precision = precision_runtime
    graph = runtime.graph()
    assert graph["total_parameters"] == original["parameters"]
    weight = next(w for w in graph["matrices"] if w["name"] == "transformer.h.0.attn.c_attn.weight")
    assert weight["elements"] == 16 * 48
    if precision == "int4":
        assert weight["shape"] == [48, 16]
        assert weight["storage_shape"] != weight["shape"]
        assert weight["dtype"] == "nf4"
    elif precision == "int8":
        assert weight["shape"] == [48, 16]
        assert weight["dtype"] == "int8"
    else:
        assert weight["shape"] == [16, 48]
        assert weight["dtype"] == precision


def test_weight_metrics_use_dequantized_values(precision_runtime):
    runtime, _, precision = precision_runtime
    metrics = runtime.matrix("transformer.h.0.attn.c_attn.weight")
    parameter = runtime.model.transformer.h[0].attn.c_attn.weight
    if precision == "int4":
        from bitsandbytes.functional import dequantize_4bit

        expected = dequantize_4bit(parameter.data, parameter.quant_state).float()
    elif precision == "int8":
        from bitsandbytes.functional import int8_vectorwise_dequant

        scales = parameter.SCB
        if scales is None:
            scales = runtime.model.transformer.h[0].attn.c_attn.state.SCB
        expected = int8_vectorwise_dequant(parameter.data, scales)
    else:
        expected = parameter.detach().float()
    assert metrics["shape"] == list(expected.shape)
    assert metrics["frobenius_norm"] == pytest.approx(float(expected.norm()), rel=1e-5)
    assert metrics["dtype"] == "float32"
    assert metrics["execution_precision"] == precision


def test_prequantized_safetensors_roundtrip(precision_runtime, tmp_path):
    runtime, metadata, precision = precision_runtime
    if precision not in ("int4", "int8"):
        pytest.skip("This case verifies packed checkpoint loading.")
    checkpoint = tmp_path / "packed"
    expected = runtime.matrix("transformer.h.0.attn.c_attn.weight")["frobenius_norm"]
    runtime.model.save_pretrained(checkpoint, safe_serialization=True)
    runtime.tokenizer.save_pretrained(checkpoint)
    raw = json.loads((checkpoint / "config.json").read_text())
    metadata = dict(metadata, quantization=raw["quantization_config"])
    assert checkpoint_precision(metadata) == precision
    assert architecture_layout(str(checkpoint))["parameters"] == metadata["parameters"]
    job = JobRecord({"status": "queued", "progress": 0})
    with patch("circuit_lantern.services.runtime.snapshot_download", return_value=str(checkpoint)):
        runtime._load(job, metadata, runtime.device, "auto", 32)
    assert job["status"] == "ready", job.snapshot()
    assert runtime.matrix("transformer.h.0.attn.c_attn.weight")["frobenius_norm"] == pytest.approx(
        expected
    )
    assert runtime.trace("the capital", 1, False, None)["execution_precision"] == precision


def test_metadata_layout_uses_meta_tensors_without_allocating_weights(tmp_path):
    model, _ = tiny_gpt2()
    model.config.save_pretrained(tmp_path)
    measured = describe_model_layout(model)
    assert architecture_layout(str(tmp_path)) == measured
    assert measured["quantizable_parameters"] < measured["parameters"]


def test_quantized_causal_sweep_keeps_canonical_weights_and_removes_hooks(precision_runtime):
    runtime, _, precision = precision_runtime
    name = "transformer.h.0.attn.c_attn.weight"
    before = runtime.matrix(name)["frobenius_norm"]
    result = activation_patch_sweep(
        runtime.model,
        runtime.tokenizer,
        runtime._blocks(),
        runtime.metadata,
        runtime.device,
        clean_prompt="the capital of France is",
        corrupted_prompt="the capital of Germany is",
        target="",
        position=None,
        job={},
        max_tokens=64,
    )
    assert result["execution_precision"] == precision
    assert len(result["layers"]) == 2
    assert result["layers"][-1]["log_probability"] == pytest.approx(
        result["clean"]["log_probability"], abs=1e-5
    )
    assert all(not block._forward_hooks for block in runtime._blocks())
    runtime.metrics_cache.clear()
    assert runtime.matrix(name)["frobenius_norm"] == pytest.approx(before)


def test_native_bnb_fp4_checkpoint_can_be_traced_and_analyzed(tmp_path):
    pytest.importorskip("bitsandbytes")
    if not quantization_capabilities()["cpu"].get("int4", {}).get("supported"):
        pytest.skip("The installed CPU quantization kernel is unavailable.")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model, tokenizer = tiny_gpt2()
    original = tmp_path / "original"
    packed = tmp_path / "packed"
    model.save_pretrained(original, safe_serialization=True)
    metadata = dict(
        id="fp4-checkpoint",
        revision="local",
        context_length=64,
        **describe_model_layout(model),
    )
    model = AutoModelForCausalLM.from_pretrained(
        original,
        dtype=torch.float32,
        device_map={"": "cpu"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4"),
        attn_implementation="eager",
        local_files_only=True,
    )
    model.save_pretrained(packed, safe_serialization=True)
    tokenizer.save_pretrained(packed)
    metadata["quantization"] = model.config.quantization_config.to_dict()
    instance = ModelRuntime(Settings(data_dir=tmp_path))
    job = {}
    try:
        with patch("circuit_lantern.services.runtime.snapshot_download", return_value=str(packed)):
            instance._load(job, metadata, "cpu")
        assert job["status"] == "ready", job
        trace = instance.trace("the capital", 1, False, None)
        assert trace["execution_config"]["quantization"]["bnb_4bit_quant_type"] == "fp4"
        metrics = instance.matrix("transformer.h.0.attn.c_attn.weight")
        assert metrics["source_format"] == "fp4"
        assert metrics["shape"] == [48, 16]
        assert metrics["frobenius_norm"] > 0
    finally:
        instance.executor.shutdown(wait=True)
