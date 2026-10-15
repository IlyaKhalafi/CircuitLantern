from circuit_lantern.schemas import CompatibilityRequest
from circuit_lantern.services.compatibility import assess_compatibility


def metadata(**changes):
    return dict(
        parameters=124_000_000,
        dtype="float32",
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        hidden_size=768,
        context_length=1024,
        has_safetensors=True,
        weight_bytes=500_000_000,
        **changes,
    )


def hardware(capability=9, memory=20 * 1024**3):
    return {
        "ram_available_bytes": memory,
        "disk_free_bytes": 30 * 1024**3,
        "cuda_available": True,
        "gpus": [{"compute_capability": capability, "free_bytes": memory}],
    }


def test_unquantized_cpu_model_fits():
    result = assess_compatibility(
        metadata(), hardware(), CompatibilityRequest(model_id="gpt2", context_length=128)
    )
    assert result["compatible"]
    assert result["required_bytes"] > result["weight_bytes"]


def test_nvfp4_rejected_on_hopper_even_with_enough_memory():
    result = assess_compatibility(
        metadata(quantization={"quant_method": "modelopt", "format": "nvfp4"}),
        hardware(),
        CompatibilityRequest(model_id="model", device="cuda", context_length=128),
    )
    assert not result["compatible"]
    assert any("Blackwell" in blocker for blocker in result["blockers"])


def test_blackwell_does_not_imply_packed_runtime_support():
    result = assess_compatibility(
        metadata(quantization={"quant_method": "modelopt", "format": "nvfp4"}),
        hardware(capability=10),
        CompatibilityRequest(model_id="model", device="cuda", context_length=128),
    )
    assert not result["compatible"]
    assert not any("requires a Blackwell" in blocker for blocker in result["blockers"])
    assert any("Packed weight" in blocker for blocker in result["blockers"])


def test_free_memory_not_total_memory_controls_fit():
    result = assess_compatibility(
        metadata(),
        hardware(memory=128 * 1024**2),
        CompatibilityRequest(model_id="gpt2", context_length=128),
    )
    assert not result["compatible"]
    assert any("Insufficient available memory" in blocker for blocker in result["blockers"])


def test_context_window_is_enforced():
    result = assess_compatibility(
        metadata(), hardware(), CompatibilityRequest(model_id="gpt2", context_length=2048)
    )
    assert any("context window" in blocker for blocker in result["blockers"])


def test_safetensors_required():
    model = metadata()
    model["has_safetensors"] = False
    result = assess_compatibility(
        model, hardware(), CompatibilityRequest(model_id="gpt2", context_length=128)
    )
    assert any("safetensors" in blocker for blocker in result["blockers"])
