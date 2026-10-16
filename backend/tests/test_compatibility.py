import pytest

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


def quantized_hardware(capability=9):
    result = hardware(capability)
    result["quantization"] = {
        device: {precision: {"supported": True} for precision in ("int4", "int8")}
        for device in ("cpu", "cuda")
    }
    return result


def test_nf4_budgets_unquantized_modules_and_float32_matrix_workspace():
    model = metadata(quantizable_parameters=100_000_000, largest_matrix_elements=10_000_000)
    result = assess_compatibility(
        model,
        quantized_hardware(),
        CompatibilityRequest(
            model_id="model",
            precision="int4",
            context_length=128,
        ),
    )
    assert result["compatible"]
    assert result["weight_bytes"] == 24_000_000 * 4 + 100_000_000 * 0.625
    assert result["analysis_workspace_bytes"] == 80_000_000
    assert result["download_bytes"] == model["weight_bytes"]


def test_selected_float_precision_changes_memory_but_not_source_download():
    reports = [
        assess_compatibility(
            metadata(largest_matrix_elements=1000),
            hardware(),
            CompatibilityRequest(
                model_id="model",
                precision=precision,
                context_length=128,
            ),
        )
        for precision in ("float32", "float16", "bfloat16")
    ]
    assert all(result["compatible"] for result in reports)
    assert reports[0]["weight_bytes"] == reports[1]["weight_bytes"] * 2
    assert reports[1]["weight_bytes"] == reports[2]["weight_bytes"]
    assert len({report["download_bytes"] for report in reports}) == 1


def test_bnb_fp4_is_distinct_from_native_nvfp4_and_preserves_native_format():
    result = assess_compatibility(
        metadata(
            quantizable_parameters=100_000_000,
            quantization={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
            },
        ),
        quantized_hardware(capability=7),
        CompatibilityRequest(
            model_id="model",
            device="cuda",
            context_length=128,
        ),
    )
    assert result["compatible"]
    assert result["precision"] == "int4"
    assert result["runtime_precision"].startswith("FP4")


def test_kernel_presence_does_not_override_int8_gpu_architecture_requirement():
    result = assess_compatibility(
        metadata(quantizable_parameters=100_000_000),
        quantized_hardware(capability=6),
        CompatibilityRequest(model_id="model", device="cuda", precision="int8", context_length=128),
    )
    assert any("Turing" in blocker for blocker in result["blockers"])


def test_execution_device_memory_is_used_instead_of_first_physical_gpu():
    machine = hardware(memory=20 * 1024**3)
    machine["execution_gpu"] = {"compute_capability": 8, "free_bytes": 64 * 1024**2}
    result = assess_compatibility(
        metadata(),
        machine,
        CompatibilityRequest(
            model_id="model",
            device="cuda",
            context_length=128,
        ),
    )
    assert not result["compatible"]
    assert result["available_bytes"] == int(64 * 1024**2 * 0.85)


def test_missing_quantization_kernel_or_layout_blocks_execution():
    result = assess_compatibility(
        metadata(),
        hardware(),
        CompatibilityRequest(
            model_id="model",
            precision="int4",
            context_length=128,
        ),
    )
    assert not result["compatible"]
    assert any("unavailable" in blocker for blocker in result["blockers"])
    assert any("layout" in blocker for blocker in result["blockers"])


def test_training_style_int8_master_weights_are_not_mistaken_for_packed_storage():
    result = assess_compatibility(
        metadata(
            quantizable_parameters=100_000_000,
            quantization={
                "quant_method": "bitsandbytes",
                "load_in_8bit": True,
                "llm_int8_has_fp16_weights": True,
            },
        ),
        quantized_hardware(),
        CompatibilityRequest(model_id="model", context_length=128),
    )
    assert not result["compatible"]
    assert any("master weights" in blocker for blocker in result["blockers"])


def test_logit_and_mlp_buffers_are_included_in_analysis_memory():
    model = metadata(vocab_size=250_000, config={"intermediate_size": 4096})
    result = assess_compatibility(
        model,
        hardware(),
        CompatibilityRequest(
            model_id="model",
            context_length=128,
            batch_size=2,
        ),
    )
    assert result["logits_bytes"] == 5 * 2 * 128 * 250_000 * 4
    assert result["activation_bytes"] == 3 * 2 * 128 * 4096 * 4
    assert result["overhead_bytes"] > result["logits_bytes"] + result["activation_bytes"]


@pytest.mark.parametrize("context_limit", [None, 0, -1, "unknown", True])
def test_unknown_context_window_blocks_fit_without_a_server_error(context_limit):
    model = metadata()
    model["context_length"] = context_limit
    result = assess_compatibility(
        model, hardware(), CompatibilityRequest(model_id="model", context_length=128)
    )
    assert not result["compatible"]
    assert any("context window could not be determined" in b for b in result["blockers"])


def test_cpu_fit_does_not_require_a_physical_gpu_inventory():
    machine = hardware()
    del machine["gpus"]
    machine["cuda_available"] = False
    result = assess_compatibility(
        metadata(), machine, CompatibilityRequest(model_id="model", context_length=128)
    )
    assert result["compatible"]


def test_cached_checkpoint_uses_remaining_download_for_disk_fit():
    machine = hardware()
    machine["disk_free_bytes"] = 20_000_000
    model = metadata()
    request = CompatibilityRequest(model_id="model", context_length=128)
    assert not assess_compatibility(model, machine, request)["compatible"]
    report = assess_compatibility(
        model,
        machine,
        request,
        cache={"cached_bytes": 490_000_000, "download_bytes": 10_000_000},
    )
    assert report["compatible"]
    assert report["source_bytes"] == 500_000_000
    assert report["download_bytes"] == 10_000_000
    assert report["cached_bytes"] == 490_000_000
    machine["disk_free_bytes"] = 0
    full = assess_compatibility(
        model,
        machine,
        request,
        cache={"cached_bytes": 500_000_000, "download_bytes": 0},
    )
    assert full["compatible"]
    machine["max_model_bytes"] = 400_000_000
    limited = assess_compatibility(
        model,
        machine,
        request,
        cache={"cached_bytes": 500_000_000, "download_bytes": 0},
    )
    assert any("configured download limit" in b for b in limited["blockers"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replacement_can_fit_using_only_the_active_model_storage(device):
    machine = hardware(memory=128 * 1024**2)
    machine["ram_available_bytes"] = 4 * 1024**3 if device == "cuda" else 128 * 1024**2
    request = CompatibilityRequest(model_id="model", context_length=128, device=device)
    assert not assess_compatibility(metadata(), machine, request)["compatible"]
    machine["reclaimable_tensor_bytes"] = {
        "cpu" if device == "cpu" else "cuda:0": 2 * 1024**3,
        "cuda:1": 100 * 1024**3,
    }
    result = assess_compatibility(metadata(), machine, request)
    if device == "cuda":
        assert result["compatible"]
        assert result["reclaimable_bytes"] == 2 * 1024**3
        assert result["available_bytes"] == int((128 * 1024**2 + 2 * 1024**3) * 0.85)
    else:
        assert not result["compatible"]
        assert result["reclaimable_bytes"] == 0


def test_gpu_replacement_budget_cannot_exceed_device_capacity():
    machine = hardware(memory=128 * 1024**2)
    machine["ram_available_bytes"] = 4 * 1024**3
    machine["gpus"][0]["total_bytes"] = 2 * 1024**3
    machine["reclaimable_tensor_bytes"] = {"cuda:0": 3 * 1024**3}
    result = assess_compatibility(
        metadata(),
        machine,
        CompatibilityRequest(model_id="model", context_length=128, device="cuda"),
    )
    assert result["available_bytes"] == int(2 * 1024**3 * 0.85)


def test_host_memory_is_not_credited_for_cuda_staging_or_unreadable_cgroup_usage():
    machine = hardware()
    machine["ram_available_bytes"] = 0
    machine["ram_constraints"] = [{"used_bytes": None}]
    machine["reclaimable_tensor_bytes"] = {"cpu": 2 * 1024**3, "cuda:1": 8 * 1024**3}
    result = assess_compatibility(
        metadata(),
        machine,
        CompatibilityRequest(model_id="model", context_length=128, device="cuda"),
    )
    assert not result["compatible"]
    assert any("host RAM" in b for b in result["blockers"])
    assert result["reclaimable_bytes"] == 0
    assert result["available_bytes"] == int(20 * 1024**3 * 0.85)


@pytest.mark.parametrize("kind", ["fp4", "nf4"])
@pytest.mark.parametrize(
    ("capability", "compute", "fits"),
    [(7.5, "bfloat16", False), (8, "bfloat16", True), (6, "float32", True), (7, "float16", True)],
)
def test_native_four_bit_compute_dtype_adds_its_own_gpu_requirement(
    kind, capability, compute, fits
):
    result = assess_compatibility(
        metadata(
            quantizable_parameters=100_000_000,
            quantization={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": kind,
                "bnb_4bit_compute_dtype": compute,
            },
        ),
        quantized_hardware(capability=capability),
        CompatibilityRequest(model_id="model", device="cuda", context_length=128),
    )
    assert result["compatible"] is fits
    if not fits:
        assert any("BF16 compute" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("compute", ["float64", "int8", "torch.float32", {"dtype": "float32"}])
def test_unsupported_native_compute_dtype_blocks_an_otherwise_fitting_checkpoint(compute):
    result = assess_compatibility(
        metadata(
            quantizable_parameters=100_000_000,
            quantization={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": compute,
            },
        ),
        quantized_hardware(),
        CompatibilityRequest(model_id="model", device="cuda", context_length=128),
    )
    assert not result["compatible"]
    assert any(
        "unsupported bitsandbytes compute dtype" in blocker for blocker in result["blockers"]
    )


@pytest.mark.parametrize("kind", [None, {"format": "fp4"}])
def test_invalid_native_codebook_returns_a_fit_report_instead_of_a_server_error(kind):
    result = assess_compatibility(
        metadata(
            quantizable_parameters=100_000_000,
            quantization={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": kind,
            },
        ),
        quantized_hardware(),
        CompatibilityRequest(model_id="model", device="cpu", context_length=128),
    )
    assert not result["compatible"]
    assert any("matrix adapter" in blocker for blocker in result["blockers"])
