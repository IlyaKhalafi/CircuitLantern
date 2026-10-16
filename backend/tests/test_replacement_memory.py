from unittest.mock import patch

import pytest
import torch
from model_fixtures import checkpoint_metadata
from torch import nn

from circuit_lantern.schemas import CompatibilityRequest
from circuit_lantern.services.runtime import RuntimeBusyError
from circuit_lantern.services.weights import resident_tensor_bytes


def test_resident_storage_counts_tied_views_once_and_ignores_meta_tensors():
    model = nn.Module()
    storage = torch.arange(20, dtype=torch.float32)
    model.register_parameter("first", nn.Parameter(storage))
    model.register_parameter("view", nn.Parameter(storage[2:10]))
    model.register_parameter("meta", nn.Parameter(torch.empty(7, device="meta")))
    model.register_buffer("shared", storage[:2])
    model.register_buffer("extra", torch.ones(3, dtype=torch.float64))
    assert resident_tensor_bytes(model) == {"cpu": 104}
    assert resident_tensor_bytes(None) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not installed")
def test_resident_memory_separates_host_and_execution_device_storage():
    model = nn.Module()
    model.register_buffer("host", torch.ones(7, dtype=torch.float32))
    model.register_parameter("weight", nn.Parameter(torch.ones(16, device="cuda:0")))
    assert resident_tensor_bytes(model) == {"cpu": 28, "cuda:0": 64}


def test_fit_serializes_free_memory_and_owned_tensor_accounting(runtime):
    checkpoint = runtime.settings.data_dir / "source"
    runtime.model.save_pretrained(checkpoint, safe_serialization=True)
    metadata = checkpoint_metadata(runtime.model, checkpoint)
    machine = {
        "ram_available_bytes": 4 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "gpus": [],
        "cuda_available": False,
    }
    request = CompatibilityRequest(model_id=metadata["id"], context_length=32)
    with patch("circuit_lantern.services.runtime.hardware_snapshot", return_value=machine):
        assert runtime.assess_fit(metadata, request)["compatible"]
        with runtime.lock, pytest.raises(RuntimeBusyError, match="busy"):
            runtime.assess_fit(metadata, request)


def test_replacement_rechecks_resources_after_download_and_preserves_the_active_model(runtime):
    checkpoint = runtime.settings.data_dir / "source"
    runtime.model.save_pretrained(checkpoint, safe_serialization=True)
    runtime.tokenizer.save_pretrained(checkpoint)
    metadata = checkpoint_metadata(runtime.model, checkpoint, "replacement")
    machine = {
        "ram_available_bytes": 4 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "gpus": [],
        "cuda_available": False,
    }
    request = CompatibilityRequest(model_id=metadata["id"], context_length=32)
    active = runtime.model
    job = {}

    def download(*args, **kwargs):
        machine["ram_available_bytes"] = 0
        return str(checkpoint)

    with (
        patch("circuit_lantern.services.runtime.hardware_snapshot", return_value=machine),
        patch("circuit_lantern.services.runtime.snapshot_download", side_effect=download),
        patch("circuit_lantern.services.runtime.AutoModelForCausalLM.from_pretrained") as loader,
    ):
        assert runtime.assess_fit(metadata, request)["compatible"]
        runtime._load(job, metadata, "cpu", context_length=32)
        assert job["status"] == "failed"
        assert "Server resources changed before loading" in job["message"]
        loader.assert_not_called()
    assert runtime.model is active
    assert runtime.trace("the capital", 1, False, None)["predictions"]
