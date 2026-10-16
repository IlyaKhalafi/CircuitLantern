from unittest.mock import patch

import pytest
import torch
from torch import nn

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


def test_snapshot_serializes_free_memory_and_owned_tensor_accounting(runtime):
    with patch("circuit_lantern.services.runtime.hardware_snapshot", return_value={"free": 7}):
        snapshot = runtime.fit_snapshot()
        assert snapshot["free"] == 7
        assert snapshot["reclaimable_tensor_bytes"]["cpu"] > 0
        with runtime.lock, pytest.raises(RuntimeBusyError, match="busy"):
            runtime.fit_snapshot()
        runtime.unload()
        assert runtime.fit_snapshot()["reclaimable_tensor_bytes"] == {}
