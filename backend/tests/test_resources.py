from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from circuit_lantern.schemas import CompatibilityRequest
from circuit_lantern.services.compatibility import assess_compatibility
from circuit_lantern.services.resources import disk_budget, memory_budget

GIB = 1024**3


def hierarchy(tmp_path, *, version=2, group="/parent/worker", mount_root="/"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "memory mount"
    mount.mkdir()
    if Path(group).is_relative_to(Path(mount_root)):
        relative = Path(group).relative_to(mount_root)
    else:
        relative = Path(group).relative_to("/")
    current = mount / relative
    current.mkdir(parents=True, exist_ok=True)
    membership = f"0::{group}" if version == 2 else f"4:memory:{group}"
    filesystem = "cgroup2 cgroup rw" if version == 2 else "cgroup memory rw,memory"
    escaped_mount = str(mount).replace(" ", r"\040")
    (proc / "self/cgroup").write_text(membership)
    (proc / "self/mountinfo").write_text(
        f"1 0 0:1 {mount_root} {escaped_mount} rw - {filesystem}\n"
    )
    return proc, mount, current


def constraint(path, limit, used, *, version=2):
    names = (
        ("memory.max", "memory.current")
        if version == 2
        else (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
        )
    )
    (path / names[0]).write_text(str(limit))
    if used is not None:
        (path / names[1]).write_text(str(used))


def test_v2_process_limit_caps_host_ram_and_available_budget(tmp_path):
    proc, _, current = hierarchy(tmp_path)
    constraint(current, 8 * GIB, 3 * GIB)
    result = memory_budget(32 * GIB, 20 * GIB, proc_root=proc)
    assert result["ram_total_bytes"] == 8 * GIB
    assert result["ram_available_bytes"] == 5 * GIB
    assert result["ram_scope"] == "cgroup"
    assert result["ram_host_total_bytes"] == 32 * GIB


def test_shared_parent_headroom_can_bind_before_the_child_limit(tmp_path):
    proc, _, current = hierarchy(tmp_path)
    constraint(current, 8 * GIB, 3 * GIB)
    constraint(current.parent, 12 * GIB, 11 * GIB)
    result = memory_budget(32 * GIB, 20 * GIB, proc_root=proc)
    assert result["ram_total_bytes"] == 8 * GIB
    assert result["ram_available_bytes"] == GIB
    assert len(result["ram_constraints"]) == 2


def test_unlimited_child_still_respects_parent_constraints(tmp_path):
    proc, _, current = hierarchy(tmp_path)
    constraint(current, "max", GIB)
    constraint(current.parent, 8 * GIB, 5 * GIB)
    assert memory_budget(32 * GIB, 20 * GIB, proc_root=proc)["ram_available_bytes"] == 3 * GIB


@pytest.mark.parametrize("group", ["/", "/worker", "/docker/container/worker"])
def test_namespaced_and_subtree_mounts_map_to_the_correct_process_group(tmp_path, group):
    proc, _, current = hierarchy(tmp_path, group=group, mount_root="/docker/container")
    constraint(current, 8 * GIB, GIB)
    assert memory_budget(32 * GIB, 20 * GIB, proc_root=proc)["ram_available_bytes"] == 7 * GIB


def test_legacy_unlimited_value_does_not_hide_a_parent_limit(tmp_path):
    proc, _, current = hierarchy(tmp_path, version=1)
    constraint(current, 9223372036854771712, GIB, version=1)
    constraint(current.parent, 12 * GIB, 6 * GIB, version=1)
    result = memory_budget(32 * GIB, 4 * GIB, proc_root=proc)
    assert result["ram_total_bytes"] == 12 * GIB
    assert result["ram_available_bytes"] == 4 * GIB
    assert result["ram_constraints"][0]["source"] == "cgroup-v1"


def test_legacy_disabled_hierarchical_accounting_does_not_apply_a_parent_cap(tmp_path):
    proc, _, current = hierarchy(tmp_path, version=1)
    constraint(current, 8 * GIB, GIB, version=1)
    constraint(current.parent, 2 * GIB, GIB, version=1)
    (current.parent / "memory.use_hierarchy").write_text("0")
    assert memory_budget(32 * GIB, 20 * GIB, proc_root=proc)["ram_available_bytes"] == 7 * GIB


@pytest.mark.parametrize("limit,used", [(0, 0), (GIB, 2 * GIB), (GIB, None)])
def test_zero_exhausted_or_unreadable_usage_cannot_claim_available_memory(tmp_path, limit, used):
    proc, _, current = hierarchy(tmp_path)
    constraint(current, limit, used)
    result = memory_budget(32 * GIB, 20 * GIB, proc_root=proc)
    assert result["ram_available_bytes"] == 0


def test_missing_platform_files_fall_back_to_system_observations(tmp_path):
    result = memory_budget(32 * GIB, 20 * GIB, proc_root=tmp_path)
    assert result["ram_total_bytes"] == 32 * GIB
    assert result["ram_available_bytes"] == 20 * GIB
    assert result["ram_scope"] == "system"
    assert result["ram_constraints"] == []


@pytest.mark.parametrize("version,counter", [(2, "inactive_file"), (1, "total_inactive_file")])
def test_inactive_file_cache_is_available_for_reclaim_within_the_limit(tmp_path, version, counter):
    proc, _, current = hierarchy(tmp_path, version=version)
    constraint(current, 8 * GIB, 7 * GIB, version=version)
    (current / "memory.stat").write_text(f"{counter} {3 * GIB}\n")
    result = memory_budget(32 * GIB, 20 * GIB, proc_root=proc)
    assert result["ram_available_bytes"] == 4 * GIB
    assert result["ram_constraints"][0]["reclaimable_bytes"] == 3 * GIB


def test_download_storage_uses_the_hub_cache_filesystem_instead_of_experiment_storage(tmp_path):
    records = tmp_path / "records"
    cache_volume = tmp_path / "models"
    records.mkdir()
    cache_volume.mkdir()

    def capacity(path):
        free = 2 * GIB if path == cache_volume else 20 * GIB
        return SimpleNamespace(total=32 * GIB, free=free)

    with patch("circuit_lantern.services.resources.shutil.disk_usage", side_effect=capacity):
        result = disk_budget(records, cache_volume / "not-created-yet" / "hub")
    assert result["disk_free_bytes"] == 2 * GIB
    assert result["experiment_disk_free_bytes"] == 20 * GIB
    assert result["checkpoint_cache_path"].endswith("not-created-yet/hub")


def test_process_limit_changes_the_real_compatibility_decision(tmp_path):
    proc, _, current = hierarchy(tmp_path)
    constraint(current, GIB, 900 * 1024**2)
    machine = {
        "gpus": [],
        "cuda_available": False,
        "disk_free_bytes": 20 * GIB,
        **memory_budget(32 * GIB, 20 * GIB, proc_root=proc),
    }
    metadata = {
        "parameters": 124_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 1024,
        "has_safetensors": True,
        "weight_bytes": 500_000_000,
    }
    result = assess_compatibility(
        metadata,
        machine,
        CompatibilityRequest(
            model_id="test",
            context_length=128,
        ),
    )
    assert not result["compatible"]
    assert any("Insufficient available memory" in blocker for blocker in result["blockers"])
    assert any("container memory limit" in warning for warning in result["warnings"])
