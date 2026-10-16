from pathlib import Path
from unittest.mock import patch

from circuit_lantern.services.cache import checkpoint_cache


def checkpoint(tmp_path):
    revision = "a" * 40
    directory = tmp_path / "models--org--model" / "snapshots" / revision
    directory.mkdir(parents=True)
    metadata = {
        "id": "org/model",
        "revision": revision,
        "download_files": ["config.json", "model.safetensors"],
        "download_file_sizes": {"config.json": 2, "model.safetensors": 100},
        "download_bytes": 102,
    }
    return metadata, directory


def test_exact_revision_reuses_only_complete_files_and_rechecks_after_download(tmp_path):
    metadata, directory = checkpoint(tmp_path)
    (directory / "config.json").write_text("{}")
    partial = directory / "model.safetensors.incomplete"
    partial.write_bytes(b"0" * 90)
    first = checkpoint_cache(metadata, tmp_path)
    assert first["cached_bytes"] == 2
    assert first["download_bytes"] == 100
    assert first["cached_files"] == 1
    partial.unlink()
    (directory / "model.safetensors").write_bytes(b"0" * 100)
    second = checkpoint_cache(metadata, tmp_path)
    assert second["cached_bytes"] == 102
    assert second["download_bytes"] == 0
    assert second["cached_files"] == second["total_files"] == 2
    assert first["download_bytes"] == 100


def test_different_revision_and_truncated_files_do_not_claim_cached_bytes(tmp_path):
    metadata, directory = checkpoint(tmp_path)
    (directory / "model.safetensors").write_bytes(b"0" * 99)
    assert checkpoint_cache(metadata, tmp_path)["cached_bytes"] == 0
    (directory / "model.safetensors").write_bytes(b"0" * 100)
    metadata["revision"] = "b" * 40
    assert checkpoint_cache(metadata, tmp_path)["cached_bytes"] == 0


def test_shared_cache_blob_is_counted_through_the_snapshot_link(tmp_path):
    metadata, directory = checkpoint(tmp_path)
    blobs = directory.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "shared-weight"
    blob.write_bytes(b"0" * 100)
    (directory / "model.safetensors").symlink_to(Path("../../blobs/shared-weight"))
    result = checkpoint_cache(metadata, tmp_path)
    assert result["cached_bytes"] == 100
    assert result["download_bytes"] == 2


def test_unreadable_cache_and_legacy_metadata_keep_the_full_budget(tmp_path):
    metadata, _ = checkpoint(tmp_path)
    with patch(
        "circuit_lantern.services.cache.try_to_load_from_cache", side_effect=PermissionError
    ):
        assert checkpoint_cache(metadata, tmp_path)["download_bytes"] == 102
    del metadata["download_file_sizes"]
    assert checkpoint_cache(metadata, tmp_path) is None


def test_unknown_file_sizes_do_not_claim_a_complete_cache(tmp_path):
    metadata, directory = checkpoint(tmp_path)
    (directory / "model.safetensors").write_bytes(b"0" * 100)
    metadata["download_file_sizes"]["config.json"] = None
    assert checkpoint_cache(metadata, tmp_path) is None
