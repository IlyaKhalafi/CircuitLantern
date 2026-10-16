"""Inspect exact revision cache entries without downloading files."""

from pathlib import Path

from huggingface_hub import try_to_load_from_cache


def checkpoint_cache(metadata: dict, cache_dir: str | Path | None = None) -> dict | None:
    files = metadata.get("download_files") or []
    sizes = metadata.get("download_file_sizes")
    source_bytes = metadata.get("download_bytes") or metadata.get("weight_bytes") or 0
    if not files or not sizes or not source_bytes:
        return None

    if any(not isinstance(sizes.get(name), int) or sizes[name] < 0 for name in files):
        return None

    cached_bytes = 0
    cached_files = 0
    for filename in files:
        expected_size = sizes.get(filename)
        try:
            cached = try_to_load_from_cache(
                metadata["id"], filename, revision=metadata["revision"], cache_dir=cache_dir
            )
            if isinstance(cached, str) and Path(cached).stat().st_size == expected_size:
                cached_bytes += expected_size
                cached_files += 1
        except (OSError, ValueError):
            # Unreadable or incomplete cache entries do not reduce the storage budget.
            continue
    cached_bytes = min(source_bytes, cached_bytes)
    return {
        "source_bytes": source_bytes,
        "cached_bytes": cached_bytes,
        "download_bytes": source_bytes - cached_bytes,
        "cached_files": cached_files,
        "total_files": len(files),
    }
