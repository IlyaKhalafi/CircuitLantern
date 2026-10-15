import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import psutil


def hardware_snapshot(data_dir: Path) -> dict:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(data_dir)
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for row in result.stdout.strip().splitlines():
            index, name, total, free, capability, driver = [item.strip() for item in row.split(",")]
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "total_bytes": int(float(total) * 1024**2),
                    "free_bytes": int(float(free) * 1024**2),
                    "compute_capability": float(capability),
                    "driver": driver,
                }
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "compressed-tensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch

    return {
        "cpu": psutil.cpu_count(logical=False) or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpus": gpus,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "runtime_device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def hardware_fingerprint(snapshot: dict) -> str:
    return json.dumps(snapshot["gpus"], sort_keys=True)
