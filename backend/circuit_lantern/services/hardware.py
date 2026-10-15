import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import psutil

from circuit_lantern.services.execution import quantization_capabilities


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

    cuda_available = torch.cuda.is_available()
    execution_gpu, cuda_error = None, None
    if cuda_available:
        try:
            # Query and execute on logical device zero, respecting CUDA_VISIBLE_DEVICES.
            sample = torch.ones(2, 2, device="cuda:0")
            if float((sample @ sample).sum()) != 8:
                raise RuntimeError("The CUDA execution probe returned an invalid result.")
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            properties = torch.cuda.get_device_properties(0)
            execution_gpu = {
                "index": 0,
                "name": properties.name,
                "total_bytes": total_bytes,
                "free_bytes": free_bytes,
                "compute_capability": properties.major + properties.minor / 10,
            }
        except (RuntimeError, OSError) as error:
            cuda_available = False
            cuda_error = str(error)[:250]
    return {
        "execution_gpu": execution_gpu,
        "cuda_error": cuda_error,
        "quantization": quantization_capabilities(),
        "cpu": psutil.cpu_count(logical=False) or 1,
        "cpu_threads": psutil.cpu_count() or 1,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpus": gpus,
        "packages": packages,
        "cuda_available": cuda_available,
        "runtime_device": "cuda" if cuda_available else "cpu",
    }
