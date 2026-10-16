#!/usr/bin/env python3
"""Build and launch the lab with an explicit GPU visibility list."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def gpu_selection(value: str) -> list[str]:
    if value in ("all", "cpu"):
        return [value]
    devices = [item.strip() for item in value.split(",")]
    if not devices or any(
        not re.fullmatch(
            r"(?:[0-9]+|GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})", item
        )
        for item in devices
    ):
        raise argparse.ArgumentTypeError(
            "Use GPU indices such as 0,1, GPU UUIDs, all, or cpu."
        )
    devices = [str(int(item)) if item.isdecimal() else item for item in devices]
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("Each GPU must appear only once.")
    return devices


def deployment_override(devices: list[str], port: int) -> dict:
    cpu = devices == ["cpu"]
    backend = {
        "environment": {
            "NVIDIA_VISIBLE_DEVICES": "void" if cpu else ",".join(devices),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "HOME": "/tmp",
            "TRITON_CACHE_DIR": "/data/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/data/cache/torch",
            "XDG_CACHE_HOME": "/data/cache",
        },
        "user": f"{os.getuid()}:{os.getgid()}",
    }
    if not cpu:
        reservation = {"driver": "nvidia", "capabilities": ["gpu"]}
        reservation.update(
            {"count": "all"} if devices == ["all"] else {"device_ids": devices}
        )
        backend["deploy"] = {"resources": {"reservations": {"devices": [reservation]}}}
    return {"services": {"backend": backend, "web": {"ports": [f"{port}:3000"]}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        type=gpu_selection,
        required=False,
        help="Comma separated GPU indices or UUIDs, all, or cpu",
    )
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("The port must be between 1 and 65535.")
    root = Path(__file__).resolve().parents[1]
    overlay = root / ".lantern-deployment.json"
    command = [
        "docker",
        "compose",
        "-f",
        str(root / "compose.yaml"),
        "-f",
        str(overlay),
    ]
    if args.stop:
        if not overlay.is_file():
            parser.error("There is no saved deployment configuration to stop.")
        subprocess.run(command + ["down"], cwd=root, check=True)
        return
    if args.gpus is None:
        parser.error("Choose GPU devices with --gpus, or use --gpus cpu.")
    config = deployment_override(args.gpus, args.port)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    environment = dict(os.environ, LANTERN_PORT=str(args.port))
    data = (
        Path(environment.get("LANTERN_DATA_PATH", root / "backend/data"))
        .expanduser()
        .resolve()
    )
    data.mkdir(parents=True, exist_ok=True)
    environment["LANTERN_DATA_PATH"] = str(data)
    # Persist the exact mount and device grant for subsequent lifecycle commands.
    config["services"]["backend"]["volumes"] = [f"{data}:/data"]
    temporary = overlay.with_suffix(".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(config, stream, indent=2)
    temporary.replace(overlay)
    command += ["up", "-d", "--wait", "--wait-timeout", "180"]
    if args.no_build:
        command += ["--no-build", "--pull", "never"]
    else:
        command.append("--build")
    try:
        subprocess.run(command, cwd=root, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "Deployment failed. Inspect docker compose logs for details."
        ) from error
    print(
        f"Circuit Lantern is available on port {args.port} with devices {','.join(args.gpus)}"
    )


if __name__ == "__main__":
    main()
