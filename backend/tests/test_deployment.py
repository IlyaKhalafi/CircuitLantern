import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "deploy", Path(__file__).parents[2] / "scripts/deploy.py"
)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


@pytest.mark.parametrize(
    "value", ["", "0,", "0,0", "0;echo x", "-1", "all,0", "cpu,1", "0,00", "GPU-"]
)
def test_invalid_device_lists_fail_before_running_docker(value):
    with pytest.raises(argparse.ArgumentTypeError):
        deploy.gpu_selection(value)


def test_specific_gpu_reservation_has_no_implicit_all_devices():
    config = deploy.deployment_override(deploy.gpu_selection("1, 3"), 9001)
    backend = config["services"]["backend"]
    devices = backend["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["1", "3"]}]
    assert backend["environment"]["NVIDIA_VISIBLE_DEVICES"] == "1,3"
    assert "CUDA_VISIBLE_DEVICES" not in backend["environment"]
    assert config["services"]["web"]["ports"] == ["9001:3000"]


def test_cpu_mode_never_requests_a_gpu():
    backend = deploy.deployment_override(["cpu"], 9002)["services"]["backend"]
    assert "deploy" not in backend
    assert backend["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert "CUDA_VISIBLE_DEVICES" not in backend["environment"]


def test_launcher_persists_exact_mount_and_devices_from_another_working_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "lab"
    (root / "scripts").mkdir(parents=True)
    working = tmp_path / "caller"
    working.mkdir()
    monkeypatch.setattr(deploy, "__file__", str(root / "scripts/deploy.py"))
    monkeypatch.chdir(working)
    monkeypatch.setenv("LANTERN_DATA_PATH", "research")
    monkeypatch.setattr("sys.argv", ["deploy.py", "--gpus", "3,1", "--port", "9012", "--no-build"])
    calls = []
    monkeypatch.setattr(
        deploy.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    deploy.main()
    overlay = root / ".lantern-deployment.json"
    saved = json.loads(overlay.read_text())
    assert saved["services"]["backend"]["volumes"] == [str(working / "research") + ":/data"]
    assert saved["services"]["backend"]["environment"]["NVIDIA_VISIBLE_DEVICES"] == "3,1"
    assert calls[0][1]["env"]["LANTERN_DATA_PATH"] == str(working / "research")
    assert calls[0][1]["env"]["LANTERN_PORT"] == "9012"
    assert str(overlay) in calls[0][0]
    assert "--build" not in calls[0][0]
    assert "--no-build" in calls[0][0]
    assert calls[0][0][-2:] == ["--pull", "never"]
    assert overlay.stat().st_mode & 0o777 == 0o600


def test_stop_uses_the_persisted_deployment_overlay(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    overlay = tmp_path / ".lantern-deployment.json"
    overlay.write_text("{}")
    monkeypatch.setattr(deploy, "__file__", str(tmp_path / "scripts/deploy.py"))
    monkeypatch.setattr("sys.argv", ["deploy.py", "--stop"])
    calls = []
    monkeypatch.setattr(deploy.subprocess, "run", lambda command, **kwargs: calls.append(command))
    deploy.main()
    assert calls == [
        ["docker", "compose", "-f", str(tmp_path / "compose.yaml"), "-f", str(overlay), "down"]
    ]


@pytest.mark.parametrize("selection", ["1,3", "all", "cpu"])
def test_real_compose_merge_preserves_device_grants_and_one_port(tmp_path, selection):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is not installed.")
    if subprocess.run(
        ["docker", "compose", "version"], capture_output=True, check=False
    ).returncode:
        pytest.skip("Docker Compose is not installed.")
    root = Path(__file__).parents[2]
    overlay = tmp_path / "devices.json"
    overlay.write_text(
        json.dumps(deploy.deployment_override(deploy.gpu_selection(selection), 9012))
    )
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(root / "compose.yaml"),
            "-f",
            str(overlay),
            "config",
            "--format",
            "json",
        ],
        env=dict(os.environ, LANTERN_PORT="9012"),
        capture_output=True,
        text=True,
        check=True,
    )
    config = json.loads(result.stdout)
    backend = config["services"]["backend"]
    assert backend["user"] == f"{os.getuid()}:{os.getgid()}"
    assert config["services"]["web"]["ports"] == [
        {"mode": "ingress", "target": 3000, "published": "9012", "protocol": "tcp"}
    ]
    if selection == "cpu":
        assert "deploy" not in backend
        assert backend["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
    else:
        devices = backend["deploy"]["resources"]["reservations"]["devices"][0]
        if selection == "all":
            assert devices["count"] == -1 or devices["count"] == "all"
        else:
            assert devices["device_ids"] == ["1", "3"]
