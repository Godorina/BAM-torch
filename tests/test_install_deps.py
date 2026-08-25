"""Deterministic PyG installer selection tests."""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _installer_module():
    spec = importlib.util.spec_from_file_location(
        "_bam_install_deps", REPO_ROOT / "install_deps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_torch(
    *, torch_version: str, cuda_available: bool, cuda_version: str | None
) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = torch_version
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    return torch


def _run_installer_as_script(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torch_version: str,
    cuda_available: bool,
    cuda_version: str | None,
) -> list[str]:
    commands: list[list[str]] = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
        ),
    )
    monkeypatch.setattr(subprocess, "check_call", commands.append)

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO_ROOT / "install_deps.py"), run_name="__main__")

    return_code = raised.value.code
    assert isinstance(return_code, int)
    assert return_code == 0
    assert len(commands) == 1
    return commands[0]


def test_cpu_torch_selects_cpu_pyg_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    command = _run_installer_as_script(
        monkeypatch,
        torch_version="2.5.1+cpu",
        cuda_available=False,
        cuda_version=None,
    )

    assert command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--find-links",
        "https://data.pyg.org/whl/torch-2.5.1+cpu.html",
        "torch_scatter",
        "torch_cluster",
    ]
    assert "Detected CUDA version: cpu" in capsys.readouterr().out


def test_cuda_torch_selects_matching_pyg_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    command = _run_installer_as_script(
        monkeypatch,
        torch_version="2.5.1+cu124",
        cuda_available=True,
        cuda_version="12.4.1",
    )

    assert command[5] == "https://data.pyg.org/whl/torch-2.5.1+cu124.html"
    assert "Detected CUDA version: cu124" in capsys.readouterr().out


def test_unsupported_torch_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(torch_version="2.9.0+cpu", cuda_available=False, cuda_version=None),
    )
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda command: pytest.fail(
            f"pip must not run for unsupported Torch: {command}"
        ),
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO_ROOT / "install_deps.py"), run_name="__main__")

    assert raised.value.code == 1
    assert capsys.readouterr().out == (
        "Unsupported version of PyTorch. Please install a supported version:\n"
        '  pip install "torch<=2.8"\n'
    )


@pytest.mark.parametrize("cuda_version", [None, "12", "12.x"])
def test_malformed_cuda_exits_with_user_facing_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cuda_version: str | None,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            torch_version="2.5.1+cu124",
            cuda_available=True,
            cuda_version=cuda_version,
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda command: pytest.fail(f"pip must not run for malformed CUDA: {command}"),
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO_ROOT / "install_deps.py"), run_name="__main__")

    assert raised.value.code == 1
    assert "Unable to determine a valid CUDA version" in capsys.readouterr().out


def test_malformed_cuda_error_survives_optimized_python(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text(
        """__version__ = "2.5.1+cu124"


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _Version:
    cuda = "12"


cuda = _Cuda()
version = _Version()
"""
    )
    result = subprocess.run(
        [sys.executable, "-O", "-B", str(REPO_ROOT / "install_deps.py")],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Unable to determine a valid CUDA version" in result.stdout
    assert "Installing from:" not in result.stdout


def test_supported_cpu_torch_selects_cpu_pyg_wheel() -> None:
    installer = _installer_module()

    decision = installer.select_pyg_wheel(
        torch_version="2.5.1+cpu",
        cuda_available=False,
        cuda_version=None,
    )

    assert decision.torch_version == "2.5.1"
    assert decision.cuda_version == "cpu"
    assert decision.find_links == "https://data.pyg.org/whl/torch-2.5.1+cpu.html"


def test_unsupported_torch_version_has_exact_nonzero_error_contract() -> None:
    installer = _installer_module()

    with pytest.raises(installer.UnsupportedTorchVersion) as raised:
        installer.select_pyg_wheel(
            torch_version="2.9.0+cpu",
            cuda_available=False,
            cuda_version=None,
        )

    assert raised.value.exit_code == 1
    assert str(raised.value) == (
        "Unsupported version of PyTorch. Please install a supported version:\n"
        '  pip install "torch<=2.8"'
    )
