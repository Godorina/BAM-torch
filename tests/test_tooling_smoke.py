"""Focused smoke checks for the distributable, CPU-safe tooling surface."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_packaging_metadata_and_lockfile_agree() -> None:
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    lock = _load_toml(REPO_ROOT / "uv.lock")

    package = next(item for item in lock["package"] if item["name"] == "bam-torch")
    assert package["version"] == pyproject["project"]["version"]
    assert lock["requires-python"] == pyproject["project"]["requires-python"]


def test_pytest_is_a_dev_dependency_only() -> None:
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")

    assert "pytest==9.0.2" in pyproject["dependency-groups"]["dev"]
    assert all(not dependency.startswith("pytest") for dependency in pyproject["project"]["dependencies"])


def test_economics_cli_help_runs_without_gpu_or_optional_extras() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "bam_torch.economics.cli", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run catalyst carbon social cost reporting" in result.stdout
    assert "--scc-data SCC_DATA" in result.stdout


def test_invalid_static_analysis_fixtures_are_not_collected() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ruff_invalid_import.py" not in result.stdout
    assert "ty_invalid_call.py" not in result.stdout
