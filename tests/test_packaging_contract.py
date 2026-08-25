"""Packaging contract for BAM-torch.

These assertions encode the invariants that make ``pyproject.toml`` the single
source of distribution metadata and keep the committed ``uv.lock`` honest.
They read machine-consumed metadata only (no prose), so they stay stable while
the project text evolves.

Run with the stdlib runner (no extra test dependency required)::

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s,;]+)$")
REQUIRED_DEV_TOOLS = ("ruff", "ty")

# Native PyG extensions must never re-enter the core requirement set: they are
# only reachable through the optional ``group_averaging`` extra, and they have
# no PyPI wheels, so pinning them as core dependencies breaks a plain install.
FORBIDDEN_CORE_DEPS = ("torch_scatter", "torch-scatter", "torch_sparse", "torch-sparse")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dep_name(requirement: str) -> str:
    """Canonical distribution name from a PEP 508 requirement string."""
    return _canonical(re.split(r"[\[<>=!~;\s]", requirement.strip(), maxsplit=1)[0])


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class SoleMetadataSourceTest(unittest.TestCase):
    def test_setup_py_is_gone(self) -> None:
        self.assertFalse(
            (REPO_ROOT / "setup.py").exists(),
            "setup.py must not exist: pyproject.toml is the sole distribution "
            "metadata source",
        )

    def test_core_dependencies_exclude_native_pyg_extensions(self) -> None:
        canonical = {_dep_name(d) for d in _load(PYPROJECT)["project"]["dependencies"]}
        for forbidden in FORBIDDEN_CORE_DEPS:
            self.assertNotIn(
                _canonical(forbidden),
                canonical,
                f"{forbidden} belongs to the group_averaging extra, not core deps",
            )

    def test_group_averaging_extra_still_owns_the_native_extensions(self) -> None:
        extras = _load(PYPROJECT)["project"]["optional-dependencies"]
        canonical = {_dep_name(d) for d in extras["group_averaging"]}
        self.assertIn("torch-scatter", canonical)
        self.assertIn("torch-cluster", canonical)

    def test_build_backend_and_python_floor_are_preserved(self) -> None:
        data = _load(PYPROJECT)
        self.assertEqual(data["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(data["project"]["requires-python"], ">=3.11")

    def test_distribution_does_not_ship_examples_as_a_top_level_package(self) -> None:
        find = _load(PYPROJECT)["tool"]["setuptools"]["packages"]["find"]
        include = find.get("include")
        self.assertTrue(include, "packages.find must declare an explicit include list")
        self.assertTrue(
            any(fnmatch.fnmatch("bam_torch", pattern) for pattern in include),
            "the bam_torch package must be distributed",
        )
        for stray in ("examples", "tests", "licenses"):
            self.assertFalse(
                any(fnmatch.fnmatch(stray, pattern) for pattern in include),
                f"{stray!r} must not be distributed as a top-level package",
            )


class PinnedDevToolingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dev_group = _load(PYPROJECT).get("dependency-groups", {}).get("dev")

    def test_dev_group_pins_ruff_and_ty_exactly(self) -> None:
        self.assertIsNotNone(self.dev_group, "[dependency-groups].dev is missing")
        pins = {}
        for spec in self.dev_group:
            match = EXACT_PIN.match(spec)
            self.assertIsNotNone(match, f"dev dependency {spec!r} is not an exact pin")
            pins[_canonical(match.group("name"))] = match.group("version")
        for tool in REQUIRED_DEV_TOOLS:
            self.assertIn(tool, pins, f"{tool} must be a pinned dev dependency")

    def test_dev_tools_are_not_runtime_dependencies(self) -> None:
        deps = {_dep_name(d) for d in _load(PYPROJECT)["project"]["dependencies"]}
        for tool in REQUIRED_DEV_TOOLS:
            self.assertNotIn(tool, deps, f"{tool} is dev-only tooling, not a runtime dep")


class LockfileTest(unittest.TestCase):
    def setUp(self) -> None:
        if not UV_LOCK.exists():
            self.fail("uv.lock must be committed alongside pyproject.toml")
        self.lock = _load(UV_LOCK)

    def test_lock_targets_the_same_python_floor(self) -> None:
        self.assertEqual(self.lock["requires-python"], ">=3.11")

    def test_lock_is_platform_neutral(self) -> None:
        # `environments` / `required-environments` are the knobs that narrow a
        # lock to specific platforms; a universal lock must declare neither.
        manifest = self.lock.get("manifest", {})
        self.assertNotIn("environments", manifest)
        self.assertNotIn("required-environments", manifest)
        # Positive proof of universality: torch is locked with wheels for every
        # major platform, not just the one this lock was generated on.
        torch = next(pkg for pkg in self.lock["package"] if pkg["name"] == "torch")
        platforms = {
            tag
            for tag in ("macosx", "manylinux", "win")
            for wheel in torch["wheels"]
            if tag in wheel["url"].rsplit("/", 1)[-1]
        }
        self.assertEqual(platforms, {"macosx", "manylinux", "win"})

    def test_lock_resolves_native_pyg_extensions_without_building_them(self) -> None:
        # The group_averaging extension sdists import torch in setup.py, so uv
        # can only resolve them from statically declared metadata. Losing these
        # entries silently reintroduces a source compile into `uv lock`.
        declared = {
            _canonical(entry["name"])
            for entry in self.lock["manifest"].get("dependency-metadata", [])
        }
        self.assertIn("torch-scatter", declared)
        self.assertIn("torch-cluster", declared)

    def test_lock_preserves_every_extra_and_the_dev_group(self) -> None:
        pyproject = _load(PYPROJECT)
        root = next(pkg for pkg in self.lock["package"] if pkg["name"] == "bam-torch")
        locked_extras = {_canonical(e) for e in root.get("optional-dependencies", {})}
        declared_extras = {
            _canonical(e) for e in pyproject["project"]["optional-dependencies"]
        }
        self.assertEqual(locked_extras, declared_extras)
        self.assertIn("dev", root.get("dev-dependencies", {}))

    def test_lock_contains_the_pinned_dev_tools_at_the_pinned_versions(self) -> None:
        dev_group = _load(PYPROJECT).get("dependency-groups", {}).get("dev") or []
        pins = {
            _canonical(m.group("name")): m.group("version")
            for m in (EXACT_PIN.match(spec) for spec in dev_group)
            if m
        }
        locked = {_canonical(pkg["name"]): pkg["version"] for pkg in self.lock["package"]}
        for tool in REQUIRED_DEV_TOOLS:
            self.assertIn(tool, locked, f"{tool} missing from uv.lock")
            self.assertEqual(
                locked[tool],
                pins.get(tool),
                f"uv.lock version for {tool} does not match the pyproject pin",
            )

    def test_lock_contains_the_project_itself_with_its_declared_version(self) -> None:
        version = _load(PYPROJECT)["project"]["version"]
        locked = {_canonical(pkg["name"]): pkg["version"] for pkg in self.lock["package"]}
        self.assertEqual(locked.get("bam-torch"), version)


if __name__ == "__main__":
    unittest.main()
