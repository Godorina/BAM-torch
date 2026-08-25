"""Regression coverage for the group-averaging FAENet module surface.

``bam_torch.group_averaging.model.gnn_model`` sits behind the optional
``group_averaging`` extra, so the core environment cannot import it: its import
chain pulls in pandas, mendeleev, and torch_scatter. That absence previously
hid a genuine import-time failure, so this module installs *narrow* stubs for
exactly the optional names the chain imports and then executes the real module.

Two properties are asserted:

1. The module imports at all. Every annotation in ``FAENet.__init__`` is
   evaluated when the class body executes, so an annotation that is not a valid
   runtime type (for example ``str | callable``, where ``callable`` is a builtin
   function rather than a type) raises ``TypeError`` on import.
2. ``pbc_preprocess`` and ``base_preprocess`` stay resolvable through the
   module globals, because ``FAENet.__init__`` turns its string ``preprocess``
   argument into a function with ``eval(self.preprocess)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import typing
from collections.abc import Iterator

import pytest

GNN_MODEL = "bam_torch.group_averaging.model.gnn_model"
GA_UTILS = "bam_torch.group_averaging.utils.ga_utils"
DYNAMIC_PREPROCESS_NAMES = ("pbc_preprocess", "base_preprocess")


def _unimportable(name: str):
    def _raise(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"stubbed optional dependency {name} was called")

    return _raise


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def _optional_dependency_stubs() -> dict[str, types.ModuleType]:
    """Narrow stand-ins for the optional imports in the gnn_model chain.

    Only the attributes that the production modules import by name are
    provided, and every one of them raises if it is ever called, so the stubs
    can satisfy imports without silently faking behavior under test.
    """
    pandas = _stub_module("pandas")
    pandas.concat = _unimportable("pandas.concat")
    pandas.isnull = _unimportable("pandas.isnull")

    mendeleev = _stub_module("mendeleev")
    mendeleev_fetch = _stub_module("mendeleev.fetch")
    mendeleev_fetch.fetch_table = _unimportable("mendeleev.fetch.fetch_table")
    mendeleev_fetch.fetch_ionization_energies = _unimportable(
        "mendeleev.fetch.fetch_ionization_energies"
    )
    mendeleev.fetch = mendeleev_fetch

    torch_scatter = _stub_module("torch_scatter")
    torch_scatter.scatter = _unimportable("torch_scatter.scatter")

    return {
        "pandas": pandas,
        "mendeleev": mendeleev,
        "mendeleev.fetch": mendeleev_fetch,
        "torch_scatter": torch_scatter,
    }


def _purge_group_averaging_modules() -> None:
    for name in [n for n in sys.modules if n.startswith("bam_torch.group_averaging")]:
        del sys.modules[name]


@pytest.fixture
def stubbed_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Make the optional import chain satisfiable without weakening the test.

    Where a real optional dependency is installed it is used as-is; only the
    missing ones are stubbed. The module under test is therefore always
    executed for real, in both the core environment and an environment with the
    ``group_averaging`` extra.
    """
    stubs = _optional_dependency_stubs()
    # Resolve availability before touching sys.modules: inserting a stub would
    # otherwise make find_spec() inspect the stub itself.
    missing = {
        top
        for top in {name.split(".")[0] for name in stubs}
        if importlib.util.find_spec(top) is None
    }
    for name, module in stubs.items():
        if name.split(".")[0] in missing:
            monkeypatch.setitem(sys.modules, name, module)

    _purge_group_averaging_modules()
    try:
        yield
    finally:
        # Never leave stub-backed modules behind for other tests to import.
        _purge_group_averaging_modules()


def test_gnn_model_imports_with_optional_dependencies_stubbed(
    stubbed_optional_dependencies: None,
) -> None:
    module = importlib.import_module(GNN_MODEL)

    assert hasattr(module, "FAENet")


def test_preprocess_names_resolve_dynamically(
    stubbed_optional_dependencies: None,
) -> None:
    module = importlib.import_module(GNN_MODEL)
    ga_utils = importlib.import_module(GA_UTILS)

    for name in DYNAMIC_PREPROCESS_NAMES:
        # This is exactly what FAENet.__init__ does with a string `preprocess`.
        resolved = eval(name, vars(module))  # noqa: S307
        assert resolved is getattr(ga_utils, name)
        assert callable(resolved)


def test_preprocess_annotation_is_a_usable_runtime_type(
    stubbed_optional_dependencies: None,
) -> None:
    module = importlib.import_module(GNN_MODEL)
    ga_utils = importlib.import_module(GA_UTILS)

    hints = typing.get_type_hints(module.FAENet.__init__)
    annotation = hints["preprocess"]

    # Both accepted input forms must satisfy the declared annotation: the
    # string name that __init__ eval()s, and the resolved callable itself.
    assert isinstance("pbc_preprocess", annotation)
    assert isinstance(ga_utils.pbc_preprocess, annotation)


def test_faenet_default_preprocess_is_the_string_name(
    stubbed_optional_dependencies: None,
) -> None:
    import inspect

    module = importlib.import_module(GNN_MODEL)

    default = inspect.signature(module.FAENet.__init__).parameters["preprocess"].default

    assert default == "pbc_preprocess"
    assert callable(eval(default, vars(module)))  # noqa: S307
