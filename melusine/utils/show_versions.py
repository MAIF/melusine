from __future__ import annotations

import sys

from melusine import __version__ as melusine_version


def show_versions() -> None:
    """
    Print out version of melusine and dependencies to stdout.

    Examples
    --------
    >>> melusine.show_versions()  # doctest: +SKIP
    --------Version info---------
    melusine:      3.0.0
    Platform:    Linux-5.15.90.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
    Python:      3.11.3 (main, Apr 15 2023, 14:44:51) [GCC 11.3.0]
    \b
    ----Optional dependencies----
    numpy:       1.24.2
    pandas:      2.0.0
    pytorch:  <not installed>
    """
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    # optional dependencies
    deps = _get_dependency_info()

    # determine key length for alignment
    keylen = max(len(x) for x in [*deps.keys(), "melusine", "Platform", "Python"]) + 1

    print("--------Version info---------")
    print(f"{'melusine:':{keylen}s} {melusine_version}")
    print(f"{'Platform:':{keylen}s} {platform.platform()}")
    print(f"{'Python:':{keylen}s} {sys.version}")

    print("\n----Optional dependencies----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info() -> dict[str, str]:
    """
    Collect information about optional dependencies.

    Returns:
        _: Dict of optional dependencies and associated versions.
    """
    # See the list of dependencies in pyproject.toml/setup.cfg
    opt_deps = [
        "tensorflow",
        "torch",
        "torchvision",
        "torchlib",
        "transformers",
    ]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    """
    Get the version of a dependency.

    Args:
        dep_name: Name of the dependency.

    Returns:
        _: Dependency version or "<not_installed>"
    """
    # import here to optimize the root melusine import
    import importlib

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = "<__version__ unavailable>"  # pragma: no cover

    return module_version
