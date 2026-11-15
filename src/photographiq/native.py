"""Access to the optional compiled engines in ``photographiq._native``.

The extension is built by scikit-build-core from ``cpp/`` during installation.
Pure-Python functionality never requires it; native backends and native sampling
routines raise ``ImportError`` with build instructions when it is missing.

``PHOTOGRAPHIQ_NATIVE_PATH`` may name a compiled ``_native`` extension file to load
instead of the installed one (used for out-of-tree development builds).
"""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType

_BUILD_HINT = (
    "Build it with a C++20 compiler: python -m pip install -e . "
    "(see docs/development/native-build.md)"
)


def _load() -> tuple[ModuleType | None, str | None]:
    path = os.environ.get("PHOTOGRAPHIQ_NATIVE_PATH")
    if path:
        spec = importlib.util.spec_from_file_location("photographiq._native", path)
        if spec is None or spec.loader is None:
            return None, f"PHOTOGRAPHIQ_NATIVE_PATH={path!r} is not an extension module"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    try:
        from . import _native  # type: ignore[attr-defined]
    except ImportError as exc:
        return None, str(exc)
    return _native, None


module, load_error = _load()
AVAILABLE = module is not None


def components() -> tuple[str, ...]:
    """Names of compiled native components (empty if the extension is missing)."""
    return tuple(module.components) if module is not None else ()


def build_info() -> dict:
    """Compiler, build type, OpenMP and Eigen information of the loaded extension."""
    return dict(module.build_info) if module is not None else {}


def require(component: str) -> ModuleType:
    """Return a compiled component submodule or raise an explicit ``ImportError``."""
    if module is None:
        raise ImportError(f"photographiq._native is not available ({load_error}). {_BUILD_HINT}")
    if component not in module.components:
        raise ImportError(
            f"Native component {component!r} was not compiled "
            f"(available: {', '.join(module.components) or 'none'}). {_BUILD_HINT}"
        )
    return getattr(module, component)
