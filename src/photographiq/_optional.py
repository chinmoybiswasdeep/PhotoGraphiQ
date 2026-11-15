"""Lazy access to optional third-party packages with explicit installation hints."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


class LazyModule:
    """Import ``name`` on first attribute access; raise ``ImportError`` naming the extra."""

    def __init__(self, name: str, extra: str):
        self._name, self._extra = name, extra
        self._module: ModuleType | None = None

    @property
    def available(self) -> bool:
        return self._module is not None or importlib.util.find_spec(self._name) is not None

    def load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError as exc:
                raise ImportError(
                    f"This feature requires the optional dependency {self._name!r}; "
                    f"install it with: pip install 'photographiq[{self._extra}]'"
                ) from exc
        return self._module

    def __getattr__(self, attribute: str):
        return getattr(self.load(), attribute)


piquasso = LazyModule("piquasso", "piquasso")
