"""Trainability metadata around PhotoGraphiQ's existing symbolic expressions."""

from collections.abc import Mapping

import numpy as np

from photographiq import Parameter


class ParameterStore:
    def __init__(self, names, *, frozen=()):
        self.names = tuple(names)
        self.symbols = {name: Parameter(name) for name in self.names}
        self.values = dict.fromkeys(self.names, 0.0)
        self.frozen = set(frozen)

    def bind(self, parameters=None):
        result = dict(self.values)
        if parameters is None:
            return result
        if isinstance(parameters, Mapping):
            if set(parameters) - result.keys():
                raise ValueError("Unknown parameter names")
            updates = dict(parameters)
        else:
            values = np.asarray(parameters, dtype=float)
            names = [n for n in self.names if n not in self.frozen]
            if values.shape != (len(names),):
                raise ValueError(f"parameters must have shape ({len(names)},)")
            updates = dict(zip(names, values))
        for name, value in updates.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("Parameters must be finite real scalars")
            if name in self.frozen and value != result[name]:
                raise ValueError(f"{name} is frozen; unfreeze it before rebinding")
            result[name] = value
        return result

    def freeze(self, names, value=None):
        names = (names,) if isinstance(names, str) else tuple(names)
        if set(names) - self.values.keys():
            raise ValueError("Unknown parameter names")
        if value is not None and not np.isfinite(value):
            raise ValueError("Frozen value must be finite")
        for name in names:
            if value is not None:
                self.values[name] = float(value)
            self.frozen.add(name)

    def unfreeze(self, names):
        names = (names,) if isinstance(names, str) else tuple(names)
        if set(names) - self.values.keys():
            raise ValueError("Unknown parameter names")
        self.frozen.difference_update(names)
