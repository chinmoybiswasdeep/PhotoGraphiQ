"""Gaussian circuits compiled to finite-squeezing teleportation patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .commands import Displace, Entangle, Output, Prepare
from .expressions import Outcome, atan2, sin
from .gaussian import rotation, squeezing, teleportation_matrix
from .measurements import Homodyne
from .pattern import Pattern


def decompose_symplectic(matrix):
    """Return k_j with S=T(k_last)...T(k_first), using four or five steps."""
    s = np.asarray(matrix, dtype=float)
    if (
        s.shape != (2, 2)
        or not np.isfinite(s).all()
        or not np.isclose(np.linalg.det(s), 1, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Expected a real single-mode symplectic matrix")
    prefix = []
    if abs(s[1, 0]) < 1e-8:
        prefix = [0.0]
        s = s @ teleportation_matrix(0).T
    a, c, d = s[0, 0], s[1, 0], s[1, 1]
    ks = prefix + [0.0, (1 - d) / c, c, (1 - a) / c]
    check = np.eye(2)
    for k in ks:
        check = teleportation_matrix(k) @ check
    if not np.allclose(check, matrix, atol=1e-8, rtol=1e-8):
        raise ValueError("Numerically unstable Gaussian decomposition")
    return ks


@dataclass
class Circuit:
    modes: int
    gates: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.modes, int) or self.modes < 1:
            raise ValueError("modes must be positive")

    def _add(self, name, modes, *params):
        if len(set(modes)) != len(modes) or any(
            not isinstance(m, int) or m < 0 or m >= self.modes for m in modes
        ):
            raise ValueError("Invalid circuit modes")
        self.gates.append((name, tuple(modes), params))
        return self

    def displace(self, mode, q=0.0, p=0.0):
        return self._add("displace", (mode,), q, p)

    def rotate(self, mode, angle):
        return self._add("symplectic", (mode,), rotation(angle))

    def squeeze(self, mode, r):
        return self._add("symplectic", (mode,), squeezing(r))

    def gaussian(self, mode, matrix):
        decompose_symplectic(matrix)
        return self._add("symplectic", (mode,), np.array(matrix, copy=True))

    def cz(self, u, v, weight=1.0):
        return self._add("cz", (u, v), weight)

    def beamsplitter(self, u, v, theta):
        return self._add("beamsplitter", (u, v), theta)

    def identity(self, mode):
        return self._add("wire", (mode,), [0.0] * 4)

    def compile(self, squeezing=1.0):
        return compile_circuit(self, squeezing=squeezing)


class _Builder:
    def __init__(self, modes, r):
        self.pattern = Pattern(inputs=tuple(range(modes)))
        self.frontier = list(range(modes))
        self.next_node = modes
        self.r = r

    def step(self, mode, k):
        old, new = self.frontier[mode], self.next_node
        self.next_node += 1
        angle = atan2(1.0, k)
        self.pattern.extend([Prepare(new, self.r), Entangle(old, new)])
        self.pattern.measure(old, Homodyne(angle))
        self.pattern.displace(new, q=-Outcome(old) / sin(angle))
        self.frontier[mode] = new

    def single(self, mode, matrix):
        for k in decompose_symplectic(matrix):
            self.step(mode, k)

    def cz(self, u, v, weight):
        self.pattern.append(Entangle(self.frontier[u], self.frontier[v], weight))
        for mode in (u, v):
            for _ in range(4):
                self.step(mode, 0.0)

    def sum(self, control, target, weight):
        self.single(target, rotation(np.pi / 2))
        self.cz(control, target, weight)
        self.single(target, rotation(-np.pi / 2))


def compile_circuit(circuit, *, squeezing=1.0):
    builder = _Builder(circuit.modes, squeezing)
    for name, modes, params in circuit.gates:
        if name == "symplectic":
            builder.single(modes[0], params[0])
        elif name == "wire":
            for k in params[0]:
                builder.step(modes[0], k)
        elif name == "displace":
            # Four Fourier teleportations form identity; displacement is an output correction.
            for _ in range(4):
                builder.step(modes[0], 0.0)
            builder.pattern.append(Displace(builder.frontier[modes[0]], *params))
        elif name == "cz":
            builder.cz(modes[0], modes[1], params[0])
        elif name == "beamsplitter":
            # Two half rotations avoid the tan(theta/2) singularity at pi.
            theta = float(params[0])
            if not np.isfinite(theta):
                raise ValueError("Nonfinite beam splitter angle")
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            u, v = modes
            for _ in range(2):
                builder.sum(v, u, -np.tan(theta / 4))
                builder.sum(u, v, np.sin(theta / 2))
                builder.sum(v, u, -np.tan(theta / 4))
        else:
            raise NotImplementedError(f"Unsupported circuit gate: {name}")
    builder.pattern.append(Output(tuple(builder.frontier)))
    return builder.pattern.validate()
