"""Trainable Gaussian CV programs: data encoding, ideal moments and compiled MBQC channels.

Conventions are PhotoGraphiQ's: hbar = 2, [q_j, p_k] = 2 i delta_jk, interleaved quadratures
(q_1, p_1, ..., q_n, p_n), vacuum covariance I. A Gaussian unitary acts as d -> S d + t,
V -> S V S^T; a Gaussian channel as d -> X d + t, V -> X V X^T + Y.

Two propagations of the same gate list are kept separate:

* ``ideal``: the intended symplectic map S (direct optical circuit, no resource noise);
* ``physical``: the unconditional channel of the finite-squeezing CV-MBQC pattern that
  ``photographiq.compile_circuit`` produces for the gate list. It mirrors the compiler's
  lowering step by step (each teleportation step T(k) adds diag(0, exp(-2 r)) to Y; see
  ``photographiq.gaussian.wire_channel``), so X equals S exactly and the finite-resource noise Y
  is reported separately. Tests check X, Y and t against ``photographiq.analysis.
  gaussian_channel`` of the actual compiled ``Pattern`` and against trajectory execution.

Every arithmetic operation here is real-analytic in the parameters (branch decisions use real
parts), so the same code evaluates complex-step derivatives d f = Im f(x + i h) / h, which are
exact to rounding for these deterministic moment predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import photographiq as pg

GATES = ("rotate", "squeeze", "shear", "displace", "beamsplitter", "cz")
ENCODINGS = ("displacement", "squeezing", "rotation")


@dataclass(frozen=True)
class Feature:
    """A data-dependent gate argument ``scale * x[index]`` (never trainable)."""

    index: int
    scale: float = 1.0

    def __post_init__(self):
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Feature index must be a nonnegative integer")
        if not np.isfinite(self.scale):
            raise ValueError("Feature scale must be finite")


def _value(arg, params, x):
    if isinstance(arg, str):
        return params[arg]
    if isinstance(arg, Feature):
        if arg.index >= len(x):
            raise ValueError(f"Feature index {arg.index} exceeds the input dimension {len(x)}")
        return arg.scale * x[arg.index]
    return arg


# Gate matrices written with plain arithmetic so that complex-step perturbations survive.
def _rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _squeezing(r):
    return np.array([[np.exp(-r), 0 * r], [0 * r, np.exp(r)]])


def _shear(s):
    return np.array([[1 + 0 * s, 0 * s], [s, 1 + 0 * s]])


def _teleport(k):
    return np.array([[-k, -1 + 0 * k], [1 + 0 * k, 0 * k]])


def _beamsplitter(theta):
    c, s = np.cos(theta), np.sin(theta)
    z = 0 * c
    return np.array([[c, z, -s, z], [z, c, z, -s], [s, z, c, z], [z, s, z, c]])


def _cz(g):
    m = np.eye(4) + 0 * g
    m[1, 2] = m[3, 0] = g
    return m


def _ks(matrix):
    """Teleportation shears of ``photographiq.compiler.decompose_symplectic`` (same branch)."""
    m, prefix = matrix, []
    if abs(np.real(m[1, 0])) < 1e-8:
        prefix = [0.0]
        m = m @ _teleport(0.0).T
    a, c, d = m[0, 0], m[1, 0], m[1, 1]
    return prefix + [0.0, (1 - d) / c, c, (1 - a) / c]


class _Moments:
    """Running affine map (X, t) and added noise Y with local (block) updates."""

    def __init__(self, modes, dtype, physical, squeezing):
        self.X = np.eye(2 * modes, dtype=dtype)
        self.t = np.zeros(2 * modes, dtype=dtype)
        self.Y = np.zeros((2 * modes, 2 * modes), dtype=dtype)
        self.physical, self.noise = (
            physical,
            (None if squeezing is None else np.exp(-2 * squeezing)),
        )
        self.steps = 0

    def apply(self, idx, m):
        self.X[idx] = m @ self.X[idx]
        self.t[idx] = m @ self.t[idx]
        if self.physical:
            self.Y[idx] = m @ self.Y[idx]
            self.Y[:, idx] = self.Y[:, idx] @ m.T

    def step(self, mode, k):  # one finite-squeezing teleportation step (compiler _Builder.step)
        idx = [2 * mode, 2 * mode + 1]
        self.apply(idx, _teleport(k))
        self.Y[2 * mode + 1, 2 * mode + 1] += self.noise
        self.steps += 1

    def single(self, mode, matrix):
        if self.physical:
            for k in _ks(matrix):
                self.step(mode, k)
        else:
            self.apply([2 * mode, 2 * mode + 1], matrix)

    def cz(self, u, v, g):
        self.apply([2 * u, 2 * u + 1, 2 * v, 2 * v + 1], _cz(g))
        if self.physical:
            for mode in (u, v):
                for _ in range(4):
                    self.step(mode, 0.0)

    def sum(self, control, target, w):
        self.single(target, _rotation(np.pi / 2))
        self.cz(control, target, w)
        self.single(target, _rotation(-np.pi / 2))


class CVFeatureMap:
    """Classical data -> product Gaussian input state, one feature per mode.

    ``displacement``: mode j holds a coherent state whose q (or p, ``quadrature="p"``) mean is
    scale * x_j, i.e. coherent amplitude alpha = scale * x_j / 2 in hbar = 2 units.
    ``squeezing``: S(r)|0> with r = scale * x_j, |r| <= ``max_squeezing`` enforced.
    ``rotation``: R(scale * x_j) applied to a fixed coherent reference with q mean ``offset``.
    Input-independent ``input_squeezing`` squeezes every mode before the data enters (it is
    part of the state preparation, not a trainable parameter).
    """

    def __init__(
        self,
        modes,
        encoding="displacement",
        *,
        scale=1.0,
        quadrature="q",
        offset=1.0,
        max_squeezing=1.5,
        input_squeezing=0.0,
    ):
        if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
            raise ValueError("modes must be a positive integer")
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if quadrature not in ("q", "p"):
            raise ValueError("quadrature must be 'q' or 'p'")
        for name, v in (
            ("scale", scale),
            ("offset", offset),
            ("max_squeezing", max_squeezing),
            ("input_squeezing", input_squeezing),
        ):
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if max_squeezing <= 0 or abs(input_squeezing) > max_squeezing:
            raise ValueError("Require max_squeezing > 0 and |input_squeezing| <= max_squeezing")
        self.modes, self.encoding, self.scale = modes, encoding, float(scale)
        self.quadrature, self.offset = quadrature, float(offset)
        self.max_squeezing, self.input_squeezing = float(max_squeezing), float(input_squeezing)

    def check(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.modes or not np.isfinite(X).all():
            raise ValueError(f"Inputs must be a finite array of shape (N, {self.modes})")
        if (
            self.encoding == "squeezing"
            and np.abs(self.scale * X).max(initial=0) > self.max_squeezing
        ):
            raise ValueError("Encoded squeezing exceeds max_squeezing; rescale the features")
        return X

    def moments(self, x):
        """(mean, covariance) of the encoded product state for one feature vector."""
        n = self.modes
        mean, cov = np.zeros(2 * n), np.eye(2 * n)
        base = _squeezing(self.input_squeezing)
        for j in range(n):
            idx = slice(2 * j, 2 * j + 2)
            v = self.scale * x[j]
            if self.encoding == "displacement":
                mean[2 * j + (self.quadrature == "p")] = v
                s = base
            elif self.encoding == "squeezing":
                s = _squeezing(v) @ base
            else:
                s = _rotation(v) @ base
                mean[idx] = _rotation(v) @ np.array([self.offset, 0.0])
            cov[idx, idx] = s @ s.T
        return mean, cov

    def mean_photons(self, x):
        """Mean photon number of the encoded state, (tr V + |d|^2)/4 - n/2 (hbar = 2)."""
        mean, cov = self.moments(x)
        return float((np.trace(cov) + mean @ mean) / 4 - self.modes / 2)

    def config(self):
        return {
            "modes": self.modes,
            "encoding": self.encoding,
            "scale": self.scale,
            "quadrature": self.quadrature,
            "offset": self.offset,
            "max_squeezing": self.max_squeezing,
            "input_squeezing": self.input_squeezing,
        }


class GaussianProgram:
    """Ordered trainable Gaussian gates; arguments are numbers, parameter names or Features.

    Trainable squeezing and displacement arguments may be bounded as v = bound * tanh(theta),
    which keeps optimisation inside a physical squeezing or energy budget without clipping
    (for Fock execution the cutoff is such a budget).
    """

    def __init__(self, modes):
        if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
            raise ValueError("modes must be a positive integer")
        self.modes, self.gates = modes, []

    def _add(self, kind, modes, *args, bound=None):
        if len(set(modes)) != len(modes) or any(
            isinstance(m, bool) or not isinstance(m, int) or not 0 <= m < self.modes for m in modes
        ):
            raise ValueError(f"Gate modes must be distinct integers in range(0, {self.modes})")
        for a in args:
            if not isinstance(a, (str, Feature)) and not np.isfinite(a):
                raise ValueError("Gate arguments must be finite numbers, names or Features")
        if bound is not None and (not np.isfinite(bound) or bound <= 0):
            raise ValueError("Gate bound must be positive and finite")
        self.gates.append((kind, tuple(modes), args, bound))
        return self

    def rotate(self, mode, angle):
        return self._add("rotate", (mode,), angle)

    def squeeze(self, mode, r, *, bound=None):
        return self._add("squeeze", (mode,), r, bound=bound)

    def shear(self, mode, s):
        """Quadratic phase exp(i s q^2 / 4): p -> p + s q."""
        return self._add("shear", (mode,), s)

    def displace(self, mode, q=0.0, p=0.0, *, bound=None):
        return self._add("displace", (mode,), q, p, bound=bound)

    def beamsplitter(self, u, v, theta):
        return self._add("beamsplitter", (u, v), theta)

    def cz(self, u, v, weight):
        return self._add("cz", (u, v), weight)

    @property
    def parameters(self):
        names = []
        for _, _, args, _ in self.gates:
            names += [a for a in args if isinstance(a, str) and a not in names]
        return tuple(names)

    @property
    def data_dependent(self):
        return any(isinstance(a, Feature) for _, _, args, _ in self.gates for a in args)

    def _resolved(self, params, x):
        for kind, modes, args, bound in self.gates:
            values = [_value(a, params, x) for a in args]
            if bound is not None:  # squeeze / displace budgets apply to trainable arguments
                values = [
                    bound * np.tanh(v) if isinstance(a, str) else v for a, v in zip(args, values)
                ]
            yield kind, modes, values

    def squeezing_values(self, params, x=()):
        """Physical squeezing r of every squeeze gate (after bounding)."""
        return [v[0] for k, _, v in self._resolved(params, x) if k == "squeeze"]

    def channel(self, params, x=(), *, mode="ideal", squeezing=None):
        """(X, t, Y, teleportation steps) of the ideal circuit or its compiled channel."""
        physical = mode == "physical"
        if mode not in ("ideal", "physical"):
            raise ValueError("mode must be 'ideal' or 'physical'")
        if physical and (squeezing is None or not np.isfinite(squeezing) or squeezing < 0):
            raise ValueError("Physical mode needs finite nonnegative resource squeezing")
        dtype = complex if any(np.iscomplexobj(v) for v in params.values()) else float
        m = _Moments(self.modes, dtype, physical, squeezing)
        for kind, modes, v in self._resolved(params, x):
            if kind == "rotate":
                m.single(modes[0], _rotation(v[0]))
            elif kind == "squeeze":
                m.single(modes[0], _squeezing(v[0]))
            elif kind == "shear":
                m.single(modes[0], _shear(v[0]))
            elif kind == "displace":
                if physical:  # four Fourier steps form the identity; the shift is a correction
                    for _ in range(4):
                        m.step(modes[0], 0.0)
                m.t[2 * modes[0]] += v[0]
                m.t[2 * modes[0] + 1] += v[1]
            elif kind == "cz":
                m.cz(modes[0], modes[1], v[0])
            elif not physical:
                m.apply(
                    [2 * modes[0], 2 * modes[0] + 1, 2 * modes[1], 2 * modes[1] + 1],
                    _beamsplitter(v[0]),
                )
            else:  # compiler: two half rotations, each three SUM gates
                theta = v[0] - 2 * np.pi * np.floor((np.real(v[0]) + np.pi) / (2 * np.pi))
                u, w = modes
                for _ in range(2):
                    m.sum(w, u, -np.tan(theta / 4))
                    m.sum(u, w, np.sin(theta / 2))
                    m.sum(w, u, -np.tan(theta / 4))
        return m.X, m.t, m.Y, m.steps

    def circuit(self, params, x=()):
        """The equivalent ``photographiq.Circuit`` (compiles to the physical CV-MBQC pattern)."""
        c = pg.Circuit(self.modes)
        for kind, modes, v in self._resolved(params, x):
            v = [float(a) for a in v]
            if kind == "rotate":
                c.rotate(modes[0], v[0])
            elif kind == "squeeze":
                c.squeeze(modes[0], v[0])
            elif kind == "shear":
                c.gaussian(modes[0], [[1.0, 0.0], [v[0], 1.0]])
            elif kind == "displace":
                c.displace(modes[0], v[0], v[1])
            elif kind == "beamsplitter":
                c.beamsplitter(modes[0], modes[1], v[0])
            else:
                c.cz(modes[0], modes[1], v[0])
        return c

    def gate_pattern(self, params, x=()):
        """Direct optical circuit as a gate-level ``Pattern`` (no measurements, no resources)."""
        p = pg.Pattern(inputs=tuple(range(self.modes)))
        for kind, modes, v in self._resolved(params, x):
            v = [float(a) for a in v]
            if kind == "rotate":
                p.append(pg.Rotate(modes[0], v[0]))
            elif kind == "squeeze":
                p.append(pg.Squeeze(modes[0], v[0]))
            elif kind == "shear":
                p.append(pg.QuadraticPhase(modes[0], v[0]))
            elif kind == "displace":
                p.append(pg.Displace(modes[0], v[0], v[1]))
            elif kind == "beamsplitter":
                p.append(pg.BeamSplitter(modes[0], modes[1], v[0]))
            else:
                p.append(pg.Entangle(modes[0], modes[1], v[0]))
        return p.append(pg.Output(tuple(range(self.modes)))).validate()

    def config(self):
        return {
            "modes": self.modes,
            "gates": [
                {
                    "kind": k,
                    "modes": list(m),
                    "bound": b,
                    "args": [
                        {"feature": a.index, "scale": a.scale} if isinstance(a, Feature) else a
                        for a in args
                    ],
                }
                for k, m, args, b in self.gates
            ],
        }

    @classmethod
    def from_config(cls, config):
        program = cls(config["modes"])
        for g in config["gates"]:
            args = [
                Feature(a["feature"], a["scale"]) if isinstance(a, dict) else a for a in g["args"]
            ]
            program._add(g["kind"], tuple(g["modes"]), *args, bound=g["bound"])
        return program


def layered_program(modes, layers, *, reupload=False, squeezing_bound=1.0, prefix="L"):
    """Standard trainable Gaussian ansatz: per layer R, S, beam-splitter chain, R, D.

    With ``reupload`` each layer after the first starts with q displacements by the features
    (scale 1), so data enters every layer; the first layer gets data from the feature map.
    """
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be a positive integer")
    program = GaussianProgram(modes)
    for layer in range(layers):
        name = f"{prefix}{layer}"
        if reupload and layer:
            for j in range(modes):
                program.displace(j, Feature(j), 0.0)
        for j in range(modes):
            program.rotate(j, f"{name}.r{j}")
            program.squeeze(j, f"{name}.s{j}", bound=squeezing_bound)
        for j in range(modes - 1):
            program.beamsplitter(j, j + 1, f"{name}.bs{j}")
        for j in range(modes):
            program.rotate(j, f"{name}.r'{j}")
            program.displace(j, f"{name}.dq{j}", f"{name}.dp{j}")
    return program
