"""Typed CV process diagrams: generators with quantum and classical ports.

A :class:`CVDiagram` is an immutable, causally ordered list of generator boxes acting on
labelled wires. Wires have one of two types:

* ``bosonic``: a quantum mode (label is any hashable mode name). Created by a preparation or
  an input, consumed by a measurement, a discard or an output. Never copied.
* ``real``: a classical real-valued record (label is the record key). Created by a
  measurement (homodyne: a real outcome with a probability *density*), read by
  feed-forward boxes and post-selection; may be read many times (classical copying).

Typing is checked by :meth:`CVDiagram.validate`: a bosonic wire is used only while alive,
classical wires are read only after they are produced, and no classical wire can feed a
bosonic port. Rewrites (``rules.py``) never mutate a diagram; they return a new one.

Conventions are PhotoGraphiQ's: hbar = 2, [q, p] = 2i, interleaved (q, p) per mode, vacuum
covariance I; ``Rotation(theta)`` is ``gaussian.rotation`` (q -> q cos - p sin),
``Squeezer(r)`` is diag(e^-r, e^r), ``Shear(s)`` is p -> p + s q, ``CZ(g)`` is p_u += g q_v,
p_v += g q_u, ``BeamSplitter`` is ``gaussian.beamsplitter``. A resource ``Prepare`` with
squeezing r is the p-squeezed state of ``GaussianBackend.prepare`` (covariance
diag(e^{2r}, e^{-2r})).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

BOSONIC, REAL = "bosonic", "real"


class DiagramTypeError(ValueError):
    """Malformed diagram: wrong wire type, use after consumption, missing record."""


def _num(x):
    return float(x) if isinstance(x, (int, float, np.floating, np.integer)) else x


# ----------------------------------------------------------------------------- boxes


@dataclass(frozen=True)
class Box:
    """Base generator. Subclasses declare their quantum/classical ports via properties."""

    gaussian = True  # Gaussian semantics available
    unitary = False  # a Gaussian unitary on its modes

    @property
    def modes(self) -> tuple:  # bosonic wires acted on (alive before and after)
        return ()

    @property
    def creates(self) -> tuple:  # bosonic wires created
        return ()

    @property
    def consumes(self) -> tuple:  # bosonic wires destroyed
        return ()

    @property
    def reads(self) -> tuple:  # classical wires read
        return ()

    @property
    def writes(self) -> tuple:  # classical wires produced
        return ()

    def params(self) -> dict:
        return {f.name: _num(getattr(self, f.name)) for f in fields(self)}

    def to_dict(self) -> dict:
        def enc(v):
            if isinstance(v, np.ndarray):
                return {"array": v.tolist()}
            if isinstance(v, tuple):
                return [enc(x) for x in v]
            if isinstance(v, dict):
                return {"dict": [[enc(k), enc(x)] for k, x in v.items()]}
            return _num(v)

        return {"box": type(self).__name__, **{k: enc(v) for k, v in self.params().items()}}


def _modes(*m):
    if len(set(m)) != len(m):
        raise DiagramTypeError(f"Repeated modes {m}")
    return tuple(m)


def _finite(**values):
    for k, v in values.items():
        if not np.all(np.isfinite(np.asarray(v, dtype=float))):
            raise DiagramTypeError(f"{k} must be finite")


# --- preparations -------------------------------------------------------------


@dataclass(frozen=True)
class Prepare(Box):
    """Single-mode Gaussian preparation: mean (q, p) and covariance (vacuum: I).

    ``Prepare.squeezed(mode, r)`` is the p-squeezed resource diag(e^{2r}, e^{-2r}).
    """

    mode: Any
    mean: tuple = (0.0, 0.0)
    covariance: tuple = ((1.0, 0.0), (0.0, 1.0))
    squeezing: float | None = None  # resource squeezing r when built by .squeezed (metadata)

    def __post_init__(self):
        cov = np.asarray(self.covariance, dtype=float)
        _finite(mean=self.mean, covariance=cov)
        if cov.shape != (2, 2) or not np.allclose(cov, cov.T):
            raise DiagramTypeError("Covariance must be a symmetric 2x2 matrix")
        # uncertainty principle (hbar = 2): V + i Omega >= 0
        if np.linalg.eigvalsh(cov + 1j * np.array([[0, 1], [-1, 0]])).min() < -1e-9:
            raise DiagramTypeError("Covariance violates the uncertainty principle")

    @classmethod
    def squeezed(cls, mode, r: float):
        if not np.isfinite(r) or r < 0:
            raise DiagramTypeError("Resource squeezing must be finite and nonnegative")
        return cls(mode, (0.0, 0.0), ((np.exp(2 * r), 0.0), (0.0, np.exp(-2 * r))), float(r))

    @classmethod
    def vacuum(cls, mode):
        return cls(mode)

    @property
    def creates(self):
        return (self.mode,)


@dataclass(frozen=True)
class PrepareState(Box):
    """Multi-mode Gaussian preparation (e.g. from a GaussianGraph): mean, covariance, labels."""

    labels: tuple
    mean: Any
    covariance: Any

    def __post_init__(self):
        object.__setattr__(self, "labels", _modes(*self.labels))
        mean, cov = np.asarray(self.mean, float), np.asarray(self.covariance, float)
        n = 2 * len(self.labels)
        if mean.shape != (n,) or cov.shape != (n, n):
            raise DiagramTypeError("PrepareState mean/covariance shape mismatch")
        _finite(mean=mean, covariance=cov)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

    @property
    def creates(self):
        return self.labels


# --- Gaussian unitaries ---------------------------------------------------------


@dataclass(frozen=True)
class Rotation(Box):
    mode: Any
    theta: float
    unitary = True

    def __post_init__(self):
        _finite(theta=self.theta)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class Squeezer(Box):
    mode: Any
    r: float
    unitary = True

    def __post_init__(self):
        _finite(r=self.r)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class Shear(Box):
    """Quadratic phase exp(i s q^2 / 4): p -> p + s q (``QuadraticPhase``)."""

    mode: Any
    s: float
    unitary = True

    def __post_init__(self):
        _finite(s=self.s)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class Displacement(Box):
    """Constant displacement (q, p) -> (q + dq, p + dp)."""

    mode: Any
    dq: float = 0.0
    dp: float = 0.0
    unitary = True

    def __post_init__(self):
        _finite(dq=self.dq, dp=self.dp)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class BeamSplitter(Box):
    u: Any
    v: Any
    theta: float
    unitary = True

    def __post_init__(self):
        _modes(self.u, self.v)
        _finite(theta=self.theta)

    @property
    def modes(self):
        return (self.u, self.v)


@dataclass(frozen=True)
class CZ(Box):
    u: Any
    v: Any
    g: float = 1.0
    unitary = True

    def __post_init__(self):
        _modes(self.u, self.v)
        _finite(g=self.g)

    @property
    def modes(self):
        return (self.u, self.v)


@dataclass(frozen=True)
class Symplectic(Box):
    """General Gaussian unitary: x -> S x + d on ``labels`` (interleaved), S symplectic."""

    labels: tuple
    S: Any
    d: Any = None
    unitary = True

    def __post_init__(self):
        from ...gaussian import omega

        object.__setattr__(self, "labels", _modes(*self.labels))
        n = len(self.labels)
        S = np.asarray(self.S, float)
        d = np.zeros(2 * n) if self.d is None else np.asarray(self.d, float)
        if S.shape != (2 * n, 2 * n) or d.shape != (2 * n,):
            raise DiagramTypeError("Symplectic shape mismatch")
        _finite(S=S, d=d)
        tol = 1e-9 * max(1.0, np.linalg.norm(S, 2) ** 2)
        if not np.allclose(S @ omega(n) @ S.T, omega(n), atol=tol, rtol=0):
            raise DiagramTypeError("Symplectic box requires S Omega S^T = Omega")
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "d", d)

    @property
    def modes(self):
        return self.labels


# --- channels, measurement, classical -------------------------------------------


@dataclass(frozen=True)
class GaussianLoss(Box):
    """Pure-loss / thermal-loss channel: x -> sqrt(eta) x, noise (1-eta)(2 n_th + 1) I."""

    mode: Any
    eta: float
    thermal: float = 0.0

    def __post_init__(self):
        if not 0 <= self.eta <= 1 or not np.isfinite(self.thermal) or self.thermal < 0:
            raise DiagramTypeError("Loss requires 0 <= eta <= 1 and finite thermal >= 0")

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class GaussianChannelBox(Box):
    """General Gaussian channel x -> X x + d with added noise Y on ``labels``.

    Complete positivity (hbar = 2): Y + i Omega - i X Omega X^T >= 0 is checked.
    """

    labels: tuple
    X: Any
    Y: Any
    d: Any = None

    def __post_init__(self):
        from ...gaussian import omega

        object.__setattr__(self, "labels", _modes(*self.labels))
        n = len(self.labels)
        X, Y = np.asarray(self.X, float), np.asarray(self.Y, float)
        d = np.zeros(2 * n) if self.d is None else np.asarray(self.d, float)
        if X.shape != (2 * n, 2 * n) or Y.shape != X.shape or d.shape != (2 * n,):
            raise DiagramTypeError("Channel shape mismatch")
        _finite(X=X, Y=Y, d=d)
        test = Y + 1j * omega(n) - 1j * X @ omega(n) @ X.T
        if not np.allclose(Y, Y.T) or np.linalg.eigvalsh(test).min() < -1e-9 * max(
            1.0, float(np.abs(Y).max())
        ):
            raise DiagramTypeError("Not a completely positive Gaussian channel")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "d", d)

    @property
    def modes(self):
        return self.labels


@dataclass(frozen=True)
class Homodyne(Box):
    """Destructive homodyne of q cos(theta) + p sin(theta) -> real record ``key``.

    The record has a probability density. ``efficiency``/``noise`` as in ``Homodyne``.
    """

    mode: Any
    theta: float
    key: Any
    efficiency: float = 1.0
    noise: float = 0.0

    def __post_init__(self):
        _finite(theta=self.theta)
        if not 0 < self.efficiency <= 1 or not np.isfinite(self.noise) or self.noise < 0:
            raise DiagramTypeError("Require 0 < efficiency <= 1 and finite noise >= 0")

    @property
    def consumes(self):
        return (self.mode,)

    @property
    def writes(self):
        return (self.key,)

    @property
    def ideal(self):
        return self.efficiency == 1 and self.noise == 0


@dataclass(frozen=True)
class Feedforward(Box):
    """Outcome-dependent displacement: q += c_q + sum_k a_k m_k, p += c_p + sum_k b_k m_k.

    ``gains`` maps record key -> (a_k, b_k). Exact affine form (no approximation).
    """

    mode: Any
    gains: tuple  # ((key, (a, b)), ...) sorted for hashing
    constant: tuple = (0.0, 0.0)
    unitary = False

    def __post_init__(self):
        gains = tuple(
            sorted(
                ((k, (float(a), float(b))) for k, (a, b) in dict(self.gains).items()),
                key=lambda kv: repr(kv[0]),
            )
        )
        object.__setattr__(self, "gains", gains)
        _finite(gains=[v for _, v in gains] or [0.0], constant=self.constant)

    @property
    def modes(self):
        return (self.mode,)

    @property
    def reads(self):
        return tuple(k for k, _ in self.gains)


@dataclass(frozen=True)
class ClassicalSource(Box):
    """A classical Gaussian random record N(mean, variance) (e.g. an eliminated ancilla)."""

    key: Any
    mean: float
    variance: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.variance) or self.variance < 0:
            raise DiagramTypeError("ClassicalSource needs a finite mean and variance >= 0")

    @property
    def writes(self):
        return (self.key,)


@dataclass(frozen=True)
class Discard(Box):
    mode: Any

    @property
    def consumes(self):
        return (self.mode,)


# --- non-Gaussian generators (Fock semantics only) -------------------------------


@dataclass(frozen=True)
class PhasePolynomial(Box):
    """exp(i f(q)) with f(q) = sum_k c_k q^k (hbar = 2 q). A 1 -> 1 q-spider.

    This is the one-input one-output q-spider of Nagayoshi et al. (PRR 7, 033141 (2025),
    Table II) with phase function f, restated in PhotoGraphiQ's hbar = 2 quadrature.
    Degree <= 2 is Gaussian (c1: p displacement 2 c1; c2: shear 4 c2); degree 3 is the
    cubic phase (PhotoGraphiQ CubicPhase(gamma) has c3 = gamma / 6).
    """

    mode: Any
    coefficients: tuple  # (c0, c1, c2, c3, ...)

    def __post_init__(self):
        c = tuple(float(x) for x in self.coefficients)
        _finite(coefficients=c or [0.0])
        while len(c) > 1 and c[-1] == 0:
            c = c[:-1]
        object.__setattr__(self, "coefficients", c)

    @property
    def gaussian(self):  # type: ignore[override]
        return len(self.coefficients) <= 3

    @property
    def unitary(self):  # type: ignore[override]
        return self.gaussian  # degree <= 2: a Gaussian unitary (shear + p displacement)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class Kerr(Box):
    """exp(i kappa n^2) (PhotoGraphiQ ``Kerr``)."""

    mode: Any
    kappa: float
    gaussian = False

    def __post_init__(self):
        _finite(kappa=self.kappa)

    @property
    def modes(self):
        return (self.mode,)


@dataclass(frozen=True)
class FockPrepare(Box):
    mode: Any
    n: int
    gaussian = False

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise DiagramTypeError("Fock number must be a nonnegative integer")

    @property
    def creates(self):
        return (self.mode,)


@dataclass(frozen=True)
class PhotonCount(Box):
    """Destructive photon-number measurement -> integer record ``key`` (a probability)."""

    mode: Any
    key: Any
    gaussian = False

    @property
    def consumes(self):
        return (self.mode,)

    @property
    def writes(self):
        return (self.key,)


@dataclass(frozen=True)
class Postselect(Box):
    """Keep only the branch where record ``key`` equals ``value`` (not a deterministic map)."""

    key: Any
    value: Any
    gaussian = False

    @property
    def reads(self):
        return (self.key,)


@dataclass(frozen=True)
class GKPPrepare(Box):
    """Finite-energy GKP codeword c0|0~> + c1|1~> of a ``photographiq.GKPCode`` (Fock only)."""

    mode: Any
    code: Any
    c0: complex = 1.0
    c1: complex = 0.0
    gaussian = False

    @property
    def creates(self):
        return (self.mode,)


@dataclass(frozen=True)
class ModularDecode(Box):
    """Classical nearest-cell parity of a homodyne record: bit = round(m / sqrt(2 pi)) mod 2."""

    source: Any
    key: Any
    spacing: float = float(np.sqrt(2 * np.pi))

    @property
    def reads(self):
        return (self.source,)

    @property
    def writes(self):
        return (self.key,)


GENERATORS = {
    cls.__name__: cls
    for cls in (
        Prepare,
        PrepareState,
        Rotation,
        Squeezer,
        Shear,
        Displacement,
        BeamSplitter,
        CZ,
        Symplectic,
        GaussianLoss,
        GaussianChannelBox,
        Homodyne,
        Feedforward,
        ClassicalSource,
        Discard,
        PhasePolynomial,
        Kerr,
        FockPrepare,
        PhotonCount,
        Postselect,
        GKPPrepare,
        ModularDecode,
    )
}


# ----------------------------------------------------------------------------- diagram


@dataclass(frozen=True)
class CVDiagram:
    """Immutable typed process diagram ``inputs -> outputs`` (ordered bosonic wires)."""

    inputs: tuple
    boxes: tuple
    outputs: tuple
    trace: tuple = field(default=(), compare=False)

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "boxes", tuple(self.boxes))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        self.validate()

    def validate(self):
        alive, records = list(self.inputs), set()
        if len(set(alive)) != len(alive):
            raise DiagramTypeError("Duplicate input wires")
        for i, b in enumerate(self.boxes):
            if not isinstance(b, Box):
                raise DiagramTypeError(f"Box {i} is not a generator: {b!r}")
            for k in b.reads:
                if k not in records:
                    raise DiagramTypeError(
                        f"Box {i} ({type(b).__name__}) reads record {k!r} before it is produced"
                    )
                if k in alive:
                    raise DiagramTypeError(f"Record {k!r} collides with a bosonic wire")
            for m in b.modes + b.consumes:
                if m not in alive:
                    raise DiagramTypeError(
                        f"Box {i} ({type(b).__name__}) uses bosonic wire "
                        f"{m!r} which is not alive"
                        + (" (it is a classical record)" if m in records else "")
                    )
            for m in b.consumes:
                alive.remove(m)
            for m in b.creates:
                if m in alive or m in records:
                    raise DiagramTypeError(f"Box {i} recreates wire {m!r}")
                alive.append(m)
            for k in b.writes:
                if k in records or k in alive:
                    raise DiagramTypeError(f"Record {k!r} written twice / collides with a mode")
                records.add(k)
        if sorted(map(repr, alive)) != sorted(map(repr, self.outputs)) or len(
            set(self.outputs)
        ) != len(self.outputs):
            raise DiagramTypeError(
                f"Outputs {self.outputs} differ from the live wires {tuple(alive)}"
            )
        return self

    # ------------------------------------------------------------------ structure
    @property
    def records(self) -> tuple:
        return tuple(k for b in self.boxes for k in b.writes)

    @property
    def is_gaussian(self) -> bool:
        return all(b.gaussian for b in self.boxes)

    @property
    def is_unitary(self) -> bool:
        return all(b.unitary for b in self.boxes) and set(self.inputs) == set(self.outputs)

    def with_boxes(self, boxes, step=None, outputs=None):
        trace = self.trace + ((step,) if step is not None else ())
        return CVDiagram(
            self.inputs, tuple(boxes), self.outputs if outputs is None else outputs, trace
        )

    def then(self, other: CVDiagram) -> CVDiagram:
        """Sequential composition (self first). Output wires of self feed inputs of other."""
        if tuple(other.inputs) != tuple(self.outputs):
            raise DiagramTypeError("Sequential composition needs matching ordered wires")
        clash = set(self.records) & set(other.records)
        if clash:
            raise DiagramTypeError(f"Record keys {clash} used in both diagrams")
        return CVDiagram(self.inputs, self.boxes + other.boxes, other.outputs)

    def tensor(self, other: CVDiagram) -> CVDiagram:
        """Parallel composition on disjoint wires and records."""
        wires = lambda d: set(d.inputs) | {m for b in d.boxes for m in b.creates}  # noqa: E731
        if wires(self) & wires(other) or set(self.records) & set(other.records):
            raise DiagramTypeError("Tensor product needs disjoint wires and records")
        return CVDiagram(
            self.inputs + other.inputs, self.boxes + other.boxes, self.outputs + other.outputs
        )

    def permute_outputs(self, order) -> CVDiagram:
        if sorted(map(repr, order)) != sorted(map(repr, self.outputs)):
            raise DiagramTypeError("Output permutation must reorder the same wires")
        return CVDiagram(self.inputs, self.boxes, tuple(order), self.trace)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "boxes": [b.to_dict() for b in self.boxes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=repr)

    @classmethod
    def from_dict(cls, data: dict) -> CVDiagram:
        def dec(v):
            if isinstance(v, dict) and "array" in v:
                return np.asarray(v["array"])
            if isinstance(v, dict) and "dict" in v:
                return tuple((dec(k), dec(x)) for k, x in v["dict"])
            if isinstance(v, list):
                return tuple(dec(x) for x in v)
            return v

        boxes = []
        for b in data["boxes"]:
            b = dict(b)
            kind = GENERATORS[b.pop("box")]
            if kind in (GKPPrepare,):
                raise NotImplementedError(
                    "GKPPrepare holds a code object; serialise the code separately"
                )
            boxes.append(kind(**{k: dec(v) for k, v in b.items()}))
        return cls(tuple(dec(data["inputs"])), tuple(boxes), tuple(dec(data["outputs"])))

    def fingerprint(self, digits: int = 10) -> str:
        """Canonical hash of the box sequence (parameters rounded to ``digits``)."""

        def canon(v):
            if isinstance(v, (float, np.floating)):
                return round(float(v), digits) + 0.0
            if isinstance(v, (list, tuple)):
                return [canon(x) for x in v]
            if isinstance(v, dict):
                return {k: canon(x) for k, x in v.items()}
            return v

        payload = json.dumps(canon(self.to_dict()), sort_keys=True, default=repr)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __repr__(self):
        return f"CVDiagram(inputs={self.inputs}, boxes={len(self.boxes)}, outputs={self.outputs})"


__all__ = [*GENERATORS, "Box", "CVDiagram", "DiagramTypeError", "BOSONIC", "REAL"]
