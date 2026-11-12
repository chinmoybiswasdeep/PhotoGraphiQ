"""Qubit measurement calculus (Danos, Kashefi and Panangaden, J. ACM 54, 8 (2007)).

Conventions (all angles in radians; Graphix 0.4 uses the same planes in units of pi):

* ``N(i)`` prepares ``|+> = (|0> + |1>)/sqrt(2)``; ``E(i, j)`` is ``CZ = diag(1, 1, 1, -1)``.
* ``M(i, plane, angle, s_domain, t_domain)`` measures the observable ``O(plane, angle)``

  - XY: ``cos(a) X + sin(a) Y``, outcome-0 state ``(|0> + e^{ia}|1>)/sqrt(2)``
  - YZ: ``cos(a) Z + sin(a) Y``, outcome-0 state ``cos(a/2)|0> + i sin(a/2)|1>``
  - XZ: ``cos(a) Z + sin(a) X``, outcome-0 state ``cos(a/2)|0> + sin(a/2)|1>``

  Outcome ``0`` is the ``+1`` eigenvalue. With signals ``s = sum(s_domain)``, ``t = sum(t_domain)``
  (mod 2) the command measures ``X^s Z^t`` *before* ``O``, i.e. the effective observable is
  ``Z^t X^s O X^s Z^t``. Pauli conjugation keeps the plane and changes the angle:

  ============  ===================  ===================
  plane         X (s-domain)         Z (t-domain)
  ============  ===================  ===================
  XY            ``a -> -a``          ``a -> a + pi``
  YZ            ``a -> a + pi``      ``a -> -a``
  XZ            ``a -> pi - a``      ``a -> -a``
  ============  ===================  ===================

  so the domains are plane independent as operators, which makes the absorption rules
  ``M_i^{s,t} X_i^d = M_i^{s xor d, t}`` and ``M_i^{s,t} Z_i^d = M_i^{s, t xor d}`` exact.
* ``X(i, domain)``/``Z(i, domain)`` apply the Pauli iff the domain parity is 1; an empty domain
  is the identity. ``S(i, domain)`` is the DKP signal command ``s_i <- s_i xor parity(domain)``
  (parities read before the update); it is purely classical.

Commands are listed in execution order (leftmost first), the reverse of DKP operator notation.
All outcome-dependent maps are compared *per branch*, keyed by the final signals, up to a
phase of that branch. Such phases are unobservable because the branch is labelled by classical
outcomes; they arise from the absorption rules (``<+_a| X = e^{-ia} <+_{-a}|`` for XY) and
from reordering X and Z on one qubit (``(-1)^{st}``). Branch probabilities (``K^dagger K``) are
compared without any phase freedom.

The dense simulator is exact (no sampling) and exponential in the number of simultaneously
live qubits plus inputs; it refuses more than ``MAX_QUBITS``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import networkx as nx
import numpy as np

from ..ir import Model, OpenGraph, Plane

__all__ = [
    "MAX_QUBITS",
    "N",
    "E",
    "M",
    "X",
    "Z",
    "S",
    "Branch",
    "DeterminismReport",
    "EquivalenceReport",
    "QubitPattern",
    "branch_operators",
    "effective_angle",
    "enumerate_branches",
    "equivalent",
    "is_deterministic",
    "measurement_basis",
    "observable",
    "pattern_from_flow",
    "rewrite_at",
    "run_branch",
    "signal_shift",
    "standardize",
]

MAX_QUBITS = 20
_PI = math.pi


def _domain(nodes: Iterable) -> frozenset:
    return frozenset(nodes)


@dataclass(frozen=True)
class N:
    """Prepare ``node`` in ``|+>``."""

    node: Any


@dataclass(frozen=True)
class E:
    """Apply CZ between ``u`` and ``v`` (symmetric)."""

    u: Any
    v: Any

    @property
    def nodes(self) -> tuple:
        return (self.u, self.v)


@dataclass(frozen=True)
class M:
    """Destructively measure ``node`` in ``plane`` at ``angle`` with s- and t-domains."""

    node: Any
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: frozenset = field(default_factory=frozenset)
    t_domain: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane", Plane(self.plane))
        object.__setattr__(self, "angle", float(self.angle))
        if not math.isfinite(self.angle):
            raise ValueError("Measurement angle must be finite")
        object.__setattr__(self, "s_domain", _domain(self.s_domain))
        object.__setattr__(self, "t_domain", _domain(self.t_domain))


@dataclass(frozen=True)
class X:
    """Pauli X on ``node`` iff the parity of ``domain`` is 1."""

    node: Any
    domain: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _domain(self.domain))


@dataclass(frozen=True)
class Z:
    """Pauli Z on ``node`` iff the parity of ``domain`` is 1."""

    node: Any
    domain: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _domain(self.domain))


@dataclass(frozen=True)
class S:
    """Signal shift ``s_node <- s_node xor parity(domain)`` (classical only)."""

    node: Any
    domain: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _domain(self.domain))


Command = N | E | M | X | Z | S
_RANK = {N: 0, E: 1, M: 2, X: 3, Z: 3, S: 4}


def _domains(c: Command) -> tuple[frozenset, ...]:
    if isinstance(c, M):
        return (c.s_domain, c.t_domain)
    if isinstance(c, (X, Z, S)):
        return (c.domain,)
    return ()


def _quantum_nodes(c: Command) -> tuple:
    if isinstance(c, E):
        return c.nodes
    return () if isinstance(c, S) else (c.node,)


class QubitPattern:
    """Ordered qubit measurement pattern with explicit inputs and outputs.

    Args:
        inputs: Ordered input qubits (not prepared by ``N``).
        commands: Commands in execution order.
        outputs: Ordered output qubits; ``None`` infers the unmeasured qubits in order of
            appearance (inputs first).
    """

    def __init__(self, inputs: Sequence = (), commands: Iterable[Command] = (), outputs=None):
        self.inputs = tuple(inputs)
        self.commands: list[Command] = list(commands)
        if outputs is None:
            measured = {c.node for c in self.commands if isinstance(c, M)}
            seen = list(self.inputs) + [c.node for c in self.commands if isinstance(c, N)]
            outputs = tuple(n for n in seen if n not in measured)
        self.outputs = tuple(outputs)
        self.validate()

    def __repr__(self) -> str:
        return (
            f"QubitPattern(inputs={self.inputs!r}, outputs={self.outputs!r}, "
            f"commands={len(self.commands)})"
        )

    def copy(self, commands: Iterable[Command] | None = None) -> QubitPattern:
        return QubitPattern(
            self.inputs, self.commands if commands is None else commands, self.outputs
        )

    @property
    def measured(self) -> tuple:
        """Measured qubits in command order."""
        return tuple(c.node for c in self.commands if isinstance(c, M))

    def validate(self) -> QubitPattern:
        """Check DKP definiteness and causality; return self.

        Every non-input qubit is prepared once before use, no qubit is used after its
        measurement, domains refer only to earlier outcomes, and the outputs are exactly the
        unmeasured qubits.
        """
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Duplicate input qubits")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Duplicate output qubits")
        live, ever = set(self.inputs), set(self.inputs)
        measured: set = set()
        for i, c in enumerate(self.commands):
            if not isinstance(c, (N, E, M, X, Z, S)):
                raise TypeError(f"Command {i} is not a qubit calculus command: {c!r}")
            for d in _domains(c):
                if not d <= measured:
                    raise ValueError(f"Command {i} depends on an unmeasured qubit: {c!r}")
            if isinstance(c, N):
                if c.node in ever:
                    raise ValueError(f"Command {i} re-prepares qubit {c.node!r}")
                live.add(c.node)
                ever.add(c.node)
            elif isinstance(c, S):
                if c.node not in measured or c.node in c.domain:
                    raise ValueError(f"Command {i} must shift a measured qubit by other signals")
            else:
                if isinstance(c, E) and c.u == c.v:
                    raise ValueError(f"Command {i} is a CZ on one qubit")
                if not set(_quantum_nodes(c)) <= live:
                    raise ValueError(f"Command {i} acts on an unprepared or measured qubit")
                if isinstance(c, M):
                    live.remove(c.node)
                    measured.add(c.node)
        if set(self.outputs) != live:
            raise ValueError(f"Outputs {self.outputs!r} differ from unmeasured qubits {live!r}")
        return self

    def dependency_graph(self) -> nx.DiGraph:
        """DAG on measured qubits: edge ``j -> i`` (attr ``kind`` in {'s','t'}) if M_i reads s_j.

        Output corrections and ``S`` commands are classical/final and are not included.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.measured)
        for c in self.commands:
            if isinstance(c, M):
                for kind, dom in (("s", c.s_domain), ("t", c.t_domain)):
                    for j in dom:
                        graph.add_edge(j, c.node, kind=kind)
        return graph

    def dependencies(self) -> dict:
        """Map each measured qubit to the outcomes its measurement basis depends on."""
        return {c.node: c.s_domain | c.t_domain for c in self.commands if isinstance(c, M)}

    def schedule(self) -> tuple[frozenset, ...]:
        """Measurement layers: qubits in a layer depend only on earlier layers."""
        return tuple(
            frozenset(layer) for layer in nx.topological_generations(self.dependency_graph())
        )

    def is_standard(self) -> bool:
        """True if the commands are ordered N*, E*, M*, (X|Z)*, S* (NEMC form)."""
        ranks = [_RANK[type(c)] for c in self.commands]
        return ranks == sorted(ranks)

    def reorder_measurements(self, order: Sequence) -> QubitPattern:
        """Permute the measurement block of a standard pattern along a causal ``order``."""
        if not self.is_standard():
            raise ValueError("Reordering requires a standard pattern")
        by_node = {c.node: c for c in self.commands if isinstance(c, M)}
        if len(order) != len(by_node) or set(order) != set(by_node):
            raise ValueError("Order must list every measured qubit once")
        rest = [c for c in self.commands if not isinstance(c, M)]
        head = [c for c in rest if isinstance(c, (N, E))]
        tail = [c for c in rest if not isinstance(c, (N, E))]
        moved: list[Command] = [*head, *(by_node[n] for n in order), *tail]
        return self.copy(moved)  # validate() checks causality


# --------------------------------------------------------------------------- semantics


def observable(plane: Plane, angle: float) -> np.ndarray:
    """2x2 measured observable ``O(plane, angle)`` (see module docstring)."""
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1.0 + 0j, -1.0])
    c, s = math.cos(angle), math.sin(angle)
    plane = Plane(plane)
    if plane is Plane.XY:
        return c * x + s * y
    if plane is Plane.YZ:
        return c * z + s * y
    return c * z + s * x


def measurement_basis(plane: Plane, angle: float, outcome: int) -> np.ndarray:
    """Eigenvector of ``O(plane, angle)`` with eigenvalue ``(-1)**outcome``."""
    a = float(angle) + _PI * (outcome & 1)
    plane = Plane(plane)
    if plane is Plane.XY:
        return np.array([1.0, np.exp(1j * a)]) / math.sqrt(2)
    if plane is Plane.YZ:
        return np.array([math.cos(a / 2), 1j * math.sin(a / 2)])
    return np.array([math.cos(a / 2), math.sin(a / 2)], dtype=complex)


def effective_angle(plane: Plane, angle: float, s: int, t: int) -> float:
    """Angle measured after the ``X^s Z^t`` frame (table in the module docstring)."""
    plane, a = Plane(plane), float(angle)
    if s & 1:
        a = {Plane.XY: -a, Plane.YZ: a + _PI, Plane.XZ: _PI - a}[plane]
    if t & 1:
        a = a + _PI if plane is Plane.XY else -a
    return a


def _parity(domain: frozenset, signals: Mapping) -> int:
    return sum(signals[j] for j in domain) & 1


@dataclass(frozen=True)
class Branch:
    """One outcome branch: raw outcomes, final signals and the (unnormalized) Kraus operator.

    ``operator`` maps ``2**len(inputs)`` to ``2**len(outputs)`` amplitudes, big-endian with the
    first listed qubit most significant. For a normalized input ``psi`` the branch probability is
    ``||operator @ psi||**2``.
    """

    outcomes: dict
    signals: dict
    operator: np.ndarray

    def probability(self, state: np.ndarray | None = None) -> float:
        """Branch probability for ``state`` (default: the maximally mixed input)."""
        k = self.operator
        if state is None:
            return float(np.real(np.trace(k.conj().T @ k)) / k.shape[1])
        out = k @ np.asarray(state)
        return float(np.real(np.vdot(out, out)))


def _apply1(state: np.ndarray, u: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(np.tensordot(u, state, axes=([1], [axis])), 0, axis)


def run_branch(pattern: QubitPattern, outcomes: Mapping) -> Branch:
    """Execute ``pattern`` for given raw ``outcomes`` (node -> 0/1) and return its Kraus map.

    The input register is carried as a batch axis, so one run yields the whole linear map.
    """
    pattern.validate()
    missing = set(pattern.measured) - set(outcomes)
    if missing:
        raise ValueError(f"Missing outcomes for {sorted(map(repr, missing))}")
    n_in = len(pattern.inputs)
    width = n_in
    live = list(pattern.inputs)
    peak = count = n_in  # simultaneously live qubits; the input batch axis adds n_in more
    for c in pattern.commands:
        count += isinstance(c, N) - isinstance(c, M)
        peak = max(peak, count)
    if peak + n_in > MAX_QUBITS:
        raise ValueError(f"Dense simulation limited to {MAX_QUBITS} qubits (needs {peak + n_in})")
    state = np.eye(2**width, dtype=complex).reshape((2,) * width + (2**width,))
    plus = np.array([1.0, 1.0]) / math.sqrt(2)
    xgate = np.array([[0, 1], [1, 0]], complex)
    zgate = np.diag([1.0 + 0j, -1.0])
    signals: dict = {}
    for c in pattern.commands:
        if isinstance(c, N):
            state = np.expand_dims(state, -2) * plus.reshape(2, 1)
            live.append(c.node)
        elif isinstance(c, E):
            state = state.copy()
            index: list = [slice(None)] * state.ndim
            index[live.index(c.u)] = index[live.index(c.v)] = 1
            state[tuple(index)] *= -1
        elif isinstance(c, M):
            angle = effective_angle(
                c.plane, c.angle, _parity(c.s_domain, signals), _parity(c.t_domain, signals)
            )
            r = int(outcomes[c.node]) & 1
            bra = measurement_basis(c.plane, angle, r).conj()
            state = np.tensordot(bra, state, axes=([0], [live.index(c.node)]))
            live.remove(c.node)
            signals[c.node] = r
        elif isinstance(c, (X, Z)):
            if _parity(c.domain, signals):
                state = _apply1(state, xgate if isinstance(c, X) else zgate, live.index(c.node))
        elif isinstance(c, S):
            signals[c.node] ^= _parity(c.domain, signals)
    perm = [live.index(n) for n in pattern.outputs] + [len(live)]
    operator = np.transpose(state, perm).reshape(2 ** len(pattern.outputs), 2**n_in)
    raw = {n: int(outcomes[n]) & 1 for n in pattern.measured}
    return Branch(raw, signals, operator)


def enumerate_branches(pattern: QubitPattern) -> list[Branch]:
    """All ``2**len(measured)`` branches of ``pattern``."""
    nodes = pattern.measured
    return [
        run_branch(pattern, dict(zip(nodes, bits)))
        for bits in itertools.product((0, 1), repeat=len(nodes))
    ]


def branch_operators(pattern: QubitPattern, nodes: Sequence | None = None) -> dict:
    """Map final-signal tuples (over ``nodes``, default measurement order) to Kraus operators."""
    nodes = pattern.measured if nodes is None else tuple(nodes)
    return {tuple(b.signals[n] for n in nodes): b.operator for b in enumerate_branches(pattern)}


@dataclass(frozen=True)
class EquivalenceReport:
    """Per-branch comparison ``K_a = exp(i phi_b) K_b`` keyed by final signals."""

    equivalent: bool
    max_deviation: float
    max_probability_deviation: float
    phases: dict

    def __bool__(self) -> bool:
        return self.equivalent


def _phase_deviation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    overlap = np.vdot(b, a)
    phase = float(np.angle(overlap)) if abs(overlap) > 1e-300 else 0.0
    return phase, float(np.max(np.abs(a - np.exp(1j * phase) * b), initial=0.0))


def equivalent(a: QubitPattern, b: QubitPattern, *, atol: float = 1e-10) -> EquivalenceReport:
    """Compare two patterns as linear maps branch by branch.

    Both must have the same ordered inputs/outputs and measured set. Each branch (fixed final
    signals) must agree up to a global phase of that branch; branch probabilities, i.e.
    ``K^dagger K``, are compared phase-free and reported separately.
    """
    if a.inputs != b.inputs or a.outputs != b.outputs:
        raise ValueError("Patterns differ in ordered inputs or outputs")
    if set(a.measured) != set(b.measured):
        raise ValueError("Patterns measure different qubits")
    nodes = a.measured
    ka, kb = branch_operators(a, nodes), branch_operators(b, nodes)
    worst = worst_p = 0.0
    phases = {}
    for key, op in ka.items():
        phase, dev = _phase_deviation(op, kb[key])
        prob = np.abs(op.conj().T @ op - kb[key].conj().T @ kb[key])
        phases[key] = phase
        worst, worst_p = max(worst, dev), max(worst_p, float(prob.max(initial=0.0)))
    return EquivalenceReport(worst <= atol and worst_p <= atol, worst, worst_p, phases)


@dataclass(frozen=True)
class DeterminismReport:
    """Strong determinism: every branch is ``2**(-m/2)`` times one isometry, up to phase."""

    deterministic: bool
    isometry: np.ndarray
    max_deviation: float


def is_deterministic(pattern: QubitPattern, *, atol: float = 1e-10) -> DeterminismReport:
    """Check strong determinism (equiprobable branches implementing the same isometry)."""
    ops = list(branch_operators(pattern).values())
    scale = 2 ** (len(pattern.measured) / 2)
    reference = ops[0] * scale
    gram = reference.conj().T @ reference
    worst = float(np.max(np.abs(gram - np.eye(gram.shape[0])), initial=0.0))
    for op in ops[1:]:
        worst = max(worst, _phase_deviation(op * scale, reference)[1])
    return DeterminismReport(worst <= atol, reference, worst)


# --------------------------------------------------------------------------- rewriting


def standardize(pattern: QubitPattern, *, trace: list | None = None) -> QubitPattern:
    """Rewrite to NEMC form with the DKP rules; the map is preserved branch by branch.

    Rules (left: execution order before, right: after):

    * EX: ``X_i^d, E_ij -> E_ij, X_i^d, Z_j^d``;  EZ: ``Z_i^d, E_ij -> E_ij, Z_i^d``
    * MX: ``X_i^d, M_i^{s,t} -> M_i^{s^d,t}``;    MZ: ``Z_i^d, M_i^{s,t} -> M_i^{s,t^d}``
    * signal: ``S_i^d, C -> C', S_i^d`` where every domain of C containing i is xor-ed with d
    * commutation of commands on disjoint qubits (never past a producer of a read outcome).

    EX, EZ, signal and commutation steps are exact operator identities; MX/MZ hold up to a
    phase of the branch. X and Z corrections keep their relative order; repeated
    corrections of one kind on one qubit are merged and empty ones dropped. ``trace`` (if
    given) receives ``(rule, index)`` pairs.
    """
    seq = list(pattern.validate().commands)
    log = trace if trace is not None else []
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(seq) - 1:
            step = _rewrite(seq[i], seq[i + 1])
            if step is not None:
                seq[i : i + 2] = step[1]
                log.append((step[0], i))
                changed = True
            i += 1
    return pattern.copy(_merge_corrections(seq))


def _rewrite(a: Command, b: Command) -> tuple[str, list] | None:
    """One DKP rule for the adjacent pair ``a, b`` if ``a`` must move right, else ``None``."""
    if _RANK[type(a)] <= _RANK[type(b)]:
        return None
    if isinstance(a, S):
        return "S-commute", [_shift_domains(b, a), a]
    if isinstance(a, X) and isinstance(b, E) and a.node in b.nodes:
        return "EX", [b, a, Z(b.v if a.node == b.u else b.u, a.domain)]
    if isinstance(a, X) and isinstance(b, M) and a.node == b.node:
        return "MX", [replace(b, s_domain=b.s_domain ^ a.domain)]
    if isinstance(a, Z) and isinstance(b, M) and a.node == b.node:
        return "MZ", [replace(b, t_domain=b.t_domain ^ a.domain)]
    # Disjoint qubits, or Z with E. M only overtakes N/E, which read no outcomes.
    shared = isinstance(a, Z) and isinstance(b, E) and a.node in b.nodes
    return ("EZ" if shared else "commute"), [b, a]


def rewrite_at(pattern: QubitPattern, index: int) -> tuple[str, QubitPattern]:
    """Apply the single standardization rule to commands ``index, index+1``."""
    seq = list(pattern.validate().commands)
    step = _rewrite(seq[index], seq[index + 1])
    if step is None:
        raise ValueError(f"No standardization rule applies at {index}")
    seq[index : index + 2] = step[1]
    return step[0], pattern.copy(seq)


def _shift_domains(c: Command, s: S) -> Command:
    def fix(d: frozenset) -> frozenset:
        return d ^ s.domain if s.node in d else d

    if isinstance(c, M):
        return replace(c, s_domain=fix(c.s_domain), t_domain=fix(c.t_domain))
    if isinstance(c, (X, Z)):
        return replace(c, domain=fix(c.domain))
    return c


def _merge_corrections(seq: list) -> list:
    out: list = []
    for c in seq:
        if isinstance(c, (X, Z)):
            if out and type(out[-1]) is type(c) and out[-1].node == c.node:
                c = replace(c, domain=out.pop().domain ^ c.domain)
            if not c.domain:
                continue
        out.append(c)
    return out


def signal_shift(pattern: QubitPattern) -> QubitPattern:
    """DKP signal shifting: remove the outcome-flipping part of every measurement domain.

    Per plane the Pauli that negates ``O`` is shifted into the signal: XY its t-domain,
    YZ its s-domain, XZ its s-domain (with ``t <- t xor s`` since ``X^s Z^t ~ Y^s Z^{s+t}``) --
    the same choice as Graphix 0.4. Later domains are rewritten in raw outcomes and ``S``
    commands appended (reverse measurement order, so each reads raw values) restore the
    original signals, so branches keep their keys. Input patterns must not contain ``S``.
    """
    pattern.validate()
    if any(isinstance(c, S) for c in pattern.commands):
        raise NotImplementedError("signal_shift expects a pattern without S commands")
    shift: dict = {}

    def expand(d: frozenset) -> frozenset:
        out: frozenset = frozenset()
        for j in d:
            out ^= shift.get(j, frozenset({j}))
        return out

    seq: list = []
    for c in pattern.commands:
        if isinstance(c, M):
            s_dom, t_dom = expand(c.s_domain), expand(c.t_domain)
            if c.plane is Plane.XY:
                moved, t_dom = t_dom, frozenset()
            else:
                moved = s_dom
                t_dom = t_dom ^ s_dom if c.plane is Plane.XZ else t_dom
                s_dom = frozenset()
            if moved:
                shift[c.node] = frozenset({c.node}) ^ moved
            seq.append(replace(c, s_domain=s_dom, t_domain=t_dom))
        elif isinstance(c, (X, Z)):
            seq.append(replace(c, domain=expand(c.domain)))
        else:
            seq.append(c)
    tail = [S(n, shift[n] - {n}) for n in reversed(pattern.measured) if n in shift]
    return pattern.copy(_merge_corrections(seq) + tail)


# --------------------------------------------------------------------------- flow -> pattern


def _correction_table(corrections: Any, nodes: Iterable = ()) -> dict:
    table = getattr(corrections, "correction_map", corrections)
    if set(table) == {"x", "z"} and all(isinstance(v, Mapping) for v in table.values()):
        if {"x", "z"} <= set(nodes):
            raise ValueError("Correction table is ambiguous for qubits named 'x' and 'z'")
        nodes = set(table["x"]) | set(table["z"])
        table = {n: {"x": table["x"].get(n, ()), "z": table["z"].get(n, ())} for n in nodes}
    return {n: (frozenset(v.get("x", ())), frozenset(v.get("z", ()))) for n, v in table.items()}


def pattern_from_flow(
    graph: OpenGraph, corrections: Any, order: Sequence | None = None
) -> QubitPattern:
    """Standard pattern of a qubit open graph with a (causal or generalized) flow.

    Args:
        graph: ``OpenGraph`` with ``Model.QUBIT`` and a ``QubitMeasurement`` per non-output.
        corrections: ``node -> {'x': set, 'z': set}`` (``X_{g(i)\\{i}}``/``Z_{Odd(g(i))\\{i}}``)
            or an object with such a ``correction_map``; a transposed
            ``{'x': {node: set}, 'z': {node: set}}`` table is also accepted. Entries naming the
            measured node itself are ignored (gflow's ``i in g(i)`` is the measured observable).
        order: Total measurement order; defaults to ``corrections.measurement_order`` (a
            :class:`photographiq.flows.FlowAnalysisResult`, which must be certified).

    Every correction in a supplied table must target a later-measured or output qubit;
    anything else raises ``ValueError``. This includes a raw Pauli-flow table that keeps
    the components of ``X_{p(i)}`` and ``Z_{Odd(p(i))}`` on earlier Pauli-measured qubits.
    A :class:`photographiq.flows.FlowAnalysisResult` from Pauli flow never contains such
    components: its ``correction_map`` (``photographiq.flows._common.correction_map``) has
    already dropped them, because Pauli flow (P1-P3) allows them only where they commute
    with the earlier Pauli measurement (X on X, Z on Z, and X with Z, i.e. XZ ~ Y, on Y).
    There they change only the global phase of a branch, so the pattern built from the
    result is accepted, and the tests check that it is strongly deterministic. Components
    on same-layer qubits placed later in the flattened order are kept as ordinary domains;
    they commute with those measurements in the same way. The ``pauli`` tag of
    a ``QubitMeasurement`` is ignored; plane and angle define the measurement. Determinism
    is *not* assumed: check it with :func:`is_deterministic`.
    """
    if graph.model is not Model.QUBIT:
        raise ValueError("pattern_from_flow needs a Model.QUBIT open graph")
    if getattr(corrections, "certified", True) is False:
        raise ValueError(f"Flow result is not certified: {corrections.status!r}")
    if order is None:
        order = getattr(corrections, "measurement_order", getattr(corrections, "order", ()))
    order = tuple(order)
    measured = set(graph.measured)
    if len(set(order)) != len(order) or set(order) != measured:
        raise ValueError("Order must list every measured qubit exactly once")
    table = _correction_table(corrections, graph.nodes)
    position = {n: k for k, n in enumerate(order)}
    s_dom: dict = {n: frozenset() for n in graph.nodes}
    t_dom: dict = {n: frozenset() for n in graph.nodes}
    for i in order:
        xs, zs = table.get(i, (frozenset(), frozenset()))
        for targets, dom in ((xs, s_dom), (zs, t_dom)):
            for target in targets - {i}:
                if target not in graph.graph:
                    raise ValueError(f"Correction of {i!r} targets unknown qubit {target!r}")
                if target in measured and position[target] < position[i]:
                    raise ValueError(f"Correction of {i!r} acts on earlier qubit {target!r}")
                dom[target] ^= {i}
    commands: list = [N(n) for n in graph.nodes if n not in graph.inputs]
    commands += [E(u, v) for u, v in graph.graph.edges]
    for i in order:
        m = graph.measurements[i]
        commands.append(M(i, m.plane, m.angle, s_dom[i], t_dom[i]))
    for o in graph.outputs:
        commands += [X(o, s_dom[o]), Z(o, t_dom[o])]
    return QubitPattern(graph.inputs, _merge_corrections(commands), graph.outputs)
