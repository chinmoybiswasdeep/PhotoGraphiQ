"""Single-qubit Clifford group, graph-state local complementation and LC equivalence.

* The 24 Cliffords are 2x2 unitaries modulo global phase, generated breadth-first from
  ``H = [[1,1],[1,-1]]/sqrt(2)`` and ``S = diag(1, i)``. ``Clifford.name`` is a shortest word in
  ``H``/``S`` read as a matrix product (``"HS"`` means ``H @ S``: S acts first).
* Graph states: ``|G> = prod_{(u,v) in E} CZ_uv |+>^n``, big-endian in the given node order.
* Local complementation ``tau_v`` toggles every edge inside ``N(v)``; the local Clifford
  ``U_v = exp(-i pi/4 X_v) prod_{w in N(v)} exp(+i pi/4 Z_w)`` satisfies
  ``U_v |G> = |tau_v(G)>`` up to a global phase (Van den Nest, Dehaene and De Moor,
  PRA 69, 022316 (2004)); this is checked against statevectors in the tests.
* Two labelled graph states are LC-equivalent iff the graphs are related by a sequence of
  local complementations (same reference). :func:`lc_equivalence` searches the LC orbit
  breadth-first (exponential; small graphs only) and verifies the resulting local Cliffords on
  statevectors. Vertex relabelling (graph isomorphism) is not considered.

CV graphs: unweighted qubit LC does not transfer to CV cluster states; see
:func:`cv_local_complementation`.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

import networkx as nx
import numpy as np

__all__ = [
    "Clifford",
    "LCEquivalence",
    "PAULIS",
    "apply_local_cliffords",
    "cv_local_complementation",
    "graph_state",
    "lc_cliffords",
    "lc_equivalence",
    "local_complement",
]

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0 + 0j, -1.0]),
}
_GENERATORS = {
    "H": np.array([[1, 1], [1, -1]], complex) / math.sqrt(2),
    "S": np.diag([1.0 + 0j, 1j]),
}


def _canonical(m: np.ndarray) -> np.ndarray:
    """Fix the global phase: first entry of non-negligible modulus becomes real positive."""
    flat = m.reshape(-1)
    pivot = flat[np.argmax(np.abs(flat) > 1e-9)]
    return m * (abs(pivot) / pivot)


def _key(m: np.ndarray) -> tuple:
    return tuple(np.round(_canonical(m).reshape(-1), 9).tolist())


@cache
def _group() -> tuple[tuple[np.ndarray, ...], tuple[str, ...], dict]:
    matrices, names, index = [np.eye(2, dtype=complex)], ["I"], {}
    index[_key(matrices[0])] = 0
    queue = deque([0])
    while queue:
        k = queue.popleft()
        for g, gm in _GENERATORS.items():
            m = _canonical(gm @ matrices[k])
            if _key(m) not in index:
                index[_key(m)] = len(matrices)
                matrices.append(m)
                names.append(g if names[k] == "I" else g + names[k])
                queue.append(len(matrices) - 1)
    if len(matrices) != 24:  # pragma: no cover - mathematical invariant
        raise RuntimeError("Clifford group generation failed")
    return tuple(matrices), tuple(names), index


@cache
def _table() -> np.ndarray:
    mats, _, index = _group()
    return np.array([[index[_key(a @ b)] for b in mats] for a in mats], dtype=int)


@dataclass(frozen=True)
class Clifford:
    """Element ``index`` (0..23) of the single-qubit Clifford group modulo phase."""

    index: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.index) < 24:
            raise ValueError("Clifford index must be in range(24)")

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Clifford:
        """Identify a 2x2 Clifford unitary (any global phase); raise if it is not one."""
        m = np.asarray(matrix, dtype=complex)
        if m.shape != (2, 2) or not np.allclose(m.conj().T @ m, np.eye(2), atol=1e-9):
            raise ValueError("Expected a 2x2 unitary")
        key = _key(m)
        if key not in _group()[2]:
            raise ValueError("Matrix is not a Clifford unitary")
        return cls(_group()[2][key])

    @classmethod
    def named(cls, word: str) -> Clifford:
        """Clifford of a word over ``I, X, Y, Z, H, S`` (matrix product, rightmost acts first)."""
        m = np.eye(2, dtype=complex)
        for letter in word:
            m = m @ {**PAULIS, **_GENERATORS}[letter]
        return cls.from_matrix(m)

    @classmethod
    def all(cls) -> tuple[Clifford, ...]:
        return tuple(cls(k) for k in range(24))

    @property
    def matrix(self) -> np.ndarray:
        """Canonical representative (phase fixed as in :func:`_canonical`)."""
        return _group()[0][self.index].copy()

    @property
    def name(self) -> str:
        return _group()[1][self.index]

    def __matmul__(self, other: Clifford) -> Clifford:
        return Clifford(int(_table()[self.index, other.index]))

    def inverse(self) -> Clifford:
        return Clifford(int(np.flatnonzero(_table()[self.index] == 0)[0]))

    def conjugate(self, pauli: str) -> tuple[int, str]:
        """Return ``(sign, P')`` with ``C P C^dagger = sign * P'`` for ``pauli`` in X, Y, Z."""
        c = self.matrix
        image = c @ PAULIS[pauli] @ c.conj().T
        for name in "XYZ":
            for sign in (1, -1):
                if np.allclose(image, sign * PAULIS[name], atol=1e-9):
                    return sign, name
        raise ValueError(f"Unknown Pauli {pauli!r}")

    def __repr__(self) -> str:
        return f"Clifford({self.index}, {self.name!r})"


# --------------------------------------------------------------------------- graph states


def graph_state(graph: nx.Graph, order: Sequence | None = None) -> np.ndarray:
    """Normalized ``|G>`` in ``order`` (default graph order), first node most significant."""
    order = tuple(graph.nodes if order is None else order)
    n = len(order)
    pos = {v: k for k, v in enumerate(order)}
    bits = (np.arange(2**n)[:, None] >> (n - 1 - np.arange(n))) & 1
    phase = np.zeros(2**n, dtype=int)
    for u, v in graph.edges:
        phase += bits[:, pos[u]] * bits[:, pos[v]]
    return (-1.0) ** phase / 2 ** (n / 2) + 0j


def apply_local_cliffords(state: np.ndarray, cliffords: Mapping, order: Sequence) -> np.ndarray:
    """Apply ``tensor_v cliffords[v]`` (canonical matrices) to ``state`` ordered by ``order``."""
    n = len(order)
    psi = np.asarray(state, dtype=complex).reshape((2,) * n)
    for k, v in enumerate(order):
        if v in cliffords:
            m = cliffords[v].matrix
            psi = np.moveaxis(np.tensordot(m, psi, axes=([1], [k])), 0, k)
    return psi.reshape(-1)


def local_complement(graph: nx.Graph, v) -> nx.Graph:
    """Return ``tau_v(G)``: complement the subgraph induced on the neighbourhood of ``v``."""
    result = nx.Graph(graph)
    for a, b in itertools.combinations(list(graph[v]), 2):
        if result.has_edge(a, b):
            result.remove_edge(a, b)
        else:
            result.add_edge(a, b)
    return result


_SQRT_MINUS_IX = Clifford.from_matrix(
    (np.eye(2) - 1j * PAULIS["X"]) / math.sqrt(2)
)  # exp(-i pi/4 X)
_SQRT_IZ = Clifford.from_matrix((np.eye(2) + 1j * PAULIS["Z"]) / math.sqrt(2))  # exp(+i pi/4 Z)


def lc_cliffords(graph: nx.Graph, v) -> dict:
    """Local Cliffords of ``U_v`` with ``U_v |G> = |tau_v(G)>`` up to phase."""
    return {v: _SQRT_MINUS_IX, **{w: _SQRT_IZ for w in graph[v]}}


@dataclass(frozen=True)
class LCEquivalence:
    """Witness that ``tensor_v cliffords[v] |G1> = exp(i phi) |G2>``.

    ``sequence`` lists the local complementations applied to ``G1`` (first first);
    ``residual`` is the statevector check ``1 - |<G2| U |G1>|`` (``None`` if not computed).
    """

    sequence: tuple
    cliffords: dict
    residual: float | None


def _edges(graph: nx.Graph) -> frozenset:
    return frozenset(frozenset(e) for e in graph.edges)


def lc_equivalence(
    g1: nx.Graph,
    g2: nx.Graph,
    *,
    max_nodes: int = 10,
    max_orbit: int = 1_000_000,
    verify_up_to: int = 12,
) -> LCEquivalence | None:
    """Find a local-complementation sequence mapping ``g1`` to ``g2`` (same vertex labels).

    Returns ``None`` only after exhausting the LC orbit of ``g1`` (then the states are not
    LC-equivalent). Raises ``ValueError`` for different vertex sets or too many nodes and
    ``RuntimeError`` if the orbit exceeds ``max_orbit`` (no silent negative answer).
    """
    if set(g1.nodes) != set(g2.nodes):
        raise ValueError("LC equivalence compares graphs on the same labelled vertices")
    if len(g1) > max_nodes:
        raise ValueError(f"Exhaustive orbit search limited to {max_nodes} nodes")
    nodes = tuple(g1.nodes)
    start, goal = _edges(g1), _edges(g2)
    parent: dict = {start: None}
    queue = deque([start])
    while queue and goal not in parent:
        edges = queue.popleft()
        graph = nx.Graph(tuple(e) for e in edges)
        graph.add_nodes_from(nodes)
        for v in nodes:
            nxt = _edges(local_complement(graph, v))
            if nxt not in parent:
                parent[nxt] = (edges, v)
                queue.append(nxt)
        if len(parent) > max_orbit:
            raise RuntimeError(f"LC orbit exceeds {max_orbit} graphs")
    if goal not in parent:
        return None
    sequence: list = []
    cur = goal
    while parent[cur] is not None:
        cur, v = parent[cur]
        sequence.append(v)
    sequence.reverse()
    total = {v: Clifford(0) for v in nodes}
    graph = nx.Graph(g1)
    for v in sequence:
        for w, c in lc_cliffords(graph, v).items():
            total[w] = c @ total[w]
        graph = local_complement(graph, v)
    residual = None
    if len(nodes) <= verify_up_to:
        psi = apply_local_cliffords(graph_state(g1, nodes), total, nodes)
        residual = 1.0 - abs(np.vdot(graph_state(g2, nodes), psi))
        if residual > 1e-9:  # pragma: no cover - would indicate a convention bug
            raise RuntimeError(f"LC witness failed statevector verification ({residual:.2e})")
    return LCEquivalence(tuple(sequence), total, residual)


def cv_local_complementation(*_args, **_kwargs):
    """Not implemented: qubit LC does not transfer to CV graph states.

    The qubit rule relies on ``exp(-i pi/4 X)`` and ``exp(i pi/4 Z)`` being *finite-order*
    Cliffords acting on an unweighted GF(2) adjacency matrix. CV cluster states have real
    (ideal) or complex (finite squeezing) weighted adjacency matrices on which local Gaussian
    unitaries act by symplectic Moebius-type maps (Menicucci, Flammia and van Loock,
    PRA 83, 042335 (2011)); a sound CV analogue must be derived and verified by
    covariance/nullifier comparison, which this package has not done.
    """
    raise NotImplementedError(cv_local_complementation.__doc__)
