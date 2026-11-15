"""Random valid qubit patterns and brute-force gflow for calculus tests (test helper)."""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np

from photographiq.calculus.qubit import E, M, N, QubitPattern, S, X, Z
from photographiq.ir import Plane

PLANES = (Plane.XY, Plane.YZ, Plane.XZ)


def _subset(rng: np.random.Generator, items: list) -> frozenset:
    return frozenset(x for x in items if rng.random() < 0.5)


def random_pattern(
    rng: np.random.Generator,
    *,
    n_inputs: int = 2,
    n_aux: int = 3,
    n_outputs: int = 2,
    steps: int = 18,
    signals: bool = False,
    planes: tuple = PLANES,
) -> QubitPattern:
    """Valid, generally non-standard pattern with interleaved N/E/M/X/Z (and S) commands."""
    inputs = list(range(n_inputs))
    live, measured, cmds = list(inputs), [], []
    fresh = iter(range(n_inputs, n_inputs + n_aux))
    remaining = n_aux
    for _ in range(steps):
        kind = rng.choice(["N", "E", "E", "M", "X", "Z", "S"])
        if kind == "N" and remaining:
            node = next(fresh)
            remaining -= 1
            cmds.append(N(node))
            live.append(node)
        elif kind == "E" and len(live) >= 2:
            u, v = rng.choice(len(live), 2, replace=False)
            cmds.append(E(live[u], live[v]))
        elif kind == "M" and live and len(live) + remaining > n_outputs:
            node = live.pop(rng.integers(len(live)))
            plane = planes[rng.integers(len(planes))]
            angle = float(rng.uniform(-math.pi, math.pi))
            cmds.append(M(node, plane, angle, _subset(rng, measured), _subset(rng, measured)))
            measured.append(node)
        elif kind in "XZ" and measured and live:
            cls = X if kind == "X" else Z
            cmds.append(
                cls(live[rng.integers(len(live))], _subset(rng, measured) or {measured[-1]})
            )
        elif kind == "S" and signals and len(measured) >= 2:
            node = measured[rng.integers(len(measured))]
            others = [m for m in measured if m != node]
            cmds.append(S(node, _subset(rng, others) or {others[0]}))
    for _ in range(remaining):
        node = next(fresh)
        cmds.append(N(node))
        live.append(node)
    while len(live) > n_outputs:
        node = live.pop(0)
        plane = planes[rng.integers(len(planes))]
        cmds.append(M(node, plane, float(rng.uniform(-3, 3)), _subset(rng, measured)))
        measured.append(node)
    return QubitPattern(inputs, cmds, tuple(live))


def odd(graph: nx.Graph, subset) -> frozenset:
    """Odd neighbourhood of ``subset``."""
    count: dict = {}
    for v in subset:
        for w in graph[v]:
            count[w] = count.get(w, 0) + 1
    return frozenset(w for w, c in count.items() if c % 2)


def brute_force_gflow(graph: nx.Graph, inputs, outputs, planes: dict):
    """Exhaustive gflow search (Browne et al. 2007 Def. 3): returns (order, corrections)."""
    measured = [v for v in graph.nodes if v not in outputs]
    for order in itertools.permutations(measured):
        corrections = {}
        for k, i in enumerate(order):
            later = [v for v in order[k + 1 :] if v not in inputs]
            allowed = later + [o for o in outputs if o not in inputs]
            if i not in inputs:
                allowed.append(i)
            found = None
            for r in range(len(allowed) + 1):
                for g in itertools.combinations(allowed, r):
                    g = frozenset(g)
                    od = odd(graph, g)
                    if od & set(order[:k]):
                        continue
                    ok = {
                        Plane.XY: i not in g and i in od,
                        Plane.XZ: i in g and i in od,
                        Plane.YZ: i in g and i not in od,
                    }[planes[i]]
                    if ok:
                        found = g
                        break
                if found is not None:
                    break
            if found is None:
                break
            corrections[i] = {"x": found - {i}, "z": odd(graph, found) - {i}}
        else:
            return order, corrections
    return None
