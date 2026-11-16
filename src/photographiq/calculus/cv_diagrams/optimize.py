"""Equivalence reports, resource costs, flow checks and bounded rewrite search for CV-MBQC.

Equivalence is reported in three separate layers (never conflated):

* ``logical``: the affine part (X, d) of the averaged channel (the ideal map of a corrected
  pattern; for flow-corrected patterns X and d do not depend on the resource squeezing);
* ``physical``: (X, Y, d) of the averaged channel, i.e. including finite-squeezing noise;
* ``conditional``: the full instrument -- the joint affine law of outputs and records,
  compared record by record (only when both diagrams have the same record keys).

All comparisons are exact linear algebra on the semantic matrices with a stated absolute and
relative tolerance; nothing is inferred from sampled inputs. ``approximate`` optimisation
accepts logical rewrites whose added-noise change max|Y_A - Y_B| (vacuum units, entrywise) is
at most a declared epsilon: an entrywise parameter residual, not a diamond-norm bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .ir import (
    CZ,
    BeamSplitter,
    ClassicalSource,
    CVDiagram,
    Displacement,
    Feedforward,
    Homodyne,
    PhotonCount,
    Prepare,
    PrepareState,
)
from .rules import GAUSSIAN_RULES, MBQC_LOGICAL_RULES, Commute
from .semantics import evaluate

MODES = ("physical", "logical", "approximate")


# ----------------------------------------------------------------------------- equivalence
@dataclass(frozen=True)
class EquivalenceReport:
    logical_equivalent: bool
    physical_equivalent: bool
    conditional_equivalent: bool | None
    symplectic_residual: float
    channel_X_residual: float
    channel_Y_residual: float
    displacement_residual: float
    instrument_residual: float | None
    measurement_record_mapping: str
    tolerance: float
    assumptions: str = (
        "Exact semantic matrices compared entrywise with tolerance "
        "atol + rtol * scale; Gaussian diagrams only."
    )

    def to_dict(self):
        return dict(self.__dict__)


def _res(a, b):
    return float(np.max(np.abs(a - b), initial=0.0))


def compare(a: CVDiagram, b: CVDiagram, *, atol=1e-9, rtol=1e-9) -> EquivalenceReport:
    """Compare two Gaussian diagrams with the same input/output wires (by label order)."""
    from ... import gaussian as gm

    if tuple(a.inputs) != tuple(b.inputs) or len(a.outputs) != len(b.outputs):
        raise ValueError("Diagrams must have the same input wires and the same number of outputs")
    positional = sorted(map(repr, a.outputs)) != sorted(map(repr, b.outputs))
    if not positional and tuple(b.outputs) != tuple(a.outputs):
        b = b.permute_outputs(a.outputs)
    sa, sb = evaluate(a), evaluate(b)
    Xa, Ya, da = sa.channel()
    Xb, Yb, db = sb.channel()
    scale = max(1.0, np.abs(Xa).max(initial=0), np.abs(Ya).max(initial=0))
    tol = atol + rtol * scale
    rx, ry, rd = _res(Xa, Xb), _res(Ya, Yb), _res(da, db)
    n = len(a.outputs)
    symplectic = _res(Xa @ gm.omega(n) @ Xa.T, gm.omega(n)) if len(a.inputs) == n else float("nan")
    conditional, inst = None, None
    if sorted(map(repr, sa.records)) == sorted(map(repr, sb.records)):
        order = list(sa.records)
        rows = lambda s: s._rows(s.outputs) + [s._index("r", k) for k in order]  # noqa: E731
        ia, ib = rows(sa), rows(sb)
        inst = max(
            _res(sa.A[ia], sb.A[ib]),
            _res(sa.b[ia], sb.b[ib]),
            _res(sa.Y[np.ix_(ia, ia)], sb.Y[np.ix_(ib, ib)]),
        )
        conditional = inst <= tol
        mapping = "identity (same record keys)"
    else:
        mapping = (
            f"records differ: {len(sa.records)} -> {len(sb.records)}; "
            "conditional equivalence not defined"
        )
    if positional:
        mapping += f"; outputs matched by position {tuple(a.outputs)} -> {tuple(b.outputs)}"
    logical = rx <= tol and rd <= tol
    return EquivalenceReport(
        logical, logical and ry <= tol, conditional, symplectic, rx, ry, rd, inst, mapping, tol
    )


# ----------------------------------------------------------------------------- costs
COST_KEYS = (
    "resource_modes",
    "ancilla_modes",
    "classical_sources",
    "entangling_edges",
    "measurement_count",
    "measurement_depth",
    "feedforward_commands",
    "feedforward_dependencies",
    "maximum_feedforward_gain",
    "squeezing_energy",
    "estimated_optical_energy",
    "added_noise",
    "boxes",
)


def resource_cost(d: CVDiagram) -> dict:
    """Structural and optical resource costs of a diagram (see docs for units).

    squeezing_energy: sum of sinh^2 r over squeezed resource preparations (mean photons of a
    squeezed vacuum); estimated_optical_energy: total mean photon number of all prepared
    Gaussian states, (tr V + |mean|^2)/4 - 1/2 per mode (hbar = 2); added_noise: mean added
    noise per output quadrature of the averaged channel (vacuum units; Gaussian diagrams).
    """
    preps = [b for b in d.boxes if isinstance(b, (Prepare, PrepareState))]
    created = [m for b in preps for m in b.creates]
    graph = nx.DiGraph()
    producer: dict = {}
    last: dict = {}
    for i, b in enumerate(d.boxes):
        graph.add_node(i)
        for k in b.reads:
            graph.add_edge(producer[k], i)
        for k in b.writes:
            producer[k] = i
        for m in set(b.modes) | set(b.creates) | set(b.consumes):
            if m in last:
                graph.add_edge(last[m], i)
            last[m] = i
    depth: dict = {}
    measure = {i for i, b in enumerate(d.boxes) if isinstance(b, (Homodyne, PhotonCount))}
    for i in nx.topological_sort(graph):
        depth[i] = (i in measure) + max((depth[p] for p in graph.predecessors(i)), default=0)
    gains = [
        abs(g) for b in d.boxes if isinstance(b, Feedforward) for _, pair in b.gains for g in pair
    ]
    energy = 0.0
    for b in preps:
        cov, mean = np.asarray(b.covariance, float), np.asarray(b.mean, float)
        energy += (np.trace(cov) + mean @ mean) / 4 - len(b.creates) / 2
    cost = {
        "resource_modes": len(created),
        "ancilla_modes": len([m for m in created if m not in d.outputs]),
        "classical_sources": sum(isinstance(b, ClassicalSource) for b in d.boxes),
        "entangling_edges": sum(isinstance(b, (CZ, BeamSplitter)) for b in d.boxes),
        "measurement_count": len(measure),
        "measurement_depth": max(depth.values(), default=0),
        "feedforward_commands": sum(isinstance(b, Feedforward) for b in d.boxes),
        "feedforward_dependencies": sum(
            len(b.gains) for b in d.boxes if isinstance(b, Feedforward)
        ),
        "maximum_feedforward_gain": max(gains, default=0.0),
        "squeezing_energy": float(
            sum(
                np.sinh(b.squeezing) ** 2
                for b in preps
                if isinstance(b, Prepare) and b.squeezing is not None
            )
        ),
        "estimated_optical_energy": float(energy),
        "boxes": len(d.boxes),
    }
    if d.is_gaussian and d.outputs:
        _, Y, _ = evaluate(d).channel()
        cost["added_noise"] = float(np.trace(Y) / len(Y))
    else:
        cost["added_noise"] = float("nan")
    return cost


DEFAULT_WEIGHTS = {
    "resource_modes": 1.0,
    "entangling_edges": 0.5,
    "measurement_depth": 0.5,
    "feedforward_dependencies": 0.1,
    "squeezing_energy": 0.1,
}


def weighted_cost(cost: dict, weights=None) -> float:
    """sum_k w_k cost_k. Units differ (counts vs photons): weights are explicit user choices,
    not a universal experimental cost; use ``pareto_front`` to avoid fixing them."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    return float(sum(w * cost[k] for k, w in weights.items()))


def pareto_front(items, keys):
    """Indices of items (dicts) not dominated on ``keys`` (all minimised)."""
    front = []
    for i, a in enumerate(items):
        dominated = any(
            all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys)
            for j, b in enumerate(items)
            if j != i
        )
        if not dominated:
            front.append(i)
    return front


# ----------------------------------------------------------------------------- flow
def pattern_flow_report(diagram: CVDiagram, *, exact=None) -> dict:
    """CV-flow analysis of the cluster underlying an executable diagram.

    The open graph is read from the diagram: CZ edges (weights), squeezed preparations,
    homodyne angles (shear cot theta) and input/output wires. ``find_cv_flow`` certifies an
    admissible order (Booth-Markham real-linear flow) and checks determinism in the ideal limit.
    Diagrams with other entangling gates or non-homodyne measurements are reported unsupported.
    """
    from ...flows import find_cv_flow
    from ...ir import HomodyneSpec, Model, OpenGraph

    graph = nx.Graph()
    measurements, squeezing = {}, {}
    for w in diagram.inputs:
        graph.add_node(w)
    for b in diagram.boxes:
        if isinstance(b, Prepare):
            graph.add_node(b.mode)
            squeezing[b.mode] = b.squeezing if b.squeezing is not None else 0.0
        elif isinstance(b, CZ):
            weight = graph.edges[b.u, b.v]["weight"] + b.g if graph.has_edge(b.u, b.v) else b.g
            graph.add_edge(b.u, b.v, weight=weight)
        elif isinstance(b, Homodyne):
            if abs(np.sin(b.theta)) < 1e-12:
                return {"status": "unsupported", "reason": "q measurement (theta = 0) in the graph"}
            measurements[b.mode] = HomodyneSpec(float(np.cos(b.theta) / np.sin(b.theta)))
        elif isinstance(b, (Feedforward, Displacement, ClassicalSource)):
            continue  # affine frame shifts: they do not change the graph or its flow
        else:
            return {
                "status": "unsupported",
                "reason": f"{type(b).__name__} is not a graph-state element",
            }
    try:
        og = OpenGraph(
            Model.CV_GAUSSIAN,
            graph,
            diagram.inputs,
            diagram.outputs,
            measurements,
            {n: squeezing.get(n, 1.0) for n in graph.nodes},
        )
    except ValueError as exc:
        return {"status": "unsupported", "reason": str(exc)}
    start = time.perf_counter()
    result = find_cv_flow(og, exact=exact)
    return {
        "status": result.status,
        "decision": result.diagnostics.get("decision"),
        "layers": len(result.layers),
        "nodes": len(graph.nodes),
        "edges": graph.number_of_edges(),
        "seconds": time.perf_counter() - start,
        "regimes": result.certificate.get("regimes") if result.certificate else None,
    }


# ----------------------------------------------------------------------------- search
@dataclass
class OptimizationResult:
    original: CVDiagram
    best: CVDiagram
    mode: str
    objective: str
    original_cost: dict
    best_cost: dict
    report: EquivalenceReport
    history: list = field(default_factory=list)
    explored: int = 0
    rejected: int = 0
    seconds: float = 0.0

    @property
    def improved(self):
        return self._score(self.best_cost) < self._score(self.original_cost)

    def _score(self, cost):
        return weighted_cost(cost) if self.objective == "weighted" else cost[self.objective]


def _rules_for(mode):
    exact = [r for r in GAUSSIAN_RULES if not isinstance(r, Commute)]
    if mode == "physical":
        return exact
    return exact + list(MBQC_LOGICAL_RULES)


def _accept(report, mode, epsilon):
    if mode == "physical":
        return report.physical_equivalent
    if mode == "logical":
        return report.logical_equivalent
    return report.logical_equivalent and report.channel_Y_residual <= epsilon


def optimize_cv_pattern(
    source,
    *,
    equivalence="physical",
    objective="resource_modes",
    strategy="greedy",
    beam_width=4,
    max_steps=50,
    epsilon=0.0,
    executable=True,
    weights=None,
):
    """Bounded deterministic rewrite search with semantic verification of every candidate.

    ``source`` is a ``Pattern`` or a ``CVDiagram``. Every candidate is compared with the
    *original* (no drift): physical (X, Y, d), logical (X, d) or approximate (logical with
    max|dY| <= epsilon). With ``executable`` each accepted candidate must recompile to an
    executable ``Pattern``. Deterministic: rules and matches are enumerated in a fixed order;
    ties broken by the diagram fingerprint; visited fingerprints are never revisited.
    """
    from .translate import from_pattern, to_pattern

    if equivalence not in MODES:
        raise ValueError(f"equivalence must be one of {MODES}")
    if strategy not in ("greedy", "beam"):
        raise ValueError("strategy must be 'greedy' or 'beam'")
    original = source if isinstance(source, CVDiagram) else from_pattern(source)
    score = (
        (lambda c: weighted_cost(c, weights))
        if objective == "weighted"
        else (lambda c: c[objective])
    )
    start = time.perf_counter()
    rules = _rules_for(equivalence)
    visited = {original.fingerprint()}
    frontier = [original]
    best, best_cost = original, resource_cost(original)
    explored = rejected = 0
    history = []
    for depth in range(max_steps):
        candidates = []
        for diagram in frontier:
            for rule in rules:
                for match in rule.find(diagram):
                    candidate = rule.apply(diagram, match)
                    fp = candidate.fingerprint()
                    explored += 1
                    if fp in visited:
                        continue
                    visited.add(fp)
                    report = compare(original, candidate)
                    ok = _accept(report, equivalence, epsilon)
                    if ok and executable:
                        try:
                            to_pattern(candidate)
                        except NotImplementedError:
                            ok = False
                    if not ok:
                        rejected += 1
                        continue
                    cost = resource_cost(candidate)
                    candidates.append((score(cost), fp, candidate, cost))
        if not candidates:
            break
        candidates.sort(key=lambda t: (t[0], t[1]))
        frontier = [c[2] for c in candidates[: (1 if strategy == "greedy" else beam_width)]]
        top = candidates[0]
        history.append(
            {
                "depth": depth,
                "candidates": len(candidates),
                "best_score": top[0],
                "rule": top[2].trace[-1].rule if top[2].trace else None,
            }
        )
        if top[0] < score(best_cost) or (
            top[0] == score(best_cost) and weighted_cost(top[3]) < weighted_cost(best_cost)
        ):
            best, best_cost = top[2], top[3]
    return OptimizationResult(
        original,
        best,
        equivalence,
        objective,
        resource_cost(original),
        best_cost,
        compare(original, best),
        history,
        explored,
        rejected,
        time.perf_counter() - start,
    )
