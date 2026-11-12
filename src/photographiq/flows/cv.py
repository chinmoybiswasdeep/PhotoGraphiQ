"""Real-linear CV flow search and numerical determinism verification (Model.CV_GAUSSIAN).

Flow notion: the correction-matrix condition of Booth and Markham, Quantum 7, 1146
(2023), Defs. 5-6 and Lemma 7, as certified for a *given* total order by
:func:`photographiq.cvflow.certify_cv_flow`: for node u, a real vector c supported
on later non-input nodes with (A c)_v = delta_{uv} for every node v measured up to u,
A the *weighted* adjacency. Qubit (GF(2), unweighted) rules are never used here.

Order search: backward layering over the reals. A node u joins the next-earlier layer
when A[W, C] c = e_u is solvable (least squares, residual <= tolerance), W the
unsolved nodes and C the solved non-input nodes. Solvability is monotone in the
solved set, so in exact arithmetic the search finds an admissible total order iff
one exists; numerically it is exact up to ``tolerance``. Cost: at most n rounds,
each one least-squares solve with n right-hand sides, O(n^4) overall. Nodes of a
layer need no corrections from one another under the layered certificate, which is
also a valid certificate for the flattened total order; the reported correction map
and the verified pattern both use it, so layers are genuinely parallel.

Determinism is then checked numerically on the pattern built from exactly the
reported correction map (see :func:`corrected_pattern`, :func:`verify_cv_determinism`).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import expm

from ..analysis import gaussian_channel
from ..backends import GaussianBackend
from ..commands import Measure, Prepare
from ..cvflow import certify_cv_flow
from ..expressions import Outcome, atan2, sin
from ..gaussian import omega
from ..graph import CVGraph
from ..ir import HomodyneSpec, Model, OpenGraph
from ..measurements import Homodyne
from ..pattern import Pattern
from ..simulator import simulate
from ..states import GaussianState
from .result import FlowAnalysisResult, unsupported

ALGORITHM = "cv-flow (Booth-Markham 2023 real-linear condition; layered order search)"


def resource_graph(og: OpenGraph) -> CVGraph:
    """CVGraph whose node order makes flow_pattern's surviving modes the outputs.

    Pattern output order is (inputs that are outputs, in input order) followed by
    the remaining outputs in node-insertion order; outputs are declared in that order.
    """
    outputs = tuple(n for n in og.inputs if n in og.outputs) + tuple(
        n for n in og.outputs if n not in og.inputs
    )
    network = nx.Graph()
    for n in [n for n in og.nodes if n not in og.outputs] + list(outputs):
        network.add_node(n, squeezing=float(og.squeezing.get(n, 1.0)))
    network.add_weighted_edges_from(
        (u, v, float(d["weight"])) for u, v, d in og.graph.edges(data=True)
    )
    return CVGraph(network, inputs=og.inputs, outputs=outputs)


def search_cv_order(og: OpenGraph, *, tolerance: float = 1e-10) -> tuple[tuple, dict, tuple, float]:
    """Layered search; return (layers earliest first, layered vectors, unsolved, max residual)."""
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    adj = og.adjacency(nodes)
    solved = set(og.outputs)
    vectors: dict = {}
    layers: list[tuple] = []
    worst = 0.0
    while unsolved := [n for n in nodes if n not in solved]:
        columns = [n for n in nodes if n in solved and n not in og.inputs]
        if not columns:
            return tuple(layers), vectors, tuple(unsolved), worst
        matrix = adj[np.ix_([index[n] for n in unsolved], [index[n] for n in columns])]
        targets = np.eye(len(unsolved))
        solution = np.linalg.lstsq(matrix, targets, rcond=None)[0]
        residuals = np.abs(matrix @ solution - targets).max(axis=0)
        layer = tuple(u for i, u in enumerate(unsolved) if residuals[i] <= tolerance)
        if not layer:
            return tuple(layers), vectors, tuple(unsolved), worst
        for i, u in enumerate(unsolved):
            if u in layer:
                vectors[u] = dict(zip(columns, solution[:, i].tolist()))
                worst = max(worst, float(residuals[i]))
        solved.update(layer)
        layers.insert(0, layer)
    return tuple(layers), vectors, (), worst


def corrected_pattern(graph: CVGraph, order, corrections: dict, shears: dict) -> Pattern:
    """Finite-squeezing pattern executing exactly ``corrections`` (a CV correction map).

    Measures each node u of ``order`` with Homodyne(atan2(1, shears[u])) and displaces
    every target t by ``q += cq * m``, ``p += cp * m`` with ``m = record / sin(theta)``.
    """
    pattern = Pattern(graph)
    for u in order:
        angle = atan2(1, shears.get(u, 0.0))
        pattern.measure(u, Homodyne(angle))
        m = Outcome(u) / sin(angle)
        for target, c in corrections[u].items():
            pattern.displace(target, q=c["q"] * m, p=c["p"] * m)
    return pattern.validate()


def _with_extra_squeezing(pattern, extra: float):
    """Copy of ``pattern`` with every resource squeezing r -> r + extra."""
    result = pattern.copy()
    result.commands = [
        replace(c, squeezing=c.squeezing + extra)
        if isinstance(c, Prepare) and c.state is None
        else c
        for c in result.commands
    ]
    return result


class _PeakGaussianBackend(GaussianBackend):
    """The NumPy Gaussian backend, also recording the largest |covariance entry| it held."""

    peak = 0.0

    @property  # type: ignore[override]
    def state(self) -> GaussianState:
        return self._state

    @state.setter
    def state(self, value: GaussianState) -> None:
        self._state = value
        self.peak = max(self.peak, float(np.abs(value.covariance).max(initial=0.0)))


def _conditional_response(pattern, keys, cov, record_probes, input_probes):
    """Exact conditional output moments given postselected records (Gaussian backend).

    Conditioning is affine: mean = A mu + G m + const with covariance independent of
    m. Returns (G @ record_probes, A @ input_probes, state at zero records and zero
    input mean, states at each record probe, largest |covariance entry| of any
    intermediate state); probe columns are random directions, which detect any
    nonzero G or A - S with probability one at a cost independent of the number of
    records.
    """
    inputs, dim = pattern.inputs, 2 * len(pattern.inputs)
    engine = _PeakGaussianBackend()

    def run(mean, records):
        state = GaussianState(mean, cov, inputs) if inputs else None
        fixed = dict(zip(keys, records.tolist()))
        result = simulate(pattern, backend=engine, initial_state=state, measurement_outcomes=fixed)
        return result.state

    zero = np.zeros(len(keys))
    base = run(np.zeros(dim), zero)
    others = [run(np.zeros(dim), r) for r in record_probes.T]
    gain = np.array([o.mean - base.mean for o in others]).T
    amap = np.array([run(u, zero).mean - base.mean for u in input_probes.T]).T
    rows = len(base.mean)
    gain = gain.reshape(rows, record_probes.shape[1])
    return gain, amap.reshape(rows, input_probes.shape[1]), base, others, engine.peak


def verify_cv_determinism(
    pattern,
    *,
    shift: float = 3.0,
    seed: int = 0,
    tolerance: float = 1e-9,
    min_exponent: float = 0.9,
    probes: int = 2,
) -> dict:
    """Numerically verify that a corrected Gaussian pattern is deterministic in the ideal limit.

    Physics: with finite resource squeezing r the records carry partial information
    about the input, so the *conditional* output is NOT exactly outcome independent:
    its mean is A mu + G m with gain G and A != S (both probed along ``probes``
    random directions, i.e. G R and (A - S) U for Gaussian random R, U). CV-flow determinism is an
    infinite-squeezing statement. Finite squeezing appears as (i) the added noise of
    the outcome-averaged channel (S, N) and (ii) the residual outcome dependence G;
    both are reported, separately from the ideal input map S.

    Ideal outputs may contain infinitely squeezed modes (e.g. graphs with more outputs
    than inputs): writing N(r + x) = e^{2x} N_plus + e^{-2x} N_minus, range(N_plus)
    holds output directions whose variance diverges as r -> oo. Displacements along
    them are invisible in the ideal limit, so outcome dependence is assessed after
    projecting onto F = null(N_plus) (all of phase space when N_plus = 0).

    Procedure: ``gaussian_channel`` gives the outcome-averaged channel of the pattern
    rebuilt with every resource squeezing raised by x in (0, ``shift``, 2 ``shift``);
    conditional moments come from ``simulate(backend='gaussian', measurement_outcomes=
    ...)`` (exact Gaussian postselection) for a vacuum and a random correlated
    squeezed input covariance. Checks, at absolute tolerance ``tolerance`` times the
    largest entry involved:

    * S is identical at every level, and N follows the two-term model above at the
      third level (fails if noise has other sources, e.g. loss or inefficiency);
    * conditional covariances do not depend on the record values;
    * law of total covariance: S V S^T + N - V_cond = Cov(E[out | m]) >= 0, up to the
      first-order float64 rounding allowance ``levels[k]['spread_rounding']`` =
      (commands) x (output dimension) x eps x (largest covariance entry of any
      intermediate state), an estimate rather than a rigorous bound (see the code); this
      identity holds for every Gaussian pattern, so it checks the numerics and the
      noise model, while nondeterminism is detected by the decay and rank checks;
    * P_F G and P_F (A - S) decay like exp(-2 kappa x) with kappa >= ``min_exponent``
      between the last two levels (kappa = 1 expected), or vanish identically;
    * S Omega S^T = Omega when S is square (the ideal map is then unitary);
    * rank(N_plus) = |outputs| - |inputs|: an ideal Gaussian isometry appends exactly
      one infinitely squeezed quadrature per extra output mode.

    ``ideal_limit_deterministic`` is the conjunction of these checks.
    """
    keys = [c.result_key for c in pattern.commands if isinstance(c, Measure)]
    rng = np.random.default_rng(seed)
    covs = [np.eye(2 * len(pattern.inputs)), _random_covariance(rng, len(pattern.inputs))]
    record_probes = rng.normal(0, 1, (len(keys), probes))
    input_probes = rng.normal(0, 1, (2 * len(pattern.inputs), probes))
    extras = (0.0, shift, 2 * shift)
    patterns = [_with_extra_squeezing(pattern, x) for x in extras]
    channels = [gaussian_channel(p) for p in patterns]
    s_ideal, noise = channels[0].matrix, channels[0].noise
    up, down = math.exp(2 * shift), math.exp(-2 * shift)
    n_plus = (channels[1].noise - down * noise) / (up - down)
    n_minus = noise - n_plus
    magnitude = max(1.0, _dist(s_ideal, 0), _dist(noise, 0), _dist(n_plus, 0))
    bound = tolerance * magnitude
    eigenvalues, vectors = np.linalg.eigh(n_plus)
    finite = vectors[:, np.abs(eigenvalues) <= bound]
    project = finite @ finite.T
    model_residual = _dist(channels[2].noise, up * up * n_plus + down * down * n_minus)
    exact = model_residual <= tolerance * max(magnitude, _dist(channels[2].noise, 0))
    levels = []
    for x, p, ch in zip(extras, patterns, channels):
        scale = tolerance * max(magnitude, _dist(ch.noise, 0))
        level = {
            "extra_squeezing": x,
            "matrix_change": _dist(ch.matrix, s_ideal),
            "gain": 0.0,
            "conditional_map_error": 0.0,
            "covariance_outcome_dependence": 0.0,
            "spread_min": math.inf,
            "finite_spread_max": 0.0,
            "spread_rounding": 0.0,
        }
        for cov in covs:
            gain, amap, base, others, peak = _conditional_response(
                p, keys, cov, record_probes, input_probes
            )
            total = ch.matrix @ cov @ ch.matrix.T + ch.noise - base.covariance
            # float64 rounding of `total`: ch.noise and base.covariance are small
            # differences (sums of products, Schur complements) of covariance entries as
            # large as `peak` (anti-squeezed variances ~e^{2r} times gate gains). Heuristic
            # first-order estimate, not a rigorous bound: each of the len(p.commands)
            # operations adds O(eps * peak) per entry, and Weyl's inequality moves the
            # eigenvalues of the d x d matrix by at most d times the largest entry error.
            # Errors along anti-squeezed output directions (|N| >> peak) fall under
            # `scale`. On 300 random weighted graphs (|w| in 0.02..6, shears up to 5,
            # r <= 3.6) spread_min never fell below -0.03 (scale + this allowance), though
            # an mpmath reference showed entry errors of ill-conditioned Schur complements
            # up to ~30 eps * peak. A fixed absolute bound already fails at r ~ 3.
            rounding = len(p.commands) * len(total) * np.finfo(float).eps * peak
            level["spread_rounding"] = max(level["spread_rounding"], rounding)
            level["gain"] = max(level["gain"], _dist(project @ gain, 0))
            level["conditional_map_error"] = max(
                level["conditional_map_error"],
                _dist(project @ (amap - ch.matrix @ input_probes), 0),
            )
            level["covariance_outcome_dependence"] = max(
                level["covariance_outcome_dependence"],
                max((_dist(o.covariance, base.covariance) for o in others), default=0.0),
            )
            spread = np.linalg.eigvalsh(total)
            level["spread_min"] = min(level["spread_min"], float(spread.min(initial=math.inf)))
            finite_spread = np.linalg.eigvalsh(project @ total @ project)
            level["finite_spread_max"] = max(
                level["finite_spread_max"], float(finite_spread.max(initial=0.0))
            )
        exact = exact and (
            level["matrix_change"] <= bound
            and level["covariance_outcome_dependence"] <= scale
            and level["spread_min"] >= -(scale + level["spread_rounding"])
        )
        levels.append(level)

    def exponent(key: str) -> float:
        a, b = levels[1][key], levels[2][key]
        return math.inf if b <= bound else math.log(max(a, bound) / b) / (2 * shift)

    exponents = {k: exponent(k) for k in ("gain", "conditional_map_error")}
    symplectic = _symplectic_residual(s_ideal)
    singular = int(len(eigenvalues) - finite.shape[1])
    expected_singular = len(channels[0].outputs) - len(pattern.inputs)
    return {
        "matrix": s_ideal,
        "noise": noise,
        "displacement": channels[0].displacement,
        "outputs": channels[0].outputs,
        "symplectic_residual": symplectic,
        "finite_squeezing_noise": n_minus,
        "ideal_singular_noise": n_plus,
        "ideal_singular_directions": singular,
        "expected_singular_directions": expected_singular,
        "noise_model_residual": model_residual,
        "finite_squeezing_gain": levels[0]["gain"],
        "finite_squeezing_conditional_map_error": levels[0]["conditional_map_error"],
        "finite_squeezing_outcome_spread": levels[0]["finite_spread_max"],
        "levels": tuple(levels),
        "decay_exponents": exponents,
        "tolerance": bound,
        "ideal_limit_deterministic": bool(
            exact
            and (symplectic is None or symplectic <= bound)
            and singular == expected_singular
            and min(exponents.values()) >= min_exponent
        ),
    }


def _dist(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max(initial=0.0))


def _symplectic_residual(s: np.ndarray) -> float | None:
    if s.shape[0] != s.shape[1] or not s.size:
        return None
    w = omega(s.shape[0] // 2)
    return _dist(s @ w @ s.T, w)


def _random_covariance(rng, modes: int) -> np.ndarray:
    """Physical thermal-squeezed covariance S diag(nu) S^T with a random symplectic S."""
    if not modes:
        return np.zeros((0, 0))
    h = rng.normal(size=(2 * modes, 2 * modes))
    s = expm(omega(modes) @ (0.2 * (h + h.T)))  # exp(Omega H) is symplectic for symmetric H
    return s @ np.diag(np.repeat(1 + rng.uniform(0, 0.5, modes), 2)) @ s.T


def find_cv_flow(
    og: OpenGraph,
    *,
    tolerance: float = 1e-10,
    verify: bool = True,
    seed: int = 0,
    verify_tolerance: float = 1e-9,
) -> FlowAnalysisResult:
    """Search an admissible order, certify it and verify determinism numerically.

    Status ``certified`` requires the Booth-Markham certificate for the found order
    (re-derived independently by ``certify_cv_flow``) and, when ``verify`` is True,
    the numerical determinism check on the corrected finite-squeezing pattern.
    """
    if og.model is not Model.CV_GAUSSIAN:
        return unsupported(
            og.model,
            ALGORITHM,
            "linear displacement feed-forward is only analysed for Model.CV_GAUSSIAN; "
            "non-Gaussian elements do not commute with displacement corrections",
        )
    missing = [n for n in og.measured if not isinstance(og.measurements.get(n), HomodyneSpec)]
    if missing:
        raise ValueError(f"Missing HomodyneSpec for measured nodes {missing}")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    layers, vectors, unsolved, worst = search_cv_order(og, tolerance=tolerance)
    assumptions: tuple[str, ...] = (
        "Booth-Markham real-linear CV flow for homodyne measurements on a weighted graph",
        "determinism is an infinite-squeezing statement; finite squeezing adds noise N and "
        "an outcome-dependent conditional shift G m, both in certificate['determinism']",
        "unit-efficiency homodyne detection",
        "missing resource squeezing defaults to r=1.0 (OpenGraph.to_cvgraph convention)",
    )
    if unsolved:
        return FlowAnalysisResult(
            "not_certified",
            og.model,
            ALGORITHM,
            layers=layers,
            assumptions=assumptions,
            diagnostics={
                "reason": "no admissible total order satisfies the real-linear CV-flow "
                f"condition (layered search complete in exact arithmetic; tolerance "
                f"{tolerance:g}); this is not a proof of nondeterminism by other means",
                "complete": True,
                "uncorrectable": unsolved,
            },
        )
    order = tuple(n for layer in layers for n in layer)
    graph = resource_graph(og)
    flow = certify_cv_flow(graph, order, tolerance=tolerance)
    diagnostics: dict[str, Any] = {"complete": True, "layer_residual_max": worst}
    if flow is None:
        diagnostics["reason"] = "certify_cv_flow rejected the searched order"
        return FlowAnalysisResult(
            "unknown",
            og.model,
            ALGORITHM,
            order,
            layers,
            assumptions=assumptions,
            diagnostics=diagnostics,
        )
    # Corrections come from the layered vectors: c is supported on later layers and
    # outputs, and (A c)_v = 0 on u's own and earlier layers, so no node is corrected
    # by a node of its own layer. Re-check the layered condition explicitly: on the
    # rows of u's own and earlier layers (a superset of u's past in the flattened
    # order) A c = e_u up to the residual, so those entries are dropped as exact zeros.
    adj = graph.adjacency()
    pos = {n: i for i, n in enumerate(graph.nodes)}
    rank = {n: i for i, layer in enumerate(layers) for n in layer}
    corrections, layered_residual = {}, 0.0
    for u in order:
        here = [n for n in graph.nodes if rank.get(n, len(layers)) <= rank[u]]
        if set(vectors[u]) & (set(here) | set(og.inputs)):
            raise AssertionError(f"layered vector of {u!r} has invalid support")  # pragma: no cover
        c = np.zeros(len(pos))
        for n, value in vectors[u].items():
            c[pos[n]] = value
        image = adj @ c
        target = np.array([float(n == u) for n in here])
        layered_residual = max(layered_residual, _dist(image[[pos[n] for n in here]], target))
        corrections[u] = {
            t: {"q": -float(c[pos[t]]), "p": -float(image[pos[t]])}
            for t in graph.nodes
            if t not in here and (c[pos[t]] or image[pos[t]])
        }
    shears = {n: float(og.measurements[n].shear) for n in order}
    certificate = {
        "total_order": flow,
        "layer_corrections": vectors,
        "shears": shears,
        "checks": (
            "layered real-linear condition",
            "layered vectors re-checked on the flattened total order",
            "certify_cv_flow on the total order",
        ),
    }
    diagnostics["certify_residual_max"] = max(flow.residuals, default=0.0)
    diagnostics["total_order_residual_max"] = layered_residual
    if layered_residual > tolerance:  # pragma: no cover - would be an implementation bug
        diagnostics["reason"] = "layered vectors fail the total-order condition"
        return FlowAnalysisResult(
            "unknown",
            og.model,
            ALGORITHM,
            order,
            layers,
            assumptions=assumptions,
            diagnostics=diagnostics,
        )
    status = "certified"
    if verify:
        try:
            report = verify_cv_determinism(
                corrected_pattern(graph, order, corrections, shears),
                seed=seed,
                tolerance=verify_tolerance,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            # e.g. float64 covariances failing V + i Omega >= 0 at the raised squeezing
            # levels (resource squeezing r >~ 3); undecided, never certified.
            report = {"ideal_limit_deterministic": False, "error": repr(error)}
        certificate["determinism"] = report
        certificate["checks"] += ("ideal-limit determinism by exact Gaussian postselection",)
        if not report["ideal_limit_deterministic"]:
            status = "unknown"
            diagnostics["reason"] = "algebraic certificate found but numerical check " + (
                f"could not run: {report['error']}" if "error" in report else "failed"
            )
    else:
        assumptions += ("numerical determinism verification skipped",)
    return FlowAnalysisResult(
        status,
        og.model,
        ALGORITHM,
        order,
        layers,
        corrections,
        certificate,
        assumptions,
        diagnostics,
    )
