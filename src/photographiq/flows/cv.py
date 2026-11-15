"""Real-linear CV flow search and numerical determinism verification (Model.CV_GAUSSIAN).

Flow notion: the correction-matrix condition of Booth and Markham, Quantum 7, 1146
(2023), Defs. 5-6 and Lemma 7, as certified for a *given* total order by
:func:`photographiq.cvflow.certify_cv_flow`: for node u, a real vector c supported
on later non-input nodes with (A c)_v = delta_{uv} for every node v measured up to u,
A the *weighted* adjacency. Qubit (GF(2), unweighted) rules are never used here.

Order search: backward layering over the reals. A node u joins the next-earlier layer
when A[W, C] c = e_u is solvable, W the unsolved nodes and C the solved non-input
nodes. Solvability is monotone in the solved set, so in exact arithmetic the search
finds an admissible total order iff one exists. Each round decides e_u in range(A[W, C])
by a float64 rank test with relative thresholds and reported margins
(:func:`_float_round`); decisions within the uncertain band are re-decided exactly in
rational arithmetic (:mod:`._exact`) or reported as uncertain. Cost: at most n rounds,
each one SVD, O(n^4) overall. Nodes of a
layer need no corrections from one another under the layered certificate, which is
also a valid certificate for the flattened total order; the reported correction map
and the verified pattern both use it, so layers are genuinely parallel.

Determinism is then checked numerically on the pattern built from exactly the
reported correction map (see :func:`corrected_pattern`, :func:`verify_cv_determinism`).
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
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
from . import _exact
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


_EPS = float(np.finfo(float).eps)
# Upper edge of the uncertain band for rank decisions, relative to ||M||: a decision
# quantity in (tau, sqrt(eps)) means correction coefficients >~ 1/sqrt(eps) whose
# float64 relative error (cond * eps) exceeds sqrt(eps) ~ 1.5e-8, i.e. the verifier's
# tolerance, and weights within relative sqrt(eps) of data with the opposite answer.
UNCERTAIN_UPPER = math.sqrt(_EPS)
EXACT_NODE_LIMIT = 40  # automatic exact re-check of uncertain rank decisions up to this size


def _pow2(x: np.ndarray) -> np.ndarray:
    """Power-of-two factors (exact in float64) scaling each nonzero x into [0.5, 1)."""
    return np.where(x > 0, np.ldexp(1.0, -np.frexp(np.where(x > 0, x, 1.0))[1]), 1.0)


def _float_round(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decide e_u in range(M) for each row u by a float64 rank test with margins.

    M is equilibrated by exact powers of two (D_r M D_c; membership of e_u is
    invariant) and normalised to ||M||_2 = 1. Decision quantities: the singular values
    sigma_i and, per row u, s_u = dist(e_u, range_r) * sigma_r, the singular value that
    appending e_u creates (up to a factor in [1/sqrt 2, 1]). Both carry a rounding error
    of about tau = max(dims) * eps; a quantity <= tau is zero, >= UNCERTAIN_UPPER is
    nonzero, in between the decision is uncertain. Exactly zero rows are decided
    exactly (no edge into the correction columns). Returns (solvable mask, vectors c
    with M c = e_u in original units, one column per row, round info).
    """
    rows = _pow2(np.abs(matrix).max(axis=1))
    cols = _pow2(np.abs(matrix * rows[:, None]).max(axis=0))
    scaled = matrix * rows[:, None] * cols[None, :]
    u, s, vt = np.linalg.svd(scaled)
    tau = max(scaled.shape) * _EPS
    sr = s / s[0] if s.size and s[0] > 0 else np.zeros_like(s)
    rank = int((sr > tau).sum())
    distance = np.linalg.norm(u[:, rank:], axis=1)
    appended = distance * (sr[rank - 1] if rank else 1.0)
    appended[~matrix.any(axis=1)] = 1.0  # zero row: exactly unsolvable
    quantities = np.concatenate([sr, appended])
    nonzero = quantities[quantities > tau]
    zero = quantities[quantities <= tau]
    gray = int(((quantities > tau) & (quantities < UNCERTAIN_UPPER)).sum())
    # Truncated pseudo-inverse: x = V_r S_r^-1 U_r^T (rows_u e_u), c = D_c x.
    x = (vt[:rank].T / s[:rank]) @ (u[:, :rank].T * rows[None, :])
    info = {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "threshold": tau,
        "min_nonzero": float(nonzero.min(initial=math.inf)),
        "max_zero": float(zero.max(initial=0.0)),
        "uncertain": gray,
    }
    return appended <= tau, cols[:, None] * x, info


def _exact_round(matrix: np.ndarray) -> tuple[list[bool], dict, dict]:
    """Exact counterpart of :func:`_float_round` on the same equilibrated system."""
    rows = _pow2(np.abs(matrix).max(axis=1))
    cols = _pow2(np.abs(matrix * rows[:, None]).max(axis=0))
    scaled = _exact.exact(matrix * rows[:, None] * cols[None, :])  # powers of 2: exact
    _, solutions, witnesses = _exact.solve_round(scaled)
    fr, fc = _exact.exact([rows])[0], _exact.exact([cols])[0]
    vectors = {i: [fc[j] * fr[i] * x for j, x in enumerate(sol)] for i, sol in solutions.items()}
    # y^T (D_r M D_c) = 0  <=>  (D_r y)^T M = 0: witnesses for the original matrix.
    witnesses = {i: [fr[k] * y for k, y in enumerate(wit)] for i, wit in witnesses.items()}
    return [i in solutions for i in range(len(matrix))], vectors, witnesses


def _search(og: OpenGraph, use_exact: bool) -> dict:
    """Backward layering (module docstring) with float64 or exact rank decisions."""
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    adj = og.adjacency(nodes)
    solved = set(og.outputs)
    out: dict[str, Any] = {
        "layers": [],
        "vectors": {},
        "exact_vectors": {},
        "unsolved": (),
        "worst": 0.0,
        "rounds": [],
        "witnesses": {},
        "obstruction": None,
        "exact": use_exact,
    }
    while unsolved := [n for n in nodes if n not in solved]:
        columns = [n for n in nodes if n in solved and n not in og.inputs]
        if not columns:
            out["unsolved"] = tuple(unsolved)
            out["obstruction"] = "no measured node can be corrected: every solved node is an input"
            break
        matrix = adj[np.ix_([index[n] for n in unsolved], [index[n] for n in columns])]
        mask, solution, info = _float_round(matrix)
        solvable = mask.tolist()
        out["rounds"].append(info)
        if use_exact:
            solvable, exact_vectors, witnesses = _exact_round(matrix)
            for i, vec in exact_vectors.items():
                out["exact_vectors"][unsolved[i]] = dict(zip(columns, vec))
                solution[:, i] = [float(v) for v in vec]
        layer = tuple(u for i, u in enumerate(unsolved) if solvable[i])
        if not layer:
            out["unsolved"] = tuple(unsolved)
            if not matrix.any():
                out["obstruction"] = f"nodes {unsolved} have no edge to any correction column"
            elif use_exact:
                out["witnesses"] = {
                    unsolved[i]: {w: y for w, y in zip(unsolved, wit) if y}
                    for i, wit in witnesses.items()
                }
            break
        residual = np.abs(matrix @ solution - np.eye(len(unsolved))).max(axis=0)
        for i, u in enumerate(unsolved):
            if u in layer:
                out["vectors"][u] = dict(zip(columns, solution[:, i].tolist()))
                out["worst"] = max(out["worst"], float(residual[i]))
        solved.update(layer)
        out["layers"].insert(0, layer)
    out["layers"] = tuple(out["layers"])
    out["uncertain"] = sum(r["uncertain"] for r in out["rounds"])
    return out


def search_cv_order(og: OpenGraph, *, tolerance: float = 1e-10) -> tuple[tuple, dict, tuple, float]:
    """Layered search; return (layers earliest first, layered vectors, unsolved, max residual).

    Solvability is decided by the float64 rank test of :func:`_float_round` (relative
    thresholds), not by ``tolerance``, which is kept for backward compatibility;
    :func:`find_cv_flow` reports the rank margins and re-checks uncertain decisions.
    """
    del tolerance
    r = _search(og, False)
    return r["layers"], r["vectors"], r["unsolved"], r["worst"]


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

    ``ideal_limit_deterministic`` is the conjunction of these checks; ``failed_checks``
    names the ones that failed (``decay`` alone is the signature of a slowly decaying,
    pre-asymptotic gain, see docs/validation/cv-flow-robustness.md).

    ``shift`` trades range for float64 headroom: the largest level has squeezing
    r + 2 ``shift``, and above about 9.5 the raised covariances fail float64 validation
    (the check then raises ValueError). A smaller shift reaches larger r but measures
    the decay exponent over a shorter range.
    """
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be positive and finite")
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
    failed = set()
    if model_residual > tolerance * max(magnitude, _dist(channels[2].noise, 0)):
        failed.add("noise_model")
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
        if level["matrix_change"] > bound:
            failed.add("matrix_change")
        if level["covariance_outcome_dependence"] > scale:
            failed.add("covariance_outcome_dependence")
        if level["spread_min"] < -(scale + level["spread_rounding"]):
            failed.add("total_covariance")
        levels.append(level)

    def exponent(key: str) -> float:
        a, b = levels[1][key], levels[2][key]
        return math.inf if b <= bound else math.log(max(a, bound) / b) / (2 * shift)

    exponents = {k: exponent(k) for k in ("gain", "conditional_map_error")}
    symplectic = _symplectic_residual(s_ideal)
    singular = int(len(eigenvalues) - finite.shape[1])
    expected_singular = len(channels[0].outputs) - len(pattern.inputs)
    if symplectic is not None and symplectic > bound:
        failed.add("symplectic")
    if singular != expected_singular:
        failed.add("singular_directions")
    if min(exponents.values()) < min_exponent:
        failed.add("decay")
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
        "shift": shift,
        "failed_checks": tuple(sorted(failed)),
        "ideal_limit_deterministic": not failed,
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
    shift: float = 3.0,
    exact: bool | None = None,
) -> FlowAnalysisResult:
    """Search an admissible order, certify it and verify determinism.

    Status ``certified`` requires the Booth-Markham certificate for the found order
    (re-derived independently by ``certify_cv_flow``, residual <= ``tolerance``) and,
    when ``verify`` is True, an ideal-limit determinism check: the float64 check
    :func:`verify_cv_determinism` (with ``shift``) or, when ``exact`` is True, the exact
    :func:`exact_ideal_limit`. ``certificate['regimes']`` states what is covered
    (``ideal_limit`` and/or ``finite_squeezing``) and on which basis;
    ``diagnostics['decision']`` is ``certified``, ``impossible`` (proof in
    ``diagnostics['proof']``) or ``numerically_uncertain``.

    Args:
        shift: Squeezing increment of the float64 check (levels r, r + shift,
            r + 2 shift). Keep r + 2 shift below about 9 (e.g. shift=1.5 up to r = 6).
        exact: ``None`` (default): on graphs of at most ``EXACT_NODE_LIMIT`` nodes, a
            search with float64 rank decisions in the uncertain band, or one that finds
            no flow, is redone in exact rational arithmetic.
            ``True``: every rank decision and the ideal-limit check are also done
            exactly (any size; cost grows quickly). ``False``: float64 only.
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
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be positive and finite")
    search = _search(og, exact is True)
    resolved = "exact" if exact else "float64"
    stuck = search["unsolved"] and not search["obstruction"]
    if exact is None and len(og.nodes) <= EXACT_NODE_LIMIT and (search["uncertain"] or stuck):
        # Cheap on small graphs: uncertain decisions become exact, and a float64
        # "no flow" (zero decisions sit at the rounding level) gets rational witnesses.
        search = _search(og, True)
        resolved = "exact (automatic)"
    rounds = search["rounds"]
    nonzero = min((r["min_nonzero"] for r in rounds), default=math.inf)
    threshold = max((r["threshold"] for r in rounds), default=0.0)
    margin = math.log10(nonzero / threshold) if threshold and nonzero < math.inf else math.inf
    diagnostics: dict[str, Any] = {
        "complete": True,
        "rank_decisions": {
            "rule": "singular values of the 2^k-equilibrated, normalised correction matrix: "
            "<= max(dims)*eps zero, >= sqrt(eps) nonzero, in between uncertain",
            "uncertain_upper": UNCERTAIN_UPPER,
            "uncertain_float64_decisions": search["uncertain"],
            "margin_decades": margin,
            "decided_by": resolved,
            "rounds": tuple(rounds),
        },
    }
    assumptions: tuple[str, ...] = (
        "Booth-Markham real-linear CV flow for homodyne measurements on a weighted graph",
        "determinism is an infinite-squeezing statement; finite squeezing adds noise N and "
        "an outcome-dependent conditional shift G m, both in certificate['determinism']",
        "unit-efficiency homodyne detection",
        "missing resource squeezing defaults to r=1.0 (OpenGraph.to_cvgraph convention)",
    )
    layers = search["layers"]

    def result(status, decision, reason=None, corrections=None, certificate=None):
        diagnostics["decision"] = decision
        if reason:
            diagnostics["reason"] = reason
        order = () if status == "not_certified" else tuple(n for lay in layers for n in lay)
        return FlowAnalysisResult(
            status,
            og.model,
            ALGORITHM,
            order,
            layers,
            corrections or {},
            certificate or {},
            assumptions,
            diagnostics,
        )

    if search["uncertain"] and not search["exact"]:
        return result(
            "unknown",
            "numerically_uncertain",
            f"{search['uncertain']} float64 rank decision(s) fall in the uncertain band "
            "(max(dims)*eps, sqrt(eps)); exact=True decides them in rational arithmetic",
        )
    if search["unsolved"]:
        diagnostics["uncorrectable"] = search["unsolved"]
        if search["obstruction"]:
            basis = "combinatorial obstruction"
            diagnostics["proof"] = f"{basis}: {search['obstruction']}"
        elif search["exact"]:
            basis = "exact rank test"
            diagnostics["proof"] = (
                f"{basis}: for each uncorrectable u a rational y with y^T A[W, C] = 0 and "
                "y_u != 0 (diagnostics['witnesses']) shows A[W, C] c = e_u is unsolvable"
            )
            diagnostics["witnesses"] = search["witnesses"]
        else:
            basis = "float64 rank test"
            diagnostics["proof"] = (
                f"{basis} (backward-error statement, not exact): the SVD is exact for weights "
                "perturbed by at most max(dims)*eps relative, and those admit no flow; no "
                f"decision in the uncertain band, nonzero ones {margin:.1f} decades above the "
                "threshold. exact=True gives rational witnesses"
            )
        return result(
            "not_certified",
            "impossible",
            "no admissible total order satisfies the real-linear CV-flow condition "
            f"({basis}; layered search complete): the pattern can only be postselected. "
            "This is not a proof of nondeterminism by other means",
        )
    order = tuple(n for layer in layers for n in layer)
    vectors = search["vectors"]
    diagnostics["layer_residual_max"] = search["worst"]
    graph = resource_graph(og)
    flow = certify_cv_flow(graph, order, tolerance=tolerance)
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
    regimes: dict[str, str] = {}
    certificate: dict[str, Any] = {
        "total_order": flow,
        "layer_corrections": vectors,
        "shears": shears,
        "checks": ("layered real-linear condition",),
        "regimes": regimes,
    }
    diagnostics["certify_residual_max"] = max(flow.residuals, default=0.0) if flow else None
    diagnostics["total_order_residual_max"] = layered_residual
    diagnostics["correction_gain_max"] = max(
        (abs(v) for c in corrections.values() for d in c.values() for v in d.values()),
        default=0.0,
    )
    if diagnostics["correction_gain_max"] > 1 / UNCERTAIN_UPPER:
        diagnostics["warning"] = (
            f"corrections amplify records by up to {diagnostics['correction_gain_max']:.3g} "
            "> 1/sqrt(eps): the weights are close to a configuration without CV flow"
        )
    float_ok = flow is not None and layered_residual <= tolerance
    if float_ok:
        certificate["checks"] += (
            "layered vectors re-checked on the flattened total order",
            "certify_cv_flow on the total order",
        )
    ideal = "unitary" if len(og.outputs) == len(og.inputs) else "isometry"
    report: dict | None = None
    if verify and float_ok:
        try:
            report = verify_cv_determinism(
                corrected_pattern(graph, order, corrections, shears),
                seed=seed,
                tolerance=verify_tolerance,
                shift=shift,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            # e.g. float64 covariances failing V + i Omega >= 0 at the raised squeezing
            # levels (r + 2 shift >~ 9.5); undecided, never certified.
            report = {"ideal_limit_deterministic": False, "error": repr(error)}
        certificate["determinism"] = report
        certificate["checks"] += ("ideal-limit determinism by exact Gaussian postselection",)
        if report["ideal_limit_deterministic"]:
            regimes["ideal_limit"] = f"deterministic {ideal}: float64 decay check (shift={shift})"
        if "error" not in report and set(report["failed_checks"]) <= _benign(report):
            r = min((float(og.squeezing.get(n, 1.0)) for n in og.nodes), default=0.0)
            regimes["finite_squeezing"] = (
                f"outcome-averaged Gaussian channel (S, N) at the given squeezing (min r = "
                f"{r:g}), S the ideal map; residual outcome dependence "
                f"{report['finite_squeezing_gain']:.3g} (float64)"
            )
    elif float_ok:
        regimes["ideal_limit"] = (
            f"deterministic {ideal}: Booth-Markham theorem on the float64 flow certificate "
            "(determinism not checked)"
        )
        assumptions += ("numerical determinism verification skipped",)
    if exact and verify:
        check = exact_ideal_limit(og, layers, search["exact_vectors"])
        certificate["exact_ideal_limit"] = check
        certificate["checks"] += ("exact rational ideal-limit check",)
        failed = set() if report is None else set(report.get("failed_checks", ()))
        if check["ideal_limit_deterministic"] and report and "error" not in report:
            if not failed <= _benign(report):  # a float64 check that ran disagrees: undecided
                regimes.clear()
                return result(
                    "unknown",
                    "numerically_uncertain",
                    f"exact ideal-limit check passes but float64 checks {sorted(failed)} fail",
                    corrections,
                    certificate,
                )
        if check["ideal_limit_deterministic"]:
            both = " and float64 decay check" if "ideal_limit" in regimes else ""
            regimes["ideal_limit"] = f"deterministic {ideal}: exact rational check{both}"
    if "ideal_limit" in regimes:
        return result("certified", "certified", None, corrections, certificate)
    if not float_ok:
        reason = "float64 re-certification of the order failed (residual above tolerance)"
    elif report is not None and "error" in report:
        reason = f"algebraic certificate found but numerical check could not run: {report['error']}"
    else:
        reason = "algebraic certificate found but numerical check failed"
    if not exact:
        reason += "; exact=True runs the exact ideal-limit check"
    return result("unknown", "numerically_uncertain", reason, corrections, certificate)


def _benign(report: dict) -> set:
    """Float64 failures that the exact ideal-limit check may overrule.

    ``decay``: a pre-asymptotic, slowly decaying gain. ``singular_directions`` only when
    too *few* anti-squeezed output directions were counted: eigenvalues of N_plus below
    ``tolerance`` times its largest entry (dynamic range > 1e9) are read as finite.
    Missing or wrong corrections give *more* directions and are never overruled.
    """
    low = report["ideal_singular_directions"] < report["expected_singular_directions"]
    return {"decay", "singular_directions"} if low else {"decay"}


def exact_ideal_limit(og: OpenGraph, layers, exact_vectors: dict) -> dict:
    """Exact (rational) ideal-limit determinism check of the layered correction map.

    Independent of the simulator and of any squeezing value. Linear Wigner variables:
    inputs, and per resource node an anti-squeezed q (variance e^{2r}) and a squeezed p
    (e^{-2r}), as prepared by ``Prepare`` (``GaussianInput.squeezed(-r)``). After all CZ
    gates, node u is measured as m_u = k_u q_u + p_u (record / sin theta) and every
    later target t gets q_t -= c_t m_u, p_t -= (A c)_t m_u. With L the output rows,
    R the record rows and the subscript Q their anti-squeezed columns:

    * the flow condition (A c_u)_v = delta_uv holds exactly on u's layer and earlier;
    * rank L_Q = |O| - |I| (anti-squeezed noise reaches exactly the ideal-squeezed
      output directions; this is the verifier's N_plus rank);
    * rank R_Q = number of records (every record combination carries anti-squeezed
      noise, so the projected gain G and A - S are O(e^{-2r}): decay exponent exactly 1);
    * S Omega S^T = Omega exactly when |O| = |I|.

    Together these imply every criterion of :func:`verify_cv_determinism` in exact
    arithmetic (docs/validation/cv-flow-robustness.md). They are sufficient, not
    necessary: a failure here is "not shown", never a proof of nondeterminism.
    """
    graph = resource_graph(og)
    nodes = list(graph.nodes)
    adj = _exact.exact(graph.adjacency())
    pos = {n: i for i, n in enumerate(nodes)}
    rank = {n: i for i, layer in enumerate(layers) for n in layer}
    inputs = list(og.inputs)
    resource = [n for n in nodes if n not in og.inputs]
    width = 2 * len(inputs) + 2 * len(resource)
    zero, one = Fraction(0), Fraction(1)

    def unit(k: int) -> list:
        row = [zero] * width
        row[k] = one
        return row

    quad = {n: [unit(2 * i), unit(2 * i + 1)] for i, n in enumerate(inputs)}
    anti = []
    for j, n in enumerate(resource):
        k = 2 * len(inputs) + 2 * j
        quad[n] = [unit(k), unit(k + 1)]
        anti.append(k)
    q0 = {n: quad[n][0][:] for n in nodes}  # CZ leaves q unchanged
    for u, v in graph.network.edges:
        w = adj[pos[u]][pos[v]]
        quad[u][1] = [a + w * b for a, b in zip(quad[u][1], q0[v])]
        quad[v][1] = [a + w * b for a, b in zip(quad[v][1], q0[u])]
    records, condition = [], True
    for u in (n for layer in layers for n in layer):
        shear = Fraction(float(og.measurements[u].shear))
        q, p = quad.pop(u)
        m = [shear * a + b for a, b in zip(q, p)]
        records.append(m)
        c = [exact_vectors[u].get(n, zero) for n in nodes]
        image = [_exact._dot(row, c) for row in adj]
        for t in nodes:
            if rank.get(t, len(layers)) <= rank[u]:
                condition = condition and image[pos[t]] == (one if t == u else zero)
            elif t in quad:
                cq, cp = c[pos[t]], image[pos[t]]
                quad[t][0] = [a - cq * b for a, b in zip(quad[t][0], m)]
                quad[t][1] = [a - cp * b for a, b in zip(quad[t][1], m)]
    out_rows = [row for n in graph.outputs for row in quad[n]]
    s = [row[: 2 * len(inputs)] for row in out_rows]
    out_rank = _exact.rank([[row[k] for k in anti] for row in out_rows]) if out_rows else 0
    rec_rank = _exact.rank([[row[k] for k in anti] for row in records]) if records else 0
    expected = len(graph.outputs) - len(inputs)
    symplectic = None
    if len(s) == 2 * len(inputs) and s:
        form = [
            [Fraction((i // 2 == j // 2) * (j - i)) for j in range(len(s))] for i in range(len(s))
        ]
        sw = [[_exact._dot(row, col) for col in zip(*form)] for row in s]
        symplectic = all(
            _exact._dot(a, b) == form[i][j] for i, a in enumerate(sw) for j, b in enumerate(s)
        )
    ok = condition and out_rank == expected and rec_rank == len(records) and symplectic is not False
    return {
        "method": "exact rational arithmetic (fractions.Fraction) on the float64 data",
        "flow_condition": condition,
        "anti_squeezed_output_rank": out_rank,
        "expected_anti_squeezed_rank": expected,
        "record_anti_squeezed_rank": rec_rank,
        "records": len(records),
        "symplectic": symplectic,
        "matrix": np.array([[float(x) for x in row] for row in s]).reshape(len(s), 2 * len(inputs)),
        "outputs": tuple(graph.outputs),
        "ideal_limit_deterministic": bool(ok),
    }
