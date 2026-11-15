"""R51: CV flow finding on weighted graphs, checked against the defining linear algebra.

Scientific question: does ``photographiq.flows.find_cv_flow`` (backward real-linear layering,
Booth and Markham, Quantum 7, 1146 (2023), Defs. 5-6 / Lemma 7) decide CV-flow existence
correctly on linear wires, 2D clusters, branched graphs, weighted random graphs and graphs
without a supported certificate, and are its layers, correction coefficients and
impossibility witnesses mathematically valid?

Flow condition (real, weighted; A the weighted adjacency): a total order of the measured
nodes admits CV flow iff for every measured u there is a real vector c_u supported on the
non-input nodes measured after u (or outputs) with (A c_u)_v = delta_uv for every v measured
up to and including u. Corrections: q_t -= c_t m, p_t -= (A c)_t m, m = record / sin(theta).

Independent references computed in this script (no photographiq flow code):

1. Exact forward search: rational arithmetic (``fractions.Fraction`` of the float64 weights)
   over prefix SETS (the condition for u only depends on the set measured up to u), with
   e_u in range(M) decided by exact rank(M) == rank([M | e_u]). Complete for existence; it
   explores the subset lattice forward, whereas the library layers backwards from outputs.
2. Least-squares re-derivation on the reported total order: c = lstsq(M, e_u), residual
   ||M c - e_u||_inf and the rank gap rank([M|e_u]) - rank(M) by SVD.
3. Layered-certificate check: the library's layer vectors c_u are tested for support (later
   layers / outputs, never inputs) and residual on u's own and earlier layers; the reported
   correction_map must equal (q, p) = (-c_t, -(A c)_t).
4. Impossibility witnesses: y^T A[W, C] = 0 exactly with y_u != 0 (W unsolved nodes,
   C the solved non-input nodes) is re-checked in rational arithmetic.

Backend: pure linear algebra plus the library's float64 Gaussian determinism check (Gaussian
backend, exact postselection) for certified graphs; resource squeezing r = 1.0, unit-
efficiency homodyne. Seed 20261151 for the random graph family.

Limitations: the brute-force reference is exponential in the number of measured nodes
(used up to 16); float64 weights are taken as exact rationals; "impossible" means no
admissible total order for the real-linear CV-flow condition, never a proof of
nondeterminism by other means; finite squeezing is never called an ideal unitary.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from photographiq.flows import find_cv_flow  # noqa: E402
from photographiq.ir import HomodyneSpec, Model, OpenGraph  # noqa: E402

SEED = 20261151
SQUEEZING = 1.0
N_RANDOM = 30  # quick; --full uses 120
BRUTE_FORCE_MAX_MEASURED = 16

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_LSTSQ_REL = 1e-9  # ||M c - e_u||_inf on the reported order (O(1)-weight graphs, cond <~ 1e3)
TOL_LAYER_RESIDUAL = 1e-10  # library layer vectors on their rows (the library's own tolerance)
TOL_CORRECTION_MAP = 1e-12  # reported map vs (-c, -A c) recomputed from the layer vectors
# Named cases with a known classification (docs/validation/cv-flow-robustness.md, derived
# by hand: K22 unit bridge has det 0; |O| < |I| cannot be a CV flow; an isolated measured
# node has no correction column; the near-singular bridge sits inside the float64 band).
EXPECTED_DECISION = {
    "wire_n3": "certified",
    "wire_n6": "certified",
    "wire_n10": "certified",
    "grid_2x3": "certified",
    "grid_3x4": "certified",
    "branched_Y": "certified",
    "branched_tree": "certified",
    "hexagon_real_invertible": "certified",
    "bridge_K22_unit": "impossible",
    "hexagon_weighted_singular": "impossible",
    "fewer_outputs_than_inputs": "impossible",
    "isolated_measured_node": "impossible",
    "near_singular_bridge_1e-10": "numerically_uncertain",
}


# ----- graph families -----------------------------------------------------------------------
def make_og(edges, inputs, outputs, shears=None, nodes=()):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    shears = shears or {}
    return OpenGraph(
        Model.CV_GAUSSIAN,
        graph,
        inputs,
        outputs,
        {n: HomodyneSpec(float(shears.get(n, 0.0))) for n in graph if n not in outputs},
        {n: SQUEEZING for n in graph},
    )


def wire(n, rng):
    shears = {i: round(float(rng.uniform(-1, 1)), 2) for i in range(n - 1)}
    return make_og([(i, i + 1, 1.0) for i in range(n - 1)], (0,), (n - 1,), shears)


def grid(rows, cols, rng):
    g = nx.grid_2d_graph(rows, cols)
    edges = [(u, v, round(float(rng.uniform(0.6, 1.4)), 2)) for u, v in g.edges]
    inputs = tuple((i, 0) for i in range(rows))
    outputs = tuple((i, cols - 1) for i in range(rows))
    shears = {n: round(float(rng.uniform(-0.5, 0.5)), 2) for n in g if n not in outputs}
    return make_og(edges, inputs, outputs, shears)


def named_cases(rng):
    hexagon = [(u, o, 1.0) for u, o in [(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "a"), (3, "c")]]
    bridge = [("u", "a", 1.0), ("u", "b", 1.0), ("v", "a", 1.0), ("v", "b", 1.0)]
    return {
        "wire_n3": ("linear", wire(3, rng)),
        "wire_n6": ("linear", wire(6, rng)),
        "wire_n10": ("linear", wire(10, rng)),
        "grid_2x3": ("2D cluster", grid(2, 3, rng)),
        "grid_3x4": ("2D cluster", grid(3, 4, rng)),
        # 1 input -> 2 outputs: an ideal isometry (one infinitely squeezed output quadrature)
        "branched_Y": ("branched", make_og([(0, 1, 1.0), (1, 2, 0.8), (1, 3, 1.3)], (0,), (2, 3))),
        "branched_tree": (
            "branched",
            make_og(
                [(0, 1, 1.0), (1, 2, 0.9), (1, 3, 1.1), (2, 4, 1.2), (3, 5, 0.7)],
                (0,),
                (4, 5),
                {0: 0.3, 1: -0.2, 2: 0.1, 3: 0.4},
            ),
        ),
        "hexagon_real_invertible": ("weighted special", make_og(hexagon, (1, 2, 3), ("a", "b", "c"))),
        "bridge_K22_unit": ("no certificate", make_og(bridge, ("u", "v"), ("a", "b"))),
        "hexagon_weighted_singular": (
            "no certificate",
            make_og(hexagon + [(3, "b", 2.0)], (1, 2, 3), ("a", "b", "c")),
        ),
        "fewer_outputs_than_inputs": (
            "no certificate",
            make_og([(0, 2, 1.0), (1, 2, 1.0)], (0, 1), (2,)),
        ),
        "isolated_measured_node": (
            "no certificate",
            make_og([(0, 1, 1.0)], (0,), (1,), nodes=(0, 1, "z")),
        ),
        "near_singular_bridge_1e-10": (
            "no certificate",
            make_og(bridge[:3] + [("v", "b", 1.0 + 1e-10)], ("u", "v"), ("a", "b")),
        ),
    }


def random_weighted(rng, count):
    out = {}
    while len(out) < count:
        n = int(rng.integers(4, 8))
        g = nx.gnp_random_graph(n, float(rng.uniform(0.35, 0.75)), seed=int(rng.integers(1 << 30)))
        if g.number_of_edges() == 0:
            continue
        nodes = [int(v) for v in rng.permutation(n)]
        k_out = int(rng.integers(1, n // 2 + 1))
        outputs = tuple(nodes[:k_out])
        k_in = int(rng.integers(1, k_out + 1))
        inputs = tuple(int(v) for v in rng.choice(nodes, k_in, replace=False))
        edges = [
            (u, v, float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(0.2), np.log(3.0)))))
            for u, v in g.edges
        ]
        shears = {v: float(rng.uniform(-1, 1)) for v in g if v not in outputs}
        out[f"random_{len(out):03d}"] = ("weighted random", make_og(edges, inputs, outputs, shears, g.nodes))
    return out


def random_layered(rng, count):
    """Random weighted 'brickwork' clusters: k wires of length L plus random chords between
    neighbouring columns (|I| = |O| = k). These usually admit CV flow, so the certified
    branch is exercised on random weights; existence is still decided independently."""
    out = {}
    for i in range(count):
        k, length = int(rng.integers(2, 4)), int(rng.integers(3, 5))
        edges = []
        for w in range(k):
            for c in range(length - 1):
                edges.append(((w, c), (w, c + 1), float(rng.choice([-1, 1]) * rng.uniform(0.3, 2.0))))
        for w in range(k - 1):
            for c in range(length - 1):
                if rng.random() < 0.4:
                    edges.append(((w, c), (w + 1, c + 1), float(rng.uniform(-1.5, 1.5))))
                if rng.random() < 0.3:
                    edges.append(((w, c), (w + 1, c), float(rng.uniform(-1.5, 1.5))))
        inputs = tuple((w, 0) for w in range(k))
        outputs = tuple((w, length - 1) for w in range(k))
        shears = {(w, c): float(rng.uniform(-1, 1)) for w in range(k) for c in range(length - 1)}
        out[f"layered_{i:03d}"] = ("weighted random cluster", make_og(edges, inputs, outputs, shears))
    return out


# ----- independent linear algebra -----------------------------------------------------------
def adjacency(og, nodes):
    """Weighted adjacency built edge by edge (no networkx matrix helper, no library code)."""
    index = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)))
    for u, v, d in og.graph.edges(data=True):
        a[index[u], index[v]] = a[index[v], index[u]] = float(d["weight"])
    return a


def exact_rank(rows):
    """Rank of a list of Fraction rows by Gaussian elimination."""
    m = [list(r) for r in rows]
    rank, cols = 0, len(m[0]) if m else 0
    for c in range(cols):
        pivot = next((i for i in range(rank, len(m)) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][c] != 0:
                f = m[i][c] / m[rank][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def exact_solvable(a_exact, rows, cols, u_row):
    """e_u in range(A[rows, cols]) exactly (rank test)."""
    if not cols:
        return False
    m = [[a_exact[r][c] for c in cols] for r in rows]
    aug = [row + [Fraction(int(r == u_row))] for row, r in zip(m, rows)]
    return exact_rank(m) == exact_rank(aug)


def brute_force_exists(og):
    """Exact forward search over prefix sets; returns (exists, an admissible order or None)."""
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    a = adjacency(og, nodes)
    a_exact = [[Fraction(float(x)) for x in row] for row in a]
    measured = [n for n in nodes if n not in og.outputs]
    non_input = [n for n in nodes if n not in og.inputs]
    seen = {frozenset(): ()}
    frontier = [frozenset()]
    while frontier:
        nxt = []
        for prefix in frontier:
            for u in measured:
                if u in prefix:
                    continue
                done = prefix | {u}
                if done in seen:
                    continue
                rows = [index[n] for n in nodes if n in done]
                cols = [index[n] for n in non_input if n not in done]
                if exact_solvable(a_exact, rows, cols, index[u]):
                    seen[done] = seen[prefix] + (u,)
                    nxt.append(done)
        frontier = nxt
    full = frozenset(measured)
    return (full in seen), seen.get(full)


def lstsq_check(og, order):
    """Least-squares re-derivation on the flattened order: max residual and max rank gap."""
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    a = adjacency(og, nodes)
    worst, gap, gains = 0.0, 0, []
    for i, u in enumerate(order):
        done = set(order[: i + 1])
        rows = [index[n] for n in nodes if n in done]
        cols = [index[n] for n in nodes if n not in done and n not in og.inputs]
        target = np.array([float(r == index[u]) for r in rows])
        m = a[np.ix_(rows, cols)]
        c, *_ = np.linalg.lstsq(m, target, rcond=None)
        worst = max(worst, float(np.abs(m @ c - target).max()))
        gap = max(gap, np.linalg.matrix_rank(np.c_[m, target]) - np.linalg.matrix_rank(m))
        gains.append(float(np.abs(c).max(initial=0.0)))
    return worst, int(gap), max(gains, default=0.0)


def layered_check(og, result):
    """Re-check the library's layer vectors and correction map from the definition."""
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    a = adjacency(og, nodes)
    layers = result.layers
    rank = {n: k for k, layer in enumerate(layers) for n in layer}
    vectors = result.certificate["layer_corrections"]
    residual = map_error = 0.0
    support_ok = True
    for u in result.measurement_order:
        here = [n for n in nodes if rank.get(n, len(layers)) <= rank[u]]
        c = np.zeros(len(nodes))
        for n, value in vectors[u].items():
            c[index[n]] = value
            if n in og.inputs or rank.get(n, len(layers)) <= rank[u]:
                support_ok = False
        image = a @ c
        target = np.array([float(n == u) for n in here])
        residual = max(residual, float(np.abs(image[[index[n] for n in here]] - target).max()))
        expected = {
            t: {"q": -c[index[t]], "p": -image[index[t]]}
            for t in nodes
            if t not in here and (c[index[t]] or image[index[t]])
        }
        reported = result.correction_map[u]
        for t in set(expected) | set(reported):
            e, r = expected.get(t, {"q": 0, "p": 0}), reported.get(t, {"q": 0, "p": 0})
            map_error = max(map_error, abs(e["q"] - r["q"]), abs(e["p"] - r["p"]))
    return residual, map_error, support_ok


def witness_check(og, result):
    """Exact y^T A[W, C] = 0 with y_u != 0 for every uncorrectable u."""
    witnesses = result.diagnostics.get("witnesses")
    if not witnesses:
        return None
    nodes = list(og.nodes)
    a = adjacency(og, nodes)
    index = {n: i for i, n in enumerate(nodes)}
    unsolved = set(result.diagnostics["uncorrectable"])
    columns = [n for n in nodes if n not in unsolved and n not in og.inputs]
    for u, y in witnesses.items():
        if Fraction(y.get(u, 0)) == 0:
            return False
        for c in columns:
            if sum(Fraction(v) * Fraction(float(a[index[w], index[c]])) for w, v in y.items()) != 0:
                return False
    return True


# ----- main ---------------------------------------------------------------------------------
def analyse(name, family, og):
    result = find_cv_flow(og, seed=0)
    decision = result.diagnostics.get("decision")
    row = {
        "graph": name,
        "family": family,
        "nodes": og.graph.number_of_nodes(),
        "edges": og.graph.number_of_edges(),
        "inputs": len(og.inputs),
        "outputs": len(og.outputs),
        "status": result.status,
        "decision": decision,
        "regimes": "; ".join(sorted(result.certificate.get("regimes", {}))),
        "num_layers": len(result.layers),
        "layers": str([list(layer) for layer in result.layers]),
        "execution_order": str(list(result.measurement_order)),
        "rank_margin_decades": result.diagnostics.get("rank_decisions", {}).get("margin_decades"),
        "decided_by": result.diagnostics.get("rank_decisions", {}).get("decided_by"),
        "proof": (result.diagnostics.get("proof") or "")[:60],
    }
    measured = len([n for n in og.nodes if n not in og.outputs])
    if measured <= BRUTE_FORCE_MAX_MEASURED:
        exists, order = brute_force_exists(og)
        row["bruteforce_exists"] = exists
        row["bruteforce_order"] = str(list(order)) if order else ""
    else:
        row["bruteforce_exists"] = None
    if decision == "certified":
        worst, gap, gain = lstsq_check(og, result.measurement_order)
        res, map_err, support = layered_check(og, result)
        det = result.certificate.get("determinism") or {}
        row.update(
            lstsq_residual=worst,
            lstsq_rank_gap=gap,
            lstsq_correction_gain=gain,
            library_layer_residual=result.diagnostics.get("layer_residual_max"),
            library_certify_residual=result.diagnostics.get("certify_residual_max"),
            layered_residual_recomputed=res,
            correction_map_error=map_err,
            layer_support_valid=support,
            correction_gain_max=result.diagnostics.get("correction_gain_max"),
            ideal_limit_deterministic=det.get("ideal_limit_deterministic"),
            decay_exponent_gain=(det.get("decay_exponents") or {}).get("gain"),
            finite_squeezing_gain=det.get("finite_squeezing_gain"),
        )
    elif decision == "impossible":
        row["witness_exact_valid"] = witness_check(og, result)
        row["uncorrectable"] = str(list(result.diagnostics.get("uncorrectable", ())))
    elif decision == "numerically_uncertain":
        exact = find_cv_flow(og, exact=True)
        row["exact_mode_status"] = exact.status
        row["exact_mode_regimes"] = "; ".join(sorted(exact.certificate.get("regimes", {})))
        row["exact_mode_correction_gain"] = exact.diagnostics.get("correction_gain_max")
    return row, result


def correction_rows(name, result):
    rows = []
    for u, targets in result.correction_map.items():
        for t, c in targets.items():
            rows.append({"graph": name, "measured": str(u), "target": str(t), "q_coeff": c["q"], "p_coeff": c["p"]})
    return rows


def _label(n):
    return "".join(str(n).split()) if isinstance(n, tuple) else str(n)


def _layered_positions(og, result):
    """x = measurement layer (outputs last), y spread within a layer."""
    columns = [list(layer) for layer in result.layers] + [list(og.outputs)]
    pos = {}
    for x, column in enumerate(columns):
        for k, n in enumerate(column):
            pos[n] = (x, (len(column) - 1) / 2 - k)
    return pos


def _draw_graph(ax, og, pos, colors):
    for u, v, d in og.graph.edges(data=True):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.plot([x1, x2], [y1, y2], color="#78909c", lw=1.4, zorder=1)
        t = 0.5 if abs(x2 - x1) + abs(y2 - y1) < 1.6 else 0.35
        ax.text(x1 + t * (x2 - x1), y1 + t * (y2 - y1), f"{float(d['weight']):.2g}", fontsize=7,
                ha="center", va="center", zorder=3,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"})
    for n in og.nodes:
        marker = "s" if n in og.inputs else "D" if n in og.outputs else "o"
        ax.scatter(*pos[n], s=330, marker=marker, color=colors[n], edgecolor="#263238", zorder=2)
        ax.text(*pos[n], _label(n), fontsize=6.5, ha="center", va="center", zorder=4)
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_axis_off()


def figure_examples(plt, examples):
    """Per example: (graph) weighted resource, (flow) layer colouring, (corrections) table."""
    fig, axes = plt.subplots(len(examples), 3, figsize=(12.5, 3.3 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    cmap = plt.get_cmap("viridis")
    for row, (name, og, result) in enumerate(examples):
        pos = _layered_positions(og, result)
        neutral = {n: "#7fd1d6" if n in og.inputs else "#f3c766" if n in og.outputs else "#cfd8dc" for n in og.nodes}
        _draw_graph(axes[row, 0], og, pos, neutral)
        axes[row, 0].set_title(f"({'ad'[row]}) {name}: weighted resource", fontsize=9)
        nl = len(result.layers)
        rank = {n: k for k, layer in enumerate(result.layers) for n in layer}
        layered = {n: cmap(0.4 + 0.55 * rank[n] / max(nl - 1, 1)) if n in rank else "#f3c766" for n in og.nodes}
        _draw_graph(axes[row, 1], og, pos, layered)
        axes[row, 1].set_title(f"({'be'[row]}) {nl} measurement layers (column = layer)", fontsize=9)
        measured = list(result.measurement_order)
        targets = [n for n in og.nodes if any(n in result.correction_map[u] for u in measured)]
        mag = np.full((len(measured), len(targets)), np.nan)
        ax = axes[row, 2]
        for i, u in enumerate(measured):
            for j, t in enumerate(targets):
                c = result.correction_map[u].get(t)
                if c:
                    mag[i, j] = max(abs(c["q"]), abs(c["p"]))
                    ax.text(j, i, f"q {c['q'] + 0.0:+.2f}\np {c['p'] + 0.0:+.2f}", ha="center", va="center", fontsize=6.3)
        ax.imshow(mag, cmap="Oranges", vmin=0, vmax=np.nanmax(mag) * 1.6, aspect="auto")
        ax.set_xticks(range(len(targets)), [_label(t) for t in targets], fontsize=7)
        ax.set_yticks(range(len(measured)), [_label(u) for u in measured], fontsize=7)
        ax.set_xlabel("corrected node t", fontsize=8)
        ax.set_ylabel("record of u (execution order)", fontsize=8)
        ax.set_title(f"({'cf'[row]}) displacement coefficients per record", fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    return fig


def figure_summary(plt, rows):
    families = list(dict.fromkeys(r["family"] for r in rows))
    decisions = ("certified", "impossible", "numerically_uncertain")
    colors = {"certified": "#2e7d32", "impossible": "#c62828", "numerically_uncertain": "#f9a825"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    left = np.zeros(len(families))
    for d in decisions:
        counts = np.array([sum(r["family"] == f and r["decision"] == d for r in rows) for f in families])
        ax1.barh(families, counts, left=left, color=colors[d], label=d.replace("_", " "))
        left += counts
    ax1.set_xlabel("graphs")
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title("(a) library decision per family")
    cert = [r for r in rows if r["decision"] == "certified"]
    x = [max(r["lstsq_residual"], 1e-18) for r in cert]
    y = [max(r["correction_gain_max"], 1e-18) for r in cert]
    ax2.scatter(x, y, s=18, color="#1565c0")
    ax2.axvline(TOL_LSTSQ_REL, color="k", ls="--", lw=1)
    ax2.text(TOL_LSTSQ_REL, max(y), " acceptance", fontsize=8, va="top")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"independent residual $\|Mc-e_u\|_\infty$ (floored at 1e-18)")
    ax2.set_ylabel("max |correction coefficient|")
    ax2.set_title("(b) certified graphs: re-derived condition")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="120 random weighted graphs")
    args = parser.parse_args()
    n_random = 120 if args.full else N_RANDOM
    plt = common.setup_style()
    rng = common.rng(SEED)
    cases = named_cases(rng)
    cases.update(random_weighted(rng, n_random))
    cases.update(random_layered(rng, n_random // 2))

    rows, corrections, results = [], [], {}
    for name, (family, og) in cases.items():
        row, result = analyse(name, family, og)
        rows.append(row)
        results[name] = (og, result)
        corrections += correction_rows(name, result)

    failures = []
    for r in rows:
        problems = []
        expected = EXPECTED_DECISION.get(r["graph"])
        if expected and r["decision"] != expected:
            problems.append(f"decision {r['decision']} != expected {expected}")
        if r["bruteforce_exists"] is not None:
            if r["decision"] == "certified" and not r["bruteforce_exists"]:
                problems.append("certified but exact brute force finds no admissible order")
            if r["decision"] == "impossible" and r["bruteforce_exists"]:
                problems.append("impossible but exact brute force finds an admissible order")
        if r["decision"] == "certified":
            if not r["lstsq_residual"] <= TOL_LSTSQ_REL or r["lstsq_rank_gap"] != 0:
                problems.append("reported order fails the least-squares condition")
            if not r["layered_residual_recomputed"] <= TOL_LAYER_RESIDUAL or not r["layer_support_valid"]:
                problems.append("layer vectors fail the layered condition")
            if not r["correction_map_error"] <= TOL_CORRECTION_MAP:
                problems.append("correction map differs from (-c, -A c)")
            if not r["ideal_limit_deterministic"]:
                problems.append("library determinism check did not pass")
        if r["decision"] == "impossible" and r.get("witness_exact_valid") is False:
            problems.append("impossibility witness fails exact re-check")
        if r["decision"] == "numerically_uncertain" and r.get("exact_mode_status") != "certified":
            problems.append("uncertain case not resolved by exact=True")
        if problems:
            failures.append({**r, "problems": "; ".join(problems)})

    counts = {d: sum(r["decision"] == d for r in rows) for d in ("certified", "impossible", "numerically_uncertain")}
    agree = sum(
        1 for r in rows if r["bruteforce_exists"] is not None and r["decision"] != "numerically_uncertain"
        and (r["decision"] == "certified") == r["bruteforce_exists"]
    )
    decided = sum(1 for r in rows if r["bruteforce_exists"] is not None and r["decision"] != "numerically_uncertain")
    cert = [r for r in rows if r["decision"] == "certified"]
    worst_lstsq = max(r["lstsq_residual"] for r in cert)
    worst_map = max(r["correction_map_error"] for r in cert)
    witnesses = [r for r in rows if r.get("witness_exact_valid") is not None]

    common.save_result(
        rows,
        "R51_cv_flow_finding",
        extra={
            "seed": SEED,
            "parameters": {"squeezing": SQUEEZING, "n_random": n_random, "full": args.full},
            "backend": "pure linear algebra + library float64 Gaussian determinism check (gaussian backend)",
            "precision": "float64 (library); exact Fraction arithmetic (brute-force reference, witnesses)",
            "acceptance": {
                "TOL_LSTSQ_REL": TOL_LSTSQ_REL,
                "TOL_LAYER_RESIDUAL": TOL_LAYER_RESIDUAL,
                "TOL_CORRECTION_MAP": TOL_CORRECTION_MAP,
                "EXPECTED_DECISION": EXPECTED_DECISION,
            },
            "summary": {
                "graphs": len(rows),
                "decision_counts": counts,
                "bruteforce_agreement": f"{agree}/{decided}",
                "worst_lstsq_residual": worst_lstsq,
                "worst_correction_map_error": worst_map,
                "witnesses_checked": len(witnesses),
                "witnesses_valid": sum(bool(r["witness_exact_valid"]) for r in witnesses),
            },
        },
    )
    common.save_csv(corrections, "R51_cv_flow_corrections")
    table = ["| graph | family | decision | layers | order | max lstsq residual | brute force |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["family"] == "weighted random":
            continue
        res = f"{r['lstsq_residual']:.1e}" if "lstsq_residual" in r else "-"
        table.append(
            f"| {r['graph']} | {r['family']} | {r['decision']} | {r['num_layers']} | "
            f"{r['execution_order']} | {res} | {r['bruteforce_exists']} |"
        )
    (common.TABLES / "R51_cv_flow_finding.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    examples = [(n, *results[n]) for n in ("grid_2x3", "branched_tree")]
    common.save_figure(figure_examples(plt, examples), "R51_cv_flow_examples")
    plt.close("all")
    grid_res = next(r for r in rows if r["graph"] == "grid_2x3")
    tree_res = next(r for r in rows if r["graph"] == "branched_tree")
    common.save_caption(
        "R51_cv_flow_examples",
        f"""
**R51 - CV flow found on a weighted 2D cluster and a branched graph.** Columns: weighted
resource graph (squares inputs, diamonds outputs, circles homodyne-measured; edge labels are
CZ weights), measurement layers found by `photographiq.flows.find_cv_flow` (backward real-
linear layering, Booth-Markham 2023), and the displacement corrections q_t -= c_t m,
p_t -= (A c)_t m with m = record/sin(theta). grid_2x3: {grid_res['num_layers']} layers, order
{grid_res['execution_order']}, independent least-squares residual {grid_res['lstsq_residual']:.1e};
branched_tree (1 input, 2 outputs: an ideal isometry): {tree_res['num_layers']} layers, residual
{tree_res['lstsq_residual']:.1e}. Resource squeezing r = {SQUEEZING} only enters the determinism
check; the flow itself is an infinite-squeezing statement. Source: R51 (51_cv_flow_finding.py).
""",
    )
    common.save_figure(figure_summary(plt, rows), "R51_cv_flow_summary")
    plt.close("all")
    common.save_caption(
        "R51_cv_flow_summary",
        f"""
**R51 - CV flow decisions against an independent exact reference.** (a) Library decision for
{len(rows)} graphs ({n_random} seeded weighted random G(n, p) graphs, n = 4..7, signed
weights |w| log-uniform in [0.2, 3]; plus named linear, 2D cluster, branched and no-certificate
cases): {counts['certified']} certified, {counts['impossible']} impossible, {counts['numerically_uncertain']}
numerically uncertain. An exact rational forward search over measured prefix sets (independent
of the library's backward layering) agrees on existence for {agree}/{decided} decided graphs;
{len(witnesses)} impossibility witnesses y^T A[W, C] = 0 were re-checked exactly. The uncertain
case (bridge with det 1e-10) is certified only by `exact=True`, ideal limit only. (b) For each
certified graph, least-squares re-derivation of c_u on the reported order (worst residual
{worst_lstsq:.1e}; dashed: acceptance {TOL_LSTSQ_REL:g}) against the largest correction coefficient.
"Impossible" means no admissible order for the real-linear CV-flow condition, not a proof of
nondeterminism by other means. Source: R51.
""",
    )

    if failures:
        common.save_csv(failures, "R51_cv_flow_finding_FAILURES")
        raise AssertionError(f"{len(failures)} graph(s) failed R51 acceptance; see FAILURES csv")
    common.print_summary(
        "R51 CV flow finding",
        graphs=len(rows),
        decisions=counts,
        bruteforce_agreement=f"{agree}/{decided}",
        worst_lstsq_residual=f"{worst_lstsq:.2e}",
        worst_correction_map_error=f"{worst_map:.2e}",
        witnesses_valid=f"{sum(bool(r['witness_exact_valid']) for r in witnesses)}/{len(witnesses)}",
    )


if __name__ == "__main__":
    main()
