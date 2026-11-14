"""R55: Local complementation and local-Clifford equivalence of graph states (qubit), and the
validated Gaussian graph rules (CV).

Scientific question: (i) does ``photographiq.calculus.clifford.local_complement`` with its local
Clifford ``U_v = exp(-i pi/4 X_v) prod_{w in N(v)} exp(+i pi/4 Z_w)`` map |G> to |tau_v(G)> up to a
global phase (Van den Nest, Dehaene, De Moor, PRA 69, 022316 (2004)); (ii) do LC sequences
found by ``lc_equivalence`` produce Clifford witnesses that are correct on statevectors, and are
negative answers consistent with an LU invariant; (iii) for CV, are the implemented
``GaussianGraph`` rules (Menicucci-Flammia-van Loock 2011: CZ, shear, squeezing, rotation, beam
splitter, ideal q/p homodyne) consistent with phase-space evolution of the covariance?

Conventions: |G> = prod_{(u,v) in E} CZ_uv |+>^n, big-endian in the stated node order; states
compared up to a global phase, which is reported (arg <tau_v G| U_v |G>); Clifford matrices are
the library's canonical representatives (phase fixed), so composed witnesses also hold only up
to a global phase. hbar = 2, interleaved (q, p), vacuum covariance I for the CV part.

Independent references in this script: graph states built by applying CZ phases to |+>^n;
local complementation by toggling neighbourhood edges; U_v built from cos/sin of Pauli matrices;
LU invariant: the vector of Schmidt ranks over all bipartitions (a labelled-LC invariant: if it
differs, no LC sequence exists). CV: covariance propagated as S V S^T with the gates' symplectic
matrices, and ideal homodyne by Gaussian conditioning (Schur complement of the covariance in the
infinite-precision limit), compared with ``GaussianGraph.to_covariance()`` and its mean.

Not covered: CV local complementation is NOT implemented (``cv_local_complementation`` raises);
no claim about LC orbits of CV graphs. Seed 20261155.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from photographiq import gaussian as gm  # noqa: E402
from photographiq.calculus import GaussianGraph  # noqa: E402
from photographiq.calculus.clifford import (  # noqa: E402
    apply_local_cliffords,
    cv_local_complementation,
    lc_cliffords,
    lc_equivalence,
    local_complement,
)

SEED = 20261155
N_RANDOM_GRAPHS = 6
SEQUENCES_PER_GRAPH = 4

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_STATE = 1e-12  # 1 - |<a|b>| for <= 7-qubit dense statevectors (float64)
TOL_CV = 1e-10  # covariance / mean agreement relative to max(1, |V|) (r <= 1, |weights| <= 1.5)

_X = np.array([[0, 1], [1, 0]], complex)
_Z = np.diag([1.0 + 0j, -1.0])


# ----- independent qubit references ---------------------------------------------------------
def graph_state(graph, order):
    n = len(order)
    psi = np.full((2,) * n, 2 ** (-n / 2), complex)
    pos = {v: k for k, v in enumerate(order)}
    for u, v in graph.edges:
        idx = [slice(None)] * n
        idx[pos[u]] = idx[pos[v]] = 1
        psi[tuple(idx)] *= -1
    return psi.reshape(-1)


def toggle(graph, v):
    g = nx.Graph(graph)
    for a, b in itertools.combinations(list(graph[v]), 2):
        if g.has_edge(a, b):
            g.remove_edge(a, b)
        else:
            g.add_edge(a, b)
    return g


def apply_local(psi, mats, order):
    n = len(order)
    t = psi.reshape((2,) * n)
    for k, v in enumerate(order):
        if v in mats:
            t = np.moveaxis(np.tensordot(mats[v], t, axes=([1], [k])), 0, k)
    return t.reshape(-1)


def lc_unitary(graph, v):
    """exp(-i pi/4 X_v) prod_{w in N(v)} exp(+i pi/4 Z_w) from cos/sin (no library code)."""
    c = math.cos(math.pi / 4)
    mats = {v: c * np.eye(2) - 1j * c * _X}
    for w in graph[v]:
        mats[w] = c * np.eye(2) + 1j * c * _Z
    return mats


def schmidt_ranks(psi, n):
    out = []
    for k in range(1, n // 2 + 1):
        for part in itertools.combinations(range(n), k):
            rest = [i for i in range(n) if i not in part]
            m = np.transpose(psi.reshape((2,) * n), list(part) + rest).reshape(2**k, -1)
            out.append(int(np.linalg.matrix_rank(m, tol=1e-9)))
    return tuple(out)


def edges(g):
    return {frozenset(e) for e in g.edges}


def graphs(rng):
    out = {
        "path_P4": nx.path_graph(4),
        "star_S4": nx.star_graph(3),
        "cycle_C5": nx.cycle_graph(5),
        "complete_K4": nx.complete_graph(4),
        "cluster_2x3": nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3), ordering="sorted"),
    }
    while len(out) < 5 + N_RANDOM_GRAPHS:
        n = int(rng.integers(4, 7))
        g = nx.gnp_random_graph(n, float(rng.uniform(0.35, 0.7)), seed=int(rng.integers(1 << 30)))
        if nx.is_connected(g):
            out[f"random_{len(out) - 5}_n{n}"] = g
    return out


def qubit_rows(rng):
    single, seqs, negatives = [], [], []
    gs = graphs(rng)
    for name, g in gs.items():
        order = tuple(sorted(g.nodes))
        psi = graph_state(g, order)
        for v in order:
            lib = local_complement(g, v)
            mine = toggle(g, v)
            target = graph_state(mine, order)
            ours_u = apply_local(psi, lc_unitary(g, v), order)
            lib_u = apply_local_cliffords(psi, lc_cliffords(g, v), order)
            overlap = np.vdot(target, ours_u)
            single.append({
                "graph": name, "vertex": v, "degree": g.degree(v),
                "edges_match_independent": edges(lib) == edges(mine),
                "toggled_edges": len(edges(lib) ^ edges(g)),
                "state_dev_independent_U": 1 - abs(overlap),
                "global_phase_rad": float(np.angle(overlap)),
                "state_dev_library_cliffords": 1 - abs(np.vdot(target, lib_u)),
            })
        for s in range(SEQUENCES_PER_GRAPH):
            length = int(rng.integers(1, 7))
            seq = [order[int(rng.integers(len(order)))] for _ in range(length)]
            tgt = g
            for v in seq:
                tgt = toggle(tgt, v)
            found = lc_equivalence(g, tgt)
            row = {"graph": name, "sequence": str(seq), "length": length, "found": found is not None}
            if found is not None:
                h = g
                for v in found.sequence:
                    h = toggle(h, v)
                mats = {v: c.matrix for v, c in found.cliffords.items()}
                row.update(
                    found_sequence=str(list(found.sequence)),
                    found_length=len(found.sequence),
                    replay_edges_match=edges(h) == edges(tgt),
                    witness_state_dev=1 - abs(np.vdot(graph_state(tgt, order), apply_local(psi, mats, order))),
                    library_residual=found.residual,
                )
            seqs.append(row)
    pairs = [("path_P4", "star_S4"), ("cycle_C5", None), ("cluster_2x3", None)]
    for a, b in pairs:
        g1 = gs[a]
        order = tuple(sorted(g1.nodes))
        if b is None:  # complete graph on the same labels: GHZ class vs a non-GHZ class
            g2, b = nx.complete_graph(order), f"complete_K{len(order)}"
        else:
            g2 = gs[b]
        found = lc_equivalence(g1, g2)
        inv1 = schmidt_ranks(graph_state(g1, order), len(order))
        inv2 = schmidt_ranks(graph_state(g2, order), len(order))
        negatives.append({"graph": a, "other": b, "library_found": found is not None,
                          "schmidt_rank_vectors_differ": inv1 != inv2,
                          "max_schmidt_rank": f"{max(inv1)} vs {max(inv2)}"})
    # positive control for the invariant: star and complete graph are LC equivalent (GHZ class)
    star, k4 = nx.star_graph(3), nx.complete_graph(4)
    found = lc_equivalence(star, k4)
    negatives.append({"graph": "star_S4", "other": "complete_K4", "library_found": found is not None,
                      "schmidt_rank_vectors_differ": schmidt_ranks(graph_state(star, range(4)), 4)
                      != schmidt_ranks(graph_state(k4, range(4)), 4),
                      "max_schmidt_rank": "GHZ class (positive control)"})
    return single, seqs, negatives, gs


# ----- CV: Gaussian graph rules vs phase-space covariance --------------------------------------
def cz_symplectic(n, u, v, g):
    s = np.eye(2 * n)
    s[2 * u + 1, 2 * v] += g
    s[2 * v + 1, 2 * u] += g
    return s


def condition(mean, cov, k, quad, outcome):
    """Ideal homodyne of quadrature `quad` (0 q, 1 p) of mode k: Gaussian conditioning (Schur)."""
    i = 2 * k + quad
    rest = [j for j in range(len(mean)) if j // 2 != k]
    c = cov[rest, i]
    new_mean = mean[rest] + c * (outcome - mean[i]) / cov[i, i]
    new_cov = cov[np.ix_(rest, rest)] - np.outer(c, c) / cov[i, i]
    return new_mean, new_cov


def cv_rows(rng):
    rows = []
    for trial in range(6):
        n = int(rng.integers(3, 5))
        r = float(rng.uniform(0.3, 1.0))
        a = np.triu(rng.uniform(-1.5, 1.5, (n, n)) * (rng.random((n, n)) < 0.6), 1)
        a = a + a.T
        gg = GaussianGraph.cluster(a, r)
        cov = np.diag(np.tile([math.exp(2 * r), math.exp(-2 * r)], n))
        for u, v in zip(*np.nonzero(np.triu(a))):
            s = cz_symplectic(n, u, v, a[u, v])
            cov = s @ cov @ s.T
        mean = np.zeros(2 * n)
        ref = {"cov": cov, "mean": mean}

        def check(rule, graph, ref_mean, ref_cov):
            scale = max(1.0, np.abs(ref_cov).max())
            rows.append({"trial": trial, "modes": len(graph.nodes), "squeezing_r": r, "rule": rule,
                         "cov_rel_error": float(np.abs(graph.to_covariance() - ref_cov).max() / scale),
                         "mean_rel_error": float(np.abs(graph.mean - ref_mean).max(initial=0.0) / scale)})

        check("cluster Z = A + i e^{-2r} I", gg, ref["mean"], ref["cov"])
        disp = rng.normal(0, 0.5, 2 * n)
        gg = GaussianGraph(gg.Z, gg.nodes, disp)
        mean = disp.copy()
        gates = [
            ("cz Z_uv += g", lambda g: g.cz(0, 1, 0.7), cz_symplectic(n, 0, 1, 0.7)),
            ("shear Z_kk += s", lambda g: g.quadratic_phase(1, -0.4), gm.embed(np.array([[1.0, 0], [-0.4, 1]]), [2, 3], 2 * n)),
            ("squeeze", lambda g: g.squeeze(2, 0.3), gm.embed(gm.squeezing(0.3), [4, 5], 2 * n)),
            ("rotate", lambda g: g.rotate(0, 0.9), gm.embed(gm.rotation(0.9), [0, 1], 2 * n)),
            ("beamsplitter", lambda g: g.beamsplitter(1, 2, 0.6), gm.embed(gm.beamsplitter(0.6), [2, 3, 4, 5], 2 * n)),
        ]
        for rule, fn, s in gates:
            gg = fn(gg)
            cov, mean = s @ cov @ s.T, s @ mean
            check(rule, gg, mean, cov)
        x = float(rng.normal(0, 1))
        gq = gg.measure_q(gg.nodes[-1], x)
        check("measure_q (vertex deletion)", gq, *condition(mean, cov, n - 1, 0, x))
        gp = gg.measure_p(gg.nodes[0], x)
        check("measure_p (Schur complement)", gp, *condition(mean, cov, 0, 1, x))
    return rows


def draw(ax, g, pos, title, highlight=None, toggled=()):
    toggled = {frozenset(e) for e in toggled}
    for u, v in g.edges:
        new = frozenset((u, v)) in toggled
        ax.plot(*zip(pos[u], pos[v]), color="#c62828" if new else "#546e7a", lw=2.2 if new else 1.5, zorder=1)
    for u, v in toggled:
        if not g.has_edge(u, v):
            ax.plot(*zip(pos[u], pos[v]), color="#c62828", lw=1.2, ls=":", zorder=1)
    for v in g.nodes:
        ax.scatter(*pos[v], s=420, color="#ffca28" if v == highlight else "#90caf9", edgecolor="#263238", zorder=2)
        ax.text(*pos[v], str(v), ha="center", va="center", fontsize=9, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    plt = common.setup_style()
    rng = common.rng(SEED)
    single, seqs, negatives, gs = qubit_rows(rng)
    cv = cv_rows(rng)
    try:
        cv_local_complementation()
        cv_lc_status = "implemented (unexpected)"
    except NotImplementedError:
        cv_lc_status = "NotImplementedError (CV local complementation not implemented; not claimed)"

    failures = []
    for r in single:
        if not r["edges_match_independent"] or r["state_dev_independent_U"] > TOL_STATE or r["state_dev_library_cliffords"] > TOL_STATE:
            failures.append({**r, "problems": "single LC"})
    for r in seqs:
        if not r["found"] or not r["replay_edges_match"] or r["witness_state_dev"] > TOL_STATE or r["found_length"] > r["length"]:
            failures.append({**r, "problems": "LC sequence witness"})
    for r in negatives:
        positive = "positive control" in r["max_schmidt_rank"]
        if positive and (not r["library_found"] or r["schmidt_rank_vectors_differ"]):
            failures.append({**r, "problems": "GHZ positive control"})
        if not positive and (r["library_found"] or not r["schmidt_rank_vectors_differ"]):
            failures.append({**r, "problems": "expected non-equivalence not confirmed by library and LU invariant"})
    for r in cv:
        if r["cov_rel_error"] > TOL_CV or r["mean_rel_error"] > TOL_CV:
            failures.append({**r, "problems": "Gaussian graph rule disagrees with phase space"})
    if cv_lc_status.startswith("implemented"):
        failures.append({"problems": "cv_local_complementation unexpectedly implemented: validate before claiming"})

    rows = [{"part": "single_lc", **r} for r in single] + [{"part": "lc_sequence", **r} for r in seqs] \
        + [{"part": "non_equivalence", **r} for r in negatives] + [{"part": "cv_gaussian_graph", **r} for r in cv]
    worst = {
        "single_lc_state_dev": max(r["state_dev_independent_U"] for r in single),
        "single_lc_library_clifford_dev": max(r["state_dev_library_cliffords"] for r in single),
        "sequence_witness_dev": max(r.get("witness_state_dev", 0.0) for r in seqs),
        "cv_cov_rel": max(r["cov_rel_error"] for r in cv),
        "cv_mean_rel": max(r["mean_rel_error"] for r in cv),
    }
    phases = sorted({round(r["global_phase_rad"] / (math.pi / 4)) for r in single})
    common.save_result(rows, "R55_local_clifford", extra={
        "seed": SEED,
        "backend": "dense statevector (NumPy) for qubits; GaussianGraph vs covariance S V S^T and Schur conditioning for CV",
        "precision": "float64",
        "parameters": {"random_graphs": N_RANDOM_GRAPHS, "sequences_per_graph": SEQUENCES_PER_GRAPH},
        "acceptance": {"TOL_STATE": TOL_STATE, "TOL_CV": TOL_CV},
        "phase_convention": "states equal up to a global phase exp(i phi); phi = arg <tau_v G|U_v|G> reported per row",
        "observed_global_phases_in_units_of_pi_over_4": phases,
        "worst": worst,
        "cv_local_complementation": cv_lc_status,
    })

    g = gs["cycle_C5"]
    pos = nx.circular_layout(g)
    g1 = local_complement(g, 0)
    g2 = local_complement(g1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    row0 = next(r for r in single if r["graph"] == "cycle_C5" and r["vertex"] == 0)
    draw(axes[0], g, pos, r"(a) $|C_5\rangle$")
    draw(axes[1], g1, pos, rf"(b) $\tau_0$: $U_0|C_5\rangle = e^{{i\,{row0['global_phase_rad']:.3f}}}|\tau_0 C_5\rangle$",
         highlight=0, toggled=edges(g1) ^ edges(g))
    draw(axes[2], g2, pos, r"(c) $\tau_2\tau_0(C_5)$", highlight=2, toggled=edges(g2) ^ edges(g1))
    common.save_figure(fig, "R55_local_clifford")
    plt.close("all")
    common.save_caption("R55_local_clifford", f"""
**R55 - local complementation as a local Clifford on graph states (qubit).** (a) Five-cycle graph
state |C5> (|G> = prod CZ |+>^n). (b) Local complementation at vertex 0 (yellow) toggles the edges
inside N(0) (red solid: added, red dotted: removed); U_0 = exp(-i pi/4 X_0) prod_(w in N(0)) exp(+i pi/4 Z_w)
maps |C5> to the new graph state up to the global phase shown. (c) A second complementation at
vertex 2. Over {len(single)} (graph, vertex) pairs on {len(gs)} graphs (path, star, cycle, complete,
2x3 cluster, {N_RANDOM_GRAPHS} random connected graphs) the independently built U_v gives
1 - |<tau_v G|U_v|G>| <= {worst['single_lc_state_dev']:.1e}; global phases are multiples of pi/4
(observed {phases} x pi/4). {len(seqs)} random LC sequences (length 1-6) were recovered by
`lc_equivalence` with Clifford witnesses verified on statevectors (worst {worst['sequence_witness_dev']:.1e}).
Non-equivalences (path vs star, cycle/cluster vs complete graph) are confirmed by differing
Schmidt-rank vectors, an LU invariant. CV: only the Gaussian graph rules of MFvL 2011 are
validated (covariance error <= {worst['cv_cov_rel']:.1e}); CV local complementation is not implemented.
Source: R55.
""")
    if failures:
        common.save_csv(failures, "R55_local_clifford_FAILURES")
        raise AssertionError(f"{len(failures)} R55 check(s) failed; see FAILURES csv")
    common.print_summary("R55 local Clifford", single=len(single), sequences=len(seqs), negatives=negatives,
                         worst={k: f"{v:.1e}" for k, v in worst.items()}, phases_pi_over_4=phases, cv_lc=cv_lc_status)


if __name__ == "__main__":
    main()
