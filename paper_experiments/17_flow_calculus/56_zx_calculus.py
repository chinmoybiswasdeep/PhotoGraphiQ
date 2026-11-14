"""R56: Qubit ZX-calculus subset: rewrite soundness with exact scalars (QUBIT ZX ONLY; no CV-ZX).

Scientific question: does every implemented rewrite of ``photographiq.calculus.zx`` (spider
fusion, identity removal incl. Hadamard-Hadamard cancellation, colour change, Pauli state copy,
2x2 bialgebra, Hopf, parallel Hadamard-edge cancellation, self-loop removal, and the
``simplify_spiders`` fixed point) preserve the linear map of the diagram EXACTLY, including
its scalar, and do graph-state and MBQC-branch diagrams have the intended semantics?

Semantics (Coecke-Kissinger, exact scalars): Z spider with phase a and n legs is
|0..0><0..0| + e^{ia} |1..1><1..1|; X spider is |+..+><+..+| + e^{ia} |-..-><-..-|; simple edge
= identity, Hadamard edge = H; the diagram's matrix is 2^|out| x 2^|in| (first boundary most
significant) times the diagram's stored scalar.

Independent reference (this script): an own tensor-network evaluator that reads only the
diagram's vertex/edge/boundary data and scalar, builds Z spiders from computational-basis
projectors and X spiders DIRECTLY from |+>/|-> outer products (not as H-conjugated Z spiders),
and contracts with ``numpy.einsum``. Closed-form matrices for the textbook examples and
statevectors built from CZ phases for graph states; ``qubit.run_branch`` Kraus operators for
the MBQC branch diagrams.

Scalar / phase convention: rewrites carry exact scalar factors (e.g. Hopf 1/2, bialgebra
sqrt 2, copy 2^{(1-n)/2} e^{i a alpha}); before/after matrices are compared with NO global
phase or normalisation freedom. Only for diagrams with an unrecorded convention (none here)
would a phase be divided out. The proportionality constant after/before is reported (must be 1).

Scope: qubit ZX subset only. Pi-copy, local complementation/pivoting as ZX rewrites, and any
CV-ZX identities are not implemented and not claimed. Dense evaluation (<= ~14 open + internal
legs per contraction step). Seed 20261156.
"""

from __future__ import annotations

import argparse
import math
import string
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from matplotlib.patches import FancyArrowPatch  # noqa: E402

from photographiq.calculus import qubit as q  # noqa: E402
from photographiq.calculus import zx  # noqa: E402
from photographiq.calculus.zx import Diagram, EdgeType, RewriteError, VertexType  # noqa: E402
from photographiq.ir import Plane  # noqa: E402

SEED = 20261156
N_RANDOM = 150  # random (diagram, rewrite) trials; --full: 600

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_EXACT = 1e-12  # max |M_after - M_before| relative to max(1, |M|) (float64 contraction, <= 16 legs)
MIN_RANDOM_APPLIED = 100  # random trials must actually exercise rewrites (quick mode)

_H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
_PLUS = np.array([1, 1], complex) / math.sqrt(2)
_MINUS = np.array([1, -1], complex) / math.sqrt(2)


# ----- independent evaluator --------------------------------------------------------------------
def spider_tensor(kind, phase, n):
    if n == 0:
        return np.array(1 + np.exp(1j * phase))
    a, b = (np.array([1, 0], complex), np.array([0, 1], complex)) if kind == "Z" else (_PLUS, _MINUS)
    return reduce_outer([a] * n) + np.exp(1j * phase) * reduce_outer([b] * n)


def reduce_outer(vs):
    t = vs[0]
    for v in vs[1:]:
        t = np.multiply.outer(t, v)
    return t


def evaluate(d: Diagram) -> np.ndarray:
    letters = iter(string.ascii_letters)
    ends = {v: [] for v in d.vertices}
    operands, subs = [], []
    for e, (a, b, t) in d.edges.items():
        la, lb = next(letters), next(letters)
        ends[a].append(la)
        ends[b].append(lb)
        operands.append(np.eye(2, dtype=complex) if t is EdgeType.SIMPLE else _H)
        subs.append(la + lb)
    for v, (kind, phase) in d.vertices.items():
        if kind is VertexType.BOUNDARY:
            continue
        operands.append(spider_tensor(kind.value, phase, len(ends[v])))
        subs.append("".join(ends[v]))
    out = "".join(ends[b][0] for b in d.outputs + d.inputs)
    t = np.einsum(",".join(subs) + "->" + out, *operands, optimize="greedy")
    return d.scalar * np.asarray(t).reshape(2 ** len(d.outputs), 2 ** len(d.inputs))


def rel(a, b):
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def proportionality(a, b):
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    return complex(a[k] / b[k]) if abs(b[k]) > 1e-14 else complex("nan")


# ----- examples ---------------------------------------------------------------------------------
def examples():
    out = []
    # fusion: in - Z(a) - Z(b) - out, Z(b) has a second output: map diag-type 1 -> 2
    d = Diagram()
    i, o1, o2 = d.add_input(), d.add_output(), d.add_output()
    za, zb = d.add_vertex("Z", 0.7), d.add_vertex("Z", 1.1)
    d.add_edge(i, za), d.add_edge(za, zb), d.add_edge(zb, o1), d.add_edge(zb, o2)
    closed = np.zeros((4, 2), complex)
    closed[0, 0], closed[3, 1] = 1, np.exp(1j * 1.8)
    out.append(("spider fusion", d, lambda x, za=za, zb=zb: zx.fuse(x, za, zb), closed))
    # identity removal with Hadamard-Hadamard cancellation: in -H- Z(0) -H- out = identity
    d = Diagram()
    i, o = d.add_input(), d.add_output()
    z = d.add_vertex("Z")
    d.add_edge(i, z, "H"), d.add_edge(z, o, "H")
    out.append(("identity removal (H.H = I)", d, lambda x, z=z: zx.remove_identity(x, z), np.eye(2)))
    # colour change: X(a) with 1 input, 2 outputs
    d = Diagram()
    i, o1, o2 = d.add_input(), d.add_output(), d.add_output()
    x = d.add_vertex("X", 0.4)
    d.add_edge(i, x), d.add_edge(x, o1), d.add_edge(x, o2)
    closed = np.outer(np.kron(_PLUS, _PLUS), _PLUS.conj()) + np.exp(0.4j) * np.outer(np.kron(_MINUS, _MINUS), _MINUS.conj())
    out.append(("colour change", d, lambda y, x=x: zx.color_change(y, x), closed))
    # bialgebra: Z(0) with 2 inputs and X(0) with 2 outputs, joined by one simple edge
    d = Diagram()
    i1, i2, o1, o2 = d.add_input(), d.add_input(), d.add_output(), d.add_output()
    z, x = d.add_vertex("Z"), d.add_vertex("X")
    d.add_edge(i1, z), d.add_edge(i2, z), d.add_edge(z, x), d.add_edge(x, o1), d.add_edge(x, o2)
    out.append(("bialgebra (2x2)", d, lambda y, z=z, x=x: zx.bialgebra(y, z, x), None))
    # Hopf: Z and X joined by two simple edges, one boundary each
    d = Diagram()
    i, o = d.add_input(), d.add_output()
    z, x = d.add_vertex("Z"), d.add_vertex("X")
    d.add_edge(i, z), d.add_edge(z, x), d.add_edge(z, x), d.add_edge(x, o)
    out.append(("Hopf", d, lambda y, z=z, x=x: zx.hopf(y, z, x), None))
    # parallel Hadamard edges between Z spiders cancel
    d = Diagram()
    i, o = d.add_input(), d.add_output()
    z1, z2 = d.add_vertex("Z", 0.3), d.add_vertex("Z", -0.8)
    d.add_edge(i, z1), d.add_edge(z1, z2, "H"), d.add_edge(z1, z2, "H"), d.add_edge(z2, o)
    out.append(("Hadamard-pair cancellation", d, lambda y, z1=z1, z2=z2: zx.cancel_hadamard_pair(y, z1, z2), None))
    # Pauli state copy: X(pi) state into Z(a) with 2 outputs
    d = Diagram()
    o1, o2 = d.add_output(), d.add_output()
    s, z = d.add_vertex("X", math.pi), d.add_vertex("Z", 0.9)
    d.add_edge(s, z), d.add_edge(z, o1), d.add_edge(z, o2)
    out.append(("Pauli state copy", d, lambda y, s=s, z=z: zx.copy_state(y, s, z), None))
    # Hadamard self-loop removal
    d = Diagram()
    i, o = d.add_input(), d.add_output()
    z = d.add_vertex("Z", 0.5)
    d.add_edge(i, z), d.add_edge(z, z, "H"), d.add_edge(z, o)
    out.append(("Hadamard self-loop", d, lambda y, z=z: zx.remove_self_loops(y, z), None))
    return out


def random_diagram(rng):
    d = Diagram()
    n_sp = int(rng.integers(2, 5))
    sp = [d.add_vertex(str(rng.choice(["Z", "X"])), float(rng.choice([0, 0, math.pi, rng.uniform(0, 2 * math.pi)])))
          for _ in range(n_sp)]
    for _ in range(int(rng.integers(n_sp - 1, n_sp + 3))):
        a, b = rng.choice(sp, 2, replace=bool(rng.random() < 0.1))
        d.add_edge(int(a), int(b), str(rng.choice(["S", "S", "H"])))
    for _ in range(int(rng.integers(0, 3))):
        d.add_edge(d.add_input(), int(rng.choice(sp)), str(rng.choice(["S", "H"])))
    for _ in range(int(rng.integers(1, 3))):
        d.add_edge(int(rng.choice(sp)), d.add_output(), str(rng.choice(["S", "H"])))
    return d


def random_rewrite(d, rng):
    """Try every rule at random vertex pairs; return (rule, new diagram) or None."""
    sp = d.spiders()
    pairs = [(u, v) for u in sp for v in sp if u != v]
    rng.shuffle(pairs)
    rules = [
        ("fuse", lambda u, v: zx.fuse(d, u, v)),
        ("identity", lambda u, v: zx.remove_identity(d, u)),
        ("color-change", lambda u, v: zx.color_change(d, u)),
        ("self-loop", lambda u, v: zx.remove_self_loops(d, u)),
        ("copy", lambda u, v: zx.copy_state(d, u, v)),
        ("bialgebra", lambda u, v: zx.bialgebra(d, u, v)),
        ("hopf", lambda u, v: zx.hopf(d, u, v)),
        ("hadamard-pair", lambda u, v: zx.cancel_hadamard_pair(d, u, v)),
    ]
    order = list(range(len(rules)))
    rng.shuffle(order)
    for k in order:
        name, fn = rules[k]
        for u, v in pairs or [(sp[0], sp[0])]:
            try:
                return name, fn(u, v)
            except RewriteError:
                continue
    return None


def graph_state_vec(g, order):
    n = len(order)
    pos = {v: k for k, v in enumerate(order)}
    psi = np.full((2,) * n, 2 ** (-n / 2), complex)
    for u, v in g.edges:
        idx = [slice(None)] * n
        idx[pos[u]] = idx[pos[v]] = 1
        psi[tuple(idx)] *= -1
    return psi.reshape(-1, 1)


# ----- drawing ---------------------------------------------------------------------------------
def draw(ax, d, title):
    g = nx.MultiGraph()
    g.add_nodes_from(d.vertices)
    for e, (a, b, t) in d.edges.items():
        g.add_edge(a, b, key=e)
    fixed = {}
    for k, b in enumerate(d.inputs):
        fixed[b] = (-1.5, (len(d.inputs) - 1) / 2 - k)
    for k, b in enumerate(d.outputs):
        fixed[b] = (1.5, (len(d.outputs) - 1) / 2 - k)
    # Deterministic layered layout: a spider sits left (-0.6) if it touches only inputs, right
    # (+0.6) if only outputs, centre otherwise; spiders in one column are spread vertically in
    # the order of their boundary neighbours.
    pos = dict(fixed)
    column = {}
    for v in d.vertices:
        if v in fixed:
            continue
        nb = [d.other(e, v) for e in d.incident(v)]
        ins, outs = any(n in d.inputs for n in nb), any(n in d.outputs for n in nb)
        x = -0.6 if ins and not outs else 0.6 if outs and not ins else 0.0
        key = np.mean([fixed[n][1] for n in nb if n in fixed]) if any(n in fixed for n in nb) else 0.0
        column.setdefault(x, []).append((-key, v))
    for x, members in column.items():
        members.sort()
        for k, (_, v) in enumerate(members):
            pos[v] = (x, 0.9 * ((len(members) - 1) / 2 - k))
    seen = {}
    for e, (a, b, t) in d.edges.items():
        key = frozenset((a, b))
        rad = 0.35 * seen.get(key, 0) * (-1) ** seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
        color = "#1565c0" if t is EdgeType.HADAMARD else "#37474f"
        if a == b:
            (x, y) = pos[a]
            ax.add_patch(FancyArrowPatch((x - 0.12, y + 0.05), (x + 0.12, y + 0.05), connectionstyle="arc3,rad=-2.2",
                                         arrowstyle="-", color=color, lw=1.4, ls="--"))
            ax.plot(x, y + 0.42, "s", ms=6, color="#ffeb3b", mec="#1565c0")
            continue
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                     color=color, lw=1.4, ls="--" if t is EdgeType.HADAMARD else "-"))
        if t is EdgeType.HADAMARD:
            mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
            ax.plot(mx + rad * dy / 2, my - rad * dx / 2, "s", ms=6, color="#ffeb3b", mec="#1565c0", zorder=3)
    for v, (kind, ph) in d.vertices.items():
        x, y = pos[v]
        if kind is VertexType.BOUNDARY:
            ax.plot(x, y, "o", ms=4, color="k", zorder=3)
            continue
        ax.plot(x, y, "o", ms=17, color="#9ccc65" if kind is VertexType.Z else "#ef5350", mec="#263238", zorder=3)
        if abs(ph) > 1e-12:
            dy = -0.3 if d.loops(v) else 0.3
            ax.text(x, y + dy, f"{ph / math.pi:.2g}π", ha="center", va="top" if dy < 0 else "bottom", fontsize=7, zorder=4)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="600 random rewrite trials")
    args = parser.parse_args()
    n_random = 600 if args.full else N_RANDOM
    plt = common.setup_style()
    rng = common.rng(SEED)
    rows, failures, figure_cases = [], [], []

    for name, d, rewrite, closed in examples():
        before = evaluate(d)
        after_d = rewrite(d)
        after = evaluate(after_d)
        lib_before, lib_after = d.matrix(), after_d.matrix()
        row = {
            "case": name, "kind": "textbook", "factor": str(after_d.trace[-1].factor),
            "after_vs_before": rel(after, before),
            "proportionality_after_over_before": str(np.round(proportionality(after, before), 12)),
            "library_vs_independent_before": rel(lib_before, before),
            "library_vs_independent_after": rel(lib_after, after),
            "closed_form_error": rel(before, closed) if closed is not None else None,
            "spiders_before": len(d.spiders()), "spiders_after": len(after_d.spiders()),
        }
        rows.append(row)
        figure_cases.append((name, d, after_d, row))

    rule_counts = {}
    for t in range(n_random):
        d = random_diagram(rng)
        step = random_rewrite(d, rng)
        if step is None:
            continue
        rule, after_d = step
        before, after = evaluate(d), evaluate(after_d)
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
        rows.append({"case": f"random_{t:04d}", "kind": "random", "rule": rule,
                     "factor": str(after_d.trace[-1].factor), "after_vs_before": rel(after, before),
                     "library_vs_independent_before": rel(d.matrix(), before),
                     "library_vs_independent_after": rel(after_d.matrix(), after)})
    for n in (3, 4, 5):
        for k in range(3):
            g = nx.gnp_random_graph(n, 0.6, seed=int(rng.integers(1 << 30)))
            order = tuple(range(n))
            d = zx.graph_state_diagram(g, order)
            simp = zx.simplify_spiders(d)
            target = graph_state_vec(g, order)
            rows.append({"case": f"graph_state_n{n}_{k}", "kind": "graph-state", "edges": g.number_of_edges(),
                         "diagram_vs_statevector": rel(evaluate(d), target),
                         "after_vs_before": rel(evaluate(simp), target),
                         "simplify_steps": len(simp.trace) - len(d.trace),
                         "library_vs_independent_before": rel(d.matrix(), evaluate(d)),
                         "library_vs_independent_after": rel(simp.matrix(), evaluate(simp))})
    # MBQC branch diagrams: a gflow pattern with XY/XZ/YZ planes, every branch
    pat = q.QubitPattern((0,), [q.N(1), q.N(2), q.E(0, 1), q.E(1, 2), q.M(0, Plane.XY, 0.3),
                                q.M(1, Plane.YZ, 0.8, {0}), q.X(2, {1}), q.Z(2, {0})], (2,))
    for b in q.enumerate_branches(pat):
        dd = zx.pattern_branch_diagram(pat, b.outcomes)
        rows.append({"case": f"mbqc_branch_{b.outcomes}", "kind": "mbqc-branch",
                     "after_vs_before": rel(evaluate(zx.simplify_spiders(dd)), b.operator),
                     "diagram_vs_statevector": rel(evaluate(dd), b.operator),
                     "library_vs_independent_before": rel(dd.matrix(), evaluate(dd))})

    for r in rows:
        bad = [k for k in ("after_vs_before", "library_vs_independent_before", "library_vs_independent_after",
                           "closed_form_error", "diagram_vs_statevector")
               if r.get(k) is not None and not r[k] <= TOL_EXACT]
        if bad:
            failures.append({**r, "problems": "exceeds tolerance: " + ", ".join(bad)})
    n_applied = sum(rule_counts.values())
    if n_applied < min(MIN_RANDOM_APPLIED, n_random // 2):
        failures.append({"case": "random", "problems": f"only {n_applied} random rewrites applied"})
    worst = {k: max((r[k] for r in rows if r.get(k) is not None), default=0.0)
             for k in ("after_vs_before", "library_vs_independent_before", "library_vs_independent_after",
                       "closed_form_error", "diagram_vs_statevector")}
    common.save_result(rows, "R56_zx_calculus", extra={
        "seed": SEED,
        "scope": "QUBIT ZX subset only (no CV-ZX, no pi-copy, no ZX local complementation/pivoting)",
        "backend": "independent einsum tensor evaluator (NumPy complex128) vs photographiq.calculus.zx",
        "precision": "complex128",
        "scalar_convention": "exact scalars tracked by every rewrite; matrices compared with no phase/normalisation freedom",
        "parameters": {"n_random": n_random},
        "acceptance": {"TOL_EXACT": TOL_EXACT, "MIN_RANDOM_APPLIED": MIN_RANDOM_APPLIED},
        "random_rule_counts": rule_counts,
        "worst": worst,
    })

    half = (len(figure_cases) + 1) // 2
    fig, axes = plt.subplots(4, half, figsize=(2.9 * half, 8.0))
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, (name, before, after, row) in enumerate(figure_cases):
        top, col = 2 * (k // half), k % half
        draw(axes[top, col], before, f"({'abcdefgh'[k]}) {name}")
        f = complex(row["factor"].strip("()"))
        fs = f"{f.real:.3g}" if abs(f.imag) < 1e-12 else f"{f:.3g}"
        draw(axes[top + 1, col], after, f"after: scalar x {fs}\ndeviation {row['after_vs_before']:.0e}")
    common.save_figure(fig, "R56_zx_calculus", tight=True)
    plt.close("all")
    common.save_caption("R56_zx_calculus", f"""
**R56 - qubit ZX-calculus subset: rewrites are exact including scalars.** Top: textbook diagrams
(green Z, red X spiders, phases in units of pi; dashed blue edges with yellow boxes are Hadamard
edges; black dots are boundaries, inputs left, outputs right). Bottom: after the library rewrite;
the exact scalar factor recorded by the rewrite and the deviation of the after-matrix from the
before-matrix are shown. Matrices come from an independent einsum evaluator that builds X spiders
directly from |+>/|-> projectors, compared with no global-phase or normalisation freedom. Over all
{len(rows)} checks (textbook rules, {n_applied} random rewrites on random diagrams with rule counts
{rule_counts}, `simplify_spiders` on graph-state diagrams vs CZ statevectors, and MBQC branch diagrams
vs `run_branch` Kraus operators) the worst relative deviation is {worst['after_vs_before']:.1e} (rewrites),
{max(worst['library_vs_independent_before'], worst['library_vs_independent_after']):.1e} (library vs independent
evaluator), {worst['diagram_vs_statevector']:.1e} (graph state / MBQC semantics). Qubit ZX subset only:
no CV-ZX, pi-copy or ZX local complementation/pivoting. Source: R56.
""")
    if failures:
        common.save_csv(failures, "R56_zx_calculus_FAILURES")
        raise AssertionError(f"{len(failures)} R56 check(s) failed; see FAILURES csv")
    common.print_summary("R56 ZX calculus", checks=len(rows), random_applied=n_applied, rules=rule_counts,
                         worst={k: f"{v:.1e}" for k, v in worst.items()})


if __name__ == "__main__":
    main()
