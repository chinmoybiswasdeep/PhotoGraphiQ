"""R62 (O2): wire shortening -- what measured intermediate cluster modes do to the graph.

Hypotheses. (A) On a finite-squeezing linear CV cluster Z = A + i e^{-2r} I, p-homodyne of one
interior node k (Schur complement Z' = Z_rr - z z^T / Z_kk, MFvL Sec. III) couples its
neighbours by -1/Z_kk = i e^{2r}: a diverging *imaginary* (squeezing-like) entry, not a CZ
edge. p-homodyne of two adjacent interior nodes couples the outer neighbours with a real edge
that tends to -1 (a CZ of weight -1) as r grows, with corrections O(e^{-2r}): the standard
two-node wire shortening holds only in the ideal limit. (B) In executable compiled patterns, a
chain of corrected teleportation steps whose product is T(k_L)...T(k_1) can be replaced by at
most five steps (four or five by the compiler's decomposition; 0-3 when the product has the
special forms checked exactly), preserving the logical map exactly while changing the noise.

Method. (A) 5-node path, r in [0, 3]; two independent computations of the post-measurement
graph: ``GaussianGraph.measure_p`` and conditioning of the diagram semantics
(``to_gaussian_graph`` with fixed outcomes). (B) 60 random single-mode gate sequences (1-6
rotations / squeezers / shears, compile_circuit at r = 1), resynthesised in logical mode.

Acceptance: (A) the two computations agree to 1e-9; the two-node edge approaches -1 with
|Re Z_ac + 1| decreasing in r. (B) logical residual <= 1e-9 for all, steps after <= 5.
"""

import numpy as np
from cvzx_support import COLORS, plt, save

import photographiq as pg
from photographiq.calculus import GaussianGraph
from photographiq.calculus.cv_diagrams import CZ, CVDiagram, Homodyne, Prepare, from_pattern, to_gaussian_graph
from photographiq.calculus.cv_diagrams.optimize import compare, resource_cost
from photographiq.calculus.cv_diagrams.rules import TeleportResynthesis, teleport_steps

NAME = "R62_cvzx_wire_shortening"


def cluster_diagram(n, r, measured):
    boxes = [Prepare.squeezed(j, r) for j in range(n)] + [CZ(j, j + 1, 1.0) for j in range(n - 1)]
    boxes += [Homodyne(k, np.pi / 2, f"m{k}") for k in measured]
    return CVDiagram((), tuple(boxes), tuple(j for j in range(n) if j not in measured))


def main():
    rows = []
    adjacency = np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1)
    for r in np.linspace(0, 3, 13):
        graph = GaussianGraph.cluster(adjacency, r)
        for measured in ((2,), (1, 2)):
            g = graph
            for k in measured:
                g = g.measure_p(k, 0.0)
            d = to_gaussian_graph(cluster_diagram(5, r, measured), {f"m{k}": 0.0 for k in measured})
            a, c = (1, 3) if measured == (2,) else (0, 3)
            ia, ic = g.nodes.index(a), g.nodes.index(c)
            rows.append({"study": "graph", "r": float(r), "measured": len(measured),
                         "re_edge": float(g.Z[ia, ic].real), "im_edge": float(g.Z[ia, ic].imag),
                         "agreement": float(np.abs(g.Z - d.Z).max() / max(1, np.abs(g.Z).max()))})
    rng = np.random.default_rng(62)
    rule = TeleportResynthesis()
    for t in range(60):
        c = pg.Circuit(1)
        for _ in range(int(rng.integers(1, 7))):
            kind = rng.integers(3)
            if kind == 0:
                c.rotate(0, rng.uniform(-np.pi, np.pi))
            elif kind == 1:
                c.squeeze(0, 0.4 * rng.normal())
            else:
                c.gaussian(0, [[1.0, 0.0], [rng.normal(), 1.0]])
        d = from_pattern(c.compile(squeezing=1.0))
        before = len(teleport_steps(d))
        matches = rule.find(d)
        out = rule.apply(d, matches[0]) if matches else d
        report = compare(d, out)
        rows.append({"study": "resynthesis", "instance": t, "gates": len(c.gates), "steps_before": before,
                     "steps_after": len(teleport_steps(out)), "logical_residual": max(report.channel_X_residual, report.displacement_residual),
                     "noise_before": resource_cost(d)["added_noise"], "noise_after": resource_cost(out)["added_noise"]})
    graph_rows = [q for q in rows if q["study"] == "graph"]
    res_rows = [q for q in rows if q["study"] == "resynthesis"]
    two = [q for q in graph_rows if q["measured"] == 2]
    gap = [abs(q["re_edge"] + 1) for q in two]
    checks = {"graph_agreement": max(q["agreement"] for q in graph_rows) <= 1e-9,
              "two_node_limit": all(b <= a + 1e-12 for a, b in zip(gap, gap[1:])) and gap[-1] < 1e-2,
              "logical": max(q["logical_residual"] for q in res_rows) <= 1e-9,
              "at_most_five": max(q["steps_after"] for q in res_rows) <= 5}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    one = [q for q in graph_rows if q["measured"] == 1]
    ax = axes[0, 0]
    ax.semilogy([q["r"] for q in one], [q["im_edge"] for q in one], "o-", color=COLORS["physical"], label="Im Z (one p measurement)")
    ax.semilogy([q["r"] for q in one], np.exp(2 * np.array([q["r"] for q in one])), "k:", lw=0.8, label="$e^{2r}$")
    ax.set(xlabel="Resource squeezing r", ylabel="Induced neighbour entry", title="One interior p measurement")
    ax.legend(fontsize=6)
    ax = axes[0, 1]
    ax.plot([q["r"] for q in two], [q["re_edge"] for q in two], "o-", color=COLORS["optimised"], label="Re Z (edge weight)")
    ax.plot([q["r"] for q in two], [q["im_edge"] for q in two], "s--", color=COLORS["reference"], label="Im Z")
    ax.axhline(-1, color="k", ls=":", lw=0.7)
    ax.set(xlabel="Resource squeezing r", ylabel="Induced edge between outer nodes", title="Two adjacent p measurements")
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    b, a = np.array([q["steps_before"] for q in res_rows]), np.array([q["steps_after"] for q in res_rows])
    ax.scatter(b + rng.uniform(-0.2, 0.2, len(b)), a + rng.uniform(-0.15, 0.15, len(a)), s=8, color=COLORS["optimised"])
    ax.plot([0, b.max()], [0, b.max()], "k-", lw=0.5)
    ax.set(xlabel="Teleportation steps before", ylabel="Steps after resynthesis", title="Executable wire shortening (60 programs)")
    ax = axes[1, 1]
    nb, na = np.array([q["noise_before"] for q in res_rows]), np.array([q["noise_after"] for q in res_rows])
    ax.loglog(nb, na, "o", ms=3, color=COLORS["logical"])
    lim = [min(nb.min(), na.min()), max(nb.max(), na.max())]
    ax.plot(lim, lim, "k-", lw=0.5)
    ax.set(xlabel="Added noise before (vacuum units)", ylabel="Added noise after", title="Noise is not preserved (logical rewrite)")
    save(rows, NAME, configuration={"path_nodes": 5, "r_grid": [0, 3, 13], "programs": 60, "seed": 62, "squeezing_B": 1.0},
         validation={"status": status, "checks": checks, "references": ["GaussianGraph.measure_p (MFvL)", "diagram conditioning"]},
         figure=fig, caption="Wire shortening: graph view of interior p measurements and executable teleport-chain resynthesis.")
    print(f"R62 status={status} {checks}; steps {b.sum()} -> {a.sum()}, noise increases in {int((na > nb + 1e-12).sum())} programs")
    if status != "pass":
        raise AssertionError("R62 acceptance failed")


if __name__ == "__main__":
    main()
