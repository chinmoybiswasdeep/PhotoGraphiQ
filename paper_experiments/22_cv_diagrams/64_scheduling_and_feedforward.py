"""R64 (O4 + O5): measurement scheduling and feed-forward simplification on CV clusters.

Hypotheses. (O5) In patterns built from a CV-flow certificate (``cvflow.flow_pattern``), every
measurement displaces every later target, so an unmeasured node accumulates many wire-adjacent
corrections; merging them (rule ``feedforward-merge``) is exact at the instrument level and
reduces the number of feed-forward commands while the record dependencies are unchanged.
(O4) Commuting independent boxes changes the sequential order but never the dependency depth
(the depth is a property of the causal DAG, which commutation preserves). The depth *is*
reduced by re-deriving the corrections from the layered Booth-Markham certificate found by
``find_cv_flow`` (nodes of one layer need no corrections from each other): different
corrections, same ideal map; whether the averaged physical channel and the conditional
instrument also agree is measured, not assumed.

Method. w x c grid clusters (w = 1..3, c = 3..7), random shears N(0, 0.5^2) (seed 64), r = 1.
O5: physical-mode search with objective feedforward_commands. O4: 50 random valid commutations
(depth tracked); flow-layered pattern ``flows.cv.corrected_pattern(order, correction_map)``.
Executed checks: ``analysis.gaussian_channel`` of the recompiled patterns.

Acceptance: O5 exact instrument and executed channel <= 1e-9, commands never increase;
O4 commutation leaves depth unchanged; layered patterns are logically equivalent (<= 1e-9).
"""

import networkx as nx
import numpy as np
from cvzx_support import COLORS, plt, save

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import compare, from_pattern, optimize_cv_pattern, resource_cost, to_pattern
from photographiq.calculus.cv_diagrams.rules import Commute
from photographiq.cvflow import flow_pattern
from photographiq.flows import find_cv_flow
from photographiq.flows.cv import corrected_pattern
from photographiq.ir import HomodyneSpec, Model, OpenGraph

NAME = "R64_cvzx_scheduling_feedforward"


def cluster(w, c, rng):
    g = nx.grid_2d_graph(w, c)
    G = nx.Graph()
    G.add_nodes_from(sorted(g.nodes))
    G.add_edges_from(g.edges)
    ins, outs = tuple((i, 0) for i in range(w)), tuple((i, c - 1) for i in range(w))
    order = [(i, j) for j in range(c - 1) for i in range(w)]
    shears = {n: float(0.5 * rng.normal()) for n in order}
    return G, ins, outs, order, shears


def main():
    rng = np.random.default_rng(64)
    rows = []
    for w in (1, 2, 3):
        for c in range(3, 8):
            G, ins, outs, order, shears = cluster(w, c, rng)
            graph = pg.CVGraph(G, squeezing=1.0, inputs=ins, outputs=outs)
            pattern = flow_pattern(graph, order, shears=shears)
            diagram = from_pattern(pattern)
            ff = optimize_cv_pattern(pattern, equivalence="physical", objective="feedforward_commands", max_steps=400)
            executed = gaussian_channel(to_pattern(ff.best))
            original = gaussian_channel(pattern)
            # O4: random commutations
            d, depth0 = diagram, resource_cost(diagram)["measurement_depth"]
            commute, depth_changes, moves = Commute(), 0, 0
            for _ in range(50):
                matches = commute.find(d)
                if not matches:
                    break
                d = commute.apply(d, matches[int(rng.integers(len(matches)))])
                moves += 1
                depth_changes += resource_cost(d)["measurement_depth"] != depth0
            og = OpenGraph(Model.CV_GAUSSIAN, G, ins, outs, {n: HomodyneSpec(shears[n]) for n in order}, {n: 1.0 for n in G.nodes})
            flow = find_cv_flow(og)
            layered = from_pattern(corrected_pattern(graph, flow.measurement_order, flow.correction_map, shears))
            rep = compare(diagram, layered)
            rows.append({"wires": w, "columns": c, "measurements": len(order),
                         "ff_commands_before": ff.original_cost["feedforward_commands"],
                         "ff_commands_after": ff.best_cost["feedforward_commands"],
                         "ff_dependencies_before": ff.original_cost["feedforward_dependencies"],
                         "ff_dependencies_after": ff.best_cost["feedforward_dependencies"],
                         "ff_instrument_exact": bool(ff.report.conditional_equivalent),
                         "ff_executed_residual": float(max(np.abs(executed.matrix - original.matrix).max(), np.abs(executed.noise - original.noise).max())),
                         "commutations": moves, "depth_changes_under_commutation": depth_changes,
                         "flow_status": flow.status, "flow_layers": len(flow.layers),
                         "depth_flow_pattern": depth0, "depth_layered": resource_cost(layered)["measurement_depth"],
                         "layered_logical": bool(rep.logical_equivalent), "layered_physical": bool(rep.physical_equivalent),
                         "layered_conditional": rep.conditional_equivalent, "layered_Y_residual": rep.channel_Y_residual,
                         "layered_instrument_residual": rep.instrument_residual})
        print(f"w={w}: ff {sum(r['ff_commands_before'] for r in rows if r['wires'] == w)} -> "
              f"{sum(r['ff_commands_after'] for r in rows if r['wires'] == w)}; depth "
              f"{sum(r['depth_flow_pattern'] for r in rows if r['wires'] == w)} -> {sum(r['depth_layered'] for r in rows if r['wires'] == w)}", flush=True)
    checks = {"ff_exact": all(r["ff_instrument_exact"] for r in rows),
              "ff_executed": max(r["ff_executed_residual"] for r in rows) <= 1e-9,
              "ff_not_worse": all(r["ff_commands_after"] <= r["ff_commands_before"] for r in rows),
              "commutation_preserves_depth": sum(r["depth_changes_under_commutation"] for r in rows) == 0,
              "layered_logical": all(r["layered_logical"] for r in rows)}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axes[0]
    for w, colour in zip((1, 2, 3), (COLORS["reference"], COLORS["optimised"], COLORS["logical"])):
        sub = [r for r in rows if r["wires"] == w]
        ax.plot([r["measurements"] for r in sub], [r["ff_commands_before"] for r in sub], "o--", color=colour, mfc="none")
        ax.plot([r["measurements"] for r in sub], [r["ff_commands_after"] for r in sub], "o-", color=colour, label=f"{w} wire{'s' if w > 1 else ''}")
    ax.set(xlabel="Measurements", ylabel="Feed-forward commands (dashed: before)", title="Feed-forward merge (exact)")
    ax.legend(fontsize=6)
    ax = axes[1]
    for w, colour in zip((1, 2, 3), (COLORS["reference"], COLORS["optimised"], COLORS["logical"])):
        sub = [r for r in rows if r["wires"] == w]
        ax.plot([r["measurements"] for r in sub], [r["depth_flow_pattern"] for r in sub], "s--", color=colour, mfc="none")
        ax.plot([r["measurements"] for r in sub], [r["depth_layered"] for r in sub], "s-", color=colour)
    ax.set(xlabel="Measurements", ylabel="Measurement depth (dashed: sequential)", title="Flow-layered schedule")
    ax = axes[2]
    ax.semilogy([r["measurements"] for r in rows], [max(r["layered_Y_residual"], 1e-17) for r in rows], "o", color=COLORS["physical"], label="averaged noise Y")
    ax.semilogy([r["measurements"] for r in rows], [max(r["layered_instrument_residual"] or 1e-17, 1e-17) for r in rows], "x", color=COLORS["logical"], label="conditional instrument")
    ax.set(xlabel="Measurements", ylabel="Residual vs sequential pattern", title="What layering preserves")
    ax.legend(fontsize=6)
    save(rows, NAME, configuration={"wires": [1, 2, 3], "columns": [3, 7], "shear_sd": 0.5, "squeezing": 1.0, "seed": 64, "commutations": 50},
         validation={"status": status, "checks": checks},
         figure=fig, caption="Feed-forward merging and flow-layered measurement scheduling on CV grid clusters.")
    print(f"R64 status={status} {checks}")
    if status != "pass":
        raise AssertionError("R64 acceptance failed")


if __name__ == "__main__":
    main()
