"""R1: Triangle-neuron and MuTA graph anatomy.

Scientific question: Do TriangleNeuron and MuTA build the exact semantic
graph anatomy (node count, edge count, input/output labels, trainable-node
count, causal DAG) that Appendix B's paper-layer construction implies, for
every combination of wire count, requested layers, one_column and
restrict_trainable?

Theory/equations: One paper (n,pivot) layer has 5n nodes (columns 0..4 per
wire) and 4n + 2(n-1) edges (4 wire-chain edges per wire, plus 2 cross edges
per non-pivot wire connecting it to the pivot's column 1). Stacking d paper
layers (d = n_layers if one_column else n_layers*n_wires) with boundary
identification gives |V| = n_wires*(4d+1), |E| = 4*n_wires*d + 2*d*(n_wires-1),
and n_parameters = (3 if restrict_trainable else 4) * n_wires * d.

Functionality tested: TriangleNeuron.graph/input_nodes/output_nodes/
trainable_nodes; MuTA.__init__ graph assembly, flow, dependency_graph,
measurement_order, n_parameters, expected_parameter_count.

Oracle and independence class: A (independent analytic oracle) -- expected
counts are hand-derived closed-form formulas, evaluated independently of the
package's own bookkeeping and compared to what the package actually built.

Exact/approximate/statistical status: exact (integer equality).

Primary metric: max absolute error between analytic and constructed counts
(nodes, edges, trainable parameters) across the full sweep.

Declared acceptance condition: max absolute error == 0 for every swept
configuration, and the dependency DAG must be acyclic with measurement_order
respecting all DAG edges (a topological order).

Expected cost: light (pure graph construction, no simulation).

Manuscript destination: Main text (Fig. 1, triangle/MuTA anatomy) and
Table I (structural definitions).

Scientific limitations: This is a structural/analytic check of the ideal
logical graph only; it says nothing about physical lowering or MentPy
agreement (see R7-R9, R36).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import networkx as nx

from photographiqml import MuTA, TriangleNeuron

EXPERIMENT_ID = "R1"


def analytic_triangle_counts(n_wires, connections):
    connected = [w for w in range(n_wires) if w != 0] if connections is None else list(connections)
    nodes = 5 * n_wires
    edges = 4 * n_wires + 2 * len(connected)
    trainable = 4 * n_wires
    return nodes, edges, trainable


def analytic_muta_counts(n_wires, n_layers, one_column, restrict_trainable):
    d = n_layers if one_column else n_layers * n_wires
    nodes = n_wires * (4 * d + 1)
    edges = 4 * n_wires * d + 2 * d * (n_wires - 1)
    trainable = (3 if restrict_trainable else 4) * n_wires * d
    return nodes, edges, trainable, d


def main():
    plt = common.setup_style()
    rows = []

    # --- TriangleNeuron anatomy -------------------------------------------------
    for n_wires in (1, 2, 3, 4):
        for connections in (None, (), tuple(range(1, n_wires))[:1]):
            if connections == () and n_wires == 1:
                pass
            try:
                triangle = TriangleNeuron(n_wires, pivot=0, connections=connections)
            except ValueError:
                continue
            expected_nodes, expected_edges, expected_trainable = analytic_triangle_counts(
                n_wires, connections
            )
            actual_nodes, actual_edges = len(triangle.graph), triangle.graph.number_of_edges()
            actual_trainable = len(triangle.trainable_nodes)
            rows.append(
                {
                    "level": "triangle",
                    "n_wires": n_wires,
                    "n_layers": None,
                    "one_column": None,
                    "restrict_trainable": None,
                    "connections": str(connections),
                    "expected_nodes": expected_nodes,
                    "actual_nodes": actual_nodes,
                    "expected_edges": expected_edges,
                    "actual_edges": actual_edges,
                    "expected_trainable": expected_trainable,
                    "actual_trainable": actual_trainable,
                    "node_error": abs(expected_nodes - actual_nodes),
                    "edge_error": abs(expected_edges - actual_edges),
                    "trainable_error": abs(expected_trainable - actual_trainable),
                    "input_nodes_ok": triangle.input_nodes == tuple((w, 0) for w in range(n_wires)),
                    "output_nodes_ok": triangle.output_nodes
                    == tuple((w, 4) for w in range(n_wires)),
                }
            )

    # --- MuTA anatomy sweep -------------------------------------------------
    for n_wires in (1, 2, 3):
        for n_layers in (1, 2, 3):
            for one_column in (False, True):
                for restrict_trainable in (False, True):
                    model = MuTA(
                        n_wires,
                        n_layers,
                        one_column=one_column,
                        restrict_trainable=restrict_trainable,
                    )
                    expected_nodes, expected_edges, expected_trainable, d = analytic_muta_counts(
                        n_wires, n_layers, one_column, restrict_trainable
                    )
                    actual_nodes, actual_edges = len(model.graph), model.graph.number_of_edges()
                    dag_acyclic = nx.is_directed_acyclic_graph(model.dependency_graph)
                    positions = {
                        v: i for i, v in enumerate(model.measurement_order + model.output_nodes)
                    }
                    order_respects_dag = all(
                        positions[u] < positions[v] for u, v in model.dependency_graph.edges
                    )
                    rows.append(
                        {
                            "level": "muta",
                            "n_wires": n_wires,
                            "n_layers": n_layers,
                            "one_column": one_column,
                            "restrict_trainable": restrict_trainable,
                            "connections": None,
                            "expected_nodes": expected_nodes,
                            "actual_nodes": actual_nodes,
                            "expected_edges": expected_edges,
                            "actual_edges": actual_edges,
                            "expected_trainable": expected_trainable,
                            "actual_trainable": model.n_parameters,
                            "node_error": abs(expected_nodes - actual_nodes),
                            "edge_error": abs(expected_edges - actual_edges),
                            "trainable_error": abs(expected_trainable - model.n_parameters),
                            "paper_depth": model.paper_depth,
                            "paper_depth_ok": model.paper_depth == d,
                            "dag_acyclic": dag_acyclic,
                            "order_respects_dag": order_respects_dag,
                            "input_nodes_ok": model.input_nodes
                            == tuple((w, 0) for w in range(n_wires)),
                            "output_nodes_ok": model.output_nodes
                            == tuple((w, 4 * model.paper_depth) for w in range(n_wires)),
                        }
                    )

    max_node_error = max(r["node_error"] for r in rows)
    max_edge_error = max(r["edge_error"] for r in rows)
    max_trainable_error = max(r["trainable_error"] for r in rows)
    all_dag_ok = all(r.get("dag_acyclic", True) and r.get("order_respects_dag", True) for r in rows)
    all_io_ok = all(r["input_nodes_ok"] and r["output_nodes_ok"] for r in rows)
    passed = (
        max_node_error == 0
        and max_edge_error == 0
        and max_trainable_error == 0
        and all_dag_ok
        and all_io_ok
    )
    status = "pass" if passed else "fail"

    common.save_result(
        rows,
        "R1_triangle_anatomy",
        extra={
            "protocol": "TriangleNeuron/MuTA graph anatomy vs. closed-form node/edge/parameter counts",
            "oracle_class": "A",
            "status_category": "exact",
            "acceptance_condition": "max_node_error == max_edge_error == max_trainable_error == 0; DAG acyclic and topologically consistent",
            "max_node_error": max_node_error,
            "max_edge_error": max_edge_error,
            "max_trainable_error": max_trainable_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    muta_rows = [r for r in rows if r["level"] == "muta"]
    for ax, key, title in zip(
        axes,
        ("node_error", "edge_error", "trainable_error"),
        ("Node count error", "Edge count error", "Trainable-parameter error"),
    ):
        values = [r[key] for r in muta_rows]
        colors = [
            common.COLORS["photographiqml"] if v == 0 else common.COLORS["piquasso"] for v in values
        ]
        # All-zero values render invisibly as bars; use markers so every
        # tested configuration is visibly plotted, not just an empty axes.
        ax.scatter(range(len(values)), values, color=colors, marker="o", s=18, zorder=3)
        ax.set_title(f"{title} ({len(values)} configurations)")
        ax.set_xlabel("configuration")
        ax.set_ylabel("|analytic - actual|")
        ax.set_xticks([])
        ax.axhline(0, color=common.COLORS["acceptance"], linewidth=0.8)
    fig.suptitle(
        f"R1: MuTA graph anatomy vs. analytic formulas ({len(muta_rows)} configurations, status={status})"
    )
    common.save_figure(fig, "R1_triangle_anatomy")
    plt.close(fig)

    common.print_summary(
        "R1 triangle/MuTA anatomy",
        configurations=len(rows),
        max_node_error=max_node_error,
        max_edge_error=max_edge_error,
        max_trainable_error=max_trainable_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R1 failed: node={max_node_error} edge={max_edge_error} trainable={max_trainable_error}"
        )


if __name__ == "__main__":
    main()
