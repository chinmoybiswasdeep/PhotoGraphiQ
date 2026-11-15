"""R8: Flow and dependency-order agreement against MentPy, comparing causal
validity rather than exact tie-breaking order.

Scientific question: Does MentPy's own causal-flow successor/correction sets
(reference.flow(node), reference.flow.correction_op(node)) agree with MuTA's
production correction map (model.corrections), and is MentPy's own
measurement_order consistent with MuTA's dependency DAG once both are
expressed in the same semantic coordinates?

Theory/equations: docs/research/mentpy-audit.md: "Flow permits multiple total
orders; compare causal validity rather than requiring identical tie
breaking." This experiment therefore checks (i) X/Z correction-target set
agreement per node (an order-independent invariant) and (ii) that MentPy's
own valid measurement_order respects every edge of MuTA's dependency_graph
(a necessary condition for causal consistency, not a demand for identical
ordering of causally independent nodes).

Functionality tested: Flow.__call__, Flow.correction_op (mentpy), vs.
MuTA.corrections and MuTA.dependency_graph (photographiqml).

Oracle and independence class: B (independent external implementation).

Exact/approximate/statistical status: exact (set equality / DAG-consistency,
both discrete).

Primary metric: number of nodes whose mapped X-correction target or Z-target
set disagrees with model.corrections; number of dependency_graph edges
violated by MentPy's own measurement_order.

Declared acceptance condition: both counts are 0 for every configuration.

Expected cost: light.

Manuscript destination: Main text (Fig. 2/3 supporting causal-flow
agreement); Table II.

Scientific limitations: Restricted to configurations where MentPy finds a
causal flow (true for every MuTA configuration tested here, since the paper
ansatz is causal-flow-friendly by construction); gflow/pflow fallbacks are
not separately exercised.

Documented convention difference (not a defect): MentPy's raw
Flow.correction_op(node) reports the *full* Pauli frame over every graph
qubit, including a formal Z-component on the just-measured node itself
(since that node is a graph-neighbor of its own flow successor, it appears
in odd_neighborhood({successor})). MuTA.corrections explicitly excludes the
measured node (`z = frozenset(graph.neighbors(successor)) - {node}` in
ansatz/muta.py), since a Z correction on an already destructively-measured
qubit has no physical effect on the surviving register. This experiment
subtracts the measured node from MentPy's mapped Z-target set before
comparing, and records that subtraction explicitly (never silently) in the
saved JSON's `self_correction_dropped` field.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import MuTA
from photographiqml.validation import mentpy_reference

EXPERIMENT_ID = "R8"


def main():
    plt = common.setup_style()
    rows = []
    for n_wires in (1, 2, 3):
        for n_layers in (1, 2):
            for one_column in (False, True):
                model = MuTA(n_wires, n_layers, one_column=one_column)
                reference, mapping = mentpy_reference(model)
                reverse = {v: k for k, v in mapping.items()}
                order_violations = 0
                correction_mismatches = 0
                n_measured = len(model.corrections)
                self_correction_dropped = 0
                mentpy_order_positions = {
                    node: i for i, node in enumerate(reference.measurement_order)
                }
                for source, (successor, z_targets) in model.corrections.items():
                    mentpy_node = reverse[source]
                    # MentPy's own causal order must place this node before its
                    # mapped successor (a necessary consequence of any valid flow).
                    successor_mentpy = reverse[successor]
                    if (
                        mentpy_order_positions[mentpy_node]
                        >= mentpy_order_positions[successor_mentpy]
                    ):
                        order_violations += 1
                    for target in z_targets:
                        target_mentpy = reverse[target]
                        if (
                            mentpy_order_positions[mentpy_node]
                            >= mentpy_order_positions[target_mentpy]
                        ):
                            order_violations += 1
                    # Correction-set agreement: MentPy's correction_op X-part must
                    # be exactly {successor}; Z-part must equal z_targets, mapped.
                    n_nodes = reference.graph.number_of_nodes()
                    pauli = reference.flow.correction_op(mentpy_node)
                    x_bits = set(pauli.matrix[0, :n_nodes].nonzero()[0].tolist())
                    z_bits = set(pauli.matrix[0, n_nodes:].nonzero()[0].tolist())
                    x_targets_semantic = {mapping[v] for v in x_bits}
                    z_targets_semantic = {mapping[v] for v in z_bits}
                    if source in z_targets_semantic:
                        z_targets_semantic = z_targets_semantic - {source}
                        self_correction_dropped += 1
                    if x_targets_semantic != {successor} or z_targets_semantic != set(z_targets):
                        correction_mismatches += 1
                rows.append(
                    {
                        "n_wires": n_wires,
                        "n_layers": n_layers,
                        "one_column": one_column,
                        "n_measured": n_measured,
                        "order_violations": order_violations,
                        "correction_mismatches": correction_mismatches,
                        "self_correction_dropped": self_correction_dropped,
                    }
                )

    max_order_violations = max(r["order_violations"] for r in rows)
    max_correction_mismatches = max(r["correction_mismatches"] for r in rows)
    status = "pass" if (max_order_violations == 0 and max_correction_mismatches == 0) else "fail"

    common.save_result(
        rows,
        "R8_flow_dependency_order",
        extra={
            "protocol": "MentPy causal flow/correction_op vs. MuTA.corrections and dependency_graph",
            "oracle_class": "B",
            "status_category": "exact",
            "acceptance_condition": "order_violations == 0 and correction_mismatches == 0 for every configuration",
            "max_order_violations": max_order_violations,
            "max_correction_mismatches": max_correction_mismatches,
            "status": status,
            "convention_note": "MentPy's raw correction_op Z-part includes a formal self-correction on the measured node itself; this is subtracted before comparison (see self_correction_dropped per row) as documented in the docstring, not treated as a defect",
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = [f"n{r['n_wires']}L{r['n_layers']}{'1c' if r['one_column'] else 'nc'}" for r in rows]
    for ax, key, title in zip(
        axes,
        ("order_violations", "correction_mismatches"),
        ("Causal-order violations", "X/Z correction-set mismatches"),
    ):
        colors = [
            common.COLORS["photographiqml"] if r[key] == 0 else common.COLORS["piquasso"]
            for r in rows
        ]
        # All-zero values render invisibly as bars; use markers so every
        # tested configuration is visibly plotted, not just an empty axes.
        ax.scatter(range(len(rows)), [r[key] for r in rows], color=colors, zorder=3)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_title(f"{title} ({len(rows)} configurations)")
    fig.suptitle(f"R8: flow/dependency-order agreement with MentPy (status={status})")
    common.save_figure(fig, "R8_flow_dependency_order")
    plt.close(fig)

    common.print_summary(
        "R8 flow/dependency order",
        n_configs=len(rows),
        max_order_violations=max_order_violations,
        max_correction_mismatches=max_correction_mismatches,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R8 failed: order_violations={max_order_violations} correction_mismatches={max_correction_mismatches}"
        )


if __name__ == "__main__":
    main()
