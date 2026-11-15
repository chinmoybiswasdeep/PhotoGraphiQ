"""R7: Semantic topology agreement between MuTA and independently constructed
MentPy circuits.

Scientific question: For every combination of wire count, requested layers,
one_column and restrict_trainable, does MentPy's own `templates.muta` build a
graph that is edge-for-edge isomorphic to PhotoGraphiQML's MuTA.graph once
both are expressed in the same semantic (wire, column) coordinates, and do
input/output node orderings agree?

Theory/equations: GraphState equality via isomorphism is insufficient for
semantic comparison (docs/research/mentpy-audit.md); the correct invariant is
edge agreement under the *causal* (wire, column) labeling recovered by
walking MentPy's own flow function from each input node
(photographiqml.validation.mentpy_reference), not an arbitrary vertex
relabeling.

Functionality tested: MuTA graph construction (ansatz/muta.py) vs.
mentpy.templates.muta, compared via validation.mentpy_reference's semantic
mapping (built by walking reference.flow(node) from each input, an operation
this experiment does not repeat itself but whose *result* -- the edge set --
it independently re-derives and checks against MuTA.graph.edges).

Oracle and independence class: B (independent external implementation --
MentPy is a separate Apache-2.0 package validating qubit structure only).

Exact/approximate/statistical status: exact (integer edge-set equality).

Primary metric: symmetric-difference size between the semantically mapped
MentPy edge set and MuTA.graph.edges; input/output ordering mismatches.

Declared acceptance condition: symmetric difference == 0 and input/output
orderings agree for every swept configuration with connections=None
(mentpy_reference only supports full connectivity).

Expected cost: light (graph construction only, no state simulation).

Manuscript destination: Main text (Fig. 2/3, MentPy cross-validation) and
Table II (structural agreement).

Scientific limitations: MentPy validates the ideal logical qubit layer only
(never a finite-GKP oracle); custom `connections` (non-default triangle
wiring) has no MentPy counterpart and is excluded from this sweep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import MuTA
from photographiqml.validation import mentpy_reference

EXPERIMENT_ID = "R7"


def main():
    plt = common.setup_style()
    rows = []
    for n_wires in (1, 2, 3):
        for n_layers in (1, 2):
            for one_column in (False, True):
                for restrict_trainable in (False, True):
                    model = MuTA(
                        n_wires,
                        n_layers,
                        one_column=one_column,
                        restrict_trainable=restrict_trainable,
                    )
                    reference, mapping = mentpy_reference(model)
                    mentpy_edges = {
                        frozenset((mapping[u], mapping[v])) for u, v in reference.graph.edges
                    }
                    muta_edges = {frozenset(e) for e in model.graph.edges}
                    symmetric_difference = mentpy_edges ^ muta_edges
                    mapped_inputs = tuple(mapping[v] for v in reference.input_nodes)
                    mapped_outputs = tuple(mapping[v] for v in reference.output_nodes)
                    rows.append(
                        {
                            "n_wires": n_wires,
                            "n_layers": n_layers,
                            "one_column": one_column,
                            "restrict_trainable": restrict_trainable,
                            "mentpy_nodes": len(reference.graph.nodes),
                            "muta_nodes": len(model.graph.nodes),
                            "mentpy_edges": len(mentpy_edges),
                            "muta_edges": len(muta_edges),
                            "symmetric_difference": len(symmetric_difference),
                            "inputs_match": mapped_inputs == model.input_nodes,
                            "outputs_match": mapped_outputs == model.output_nodes,
                        }
                    )

    max_symdiff = max(r["symmetric_difference"] for r in rows)
    all_io_ok = all(r["inputs_match"] and r["outputs_match"] for r in rows)
    status = "pass" if (max_symdiff == 0 and all_io_ok) else "fail"

    common.save_result(
        rows,
        "R7_semantic_topology",
        extra={
            "protocol": "MuTA.graph vs. MentPy templates.muta, compared via semantic (wire,column) mapping",
            "oracle_class": "B",
            "status_category": "exact",
            "acceptance_condition": "symmetric_difference == 0 and input/output ordering matches for every configuration",
            "max_symmetric_difference": max_symdiff,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [
        f"n{r['n_wires']}L{r['n_layers']}{'1c' if r['one_column'] else 'nc'}{'r' if r['restrict_trainable'] else 'f'}"
        for r in rows
    ]
    colors = [
        common.COLORS["photographiqml"]
        if r["symmetric_difference"] == 0 and r["inputs_match"] and r["outputs_match"]
        else common.COLORS["piquasso"]
        for r in rows
    ]
    # All-zero values render invisibly as bars; use markers so every tested
    # configuration is visibly plotted, not just an empty axes.
    ax.scatter(range(len(rows)), [r["symmetric_difference"] for r in rows], color=colors, zorder=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set(
        title=f"R7: MuTA vs. MentPy semantic edge-set agreement, {len(rows)} configurations (status={status})",
        ylabel="|symmetric difference|",
    )
    common.save_figure(fig, "R7_semantic_topology")
    plt.close(fig)

    common.print_summary(
        "R7 semantic topology",
        n_configs=len(rows),
        max_symmetric_difference=max_symdiff,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R7 failed: max_symdiff={max_symdiff} all_io_ok={all_io_ok}")


if __name__ == "__main__":
    main()
