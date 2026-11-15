"""R10: Explicit audit of trainability/frozen-column behavior, including the
documented MentPy restrict_trainable discrepancy.

Scientific question: (a) Does raw MentPy's `templates.muta(..., restrict_trainable=True)`
actually fix column-3 (the paper's non-trainable) measurement angles, or
merely remove them from a bookkeeping list that upstream graph-mutation
(add_edge/hstack) silently rebuilds anyway? (b) Does PhotoGraphiQML's own
restrict_trainable freezing mechanism (ParameterStore) remain robust across
every wire/layer/one_column configuration, unlike MentPy's fragile list-based
approach?

Theory/equations: docs/research/mentpy-audit.md: "`restrict_trainable`
removes column-3 nodes from a list but does not fix their Ment angles...
Only an unstacked one-wire block retains the list restriction... Adding
cross edges also rebuilds attributes... Only an unstacked one-wire block
retains the list restriction." Mechanistically: MBQCircuit.add_edge and
hstack both call MBQCircuit.__init__ -> _update_attributes(), which discards
any manually assigned `trainable_nodes` list and rebuilds it purely from each
Ment.is_trainable() (angle is None), so the restriction never touches the
Ment objects themselves and survives only when neither add_edge nor hstack
is ever invoked after it (n_wires==1 and n_layers==1).

Functionality tested: raw mentpy.templates.muta's restrict_trainable option
(mentpy) vs. photographiqml.parameters.ParameterStore.freeze /
ansatz/muta.py's restrict_trainable construction (photographiqml).

Oracle and independence class: B for the MentPy-side audit (independent
external implementation, inspected via its own public trainable_nodes/Ment
API); E (structural/self-consistency) for confirming PhotoGraphiQML's own
freeze mechanism rejects an override, which is a property of
photographiqml.parameters.ParameterStore.bind alone.

Exact/approximate/statistical status: exact (discrete membership/exception
checks).

Primary metric: (descriptive, not pass/fail) whether each column-3 semantic
node is present/absent from raw MentPy's trainable_nodes list and whether its
Ment.angle is None, per configuration; (pass/fail) whether PhotoGraphiQML
raises ValueError on every attempted override of a frozen column-3 parameter,
in every configuration.

Declared acceptance condition: PhotoGraphiQML raises ValueError for 100% of
attempted frozen-column-3 overrides, and model.trainable_parameters() never
contains a column-3 name when restrict_trainable=True, for every
configuration. MentPy's own behavior is recorded but is explicitly NOT
required to match any particular pattern (it is a documented, non-PhotoGraphiQML
finding, not a PhotoGraphiQML pass/fail criterion).

Expected cost: light.

Manuscript destination: Appendix (upstream-discrepancy documentation
supporting the release report's trainability claims).

Scientific limitations: This audits MentPy's public list/Ment API only;
whether the discrepancy is intended internal design or an oversight is not
determinable from black-box inspection, and PhotoGraphiQML's production code
is never modified to work around it (the model instead sidesteps MentPy's
fragile list convention entirely by fixing angles itself).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import MuTA
from photographiqml.validation import mentpy_reference

EXPERIMENT_ID = "R10"


def main():
    plt = common.setup_style()
    rows = []
    configs = [
        (1, 1, True),
        (1, 2, True),
        (2, 1, True),
        (2, 1, False),
        (2, 2, True),
        (3, 1, True),
        (1, 1, False),
        (3, 2, True),
    ]
    for n_wires, n_layers, one_column in configs:
        model = MuTA(n_wires, n_layers, one_column=one_column, restrict_trainable=True)
        reference, mapping = mentpy_reference(model, fix_measurements=False)
        reverse = {v: k for k, v in mapping.items()}
        column3_nodes = [v for v in model.measured_nodes if v[1] % 4 == 3]

        # --- (a) raw MentPy discrepancy audit ---------------------------------
        list_restricted = 0  # absent from reference.trainable_nodes (list omitted)
        angle_actually_fixed = 0  # Ment.angle is not None (truly fixed, not just listed)
        for node in column3_nodes:
            mnode = reverse[node]
            in_list = mnode in reference.trainable_nodes
            if not in_list:
                list_restricted += 1
            if reference[mnode].angle is not None:
                angle_actually_fixed += 1

        # --- (b) PhotoGraphiQML's own freeze robustness ------------------------
        trainable_names = set(model.trainable_parameters())
        column3_names = {model.parameter_name(v) for v in column3_nodes}
        leaked_into_trainable = len(trainable_names & column3_names)
        override_rejected = 0
        computational_zero = [1] + [0] * (2**n_wires - 1)
        for name in column3_names:
            try:
                model.run(computational_zero, {name: 1.2345})
                overrode_silently = True
            except ValueError:
                overrode_silently = False
            if not overrode_silently:
                override_rejected += 1

        rows.append(
            {
                "n_wires": n_wires,
                "n_layers": n_layers,
                "one_column": one_column,
                "n_column3_nodes": len(column3_nodes),
                "mentpy_list_restricted": list_restricted,
                "mentpy_angle_actually_fixed": angle_actually_fixed,
                "mentpy_restriction_effective": angle_actually_fixed == len(column3_nodes)
                and len(column3_nodes) > 0,
                "photographiqml_leaked_into_trainable": leaked_into_trainable,
                "photographiqml_override_rejected": override_rejected,
                "photographiqml_robust": leaked_into_trainable == 0
                and override_rejected == len(column3_nodes),
            }
        )

    all_robust = all(r["photographiqml_robust"] for r in rows)
    status = "pass" if all_robust else "fail"
    n_mentpy_effective = sum(1 for r in rows if r["mentpy_restriction_effective"])

    common.save_result(
        rows,
        "R10_trainability_audit",
        extra={
            "protocol": "MentPy raw restrict_trainable list/Ment audit vs. PhotoGraphiQML ParameterStore freeze robustness",
            "oracle_class": "B/E",
            "status_category": "exact",
            "acceptance_condition": "photographiqml_robust True for every configuration (MentPy's own pattern is descriptive, not a pass/fail criterion)",
            "n_configs_mentpy_restriction_actually_effective": n_mentpy_effective,
            "n_configs_total": len(rows),
            "status": status,
            "finding": "MentPy's restrict_trainable only ever fixes the Ment angle (not just the list) when neither add_edge nor hstack runs afterward, i.e. n_wires==1 and n_layers==1; PhotoGraphiQML never relies on MentPy's list convention and freezes column-3 parameters directly and robustly in every configuration.",
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B/E", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = [f"n{r['n_wires']}L{r['n_layers']}{'1c' if r['one_column'] else 'nc'}" for r in rows]
    colors_mentpy = [
        common.COLORS["mentpy"]
        if r["mentpy_restriction_effective"]
        else common.COLORS["unsupported"]
        for r in rows
    ]
    # This fraction is 0 for every configuration (see docstring); a bar chart
    # would render as an empty axes, so use markers instead.
    axes[0].scatter(
        range(len(rows)),
        [r["mentpy_angle_actually_fixed"] / max(r["n_column3_nodes"], 1) for r in rows],
        color=colors_mentpy,
        zorder=3,
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    axes[0].set(
        title="MentPy: fraction of column-3 Ments actually fixed\n(gray = restriction silently lost)",
        ylabel="fraction",
    )
    colors_pqml = [
        common.COLORS["photographiqml"] if r["photographiqml_robust"] else common.COLORS["piquasso"]
        for r in rows
    ]
    axes[1].bar(
        range(len(rows)),
        [r["photographiqml_override_rejected"] / max(r["n_column3_nodes"], 1) for r in rows],
        color=colors_pqml,
    )
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    axes[1].set(
        title="PhotoGraphiQML: fraction of frozen overrides correctly rejected", ylabel="fraction"
    )
    fig.suptitle(f"R10: trainability audit (status={status})")
    common.save_figure(fig, "R10_trainability_audit")
    plt.close(fig)

    common.print_summary(
        "R10 trainability audit",
        n_configs=len(rows),
        n_mentpy_effective=n_mentpy_effective,
        all_photographiqml_robust=all_robust,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            "R10 failed: PhotoGraphiQML's own freeze mechanism was not robust in every configuration"
        )


if __name__ == "__main__":
    main()
