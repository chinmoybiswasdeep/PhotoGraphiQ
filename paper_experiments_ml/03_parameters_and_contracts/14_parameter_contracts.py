"""R14: Parameter ordering, partial/exact binding, freeze/unfreeze, frozen
override rejection, and seeded initialization.

Scientific question: Does photographiqml.parameters.ParameterStore (exposed
through MuTA.parameters/trainable_parameters/freeze/unfreeze/initialize)
honor its documented contract: paper-block/wire/column name ordering,
partial dict binding, exact-length vector binding in trainable_parameters()
order, rejection of any attempt to override a frozen value, and
seed-reproducible/seed-distinct initialization?

Theory/equations: none (API contract verification).

Functionality tested: photographiqml.parameters.ParameterStore.bind/freeze/
unfreeze; MuTA.parameters/trainable_parameters/initialize/freeze/unfreeze.

Oracle and independence class: E (structural/self-consistency test of the
public contract, per docs/api.md).

Exact/approximate/statistical status: exact (discrete pass/fail per
sub-check).

Primary metric: number of contract sub-checks failed (out of the declared
set below).

Declared acceptance condition: 0 failed sub-checks.

Expected cost: light.

Manuscript destination: Appendix (API contract table).

Scientific limitations: This is a self-consistency check of the public
interface, not a physics validation (see R1-R12 for that).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA

EXPERIMENT_ID = "R14"


def main():
    plt = common.setup_style()
    checks = {}

    model = MuTA(2, 2, one_column=True)
    names = list(model.parameters())
    # Ordering: model.parameters() must enumerate exactly model.measured_nodes
    # in order, and that order is documented as paper block, wire, local column.
    measured = model.measured_nodes
    checks["measured_nodes_sorted_block_wire_col"] = measured == tuple(
        sorted(measured, key=lambda v: (v[1] // 4, v[0], v[1] % 4))
    )
    checks["parameters_dict_order_matches_measured_nodes"] = names == [
        model.parameter_name(v) for v in measured
    ]

    # Partial dict binding: only some names given; rest keep stored defaults.
    subset_name = names[0]
    bound = model._parameters.bind({subset_name: 3.21})
    checks["partial_dict_binding_updates_only_given_names"] = bound[subset_name] == 3.21 and all(
        bound[n] == 0.0 for n in names if n != subset_name
    )

    # Exact vector binding must match trainable_parameters() length/order.
    trainable_names = list(model.trainable_parameters())
    vector = np.arange(1, len(trainable_names) + 1, dtype=float)
    bound_vec = model._parameters.bind(vector)
    checks["exact_vector_binding_matches_trainable_order"] = all(
        bound_vec[n] == vector[i] for i, n in enumerate(trainable_names)
    )
    try:
        model._parameters.bind(np.arange(len(trainable_names) + 1, dtype=float))
        checks["wrong_length_vector_rejected"] = False
    except ValueError:
        checks["wrong_length_vector_rejected"] = True

    # Freeze/unfreeze.
    frozen_name = trainable_names[0]
    model.freeze(frozen_name, 0.0)
    checks["freeze_removes_from_trainable"] = frozen_name not in model.trainable_parameters()
    checks["freeze_reduces_n_parameters"] = model.n_parameters == len(trainable_names) - 1
    try:
        model.run([1, 0, 0, 0], {frozen_name: 1.0})
        checks["frozen_override_rejected"] = False
    except ValueError:
        checks["frozen_override_rejected"] = True
    model.unfreeze(frozen_name)
    checks["unfreeze_restores_trainable"] = frozen_name in model.trainable_parameters()
    checks["unfreeze_restores_n_parameters"] = model.n_parameters == len(trainable_names)

    # Unknown parameter name rejected.
    try:
        model._parameters.bind({"alpha.w9.c9": 1.0})
        checks["unknown_name_rejected"] = False
    except ValueError:
        checks["unknown_name_rejected"] = True

    # Seeded initialization: reproducible under the same seed, distinct
    # (with overwhelming probability) under different seeds.
    init_a = model.initialize(seed=42, scale=0.5)
    init_b = model.initialize(seed=42, scale=0.5)
    init_c = model.initialize(seed=43, scale=0.5)
    checks["same_seed_reproducible"] = init_a == init_b
    checks["different_seed_differs"] = init_a != init_c

    rows = [{"check": name, "passed": bool(value)} for name, value in checks.items()]
    n_failed = sum(1 for r in rows if not r["passed"])
    status = "pass" if n_failed == 0 else "fail"

    common.save_result(
        rows,
        "R14_parameter_contracts",
        extra={
            "protocol": "ParameterStore/MuTA public parameter-contract checklist",
            "oracle_class": "E",
            "status_category": "exact",
            "acceptance_condition": "0 failed sub-checks",
            "n_checks": len(rows),
            "n_failed": n_failed,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        common.COLORS["photographiqml"] if r["passed"] else common.COLORS["piquasso"] for r in rows
    ]
    ax.barh(range(len(rows)), [1] * len(rows), color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["check"] for r in rows], fontsize=7)
    ax.set_xticks([])
    ax.set_title(f"R14: parameter-contract checklist (status={status}, {n_failed} failed)")
    common.save_figure(fig, "R14_parameter_contracts")
    plt.close(fig)

    common.print_summary(
        "R14 parameter contracts", n_checks=len(rows), n_failed=n_failed, status=status
    )
    if status != "pass":
        raise AssertionError(f"R14 failed: {[r['check'] for r in rows if not r['passed']]}")


if __name__ == "__main__":
    main()
