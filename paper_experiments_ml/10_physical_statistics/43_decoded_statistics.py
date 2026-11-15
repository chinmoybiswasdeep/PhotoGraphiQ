"""R43: Logical versus physical decoded statistics for the signed-X subset.

Scientific question: For the signed-X physical subset, how large is the
total-variation distance between physical decoded joint probabilities and
the ideal logical target, what are the per-wire observable differences and
marginal code-subspace leakages, and does compare_logical_physical ever
invent a finite decoded-state fidelity where none is justified?

Theory/equations: compare_logical_physical(result) is documented (physical.py)
to leave finite_state_fidelity explicitly None; this experiment verifies
that contract directly rather than assuming it.

Functionality tested: photographiqml.physical.compare_logical_physical, for
1-wire and 2-wire zero-angle physical-conditional cases (matching the
resource baseline in docs/physical/evidence.json).

Oracle and independence class: E (structural/self-consistency -- checks the
function's own documented contract: TV distance in [0,1], leakage in [0,1],
finite_state_fidelity is None, retained norm reported by the backend).

Exact/approximate/statistical status: exact (single deterministic
conditional trajectory per configuration; no sampling).

Primary metric: total_variation_distance, observable_differences, marginal
code-subspace leakage, retained Fock norm, per configuration.

Declared acceptance condition: TV distance in [0,1]; leakage values in
[0,1]; finite_state_fidelity is exactly None for every case (never
silently invented).

Expected cost: moderate (1- and 2-wire physical-conditional Fock
simulations, cutoff=24).

Manuscript destination: Main text (Fig. 11, decoded-statistics panel).

Scientific limitations: Restricted signed-X, zero-angle, physical-
conditional cases only, matching the tractable resource baseline; broader
shot-based statistics are R44's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.physical import compare_logical_physical

EXPERIMENT_ID = "R43"


def main():
    plt = common.setup_style()
    config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    rows = []
    for n_wires in (1, 2):
        model = PhysicalMuTA(n_wires, physical_config=config)
        result = model.run(
            [1] + [0] * (2**n_wires - 1),
            mode="physical-conditional",
            analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
        )
        comparison = compare_logical_physical(result)
        marginal_leakage = list(result.diagnostics["marginal_code_subspace_leakage"][0].values())
        retained_norms = [d["retained_norm"] for d in result.diagnostics["backend_diagnostics"][0]]
        rows.append(
            {
                "n_wires": n_wires,
                "total_variation_distance": comparison["total_variation_distance"],
                "observable_differences": comparison["observable_differences"],
                "marginal_leakage": marginal_leakage,
                "finite_state_fidelity_is_none": comparison["finite_state_fidelity"] is None,
                "min_retained_norm": min(retained_norms),
                "max_retained_norm": max(retained_norms),
                "certified": comparison["convergence"]["certified"],
            }
        )

    tv_valid = all(0 <= r["total_variation_distance"] <= 1 + 1e-9 for r in rows)
    leakage_valid = all(all(0 <= v <= 1 for v in r["marginal_leakage"]) for r in rows)
    fidelity_never_invented = all(r["finite_state_fidelity_is_none"] for r in rows)
    never_certified = all(not r["certified"] for r in rows)
    status = (
        "pass"
        if (tv_valid and leakage_valid and fidelity_never_invented and never_certified)
        else "fail"
    )

    common.save_result(
        rows,
        "R43_decoded_statistics",
        extra={
            "protocol": "compare_logical_physical contract check, 1- and 2-wire zero-angle physical-conditional cases",
            "oracle_class": "E",
            "status_category": "exact",
            "resource_config": config.to_dict(),
            "acceptance_condition": "TV distance and leakage in [0,1]; finite_state_fidelity always None; convergence never silently certified",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(
        [f"{r['n_wires']}-wire" for r in rows],
        [r["total_variation_distance"] for r in rows],
        color=common.COLORS["photographiqml"],
    )
    axes[0].set(title="Decoded TV distance from ideal logical target", ylabel="TV distance")
    axes[1].bar(
        [f"{r['n_wires']}-wire" for r in rows],
        [max(r["marginal_leakage"]) for r in rows],
        color=common.COLORS["piquasso"],
    )
    axes[1].set(title="Max marginal code-subspace leakage", ylabel="leakage")
    fig.suptitle(f"R43: logical vs. physical decoded statistics (status={status})")
    common.save_figure(fig, "R43_decoded_statistics")
    plt.close(fig)

    common.print_summary("R43 decoded statistics", n_configs=len(rows), status=status)
    if status != "pass":
        raise AssertionError(
            f"R43 failed: tv_valid={tv_valid} leakage_valid={leakage_valid} fidelity_ok={fidelity_never_invented}"
        )


if __name__ == "__main__":
    main()
