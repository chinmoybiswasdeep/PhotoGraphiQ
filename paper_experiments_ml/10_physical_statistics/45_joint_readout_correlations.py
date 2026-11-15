"""R45: Two-wire joint readout, connected correlations, and direct
verification that multiplying marginals does not reproduce the joint
distribution.

Scientific question: For a two-wire physical-conditional run, does the
decoded joint distribution genuinely differ from the product of its own
decoded marginals (confirming real correlations are retained, not silently
discarded), and does the diagnostics' pair_correlations entry match an
independently recomputed connected correlation from the joint distribution
itself?

Theory/equations: connected correlation <Z0 Z1> = sum_{b0,b1} (-1)^(b0+b1)
P(b0,b1); if the joint equals the product of its marginals, this equals
<Z0><Z1> exactly (zero connected correlation). This experiment recomputes
<Z0 Z1> independently from decoded_joint_probabilities and compares it to
the production diagnostics["pair_correlations"][(0,1)] entry, and
separately computes the total-variation distance between the actual joint
and the product-of-marginals distribution.

Functionality tested: PhysicalMuTA(2).run's decoded_joint_probabilities/
decoded_marginals/diagnostics["pair_correlations"] (photographiqml.physical).

Oracle and independence class: A for the independent connected-correlation
recomputation (a closed-form formula evaluated fresh in this script); E for
the joint-vs-product-of-marginals structural comparison.

Exact/approximate/statistical status: exact (single deterministic
physical-conditional trajectory).

Primary metric: |independent pair correlation - reported pair_correlations|;
total-variation distance between the actual joint and the product-of-
marginals distribution.

Declared acceptance condition: correlation error < tol
(tol = declare_tolerance(scale=1, safety_factor=100)); product-of-marginals
TV distance > 0 (a structural fact for this finite-resource case, not
asserted to be large).

Expected cost: moderate (one 2-wire physical-conditional Fock simulation,
cutoff=24).

Manuscript destination: Main text (Fig. 11, joint-readout panel).

Scientific limitations: Single zero-angle 2-wire configuration; this does
not sweep over general MuTA circuits with different entangling angles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common

from photographiqml import GKPPhysicalConfig, PhysicalMuTA

EXPERIMENT_ID = "R45"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=100.0)
    config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    model = PhysicalMuTA(2, physical_config=config)
    result = model.run(
        [1, 0, 0, 0],
        mode="physical-conditional",
        analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
    )

    joint = result.decoded_joint_probabilities
    marginals = result.decoded_marginals
    labels = list(joint.keys())

    independent_correlation = float(
        sum(((-1) ** (bits[0] + bits[1])) * p for bits, p in joint.items())
    )
    reported_correlation = result.diagnostics["pair_correlations"][(0, 1)]
    correlation_error = abs(independent_correlation - reported_correlation)

    wire0_marginal, wire1_marginal = (
        marginals[model.output_nodes[0]],
        marginals[model.output_nodes[1]],
    )
    product_distribution = {
        bits: wire0_marginal[bits[0]] * wire1_marginal[bits[1]] for bits in labels
    }
    tv_distance_product = float(
        sum(abs(joint[bits] - product_distribution[bits]) for bits in labels) / 2
    )

    rows = [
        {
            "label": str(bits),
            "joint_probability": joint[bits],
            "product_of_marginals": product_distribution[bits],
            "difference": joint[bits] - product_distribution[bits],
        }
        for bits in labels
    ]

    status = "pass" if (correlation_error < tol and tv_distance_product > 0) else "fail"

    common.save_result(
        rows,
        "R45_joint_readout_correlations",
        extra={
            "protocol": "Independent connected-correlation recomputation + joint-vs-product-of-marginals check, 2-wire",
            "oracle_class": "A/E",
            "status_category": "exact",
            "tolerance": tol,
            "independent_pair_correlation": independent_correlation,
            "reported_pair_correlation": reported_correlation,
            "correlation_error": correlation_error,
            "tv_distance_joint_vs_product_of_marginals": tv_distance_product,
            "acceptance_condition": f"correlation error < {tol:.3e}; product-of-marginals TV distance > 0",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A/E", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = range(len(rows))
    axes[0].bar(
        [i - 0.15 for i in x],
        [r["joint_probability"] for r in rows],
        width=0.3,
        label="actual joint",
        color=common.COLORS["photographiqml"],
    )
    axes[0].bar(
        [i + 0.15 for i in x],
        [r["product_of_marginals"] for r in rows],
        width=0.3,
        label="product of marginals",
        color=common.COLORS["classical_baseline"],
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([r["label"] for r in rows])
    axes[0].set(
        title=f"Joint vs. product of marginals (TV={tv_distance_product:.4f})", ylabel="probability"
    )
    axes[0].legend(fontsize=7)
    axes[1].bar(
        ["independent", "reported"],
        [independent_correlation, reported_correlation],
        color=common.COLORS["photographiqml"],
    )
    axes[1].set(
        title=f"<Z0 Z1> connected correlation (error={correlation_error:.2e})", ylabel="correlation"
    )
    fig.suptitle(f"R45: joint readout correlations (status={status})")
    common.save_figure(fig, "R45_joint_readout_correlations")
    plt.close(fig)

    common.print_summary(
        "R45 joint readout correlations",
        correlation_error=correlation_error,
        tv_distance_product=tv_distance_product,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R45 failed: correlation_error={correlation_error} tv_distance={tv_distance_product}"
        )


if __name__ == "__main__":
    main()
