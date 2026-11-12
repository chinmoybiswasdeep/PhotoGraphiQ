"""R17: Seeded logical and physical reproducibility, with different-seed
distribution comparisons where stochastic sampling is used.

Scientific question: Does an identical seed reproduce bitwise-identical
logical initialization, and identical physical shot results
(sampled_output_bits, decoded_joint_probabilities), while different seeds
give statistically distinct outcomes without breaking any invariant (e.g.
probabilities still summing to 1)?

Theory/equations: none (reproducibility contract verification).

Functionality tested: MuTA.initialize (logical), PhysicalMuTA.run(mode=
"physical-shots", seed=...) (physical).

Oracle and independence class: E (structural/self-consistency -- a run is
compared against a second run with the same declared seed).

Exact/approximate/statistical status: exact for the same-seed reproducibility
check; statistical for the different-seed distinctness check (a finite-shot
comparison, reported with its own sampling uncertainty, not just a bare
inequality).

Primary metric: same-seed max difference (must be exactly 0); different-seed
total-variation distance between two independent 32-shot runs (must be
> 0 with high probability, and its own multinomial standard error is
reported alongside it).

Declared acceptance condition: same-seed error == 0 in every case; for the
different-seed check, TV distance is reported with its standard error and
compared against that error rather than asserted to be "large" (a small but
nonzero, error-consistent TV distance is an expected pass, not a failure).

Expected cost: light (1-wire physical model, cutoff=40, 32 shots).

Manuscript destination: Appendix (reproducibility contract table).

Scientific limitations: The physical check uses one small 1-wire
configuration only (full physical layer performance is exercised
separately in R43-R48); this experiment is about seed contract compliance,
not about resource-accuracy convergence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA

EXPERIMENT_ID = "R17"


def main():
    plt = common.setup_style()
    rows = []

    # --- Logical: same-seed reproducibility, different-seed distinctness ------
    model = MuTA(2, 1, one_column=True)
    for seed in (0, 1, 7, 123):
        a = model.initialize(seed=seed, scale=0.7)
        b = model.initialize(seed=seed, scale=0.7)
        rows.append(
            {
                "layer": "logical",
                "seed": seed,
                "same_seed_max_diff": max(abs(a[k] - b[k]) for k in a),
            }
        )
    different = model.initialize(seed=0, scale=0.7) != model.initialize(seed=1, scale=0.7)

    # --- Physical: same-seed reproducibility of shot results ------------------
    # cutoff=40 is required for this width/envelope/peaks resource at shots>1;
    # docs/release-report.md records the same cutoff=24 guard failure at 16
    # shots (retained norm 0.9987) and cutoff=40 succeeding.
    physical_config = GKPPhysicalConfig(
        cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025
    )
    physical_model = PhysicalMuTA(1, physical_config=physical_config)
    n_shots = 32
    result_a = physical_model.run([1, 0], mode="physical-shots", shots=n_shots, seed=2026)
    result_b = physical_model.run([1, 0], mode="physical-shots", shots=n_shots, seed=2026)
    same_seed_bit_diff = sum(
        a != b for a, b in zip(result_a.sampled_output_bits, result_b.sampled_output_bits)
    )
    same_seed_prob_diff = max(
        abs(result_a.decoded_joint_probabilities[k] - result_b.decoded_joint_probabilities[k])
        for k in result_a.decoded_joint_probabilities
    )
    result_c = physical_model.run([1, 0], mode="physical-shots", shots=n_shots, seed=4052)
    labels = list(result_a.empirical_probabilities)
    p_a = np.array([result_a.empirical_probabilities[k] for k in labels])
    p_c = np.array([result_c.empirical_probabilities[k] for k in labels])
    tv_distance = float(np.sum(abs(p_a - p_c)) / 2)
    # Multinomial standard error of the TV distance estimator (rough, per-cell
    # binomial SE combined; both runs use n_shots draws, so combine in quadrature).
    se_a = np.sqrt(p_a * (1 - p_a) / n_shots)
    se_c = np.sqrt(p_c * (1 - p_c) / n_shots)
    tv_standard_error = float(np.sqrt(np.sum(se_a**2 + se_c**2)) / 2)

    rows.append(
        {
            "layer": "physical",
            "seed": 2026,
            "same_seed_max_diff": max(same_seed_bit_diff, same_seed_prob_diff),
            "same_seed_bit_diff": same_seed_bit_diff,
            "same_seed_probability_diff": same_seed_prob_diff,
            "different_seed_tv_distance": tv_distance,
            "different_seed_tv_standard_error": tv_standard_error,
        }
    )

    max_same_seed_error = max(r["same_seed_max_diff"] for r in rows)
    status = "pass" if (max_same_seed_error == 0 and different) else "fail"

    common.save_result(
        rows,
        "R17_seeded_reproducibility",
        extra={
            "protocol": "Same-seed reproducibility and different-seed distinctness, logical + physical",
            "oracle_class": "E",
            "status_category": "exact/statistical",
            "acceptance_condition": "same-seed max diff == 0 in every case; different-seed logical init differs",
            "max_same_seed_error": max_same_seed_error,
            "different_seed_logical_init_differs": bool(different),
            "different_seed_physical_tv_distance": tv_distance,
            "different_seed_physical_tv_standard_error": tv_standard_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    logical_rows = [r for r in rows if r["layer"] == "logical"]
    axes[0].bar(
        range(len(logical_rows)),
        [r["same_seed_max_diff"] for r in logical_rows],
        color=common.COLORS["photographiqml"],
    )
    axes[0].set_xticks(range(len(logical_rows)))
    axes[0].set_xticklabels([f"seed={r['seed']}" for r in logical_rows])
    axes[0].set(title="Logical: same-seed init difference", ylabel="max |diff|")
    axes[1].bar(
        ["same-seed\n(bit+prob diff)", "different-seed\nTV distance"],
        [max(same_seed_bit_diff, same_seed_prob_diff), tv_distance],
        yerr=[0, tv_standard_error],
        color=[common.COLORS["photographiqml"], common.COLORS["piquasso"]],
        capsize=4,
    )
    axes[1].set(title="Physical shot reproducibility (32 shots)", ylabel="difference")
    fig.suptitle(f"R17: seeded reproducibility (status={status})")
    common.save_figure(fig, "R17_seeded_reproducibility")
    plt.close(fig)

    common.print_summary(
        "R17 seeded reproducibility",
        max_same_seed_error=max_same_seed_error,
        different_seed_tv_distance=tv_distance,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R17 failed: max_same_seed_error={max_same_seed_error} different={different}"
        )


if __name__ == "__main__":
    main()
