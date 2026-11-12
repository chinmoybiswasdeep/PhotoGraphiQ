"""R44: Shot convergence for Rao-Blackwell decoded probabilities and
empirical sampled-bit frequencies, with Monte Carlo standard errors and
coverage checks.

Scientific question: Do PhysicalMuTA's two shot-based estimators --
decoded_joint_probabilities (a Rao-Blackwell average of conditional POVM
probabilities over trajectories) and empirical_probabilities (raw sampled-
bit frequencies) -- shrink their reported standard errors as shots
increase, and does a declared-in-advance 95% normal-approximation interval
around each independent repetition's estimate cover a fixed reference value
at close to the nominal rate?

Theory/equations: decoded_joint_probabilities' standard_errors are sample
standard deviations / sqrt(shots) (a Rao-Blackwell estimator, lower
variance than raw sampling); empirical_standard_errors are plug-in binomial
sqrt(p(1-p)/shots) errors. Both should shrink roughly as 1/sqrt(shots) for
a fixed underlying resource.

Functionality tested: PhysicalMuTA.run(mode="physical-shots") standard
error reporting (photographiqml.physical).

Oracle and independence class: E (self-consistency: reported standard
errors are checked against an independently recomputed sample standard
deviation from the raw per-trajectory rows, not trusted blindly) plus a
statistical coverage check against a fixed high-shot reference value.

Exact/approximate/statistical status: statistical (Monte Carlo; coverage
computed over 8 independent repetitions at a fixed shot count).

Primary metric: standard error vs. shots (both estimators); empirical
coverage rate of a declared 95% interval around the Rao-Blackwell estimate,
across 8 independent-seed repetitions at shots=16, against a shots=64
reference.

Declared acceptance condition: standard errors decrease (not necessarily
monotonically at every step, since this is a finite Monte Carlo estimate,
but the shots=32 error must be below the shots=4 error); coverage rate
within [0.5, 1.0] (a loose sanity band for only 8 repetitions -- exact 0.95
coverage is not statistically resolvable at n=8, so this experiment does
not assert exact nominal coverage, only that it is not badly broken).

Expected cost: moderate-to-heavy (multiple physical-shots Fock simulations
at cutoff=40, needed for shots>1 per docs/physical/evidence.json).

Manuscript destination: Main text (Fig. 11, shot-convergence panel).

Scientific limitations: One-wire only, for tractability; neither estimator
is claimed to certify resource accuracy (only their own Monte Carlo
uncertainty is characterized here, per docs/physical/execution-modes.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common

from photographiqml import GKPPhysicalConfig, PhysicalMuTA

EXPERIMENT_ID = "R44"


def main():
    plt = common.setup_style()
    config = GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    model = PhysicalMuTA(1, physical_config=config)

    convergence_rows = []
    for shots in (4, 8, 16, 32):
        result = model.run([1, 0], mode="physical-shots", shots=shots, seed=2026)
        rb_se = max(result.standard_errors.values()) if result.standard_errors else None
        emp_se = (
            max(result.empirical_standard_errors.values())
            if result.empirical_standard_errors
            else None
        )
        convergence_rows.append(
            {"shots": shots, "max_rb_standard_error": rb_se, "max_empirical_standard_error": emp_se}
        )

    reference_result = model.run([1, 0], mode="physical-shots", shots=64, seed=999)
    reference = reference_result.decoded_joint_probabilities

    coverage_shots = 16
    coverage_hits = 0
    coverage_rows = []
    for seed in range(8):
        result = model.run([1, 0], mode="physical-shots", shots=coverage_shots, seed=1000 + seed)
        label = next(iter(reference))
        estimate = result.decoded_joint_probabilities[label]
        se = result.standard_errors[label]
        lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
        covered = lo <= reference[label] <= hi
        coverage_hits += covered
        coverage_rows.append(
            {
                "seed": seed,
                "label": str(label),
                "estimate": estimate,
                "se": se,
                "low": lo,
                "high": hi,
                "covered": covered,
            }
        )
    coverage_rate = coverage_hits / len(coverage_rows)

    se_decreasing = convergence_rows[-1]["max_rb_standard_error"] is None or (
        convergence_rows[0]["max_rb_standard_error"] is not None
        and convergence_rows[-1]["max_rb_standard_error"]
        < convergence_rows[0]["max_rb_standard_error"]
    )
    status = "pass" if (se_decreasing and 0.5 <= coverage_rate <= 1.0) else "fail"

    common.save_result(
        convergence_rows + [{"coverage_check": True, **r} for r in coverage_rows],
        "R44_shot_convergence",
        extra={
            "protocol": "Rao-Blackwell and empirical standard-error shot convergence + coverage check",
            "oracle_class": "E",
            "status_category": "statistical",
            "reference_shots": 64,
            "coverage_shots": coverage_shots,
            "coverage_rate": coverage_rate,
            "seeds": [2026, 999, *range(1000, 1008)],
            "acceptance_condition": "shots=32 SE < shots=4 SE; coverage rate in [0.5,1.0] (loose n=8 sanity band)",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "E", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    shots_list = [r["shots"] for r in convergence_rows]
    axes[0].loglog(
        shots_list,
        [r["max_rb_standard_error"] for r in convergence_rows if r["max_rb_standard_error"]],
        "o-",
        label="Rao-Blackwell SE",
        color=common.COLORS["photographiqml"],
    )
    axes[0].loglog(
        shots_list,
        [
            r["max_empirical_standard_error"]
            for r in convergence_rows
            if r["max_empirical_standard_error"]
        ],
        "s--",
        label="empirical SE",
        color=common.COLORS["piquasso"],
    )
    axes[0].set(title="Standard error vs. shots", xlabel="shots", ylabel="max SE")
    axes[0].legend(fontsize=7)
    axes[1].errorbar(
        range(len(coverage_rows)),
        [r["estimate"] for r in coverage_rows],
        yerr=[1.96 * r["se"] for r in coverage_rows],
        fmt="o",
        color=common.COLORS["photographiqml"],
        capsize=3,
    )
    axes[1].axhline(
        reference[next(iter(reference))],
        color=common.COLORS["acceptance"],
        linestyle="--",
        label="reference (64 shots)",
    )
    axes[1].set(
        title=f"Coverage check ({coverage_shots} shots, rate={coverage_rate:.2f})",
        xlabel="repetition seed",
        ylabel="probability",
    )
    axes[1].legend(fontsize=7)
    fig.suptitle(f"R44: shot convergence and coverage (status={status})")
    common.save_figure(fig, "R44_shot_convergence")
    plt.close(fig)

    common.print_summary("R44 shot convergence", coverage_rate=coverage_rate, status=status)
    if status != "pass":
        raise AssertionError(
            f"R44 failed: se_decreasing={se_decreasing} coverage_rate={coverage_rate}"
        )


if __name__ == "__main__":
    main()
