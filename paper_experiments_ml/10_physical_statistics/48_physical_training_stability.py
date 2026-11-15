"""R48: Discrete physical training stability across training seeds, fresh
validation trajectories, and shot counts.

Scientific question: How stable is DiscreteSearch's selected categorical
0/pi angle configuration across independent training seeds, and how much
does the resulting classifier's accuracy change when re-evaluated on fresh
(independently seeded) validation trajectories versus the training
trajectories used for selection?

Theory/equations: none (empirical stability study). The 0/pi family is a
categorical relabeling, not a continuously tunable variational family; this
experiment explicitly reports selection instability rather than describing
it as continuous training convergence (see docs/physical/physical-training.md).

Functionality tested: MuTAClassifier(PhysicalMuTA, trainer=DiscreteSearch)
(photographiqml.models, physical_training), across 6 training seeds, each
re-evaluated on an independent validation-trajectory seed.

Oracle and independence class: N/A (descriptive stability measurement -- no
correctness oracle for "the right selected configuration"; this documents
instability itself as the finding, matching docs/physical/physical-training.md's
own admission that a 2-shot demo does not support a robust-classifier
claim).

Exact/approximate/statistical status: statistical (6 independent training
seeds, each with one independent fresh-trajectory validation seed; small-n,
reported with explicit sample size, no claimed generalization estimate).

Primary metric: training-trajectory accuracy vs. fresh-validation-trajectory
accuracy per seed; fraction of seeds where the two differ (selection
instability rate); fraction of seeds selecting each distinct angle
configuration (selection diversity).

Declared acceptance condition: none required against a correctness oracle
(N/A); the only enforced pass/fail is that every reported accuracy is
finite and in [0,1] and that DiscreteSearchResult.parameters are always
exact categorical {0,pi} values (never rounded from an attempted continuous
solution).

Expected cost: heavy (6 seeds x DiscreteSearch(sweeps=1) x 4 training
inputs x 2 shots x cutoff=40 Fock simulations, plus fresh-trajectory
validation runs).

Manuscript destination: Main text (Fig. 12, physical training stability --
explicitly an instability/negative result, not a performance claim).

Scientific limitations: Small shot count (2) and small training set (4
points) by design, matching the tractable demonstration scope in
experiments/restricted_physical.py; this is execution evidence, not a
held-out generalization estimate (per docs/physical/physical-training.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.models import MuTAClassifier
from photographiqml.physical_training import DiscreteSearch

EXPERIMENT_ID = "R48"
SEEDS = (7, 19, 23, 31, 42, 101)
X = [[0.0], [0.2], [2.9], [np.pi]]
Y = [0, 0, 1, 1]


def main():
    plt = common.setup_style()
    config = GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    rows = []
    for seed in SEEDS:
        model = PhysicalMuTA(1, physical_config=config)
        classifier = MuTAClassifier(
            model,
            trainer=DiscreteSearch(sweeps=1, seed=seed),
            physical_options={"mode": "physical-shots", "shots": 2, "seed": seed},
        )
        classifier.fit(X, Y)
        training_predictions = classifier.predict(X)
        training_accuracy = float(np.mean(training_predictions == np.array(Y)))

        classifier.physical_options["seed"] = seed + 1000  # independent fresh trajectories
        fresh_predictions = classifier.predict(X)
        fresh_accuracy = float(np.mean(fresh_predictions == np.array(Y)))

        selected = tuple(np.round(classifier.history.parameters, 6).tolist())
        exact_categorical = all(
            np.isclose(v, 0.0) or np.isclose(v, np.pi) for v in classifier.history.parameters
        )

        rows.append(
            {
                "seed": seed,
                "selected_parameters": selected,
                "exact_categorical": exact_categorical,
                "training_trajectory_accuracy": training_accuracy,
                "fresh_trajectory_accuracy": fresh_accuracy,
                "accuracy_changed": training_accuracy != fresh_accuracy,
            }
        )

    all_finite = all(
        0 <= r["training_trajectory_accuracy"] <= 1 and 0 <= r["fresh_trajectory_accuracy"] <= 1
        for r in rows
    )
    all_categorical = all(r["exact_categorical"] for r in rows)
    instability_rate = float(np.mean([r["accuracy_changed"] for r in rows]))
    distinct_selections = len({r["selected_parameters"] for r in rows})
    status = "pass" if (all_finite and all_categorical) else "fail"

    common.save_result(
        rows,
        "R48_physical_training_stability",
        extra={
            "protocol": "DiscreteSearch classifier stability across training seeds, fresh-trajectory re-evaluation",
            "oracle_class": "N/A",
            "status_category": "statistical",
            "n_seeds": len(SEEDS),
            "instability_rate": instability_rate,
            "distinct_selected_configurations": distinct_selections,
            "acceptance_condition": "all accuracies finite and in [0,1]; every selected configuration exactly categorical {0,pi}",
            "finding": "Selected configurations and their fresh-trajectory accuracy vary across training seeds; this documents instability, not a robust classifier-performance claim (per docs/physical/physical-training.md).",
            "structural_status": status,
            "scientific_outcome": "negative",
            "claim_supported": "Fresh-trajectory physical-training instability was observed under the declared six-seed protocol.",
            "claim_not_supported": "Robust physical classifier performance is not demonstrated.",
            "seeds": list(SEEDS),
            "uncertainty_method": "six-seed empirical instability rate (small-n; no population CI claim)",
            "status": status,
        },
        meta_extra={
            "experiment_id": EXPERIMENT_ID,
            "oracle_class": "N/A",
            "status": status,
            "scientific_outcome": "negative",
            "seed_set": list(SEEDS),
            "n_repetitions": len(SEEDS),
            "uncertainty_method": "six-seed empirical instability rate",
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(rows))
    ax.bar(
        [i - 0.15 for i in x],
        [r["training_trajectory_accuracy"] for r in rows],
        width=0.3,
        label="training trajectories",
        color=common.COLORS["photographiqml"],
    )
    ax.bar(
        [i + 0.15 for i in x],
        [r["fresh_trajectory_accuracy"] for r in rows],
        width=0.3,
        label="fresh trajectories",
        color=common.COLORS["piquasso"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in SEEDS])
    # A zero-height bar (seed 7's fresh-trajectory accuracy) must not read as
    # "no data"; label every bar's value explicitly.
    for i, r in enumerate(rows):
        ax.text(
            i - 0.15,
            r["training_trajectory_accuracy"] + 0.02,
            f"{r['training_trajectory_accuracy']:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
        ax.text(
            i + 0.15,
            r["fresh_trajectory_accuracy"] + 0.02,
            f"{r['fresh_trajectory_accuracy']:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_ylim(0, 1.15)
    ax.set(
        title=f"R48: discrete physical training instability (instability_rate={instability_rate:.2f}, {distinct_selections} distinct configs)",
        xlabel="training seed",
        ylabel="accuracy",
    )
    ax.legend(fontsize=7)
    common.save_figure(fig, "R48_physical_training_stability")
    plt.close(fig)

    common.print_summary(
        "R48 physical training stability",
        instability_rate=instability_rate,
        distinct_selections=distinct_selections,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R48 failed: all_finite={all_finite} all_categorical={all_categorical}"
        )


if __name__ == "__main__":
    main()
