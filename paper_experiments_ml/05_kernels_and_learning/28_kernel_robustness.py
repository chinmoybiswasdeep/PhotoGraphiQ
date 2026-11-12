"""R28: Kernel-SVM robustness to dataset size, noise, SVM C, and coordinate
scale, plus a leakage-safe nested hyperparameter-selection demonstration.

Scientific question: How sensitive is the MuTA-kernel SVM's held-out
accuracy on the "moons" dataset to dataset size, label noise, the SVM
regularization C, and a coordinate-scale rescaling of the raw features
(which changes the effective angle magnitude MuTAKernel encodes), and does
model selection performed strictly inside a validation fold (never touching
the test fold) avoid test-set leakage?

Theory/equations: none (empirical sensitivity study); coordinate_scale
directly rescales X before angle-encoding, so it is a genuine physical
change to the encoded angles, not a data-preprocessing artifact.

Functionality tested: MuTAKernel + sklearn.svm.SVC(kernel="precomputed") on
make_moons, swept one axis at a time from a fixed baseline
(n_samples=120, noise=0.1, C=1.0, coordinate_scale=1.0).

Oracle and independence class: N/A (descriptive sensitivity study; no
correctness oracle for "the right accuracy" at any setting).

Exact/approximate/statistical status: statistical (4 seeds per swept point
for the one-axis-at-a-time sweep; 6 independent train/val/test splits for
the nested-selection demonstration).

Primary metric: held-out test accuracy vs. each swept axis value (median
across seeds); for the nested-selection demonstration, the C chosen per
split (via validation accuracy only) and the resulting test accuracy.

Structural acceptance condition: every reported accuracy is finite and in
[0,1], and nested selection is confined to validation data. Scientific
outcome: descriptive; no observed accuracy is a performance pass condition.

Expected cost: moderate (4 axes x 3 values x 4 seeds, plus 6 nested splits).

Manuscript destination: Appendix (Fig. 7 supporting robustness panel).

Scientific limitations: One dataset family (moons) and one baseline
configuration; not an exhaustive hyperparameter search, and no claim that
any chosen C is globally optimal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from photographiqml import MuTAKernel

EXPERIMENT_ID = "R28"
SEEDS_SWEEP = tuple(range(4))
SEEDS_NESTED = tuple(range(6))
BASELINE = {"n_samples": 120, "noise": 0.1, "C": 1.0, "coordinate_scale": 1.0}


def evaluate(n_samples, noise, C, coordinate_scale, seed):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X * coordinate_scale
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    kernel = MuTAKernel()
    gram = kernel.gram_matrix(train_x)
    model = SVC(kernel="precomputed", C=C).fit(gram, train_y)
    predictions = model.predict(kernel(test_x, train_x))
    return float(accuracy_score(test_y, predictions))


def main():
    plt = common.setup_style()
    rows = []
    axes_sweep = {
        "n_samples": [60, 120, 240],
        "noise": [0.05, 0.15, 0.3],
        "C": [0.1, 1.0, 10.0],
        "coordinate_scale": [0.5, 1.0, 2.0],
    }
    for axis, values in axes_sweep.items():
        for value in values:
            config = dict(BASELINE)
            config[axis] = value
            accuracies = [
                evaluate(
                    config["n_samples"],
                    config["noise"],
                    config["C"],
                    config["coordinate_scale"],
                    seed,
                )
                for seed in SEEDS_SWEEP
            ]
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "median_accuracy": float(np.median(accuracies)),
                    "accuracies": accuracies,
                }
            )

    # --- Leakage-safe nested hyperparameter selection ---------------------
    nested_rows = []
    candidate_Cs = [0.1, 1.0, 10.0]
    for seed in SEEDS_NESTED:
        X, y = make_moons(n_samples=150, noise=0.15, random_state=seed)
        train_x, rest_x, train_y, rest_y = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=seed
        )
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=0.5, stratify=rest_y, random_state=seed
        )
        kernel = MuTAKernel()
        gram_train = kernel.gram_matrix(train_x)
        val_scores = {}
        for C in candidate_Cs:
            model = SVC(kernel="precomputed", C=C).fit(gram_train, train_y)
            val_scores[C] = float(accuracy_score(val_y, model.predict(kernel(val_x, train_x))))
        best_C = max(val_scores, key=val_scores.get)
        final_model = SVC(kernel="precomputed", C=best_C).fit(gram_train, train_y)
        test_accuracy = float(accuracy_score(test_y, final_model.predict(kernel(test_x, train_x))))
        nested_rows.append(
            {
                "seed": seed,
                "selected_C": best_C,
                "validation_scores": val_scores,
                "test_accuracy": test_accuracy,
            }
        )

    all_finite = all(0 <= r["median_accuracy"] <= 1 for r in rows) and all(
        0 <= r["test_accuracy"] <= 1 for r in nested_rows
    )
    structural_status = "pass" if all_finite else "fail"
    scientific_outcome = "descriptive"

    common.save_result(
        rows + [{"nested_demonstration": True, **r} for r in nested_rows],
        "R28_kernel_robustness",
        extra={
            "protocol": "One-axis-at-a-time robustness sweep + leakage-safe nested C selection, moons dataset",
            "seeds": list(SEEDS_NESTED),
            "sweep_seeds": list(SEEDS_SWEEP),
            "oracle_class": "N/A",
            "status_category": "statistical",
            "structural_status": structural_status,
            "scientific_outcome": scientific_outcome,
            "baseline": BASELINE,
            "nested_selection_rows": nested_rows,
            "acceptance_condition": "structural: reported accuracies finite and in [0,1]; scientific outcome: descriptive, no performance threshold",
            "status": structural_status,
        },
        meta_extra={
            "experiment_id": EXPERIMENT_ID,
            "oracle_class": "N/A",
            "status": structural_status,
            "scientific_outcome": scientific_outcome,
        },
    )

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for ax, (axis, values) in zip(axes[:4], axes_sweep.items()):
        medians = [
            next(r["median_accuracy"] for r in rows if r["axis"] == axis and r["value"] == v)
            for v in values
        ]
        ax.plot(values, medians, "o-", color=common.COLORS["photographiqml"])
        ax.set(title=axis, xlabel=axis, ylabel="median accuracy", ylim=(0, 1.05))
    axes[4].bar(
        [str(r["seed"]) for r in nested_rows],
        [r["test_accuracy"] for r in nested_rows],
        color=common.COLORS["photographiqml"],
    )
    axes[4].set(
        title="Nested C-selection test accuracy",
        xlabel="split seed",
        ylabel="test accuracy",
        ylim=(0, 1.05),
    )
    fig.suptitle(
        f"R28: kernel-SVM robustness sweep (structural={structural_status}; outcome={scientific_outcome})"
    )
    common.save_figure(fig, "R28_kernel_robustness")
    plt.close(fig)

    common.print_summary(
        "R28 kernel robustness",
        n_sweep_points=len(rows),
        n_nested_splits=len(nested_rows),
        structural_status=structural_status,
        scientific_outcome=scientific_outcome,
    )
    if structural_status != "pass":
        raise AssertionError("R28 failed: some reported accuracy was out of [0,1] or non-finite")


if __name__ == "__main__":
    main()
