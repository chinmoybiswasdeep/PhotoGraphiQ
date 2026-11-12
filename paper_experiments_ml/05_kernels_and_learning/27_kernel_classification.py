"""R27: Kernel classification on circles, moons, and blobs with predeclared
generators, repeated splits, confusion matrices, and classical baselines.

Scientific question: Does an SVM with the MuTA Eq. 5 kernel classify three
standard synthetic 2-D datasets (circles, moons, blobs) at accuracy
comparable to classical RBF-SVM and logistic-regression baselines fit on
the same raw coordinates, under repeated stratified splits?

Theory/equations: raw 2-D coordinates are interpreted directly as angles
(radians) by MuTAKernel; no standardization or clipping is applied. This is
a functionality/statistics comparison, not a claim of quantum advantage --
the classical baselines see the identical raw features.

Functionality tested: MuTAKernel + sklearn.svm.SVC(kernel="precomputed") vs.
SVC(kernel="rbf") and LogisticRegression baselines (sklearn), on
sklearn.datasets.make_circles/make_moons/make_blobs.

Oracle and independence class: N/A for the accuracy comparison itself (no
"correct" answer to check against; classical baselines are a comparison
point, not a ground truth) with a structural class-A check folded in (Gram
PSD, reused from R26's already-declared tolerance).

Exact/approximate/statistical status: statistical (8 independent stratified
splits per dataset; median accuracy and bootstrap 95% CI, paired
differences vs. each baseline with their own CIs).

Primary metric: held-out accuracy per (dataset, model, split); paired
accuracy difference (MuTA - baseline) with bootstrap 95% CI.

Structural acceptance condition: every held-out accuracy is finite and in
[0,1], and every Gram matrix is PSD within R26's declared tolerance.
Scientific outcome: descriptive; no observed performance is classified as a
scientific pass merely because these structural checks succeed.

Expected cost: moderate (3 datasets x 8 seeds x 3 models).

Manuscript destination: Main text (Fig. 7, kernel classification).

Scientific limitations: No quantum-advantage claim; hyperparameter (C)
sensitivity is R28's job; raw coordinates as angles is a specific, arbitrary
encoding choice, not the only possible one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from photographiqml import MuTAKernel

EXPERIMENT_ID = "R27"
SEEDS = tuple(range(8))
N_SAMPLES, SVM_C, TEST_FRACTION = 120, 1.0, 0.3
DATASETS = ("circles", "moons", "blobs")


def generate(name, seed):
    if name == "circles":
        return make_circles(n_samples=N_SAMPLES, random_state=seed, factor=0.5, noise=0.08)
    if name == "moons":
        return make_moons(n_samples=N_SAMPLES, random_state=seed, noise=0.1)
    return make_blobs(n_samples=N_SAMPLES, random_state=seed, centers=2, cluster_std=1.2)


def main():
    plt = common.setup_style()
    # Larger safety_factor than R26's since Gram matrices here are up to
    # ~84x84 (vs R26's max 64x64 but at smaller N per point); eigenvalue
    # roundoff for a PSD-by-construction matrix grows with matrix size.
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e5)
    rows = []
    confusions = {name: np.zeros((2, 2), dtype=int) for name in DATASETS}
    min_gram_eigenvalue = np.inf

    for name in DATASETS:
        for seed in SEEDS:
            X, y = generate(name, seed)
            train_x, test_x, train_y, test_y = train_test_split(
                X, y, test_size=TEST_FRACTION, stratify=y, random_state=seed
            )
            kernel = MuTAKernel()
            gram = kernel.gram_matrix(train_x)
            min_gram_eigenvalue = min(min_gram_eigenvalue, float(np.linalg.eigvalsh(gram).min()))
            muta_model = SVC(kernel="precomputed", C=SVM_C).fit(gram, train_y)
            muta_predictions = muta_model.predict(kernel(test_x, train_x))
            confusions[name] += confusion_matrix(test_y, muta_predictions, labels=[0, 1])
            accuracies = {"muta": float(accuracy_score(test_y, muta_predictions))}
            for label, baseline in [
                ("rbf_svm", SVC(C=SVM_C)),
                ("logistic", LogisticRegression(max_iter=1000, random_state=seed)),
            ]:
                baseline.fit(train_x, train_y)
                accuracies[label] = float(accuracy_score(test_y, baseline.predict(test_x)))
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    **{f"accuracy_{k}": v for k, v in accuracies.items()},
                }
            )

    summary = {}
    for name in DATASETS:
        subset = [r for r in rows if r["dataset"] == name]
        muta_acc = [r["accuracy_muta"] for r in subset]
        rbf_acc = [r["accuracy_rbf_svm"] for r in subset]
        log_acc = [r["accuracy_logistic"] for r in subset]
        summary[name] = {
            "muta_ci": common.bootstrap_ci(muta_acc, statistic=np.median, seed=0),
            "rbf_svm_ci": common.bootstrap_ci(rbf_acc, statistic=np.median, seed=0),
            "logistic_ci": common.bootstrap_ci(log_acc, statistic=np.median, seed=0),
            "muta_minus_rbf_ci": common.paired_difference_ci(
                muta_acc, rbf_acc, statistic=np.median, seed=0
            ),
            "muta_minus_logistic_ci": common.paired_difference_ci(
                muta_acc, log_acc, statistic=np.median, seed=0
            ),
        }

    all_finite = all(0 <= r[k] <= 1 for r in rows for k in r if k.startswith("accuracy_"))
    structural_status = "pass" if (all_finite and min_gram_eigenvalue > -tol) else "fail"
    scientific_outcome = "descriptive"

    common.save_result(
        rows,
        "R27_kernel_classification",
        extra={
            "protocol": "MuTA-kernel SVM vs. RBF-SVM/logistic baselines on circles/moons/blobs, repeated splits",
            "oracle_class": "N/A",
            "status_category": "statistical",
            "structural_status": structural_status,
            "scientific_outcome": scientific_outcome,
            "seeds": list(SEEDS),
            "svm_C": SVM_C,
            "summary": summary,
            "confusion_matrices": {k: v.tolist() for k, v in confusions.items()},
            "min_gram_eigenvalue_overall": min_gram_eigenvalue,
            "acceptance_condition": "structural: every accuracy in [0,1] and every training Gram matrix PSD within declared tolerance; scientific outcome: descriptive, no performance threshold",
            "status": structural_status,
        },
        meta_extra={
            "experiment_id": EXPERIMENT_ID,
            "oracle_class": "N/A",
            "status": structural_status,
            "scientific_outcome": scientific_outcome,
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, DATASETS):
        s = summary[name]
        labels = ["MuTA", "RBF-SVM", "Logistic"]
        points = [s["muta_ci"]["point"], s["rbf_svm_ci"]["point"], s["logistic_ci"]["point"]]
        errs = [
            [
                s["muta_ci"]["point"] - s["muta_ci"]["low"],
                s["rbf_svm_ci"]["point"] - s["rbf_svm_ci"]["low"],
                s["logistic_ci"]["point"] - s["logistic_ci"]["low"],
            ],
            [
                s["muta_ci"]["high"] - s["muta_ci"]["point"],
                s["rbf_svm_ci"]["high"] - s["rbf_svm_ci"]["point"],
                s["logistic_ci"]["high"] - s["logistic_ci"]["point"],
            ],
        ]
        colors = [
            common.COLORS["photographiqml"],
            common.COLORS["classical_baseline"],
            common.COLORS["analytic"],
        ]
        ax.bar(labels, points, yerr=errs, color=colors, capsize=4)
        ax.set(title=name, ylabel="held-out accuracy", ylim=(0, 1.05))
    fig.suptitle(
        f"R27: kernel classification, median +/- 95% bootstrap CI, {len(SEEDS)} seeds (structural={structural_status}; outcome={scientific_outcome})"
    )
    common.save_figure(fig, "R27_kernel_classification")
    plt.close(fig)

    common.print_summary(
        "R27 kernel classification",
        n_datasets=len(DATASETS),
        n_seeds=len(SEEDS),
        min_gram_eigenvalue=min_gram_eigenvalue,
        structural_status=structural_status,
        scientific_outcome=scientific_outcome,
    )
    if structural_status != "pass":
        raise AssertionError(
            f"R27 failed: all_finite={all_finite} min_gram_eigenvalue={min_gram_eigenvalue}"
        )


if __name__ == "__main__":
    main()
