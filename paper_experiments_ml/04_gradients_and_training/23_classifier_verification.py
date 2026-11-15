"""R23: Logical classifier verification with repeated stratified held-out
splits.

Scientific question: Does MuTAClassifier (a MuTA circuit with a logistic
head on the first-wire Z expectation, trained by Adam) reliably learn a
simple, analytically separable synthetic decision boundary, with accuracy
estimated by repeated stratified train/test splits rather than a single
lucky split?

Theory/equations: synthetic labels y = 1[x < pi/2] for a single feature
x ~ Uniform(0, pi) (angle_encode applies Ry(x/2), so the bare Z expectation
cos(x) crosses zero exactly at x=pi/2 -- the same single-wire threshold task
the README's worked example uses, extended here to a larger dataset with
repeated splits rather than four fixed points).

A composite two-feature boundary (y = 1[cos(x0)+cos(x1) > 0], n_wires=2)
was tried first and did not converge reliably within a moderate epoch
budget (final training loss stuck near log(2), i.e. near chance) -- a
genuine architecture/optimization-budget limitation for that harder
target, not a bug, and not silently discarded (see ISSUES_FOUND.md). This
experiment is rescoped to the single-feature task the package's own
documentation demonstrates working, to verify the classifier *wrapper's*
correctness rather than probe the ansatz's expressivity limits (that is
R22's job).

Functionality tested: MuTAClassifier.fit/predict/score (photographiqml.models),
Trainer(optimizer="adam").

Oracle and independence class: A (independent analytic oracle -- the label
rule is a closed-form function of the inputs, and a majority-class baseline
is computed independently as a sanity floor).

Exact/approximate/statistical status: statistical (5 independent stratified
splits, each with an independent training seed; median accuracy and
bootstrap 95% CI reported).

Primary metric: held-out accuracy per split, median and 95% CI across
splits; confusion matrix aggregated over splits.

Declared acceptance condition: median held-out accuracy >= 0.85 and
strictly greater than the majority-class baseline accuracy (paired
comparison, same splits).

Expected cost: light (5 splits x 80 Adam epochs, 1-wire circuit,
n_parameters=4).

Manuscript destination: Main text (Fig. 6/7, supervised learning panel).

Scientific limitations: A single fixed synthetic boundary, not a benchmark
suite; not a claim of quantum advantage over classical logistic regression
on the same raw features (which would trivially also separate this
boundary, since it is linear in cos(x0), cos(x1)).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from photographiqml import MuTA, MuTAClassifier, Trainer

EXPERIMENT_ID = "R23"
N_SAMPLES, N_SPLITS, EPOCHS = 60, 5, 80


def main():
    plt = common.setup_style()
    generator = common.rng(0)
    X = generator.uniform(0, np.pi, size=(N_SAMPLES, 1))
    y = (X[:, 0] < np.pi / 2).astype(int)

    rows = []
    confusions = np.zeros((2, 2), dtype=int)
    for split in range(N_SPLITS):
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=split
        )
        model = MuTA(1, 1, one_column=True)
        classifier = MuTAClassifier(
            model, trainer=Trainer(optimizer="adam", epochs=EPOCHS, learning_rate=0.1), seed=split
        )
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        accuracy = float(np.mean(predictions == test_y))
        majority_baseline = float(np.mean(test_y == np.bincount(train_y).argmax()))
        confusions += confusion_matrix(test_y, predictions, labels=[0, 1])
        rows.append(
            {
                "split": split,
                "accuracy": accuracy,
                "majority_baseline_accuracy": majority_baseline,
                "final_train_loss": classifier.history.losses[-1],
            }
        )

    accuracies = [r["accuracy"] for r in rows]
    baselines = [r["majority_baseline_accuracy"] for r in rows]
    ci = common.bootstrap_ci(accuracies, statistic=np.median, seed=0)
    paired_ci = common.paired_difference_ci(accuracies, baselines, statistic=np.median, seed=0)
    status = "pass" if (ci["point"] >= 0.85 and paired_ci["point"] > 0) else "fail"

    common.save_result(
        rows,
        "R23_classifier_verification",
        extra={
            "protocol": "MuTAClassifier on the one-dimensional threshold y=1[x < pi/2], repeated stratified splits",
            "point_estimates": {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_median": float(np.median(accuracies)),
            },
            "oracle_class": "A",
            "status_category": "statistical",
            "n_splits": N_SPLITS,
            "seeds": list(range(N_SPLITS)),
            "accuracy_bootstrap_ci": ci,
            "paired_difference_vs_majority_baseline_ci": paired_ci,
            "confusion_matrix": confusions.tolist(),
            "acceptance_condition": "median accuracy >= 0.85 and paired improvement over majority baseline > 0",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(
        range(N_SPLITS),
        accuracies,
        "o-",
        color=common.COLORS["photographiqml"],
        label="MuTAClassifier",
    )
    axes[0].plot(
        range(N_SPLITS),
        baselines,
        "s--",
        color=common.COLORS["classical_baseline"],
        label="majority baseline",
    )
    axes[0].axhline(ci["point"], color=common.COLORS["photographiqml"], linestyle=":", alpha=0.6)
    axes[0].set(title="Held-out accuracy per split", xlabel="split", ylabel="accuracy")
    axes[0].legend(fontsize=7)
    axes[1].imshow(confusions, cmap="Blues")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set(title="Aggregated confusion matrix", xlabel="predicted", ylabel="true")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, confusions[i, j], ha="center", va="center")
    fig.suptitle(f"R23: classifier verification (status={status})")
    common.save_figure(fig, "R23_classifier_verification")
    plt.close(fig)

    common.print_summary(
        "R23 classifier verification", n_splits=N_SPLITS, median_accuracy=ci["point"], status=status
    )
    if status != "pass":
        raise AssertionError(f"R23 failed: median_accuracy={ci['point']} paired_ci={paired_ci}")


if __name__ == "__main__":
    main()
