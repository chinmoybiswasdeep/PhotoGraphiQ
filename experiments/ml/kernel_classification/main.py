"""MuTA Eq. 5 SVM with explicit classical baselines and held-out splits."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from photographiqml import MuTAKernel


def main():
    root = Path(__file__).parent
    config = json.loads((root / "config.json").read_text())
    output = root / "results"
    output.mkdir(exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for index, name in enumerate(config["datasets"]):
        for seed in config["seeds"]:
            kwargs = {"n_samples": config["samples"], "random_state": seed}
            if name == "circles":
                X, y = make_circles(**kwargs, factor=0.5, noise=0.08)
            elif name == "moons":
                X, y = make_moons(**kwargs, noise=0.1)
            else:
                X, y = make_blobs(**kwargs, centers=2, cluster_std=1)
            train_x, test_x, train_y, test_y = train_test_split(
                X, y, test_size=config["test_fraction"], stratify=y, random_state=seed
            )
            kernel = MuTAKernel()
            gram = kernel.gram_matrix(train_x)
            model = SVC(kernel="precomputed", C=config["svm_C"]).fit(gram, train_y)
            predictions = model.predict(kernel(test_x, train_x))
            results = {"muta": float(accuracy_score(test_y, predictions))}
            for label, baseline in [
                ("rbf_svm", SVC(C=config["svm_C"])),
                ("logistic", LogisticRegression(max_iter=1000, random_state=seed)),
            ]:
                baseline.fit(train_x, train_y)
                results[label] = float(accuracy_score(test_y, baseline.predict(test_x)))
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "accuracy": results,
                    "muta_confusion": confusion_matrix(test_y, predictions).tolist(),
                    "minimum_gram_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
                }
            )
            if seed == config["seeds"][0]:
                axes[index].scatter(
                    test_x[:, 0], test_x[:, 1], c=predictions, cmap="coolwarm", edgecolor="black"
                )
                axes[index].set(title=f"{name}: held-out MuTA labels", xlabel="x0", ylabel="x1")
    fig.tight_layout()
    fig.savefig(output / "kernel_classification.svg")
    plt.close(fig)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "config": config,
                "runs": rows,
                "scope": "Eq. 5 feature map; locally specified data distributions and C, not exact Fig. 8 reproduction",
            },
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
