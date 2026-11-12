"""Reproducible logical gate learning with independent reference checkpoints."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

from photographiqml import MuTA, Trainer
from photographiqml.logical import X
from photographiqml.models import haar_states, infidelity
from photographiqml.validation import MENTPY_COMMIT, compare_mentpy


def main():
    root = Path(__file__).parent
    config = json.loads((root / "config.json").read_text())
    runs = []
    for target_name in ("haar_first_wire", "ising_xx"):
        for seed in config["seeds"]:
            model = MuTA(2, one_column=True)
            if target_name == "haar_first_wire":
                target = np.kron(
                    unitary_group.rvs(2, random_state=np.random.default_rng(seed)), np.eye(2)
                )
            else:
                target = expm(-0.25j * np.pi * np.kron(X, X))
            states = haar_states(2, config["n_states"], seed + 100)
            targets = states @ target.T
            n_train = config["n_train"]

            def objective(values):
                predictions = states[:n_train] @ model.unitary(values).T
                return infidelity(predictions, targets[:n_train])

            def validation(values):
                return infidelity(states[n_train:] @ model.unitary(values).T, targets[n_train:])

            checkpoints = []

            def checkpoint(values, history):
                if (len(history.losses) - 1) % 40 == 0:
                    checkpoints.append(
                        {
                            "epoch": len(history.losses) - 1,
                            "density_max_error": compare_mentpy(model, states[-1], values),
                        }
                    )

            trainer = Trainer(epochs=config["epochs"], learning_rate=config["learning_rate"])
            values, history = trainer.fit(
                objective,
                list(model.initialize(seed, 0.3).values()),
                validation=validation,
                callback=checkpoint,
            )
            run = {
                "target": target_name,
                "seed": seed,
                "parameters": values.tolist(),
                "train_loss": history.losses,
                "test_loss": history.validation_losses,
                "seconds": history.seconds[-1],
                "mentpy_checkpoints": checkpoints,
            }
            runs.append(run)
            print(target_name, seed, history.losses[-1], history.validation_losses[-1], flush=True)
    (root / "results").mkdir(exist_ok=True)
    report = {
        "config": config,
        "mentpy_commit": MENTPY_COMMIT,
        "runs": runs,
        "discrepancies": [
            "Seed identities and optimizer hyperparameters are local to this reproduction",
            "Explicit finite-difference gradients and local hyperparameters",
            "Ideal logical execution only; no finite-energy GKP curve",
            "MentPy output checked at checkpoints, not independently retrained",
        ],
    }
    (root / "results/gate_learning.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target_name in zip(axes, ("haar_first_wire", "ising_xx")):
        losses = np.array([r["test_loss"] for r in runs if r["target"] == target_name])
        mean, std = losses.mean(axis=0), losses.std(axis=0)
        ax.plot(mean, label="test mean")
        ax.fill_between(range(len(mean)), np.maximum(0, mean - std), mean + std, alpha=0.2)
        ax.set(title=target_name, xlabel="Adam step", ylabel="Mean infidelity")
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "results/gate_learning.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
