"""R21: Ising-XX gate learning across 20 declared seeds, same statistical
protocol as R20.

Scientific question: Can Adam training of a two-wire, one-layer MuTA circuit
learn the entangling Ising-XX target exp(-i*pi*XX/4) from a finite training
set of Haar states, generalize to held-out Haar states, and does the trained
model still agree with an independently executed MentPy circuit at the
final checkpoint?

Theory/equations: target = expm(-i*pi/4 * X⊗X) (scipy.linalg.expm, an
analytic closed-form target independent of MuTA); infidelity as in R20.

Functionality tested: same as R20, with a fixed entangling target instead of
a random single-wire target -- this is R3's entangling identity used as a
*training target*, not re-verified as an identity here (R3 already did
that).

Oracle and independence class: A for the target (scipy.linalg.expm); B for
the MentPy training checkpoint.

Exact/approximate/statistical status: statistical (20 independent seeds).

Primary metric: held-out test infidelity at the final epoch, across seeds.

Declared acceptance condition: median final test infidelity < 0.05 (this
target is exactly representable by the ansatz at a *single* trainable angle,
alpha.w1.c1 = pi/2, per R3, so it is expected to converge at least as
reliably as R20's generic Haar target); max MentPy checkpoint error <
tol = declare_tolerance(scale=1, safety_factor=1000).

Expected cost: moderate (20 seeds x 120 epochs).

Manuscript destination: Main text (Fig. 6, gate-learning statistics,
companion panel to R20).

Scientific limitations: Same as R20; a functionality/statistics
demonstration, not a quantum-advantage or general-learnability claim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from scipy.linalg import expm

from photographiqml import MuTA, Trainer
from photographiqml.logical import X
from photographiqml.models import haar_states, infidelity
from photographiqml.validation import MENTPY_COMMIT, compare_mentpy

EXPERIMENT_ID = "R21"
SEEDS = tuple(range(20))
N_STATES, N_TRAIN, EPOCHS, LEARNING_RATE = 12, 8, 120, 0.05


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1000.0)
    target = expm(-0.25j * np.pi * np.kron(X, X))
    rows = []
    trajectories = []
    max_checkpoint_error = 0.0

    for seed in SEEDS:
        model = MuTA(2, one_column=True)
        states = haar_states(2, N_STATES, seed + 1000)
        targets = states @ target.T

        def objective(values):
            predictions = states[:N_TRAIN] @ model.unitary(values).T
            return infidelity(predictions, targets[:N_TRAIN])

        def validation(values):
            return infidelity(states[N_TRAIN:] @ model.unitary(values).T, targets[N_TRAIN:])

        checkpoints = []

        def checkpoint(values, history):
            if (len(history.losses) - 1) % 40 == 0:
                checkpoints.append(compare_mentpy(model, states[-1], values))

        trainer = Trainer(optimizer="adam", epochs=EPOCHS, learning_rate=LEARNING_RATE)
        final_values, history = trainer.fit(
            objective,
            list(model.initialize(seed, 0.3).values()),
            validation=validation,
            callback=checkpoint,
        )
        max_checkpoint_error = max(max_checkpoint_error, max(checkpoints))
        trajectories.append(history.validation_losses)
        rows.append(
            {
                "seed": seed,
                "final_train_infidelity": history.losses[-1],
                "final_test_infidelity": history.validation_losses[-1],
                "seconds": history.seconds[-1],
                "max_mentpy_checkpoint_error": max(checkpoints),
            }
        )

    final_test = [r["final_test_infidelity"] for r in rows]
    ci = common.bootstrap_ci(final_test, statistic=np.median, seed=0)
    status = "pass" if (ci["point"] < 0.05 and max_checkpoint_error < tol) else "fail"

    common.save_result(
        rows,
        "R21_ising_gate_learning",
        extra={
            "protocol": "Adam-trained MuTA(2,one_column=True) vs. Ising-XX target, 20 seeds",
            "oracle_class": "A/B",
            "status_category": "statistical",
            "mentpy_commit": MENTPY_COMMIT,
            "seeds": list(SEEDS),
            "config": {
                "n_states": N_STATES,
                "n_train": N_TRAIN,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
            },
            "final_test_infidelity_bootstrap_ci": ci,
            "acceptance_condition": f"median final test infidelity < 0.05; max MentPy checkpoint error < {tol:.3e}",
            "max_mentpy_checkpoint_error": max_checkpoint_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A/B", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for traj in trajectories:
        axes[0].plot(traj, color=common.COLORS["photographiqml"], alpha=0.25, linewidth=0.8)
    median_traj = np.median(np.array(trajectories), axis=0)
    axes[0].plot(median_traj, color=common.COLORS["photographiqml"], linewidth=2.2, label="median")
    axes[0].set_yscale("log")
    axes[0].set(
        title="Held-out test infidelity trajectories (20 seeds)",
        xlabel="Adam step",
        ylabel="infidelity",
    )
    axes[0].legend()
    axes[1].hist(final_test, bins=10, color=common.COLORS["photographiqml"], alpha=0.8)
    axes[1].axvline(
        ci["point"],
        color=common.COLORS["acceptance"],
        linestyle="-",
        label=f"median={ci['point']:.3f}",
    )
    axes[1].axvspan(
        ci["low"],
        ci["high"],
        color=common.COLORS["acceptance"],
        alpha=0.15,
        label="95% bootstrap CI",
    )
    axes[1].set(title="Final test-infidelity distribution", xlabel="infidelity", ylabel="count")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"R21: Ising-XX gate learning, 20 seeds (status={status})")
    common.save_figure(fig, "R21_ising_gate_learning")
    plt.close(fig)

    common.print_summary(
        "R21 Ising gate learning",
        n_seeds=len(SEEDS),
        median_final_test_infidelity=ci["point"],
        ci=(ci["low"], ci["high"]),
        max_mentpy_checkpoint_error=max_checkpoint_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R21 failed: median={ci['point']} checkpoint_error={max_checkpoint_error}"
        )


if __name__ == "__main__":
    main()
