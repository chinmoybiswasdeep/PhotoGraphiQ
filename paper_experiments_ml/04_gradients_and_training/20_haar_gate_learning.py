"""R20: Haar single-wire gate learning across 20 declared seeds.

Scientific question: Can Adam training of a two-wire, one-layer MuTA circuit
learn a random Haar single-wire unitary (acting on wire 0, identity on wire
1) from a finite training set of Haar states, generalize to held-out Haar
states, and does the trained model still agree with an independently
executed MentPy circuit at the final checkpoint?

Theory/equations: infidelity(predictions, targets) = 1 - mean|<pred|target>|^2
(photographiqml.models.infidelity); target = kron(U_Haar, I_2) for a fresh
scipy.stats.unitary_group.rvs(2) draw per seed.

Functionality tested: Trainer(optimizer="adam").fit, MuTA.unitary,
models.haar_states/infidelity, validation.compare_mentpy (checkpoint only).

Oracle and independence class: A for the target (scipy.stats.unitary_group,
an independent analytic/statistical oracle for the Haar target itself); B
for the MentPy training checkpoint (independent external implementation).

Exact/approximate/statistical status: statistical (20 independent seeds;
median and bootstrap 95% CI reported, individual trajectories shown, not
only the best run).

Primary metric: held-out test infidelity at the final epoch, across seeds.

Declared acceptance condition: median final test infidelity < 0.05, with a
declared bootstrap 95% CI reported alongside it (not asserted equivalence,
only a descriptive threshold on the median); MentPy checkpoint density-matrix
error stays below tol = declare_tolerance(scale=1, safety_factor=1000) at
every recorded checkpoint (training must never desynchronize
PhotoGraphiQML's own execution from MentPy's independent execution of the
same bound angles).

Expected cost: moderate (20 seeds x 120 epochs x central-difference gradient
over 3 trainable parameters).

Manuscript destination: Main text (Fig. 6, gate-learning statistics).

Scientific limitations: This is a functionality/statistics demonstration on
a small circuit and dataset, not a claim of quantum advantage or general
learnability; seeds are declared in advance and never cherry-picked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from scipy.stats import unitary_group

from photographiqml import MuTA, Trainer
from photographiqml.models import haar_states, infidelity
from photographiqml.validation import MENTPY_COMMIT, compare_mentpy

EXPERIMENT_ID = "R20"
SEEDS = tuple(range(20))
N_STATES, N_TRAIN, EPOCHS, LEARNING_RATE = 12, 8, 120, 0.05


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1000.0)
    rows = []
    trajectories = []
    max_checkpoint_error = 0.0

    for seed in SEEDS:
        model = MuTA(2, one_column=True)
        target = np.kron(unitary_group.rvs(2, random_state=np.random.default_rng(seed)), np.eye(2))
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
        "R20_haar_gate_learning",
        extra={
            "protocol": "Adam-trained MuTA(2,one_column=True) vs. Haar single-wire target, 20 seeds",
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
    fig.suptitle(f"R20: Haar gate learning, 20 seeds (status={status})")
    common.save_figure(fig, "R20_haar_gate_learning")
    plt.close(fig)

    common.print_summary(
        "R20 Haar gate learning",
        n_seeds=len(SEEDS),
        median_final_test_infidelity=ci["point"],
        ci=(ci["low"], ci["high"]),
        max_mentpy_checkpoint_error=max_checkpoint_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R20 failed: median={ci['point']} checkpoint_error={max_checkpoint_error}"
        )


if __name__ == "__main__":
    main()
