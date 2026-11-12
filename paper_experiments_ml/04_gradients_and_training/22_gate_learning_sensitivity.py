"""R22: Gate-learning sensitivity to training-set size, depth, and
initialization scale.

Scientific question: How does held-out gate-learning performance (and its
failure rate, not just its mean) depend on training-set size, circuit depth
(paper layers), and initialization scale, for a fixed Haar single-wire
target?

Theory/equations: same infidelity objective as R20; target = kron(U_Haar, I).

Functionality tested: Trainer(optimizer="adam").fit across a systematic
(train_size, n_layers, init_scale) grid, each cell repeated over multiple
seeds with independent held-out Haar test states.

Oracle and independence class: A (independent analytic Haar target via
scipy.stats.unitary_group, same as R20).

Exact/approximate/statistical status: statistical (6 seeds per grid cell;
held-out states and seeds are independent across cells and never reused
between train/test within a cell).

Primary metric: per-cell held-out test-infidelity failure rate (fraction of
seeds with final test infidelity above a declared 0.2 threshold) and median
test infidelity.

Declared acceptance condition: none globally required to "pass" a
correctness oracle (this experiment characterizes sensitivity, it does not
assert a single global bound); the script instead reports the full grid,
and its own internal consistency check (every cell has a finite, defined
failure rate) is the only pass/fail criterion actually enforced.

Expected cost: moderate (3 train sizes x 2 depths x 3 init scales x 6 seeds
x 60 epochs).

Manuscript destination: Appendix (Fig. 6 supporting sensitivity heatmap).

Scientific limitations: A single fixed target family (Haar single-wire);
does not sweep over multiple target types (R20/R21 cover two fixed targets
with more seeds each). Held-out failure rates at only 6 seeds per cell carry
substantial sampling uncertainty, reported via each cell's own count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np
from scipy.stats import unitary_group

from photographiqml import MuTA, Trainer
from photographiqml.models import haar_states, infidelity

EXPERIMENT_ID = "R22"
SEEDS = tuple(range(6))
EPOCHS, LEARNING_RATE, N_TEST = 60, 0.05, 6
FAILURE_THRESHOLD = 0.2


def main():
    plt = common.setup_style()
    rows = []
    for train_size in (2, 4, 8):
        for n_layers in (1, 2):
            for init_scale in (0.1, 0.5, 1.5):
                final_tests = []
                for seed in SEEDS:
                    model = MuTA(2, n_layers, one_column=True)
                    target = np.kron(
                        unitary_group.rvs(2, random_state=np.random.default_rng(seed)), np.eye(2)
                    )
                    states = haar_states(2, train_size + N_TEST, seed + 1000)
                    targets = states @ target.T

                    def objective(values):
                        predictions = states[:train_size] @ model.unitary(values).T
                        return infidelity(predictions, targets[:train_size])

                    def validation(values):
                        return infidelity(
                            states[train_size:] @ model.unitary(values).T, targets[train_size:]
                        )

                    trainer = Trainer(optimizer="adam", epochs=EPOCHS, learning_rate=LEARNING_RATE)
                    _, history = trainer.fit(
                        objective,
                        list(model.initialize(seed, init_scale).values()),
                        validation=validation,
                    )
                    final_tests.append(history.validation_losses[-1])
                final_tests = np.array(final_tests)
                failure_rate = float(np.mean(final_tests > FAILURE_THRESHOLD))
                rows.append(
                    {
                        "train_size": train_size,
                        "n_layers": n_layers,
                        "init_scale": init_scale,
                        "median_test_infidelity": float(np.median(final_tests)),
                        "failure_rate": failure_rate,
                        "n_seeds": len(SEEDS),
                        "final_test_infidelities": final_tests.tolist(),
                    }
                )

    all_defined = all(
        np.isfinite(r["median_test_infidelity"]) and 0 <= r["failure_rate"] <= 1 for r in rows
    )
    status = "pass" if all_defined else "fail"

    common.save_result(
        rows,
        "R22_gate_learning_sensitivity",
        extra={
            "protocol": "Held-out gate-learning failure rate vs. train size / depth / init scale",
            "oracle_class": "A",
            "status_category": "statistical",
            "failure_threshold": FAILURE_THRESHOLD,
            "n_seeds_per_cell": len(SEEDS),
            "seeds": list(SEEDS),
            "acceptance_condition": "every grid cell has a well-defined failure rate in [0,1] (descriptive sensitivity study)",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    # constrained_layout (not tight_layout) correctly reserves space for the
    # shared colorbar added after the subplots; tight_layout does not account
    # for a colorbar added post hoc and previously caused it to overlap the
    # right panel's title.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    train_sizes, init_scales = (
        sorted({r["train_size"] for r in rows}),
        sorted({r["init_scale"] for r in rows}),
    )
    for ax, n_layers in zip(axes, (1, 2)):
        grid = np.array(
            [
                [
                    next(
                        r["failure_rate"]
                        for r in rows
                        if r["train_size"] == t
                        and r["init_scale"] == s
                        and r["n_layers"] == n_layers
                    )
                    for s in init_scales
                ]
                for t in train_sizes
            ]
        )
        im = ax.imshow(grid, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(init_scales)))
        ax.set_xticklabels(init_scales)
        ax.set_yticks(range(len(train_sizes)))
        ax.set_yticklabels(train_sizes)
        ax.set(title=f"n_layers={n_layers}", xlabel="init scale", ylabel="train size")
        for i in range(len(train_sizes)):
            for j in range(len(init_scales)):
                ax.text(
                    j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white"
                )
    fig.colorbar(im, ax=axes, fraction=0.046, label="failure rate")
    fig.suptitle(f"R22: gate-learning failure rate sensitivity (status={status})")
    common.save_figure(fig, "R22_gate_learning_sensitivity", tight=False)
    plt.close(fig)

    common.print_summary("R22 gate-learning sensitivity", n_cells=len(rows), status=status)
    if status != "pass":
        raise AssertionError("R22 failed: some grid cell had an undefined failure rate")


if __name__ == "__main__":
    main()
