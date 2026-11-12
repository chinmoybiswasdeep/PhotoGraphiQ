"""R19: Adam, SGD, and L-BFGS convergence on independent analytic objectives.

Scientific question: Does photographiqml.training.Trainer's Adam, SGD and
L-BFGS optimizers converge to the known analytic global minimum of standard
test objectives (a convex quadratic bowl and the non-convex Rosenbrock
function), with correct gradient/loss diagnostics recorded in History?

Theory/equations: quadratic bowl f(x) = ||x - x*||^2 has global minimum 0 at
x=x*, gradient 2(x-x*); Rosenbrock f(x) = sum_i [100(x_{i+1}-x_i^2)^2 +
(1-x_i)^2] has global minimum 0 at x=(1,...,1). Both are classical
optimization test functions, independent of any MuTA circuit.

Functionality tested: photographiqml.training.Trainer.fit (optimizer=
"adam"/"sgd"/"lbfgs") and History (photographiqml/training.py).

Oracle and independence class: A (independent analytic oracle -- the target
minimum and its location are closed-form and unrelated to MuTA).

Exact/approximate/statistical status: exact target, approximate convergence
(iterative optimization; final loss compared to a declared threshold).

Primary metric: final loss value and distance to the analytic minimizer,
for each (optimizer, objective) pair.

Declared acceptance condition: quadratic bowl reaches loss < 1e-6 for every
optimizer; Rosenbrock reaches loss < 0.5 for L-BFGS (expected to nail this
well-conditioned-at-the-end valley) and loss < 5.0 for Adam/SGD (fixed
small-learning-rate first-order methods are not required to fully converge
on this narrow non-convex valley in a fixed epoch budget -- substantial
descent from the initial loss (~890) is the meaningful, honestly-scoped
pass condition for them, not near-exact convergence).

Expected cost: light.

Manuscript destination: Appendix (optimizer sanity-check table supporting
Fig. 5/6).

Scientific limitations: This validates the optimizer loop's mechanics on
classical, well-understood objectives; MuTA-specific gate-learning
convergence statistics are R20-R22's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import Trainer

EXPERIMENT_ID = "R19"


def quadratic_bowl(x, target):
    return float(np.sum((x - target) ** 2))


def quadratic_gradient(x, target):
    return 2 * (x - target)


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return grad


def main():
    plt = common.setup_style()
    rows = []
    generator = common.rng(0)
    target = generator.uniform(-2, 2, size=4)
    initial_bowl = generator.uniform(-3, 3, size=4)
    initial_rosen = np.array([-1.5, 1.5, -1.0, 1.0])

    configs = [
        {
            "objective": "quadratic_bowl",
            "optimizer": "adam",
            "epochs": 300,
            "lr": 0.1,
            "threshold": 1e-6,
        },
        {
            "objective": "quadratic_bowl",
            "optimizer": "sgd",
            "epochs": 300,
            "lr": 0.1,
            "threshold": 1e-6,
        },
        {
            "objective": "quadratic_bowl",
            "optimizer": "lbfgs",
            "epochs": 100,
            "lr": 0.1,
            "threshold": 1e-6,
        },
        {
            "objective": "rosenbrock",
            "optimizer": "adam",
            "epochs": 3000,
            "lr": 0.01,
            "threshold": 5.0,
        },
        {
            "objective": "rosenbrock",
            "optimizer": "sgd",
            "epochs": 3000,
            "lr": 0.001,
            "threshold": 5.0,
        },
        {
            "objective": "rosenbrock",
            "optimizer": "lbfgs",
            "epochs": 200,
            "lr": 0.01,
            "threshold": 0.5,
        },
    ]
    histories = {}
    for cfg in configs:
        trainer = Trainer(optimizer=cfg["optimizer"], epochs=cfg["epochs"], learning_rate=cfg["lr"])
        if cfg["objective"] == "quadratic_bowl":
            objective = lambda x: quadratic_bowl(x, target)  # noqa: E731
            gradient = lambda x: quadratic_gradient(x, target)  # noqa: E731
            x0 = initial_bowl
        else:
            objective, gradient, x0 = rosenbrock, rosenbrock_gradient, initial_rosen
        final_x, history = trainer.fit(objective, x0, gradient=gradient)
        final_loss = history.losses[-1]
        distance_to_target = (
            float(np.linalg.norm(final_x - target))
            if cfg["objective"] == "quadratic_bowl"
            else float(np.linalg.norm(final_x - 1))
        )
        passed = final_loss < cfg["threshold"]
        histories[f"{cfg['objective']}_{cfg['optimizer']}"] = history
        rows.append(
            {
                "objective": cfg["objective"],
                "optimizer": cfg["optimizer"],
                "epochs": cfg["epochs"],
                "final_loss": final_loss,
                "threshold": cfg["threshold"],
                "distance_to_analytic_minimizer": distance_to_target,
                "final_gradient_norm": history.gradient_norms[-1],
                "passed": passed,
            }
        )

    n_failed = sum(1 for r in rows if not r["passed"])
    status = "pass" if n_failed == 0 else "fail"

    common.save_result(
        rows,
        "R19_optimizer_convergence",
        extra={
            "protocol": "Trainer(adam/sgd/lbfgs).fit on quadratic bowl and Rosenbrock, vs. analytic minimum",
            "oracle_class": "A",
            "status_category": "approximate",
            "acceptance_condition": "final_loss < declared threshold for every (objective, optimizer) pair",
            "n_failed": n_failed,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, history in histories.items():
        ax = axes[0] if "quadratic" in name else axes[1]
        ax.semilogy(history.losses, label=name.split("_")[-1])
    axes[0].set(title="Quadratic bowl", xlabel="iteration", ylabel="loss")
    axes[1].set(title="Rosenbrock", xlabel="iteration", ylabel="loss")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"R19: optimizer convergence on analytic objectives (status={status})")
    common.save_figure(fig, "R19_optimizer_convergence")
    plt.close(fig)

    common.print_summary(
        "R19 optimizer convergence", n_configs=len(rows), n_failed=n_failed, status=status
    )
    if status != "pass":
        raise AssertionError(
            f"R19 failed cases: {[(r['objective'], r['optimizer']) for r in rows if not r['passed']]}"
        )


if __name__ == "__main__":
    main()
