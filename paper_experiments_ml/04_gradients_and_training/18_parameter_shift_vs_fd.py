"""R18: Central finite-difference convergence versus the analytic two-term
parameter-shift rule.

Scientific question: Does photographiqml.training.finite_difference's
central-difference gradient converge, as the step size shrinks, to the exact
analytic gradient given by the parameter-shift rule -- valid here because
every MuTA measurement angle enters only through gate(a) = H @ exp(i*a*Z/2),
a generator with eigenvalues +-1/2 -- and does the classic U-shaped
finite-difference error curve (decreasing truncation error, then increasing
floating-point cancellation error) appear as expected?

Theory/equations: parameter-shift rule for a generator G with eigenvalues
+-1/2: d/da <O(a)> = <O(a+pi/2)> - <O(a-pi/2)>, exact for ANY scalar
observable expectation, regardless of surrounding circuit complexity, since
Rz(a)=exp(i*a*Z/2) is the only a-dependent factor in gate(a) and Z has
eigenvalues +-1. This experiment computes this analytic gradient by two
extra model.run evaluations at each shifted point (an independent closed-form
rule, not automatic differentiation of the codebase's internals) and compares
it to central finite differences at a grid of step sizes.

Functionality tested: photographiqml.training.finite_difference vs. an
independently coded parameter-shift evaluation, both applied to
MuTA.run(...).expectation(observable) for a fixed random Hermitian
observable.

Oracle and independence class: A (independent analytic oracle -- the
parameter-shift rule is a closed-form theorem, evaluated by fresh code in
this script, not reused from training.py or logical.py).

Exact/approximate/statistical status: exact (up to the well-understood
floating-point floor of finite differences, which this experiment expects
and reports rather than treating as failure).

Primary metric: ||finite_difference_gradient(step) - parameter_shift_gradient||
as a function of step, at several random parameter points.

Declared acceptance condition: minimum error over the step grid is below
tol = declare_tolerance(scale=1, safety_factor=1e8) (chosen loosely to
absorb the eps**(2/3)-scale floor of central differences at their optimal
step, not machine epsilon).

Expected cost: light.

Manuscript destination: Main text (Fig. 5, gradient verification).

Scientific limitations: Validates gradient correctness for a fixed
Hermitian-expectation objective only; training-loop-level convergence
(Adam/SGD/L-BFGS) is R19's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.training import finite_difference

EXPERIMENT_ID = "R18"

STEPS = np.geomspace(1e-1, 1e-8, 15)


def main():
    plt = common.setup_style()
    # Central differences have an intrinsic floor near eps**(2/3) ~ 3.6e-11
    # (truncation ~step^2 balanced against roundoff ~eps/step at the optimal
    # step); safety_factor is set generously above that floor, not near eps.
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e8)
    model = MuTA(2, 1, one_column=True)
    n = model.n_parameters
    generator = common.rng(0)
    hermitian = generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
    observable = hermitian + hermitian.conj().T
    input_state = np.array([1, 0, 0, 0], dtype=complex)

    def objective(x):
        return model.run(input_state, x).expectation(observable)

    def parameter_shift(x):
        grad = np.empty(n)
        for i in range(n):
            shift = np.zeros(n)
            shift[i] = np.pi / 2
            grad[i] = 0.5 * (objective(x + shift) - objective(x - shift))
        return grad

    rows = []
    trial_points = [generator.uniform(-2 * np.pi, 2 * np.pi, n) for _ in range(5)]
    for trial, x0 in enumerate(trial_points):
        analytic_grad = parameter_shift(x0)
        for step in STEPS:
            fd_grad = finite_difference(objective, x0, step=step)
            error = float(np.linalg.norm(fd_grad - analytic_grad))
            rows.append({"trial": trial, "step": float(step), "fd_error": error})

    min_error_per_trial = {
        t: min(r["fd_error"] for r in rows if r["trial"] == t) for t in range(len(trial_points))
    }
    worst_min_error = max(min_error_per_trial.values())
    status = "pass" if worst_min_error < tol else "fail"

    common.save_result(
        rows,
        "R18_parameter_shift_vs_fd",
        extra={
            "protocol": "Central finite differences vs. analytic parameter-shift rule, MuTA expectation objective",
            "oracle_class": "A",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"min(fd_error over step grid) < {tol:.3e} for every trial",
            "worst_min_error_across_trials": worst_min_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A", "status": status},
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for trial in range(len(trial_points)):
        trial_rows = [r for r in rows if r["trial"] == trial]
        ax.loglog(
            [r["step"] for r in trial_rows],
            [r["fd_error"] for r in trial_rows],
            "o-",
            alpha=0.7,
            label=f"trial {trial}",
        )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set(
        title=f"R18: central-FD vs. parameter-shift gradient error (status={status})",
        xlabel="step size",
        ylabel="||fd_grad - analytic_grad||",
    )
    ax.legend(fontsize=7)
    common.save_figure(fig, "R18_parameter_shift_vs_fd")
    plt.close(fig)

    common.print_summary(
        "R18 parameter-shift vs FD",
        n_trials=len(trial_points),
        tol=tol,
        worst_min_error=worst_min_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R18 failed: worst_min_error={worst_min_error} tol={tol}")


if __name__ == "__main__":
    main()
