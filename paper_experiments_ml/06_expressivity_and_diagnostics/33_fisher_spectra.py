"""R33: Local state-Fisher spectra, rank, condition number, and
finite-difference step convergence versus wires and depth.

Scientific question: Is photographiqml.expressivity.state_fisher's local
pure-state QFI matrix positive-semidefinite (as any Fisher information
matrix must be, an analytic invariant), does its central-difference
construction converge as the step size shrinks toward a step-independent
limit, and how do its rank and condition number scale with wire count and
circuit depth?

Theory/equations: state_fisher's docstring states this is a *local*
diagnostic (rank is not a statistical effective dimension or finite-depth
inclusion proof, per docs). This experiment checks the analytic invariant
(PSD) and the numerical-convergence invariant (step-size stability), both
properties any correct implementation must satisfy, independent of any
external oracle.

Functionality tested: photographiqml.expressivity.state_fisher.

Oracle and independence class: A for the PSD check (an analytic invariant
of any Fisher information matrix); E for the step-convergence check
(self-consistency against a much smaller reference step).

Exact/approximate/statistical status: exact for PSD (up to roundoff);
approximate for step convergence (central-difference truncation error).

Primary metric: minimum eigenvalue (PSD check); ||F(step) - F(step_ref)||
vs. step (convergence check); rank (numerical, tol=1e-8) and condition
number vs. (n_wires, n_layers).

Declared acceptance condition: minimum eigenvalue > -tol
(tol = declare_tolerance(scale=4, safety_factor=1e4)); convergence error
strictly decreases as step shrinks from 1e-2 to 1e-5 (before the expected
central-difference roundoff floor at very small steps).

Expected cost: light.

Manuscript destination: Appendix (expressivity diagnostics table, Fig. 8).

Scientific limitations: state_fisher's rank is explicitly a *local*
diagnostic at one parameter point; it is not evidence of finite-depth
expressivity or a global statement about the parameter landscape (see
docstring and R32's Lie-closure caveat).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.expressivity import state_fisher

EXPERIMENT_ID = "R33"


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=4.0, safety_factor=1e4)
    rows_psd = []
    rows_scaling = []
    configs = [(1, 1), (2, 1), (2, 2), (3, 1)]
    for n_wires, n_layers in configs:
        model = MuTA(n_wires, n_layers, one_column=True)
        input_state = np.eye(2**n_wires, dtype=complex)[0]
        parameters = common.rng(n_wires * 10 + n_layers).uniform(-np.pi, np.pi, model.n_parameters)
        fisher = state_fisher(model, input_state, parameters, step=1e-5)
        eigenvalues = np.linalg.eigvalsh(fisher)
        rank = int(np.sum(eigenvalues > 1e-8 * max(eigenvalues.max(), 1e-30)))
        condition_number = (
            float(
                eigenvalues.max()
                / max(
                    eigenvalues.min(where=eigenvalues > 1e-12, initial=np.inf), np.finfo(float).eps
                )
            )
            if rank
            else float("nan")
        )
        rows_psd.append(
            {"n_wires": n_wires, "n_layers": n_layers, "min_eigenvalue": float(eigenvalues.min())}
        )
        rows_scaling.append(
            {
                "n_wires": n_wires,
                "n_layers": n_layers,
                "n_parameters": model.n_parameters,
                "rank": rank,
                "condition_number": condition_number,
            }
        )

    # --- step convergence, fixed config ------------------------------------
    model = MuTA(2, 1, one_column=True)
    input_state = np.eye(4, dtype=complex)[0]
    parameters = common.rng(0).uniform(-np.pi, np.pi, model.n_parameters)
    reference = state_fisher(model, input_state, parameters, step=1e-6)
    steps = [1e-2, 1e-3, 1e-4, 1e-5]
    convergence_rows = []
    for step in steps:
        fisher = state_fisher(model, input_state, parameters, step=step)
        convergence_rows.append(
            {"step": step, "error_vs_reference": common.frobenius_error(fisher, reference)}
        )

    min_eig_overall = min(r["min_eigenvalue"] for r in rows_psd)
    errors = [r["error_vs_reference"] for r in convergence_rows]
    monotonic_decrease = all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    status = "pass" if (min_eig_overall > -tol and monotonic_decrease) else "fail"

    common.save_result(
        rows_psd
        + [{"step_convergence": True, **r} for r in convergence_rows]
        + [{"scaling": True, **r} for r in rows_scaling],
        "R33_fisher_spectra",
        extra={
            "protocol": "state_fisher PSD check and step-convergence, plus rank/conditioning vs wires/depth",
            "oracle_class": "A/E",
            "status_category": "exact/approximate",
            "tolerance": tol,
            "step_convergence_rows": convergence_rows,
            "scaling_rows": rows_scaling,
            "acceptance_condition": f"min eigenvalue > -{tol:.3e}; convergence error monotonically decreases from step=1e-2 to 1e-5",
            "min_eigenvalue_overall": min_eig_overall,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A/E", "status": status},
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(
        range(len(rows_psd)),
        [r["min_eigenvalue"] for r in rows_psd],
        color=common.COLORS["photographiqml"],
    )
    axes[0].axhline(0, color=common.COLORS["acceptance"], linewidth=0.8)
    axes[0].set_xticks(range(len(rows_psd)))
    axes[0].set_xticklabels([f"n{r['n_wires']}L{r['n_layers']}" for r in rows_psd], fontsize=7)
    axes[0].set(title="Min eigenvalue (PSD check)", ylabel="eigenvalue")
    axes[1].loglog(steps, errors, "o-", color=common.COLORS["photographiqml"])
    axes[1].set(title="Step convergence", xlabel="step", ylabel="||F(step)-F(ref)||")
    axes[2].bar(
        range(len(rows_scaling)),
        [r["rank"] for r in rows_scaling],
        color=common.COLORS["photographiqml"],
        alpha=0.7,
        label="rank",
    )
    axes[2].plot(
        range(len(rows_scaling)),
        [r["n_parameters"] for r in rows_scaling],
        "s--",
        color=common.COLORS["analytic"],
        label="n_parameters",
    )
    axes[2].set_xticks(range(len(rows_scaling)))
    axes[2].set_xticklabels([f"n{r['n_wires']}L{r['n_layers']}" for r in rows_scaling], fontsize=7)
    axes[2].set(title="Local Fisher rank vs. n_parameters", ylabel="count")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"R33: local state-Fisher spectra (status={status})")
    common.save_figure(fig, "R33_fisher_spectra")
    plt.close(fig)

    common.print_summary(
        "R33 Fisher spectra",
        n_configs=len(rows_psd),
        min_eigenvalue=min_eig_overall,
        monotonic_decrease=monotonic_decrease,
        status=status,
    )
    if status != "pass":
        raise AssertionError(
            f"R33 failed: min_eig={min_eig_overall} monotonic={monotonic_decrease}"
        )


if __name__ == "__main__":
    main()
