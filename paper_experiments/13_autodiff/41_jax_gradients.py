"""R41: JAX deterministic gradient validation (experimental; runs only if JAX is installed).

For rotation, displacement, squeezing, cubic phase, Kerr and loss gates
(each with one JAX-differentiable ``Parameter``), compares
``jax.grad`` of ``photographiq.autodiff.expectation`` against central finite
differences of the SAME function at several step sizes h, per the package's
documented validation approach (docs/validation/feature-matrix.md:
"Estimator convergence: Four finite-difference steps"). This checks internal
autodiff/finite-difference consistency of the differentiable Fock generator,
not agreement with an external analytic formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

if not common.HAS_JAX:
    print("SKIPPED: jax is not installed; see requirements_notes.md")
    sys.exit(0)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.autodiff import expectation  # noqa: E402

CUTOFF = 12
ALPHA = 0.3 + 0.2j
T0 = 0.4
STEP_SIZES = [1e-2, 1e-3, 1e-4, 1e-5]

GATES = {
    "rotation": (lambda t: pg.Rotate(0, t), "q"),
    "displacement": (lambda t: pg.Displace(0, q=t), "q"),
    "squeezing": (lambda t: pg.Squeeze(0, t), "photon_number"),
    "cubic_phase": (lambda t: pg.CubicPhase(0, t), "p"),
    "kerr": (lambda t: pg.Kerr(0, t), "q"),
    "loss": (lambda t: pg.Loss(0, t, 0.0), "photon_number"),
}


def main():
    plt = common.setup_style()
    inputs = {0: pg.GaussianInput.coherent(ALPHA)}
    rows = []

    for name, (gate_builder, observable) in GATES.items():
        gate = gate_builder(pg.Parameter("t"))
        pattern = pg.Pattern(inputs=(0,)).append(gate).append(pg.Output((0,)))
        # Keep the loss transmissivity in (0,1]: evaluate around t0 scaled down.
        t0 = T0 if name != "loss" else 0.5

        def f(t, pattern=pattern, observable=observable):
            return expectation(pattern, {"t": t}, cutoff=CUTOFF, inputs=inputs, observable=observable)

        grad_value = float(jax.grad(f)(jnp.asarray(t0)))
        for h in STEP_SIZES:
            fd_value = float((f(t0 + h) - f(t0 - h)) / (2 * h))
            relative_error = abs(grad_value - fd_value) / max(abs(grad_value), 1e-12)
            rows.append(
                {
                    "gate": name,
                    "observable": observable,
                    "t0": t0,
                    "step_size": h,
                    "jax_gradient": grad_value,
                    "finite_difference_gradient": fd_value,
                    "absolute_error": abs(grad_value - fd_value),
                    "relative_error": relative_error,
                }
            )
        print(f"  {name}: jax_grad={grad_value:.8f}", flush=True)

    # The finite-difference approximation should agree with autodiff to a few
    # times h^2 (central-difference truncation error) for a well-behaved h;
    # too-small h instead suffers floating-point cancellation. Require the
    # BEST of the tested step sizes to be accurate, not every one of them.
    tolerance = 1e-5
    best_by_gate = {}
    for row in rows:
        best_by_gate[row["gate"]] = min(best_by_gate.get(row["gate"], float("inf")), row["relative_error"])
    failures = [gate for gate, err in best_by_gate.items() if err > tolerance]
    if failures:
        common.save_csv([r for r in rows if r["gate"] in failures], "R41_jax_gradients_FAILURES")
        raise AssertionError(f"No tested step size matched autodiff for gates: {failures}; see FAILURES csv")

    common.save_result(rows, "R41_jax_gradients", extra={"cutoff": CUTOFF, "alpha": [ALPHA.real, ALPHA.imag]})

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in GATES:
        subset = [r for r in rows if r["gate"] == name]
        ax.loglog([r["step_size"] for r in subset], [max(1e-16, r["relative_error"]) for r in subset], "o-", label=name)
    ax.set_xlabel("Finite-difference step size h")
    ax.set_ylabel("Relative gradient error vs. JAX autodiff")
    ax.legend(fontsize=7)
    ax.set_title("JAX autodiff vs. central finite differences (experimental)")
    common.save_figure(fig, "R41_jax_gradients")
    plt.close(fig)

    common.print_summary("R41 JAX deterministic gradient validation", gates=list(GATES), best_relative_errors=best_by_gate)


if __name__ == "__main__":
    main()
