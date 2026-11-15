"""R41: JAX deterministic gradient validation (experimental; runs only if JAX is installed).

For rotation, displacement, squeezing, cubic phase, Kerr and loss gates
(each with one JAX-differentiable ``Parameter``), compares
``jax.grad`` of ``photographiq.autodiff.expectation`` against central finite
differences of the SAME function at several step sizes h, per the package's
documented validation approach (docs/validation/feature-matrix.md:
"Estimator convergence: Four finite-difference steps"). This checks internal
autodiff/finite-difference consistency of the differentiable Fock generator,
not agreement with an external analytic formula.

Upgrade migration (2026-11): the native C++ engines have NO automatic
differentiation (they are not JAX-traceable; ``photographiq.autodiff`` builds
its own dense JAX generators and prepares inputs with the Piquasso-backed
``MixedFockBackend``). They are used here as an INDEPENDENT numerical
realisation: (1) the forward value of ``autodiff.expectation`` at t0 is
compared with the same observable computed by ``simulate`` on ``native-fock``
(``native-mixed-fock`` for the loss channel) at the same cutoff; (2) a central
FINITE DIFFERENCE of the native-engine observable (step NATIVE_FD_STEP) is
compared with the JAX gradient. Finite differences are labelled as such and
never reported as automatic differentiation. Declared tolerances:
FORWARD_TOLERANCE and NATIVE_FD_TOLERANCE (central-difference truncation
~h^2 times the third derivative plus rounding ~eps/h).
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
NATIVE_FD_STEP = 1e-4
# The two realisations differ by truncation at a finite cutoff (projected dense generator vs
# exact projected matrix elements; e.g. squeezing differs by 6e-5 at CUTOFF = 12), so the
# cross-check runs both at a converged cutoff; the CUTOFF = 12 difference is reported too.
CROSS_CUTOFF = 40
FORWARD_TOLERANCE = 1e-9
NATIVE_FD_TOLERANCE = 1e-5  # relative
SUPPORTED_COMBINATIONS = {
    "jax.grad(autodiff.expectation)": "automatic differentiation (JAX dense generators; Piquasso-prepared inputs)",
    "native-fock / native-mixed-fock": "values only; finite differences possible, no autodiff",
    "native-gaussian": "values only; no autodiff",
    "piquasso-fock": "values only through simulate(); differentiable path is photographiq.autodiff",
}


def native_observable(pattern, t, observable, gate_name, cutoff=CROSS_CUTOFF):
    import numpy as np

    backend = "native-mixed-fock" if gate_name == "loss" else "native-fock"
    state = pg.simulate(pattern, backend=backend, cutoff=cutoff, parameters={"t": float(t)},
                        inputs={0: pg.GaussianInput.coherent(ALPHA)}, seed=0).state
    if observable == "photon_number":
        return state.photon_number(0)
    return state.quadrature(0, 0.0 if observable == "q" else np.pi / 2)[0]
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
    native_rows = []
    for name, (gate_builder, observable) in GATES.items():
        gate = gate_builder(pg.Parameter("t"))
        pattern = pg.Pattern(inputs=(0,)).append(gate).append(pg.Output((0,)))
        # Keep the loss transmissivity in (0,1]: evaluate around t0 scaled down.
        t0 = T0 if name != "loss" else 0.5

        def f(t, pattern=pattern, observable=observable):
            return expectation(pattern, {"t": t}, cutoff=CUTOFF, inputs=inputs, observable=observable)

        grad_value = float(jax.grad(f)(jnp.asarray(t0)))
        def f_cross(t, pattern=pattern, observable=observable):
            return expectation(pattern, {"t": t}, cutoff=CROSS_CUTOFF, inputs=inputs, observable=observable)

        forward_at_cutoff = abs(float(f(t0)) - native_observable(pattern, t0, observable, name, CUTOFF))
        grad_cross = float(jax.grad(f_cross)(jnp.asarray(t0)))
        forward_jax = float(f_cross(t0))
        forward_native = native_observable(pattern, t0, observable, name)
        native_fd = (native_observable(pattern, t0 + NATIVE_FD_STEP, observable, name)
                     - native_observable(pattern, t0 - NATIVE_FD_STEP, observable, name)) / (2 * NATIVE_FD_STEP)
        native_rows.append({
            "gate": name, "observable": observable, "t0": t0,
            "native_backend": "native-mixed-fock" if name == "loss" else "native-fock",
            "jax_forward_value": forward_jax, "native_forward_value": forward_native,
            "cross_check_cutoff": CROSS_CUTOFF,
            "forward_difference": abs(forward_jax - forward_native),
            "forward_difference_at_main_cutoff": forward_at_cutoff,
            "jax_autodiff_gradient": grad_cross,
            "native_finite_difference_gradient": native_fd, "finite_difference_step": NATIVE_FD_STEP,
            "relative_difference_autodiff_vs_native_fd": abs(grad_cross - native_fd) / max(abs(grad_cross), 1e-12),
        })
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
    common.save_csv(native_rows, "R41_jax_gradients_native_crosscheck")
    failures += [
        f"{r['gate']} (native cross-check)" for r in native_rows
        if r["forward_difference"] > FORWARD_TOLERANCE
        or r["relative_difference_autodiff_vs_native_fd"] > NATIVE_FD_TOLERANCE
    ]
    if failures:
        common.save_csv([r for r in rows if r["gate"] in failures] + native_rows, "R41_jax_gradients_FAILURES")
        raise AssertionError(f"No tested step size matched autodiff for gates: {failures}; see FAILURES csv")

    common.save_result(rows, "R41_jax_gradients", extra={
        "cutoff": CUTOFF, "alpha": [ALPHA.real, ALPHA.imag], "native_crosscheck": native_rows,
        "supported_combinations": SUPPORTED_COMBINATIONS, "forward_tolerance": FORWARD_TOLERANCE,
        "native_fd_tolerance": NATIVE_FD_TOLERANCE,
        "note": "finite differences are reported as finite differences; native engines provide no autodiff"})

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

    common.print_summary(
        "R41 JAX deterministic gradient validation",
        gates=list(GATES),
        best_relative_errors=best_by_gate,
        max_forward_difference_vs_native=max(r["forward_difference"] for r in native_rows),
        max_autodiff_vs_native_fd=max(r["relative_difference_autodiff_vs_native_fd"] for r in native_rows),
    )


if __name__ == "__main__":
    main()
