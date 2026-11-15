"""R25: Eq. 5 feature-state agreement against an independent analytic/SciPy
construction.

Scientific question: Does MuTAKernel.features's output agree with an
independent, freshly written full graph-state contraction of the same
5-angle circuit, built via kron-embedded CZ and tensor-reshape projective
measurement (the same independent-contraction method validated generally in
R5, applied here specifically to the kernel's fixed circuit and angle
formula)?

Theory/equations: MuTAKernel.features assigns
{alpha.w0.c0:x0, alpha.w1.c0:x1, alpha.w1.c1:cos(x0)cos(x1), alpha.w0.c2:x0,
alpha.w1.c2:x1} to MuTA(2,1,one_column=True) and reads the deterministic
zero-outcome logical output state.

Functionality tested: MuTAKernel.features (photographiqml.kernels) vs. an
independently coded kron/tensor-reshape contraction (not
photographiqml.logical.cz/local_gate, not validation.contract_branch).

Oracle and independence class: D (independent code path within the
package -- distinct from R11's class-B MentPy oracle for the same feature
map, giving a second, differently-sourced cross-check).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max density-matrix Frobenius error between
MuTAKernel.features's state and the independent contraction's state, over a
grid of (x0, x1) samples.

Declared acceptance condition: max error < tol
(tol = declare_tolerance(scale=1, safety_factor=200)).

Expected cost: light.

Manuscript destination: Main text (Fig. 4, companion analytic cross-check to
R11's MentPy agreement).

Scientific limitations: Fixed to the kernel's specific 2-wire circuit and
angle formula; general MuTA agreement across configurations is R5's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTAKernel

EXPERIMENT_ID = "R25"


def independent_contraction(model, input_state, angles):
    """Fresh kron-embedded-CZ + tensor-reshape-measurement contraction on the
    deterministic zero-outcome branch (see R5 for the general, validated
    version and its correctness argument)."""
    order = list(model.input_nodes) + [v for v in model.graph.nodes if v not in model.input_nodes]
    total = len(order)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    state = np.asarray(input_state, dtype=complex)
    for _ in range(total - model.n_wires):
        state = np.kron(state, plus)
    tensor = state.reshape([2] * total)
    for u, v in model.graph.edges:
        pu, pv = order.index(u), order.index(v)
        lo, hi = sorted((pu, pv))
        index = [slice(None)] * total
        index[lo], index[hi] = 1, 1
        tensor[tuple(index)] *= -1
    state = tensor.reshape(-1)
    remaining = list(order)
    for node in model.measurement_order:
        # Unset names default to 0, matching ParameterStore.bind's own default
        # (model.parameters() starts every stored value at 0.0).
        alpha = angles.get(model.parameter_name(node), 0.0)
        p0 = np.array([1.0, np.exp(-1j * alpha)], dtype=complex) / np.sqrt(2)
        position = remaining.index(node)
        n_now = len(remaining)
        reshaped = state.reshape(2**position, 2, 2 ** (n_now - position - 1))
        state = np.tensordot(reshaped, p0, axes=([1], [0])).reshape(-1)
        mass = float(np.vdot(state, state).real)
        state = state / np.sqrt(mass)
        remaining.remove(node)
    output_positions = [remaining.index(v) for v in model.output_nodes]
    return state.reshape([2] * model.n_wires).transpose(output_positions).reshape(-1)


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=200.0)
    kernel = MuTAKernel()
    generator = common.rng(1)
    X = generator.uniform(-2 * np.pi, 2 * np.pi, size=(10, 2))
    rows = []
    for i, (x0, x1) in enumerate(X):
        angles = {
            "alpha.w0.c0": x0,
            "alpha.w1.c0": x1,
            "alpha.w1.c1": np.cos(x0) * np.cos(x1),
            "alpha.w0.c2": x0,
            "alpha.w1.c2": x1,
        }
        pqml_state = kernel.model.run([1, 0, 0, 0], angles).state
        independent_state = independent_contraction(kernel.model, [1, 0, 0, 0], angles)
        error = common.frobenius_error(
            np.outer(pqml_state, pqml_state.conj()),
            np.outer(independent_state, independent_state.conj()),
        )
        rows.append({"sample": i, "x0": float(x0), "x1": float(x1), "density_matrix_error": error})

    max_error = max(r["density_matrix_error"] for r in rows)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R25_kernel_feature_state",
        extra={
            "protocol": "MuTAKernel.features vs. independent kron/tensor-reshape contraction",
            "oracle_class": "D",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max density-matrix error < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "D", "status": status},
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(
        [r["density_matrix_error"] for r in rows], "o-", color=common.COLORS["photographiqml"]
    )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set(
        title=f"R25: kernel feature-state vs. independent contraction (status={status})",
        xlabel="sample",
        ylabel="density-matrix error",
    )
    ax.legend()
    common.save_figure(fig, "R25_kernel_feature_state")
    plt.close(fig)

    common.print_summary(
        "R25 kernel feature state", n_samples=len(X), tol=tol, max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(f"R25 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
