"""R11: MuTA Eq. 5 kernel state and Gram-matrix agreement against
independently executed MentPy circuits.

Scientific question: Does MuTAKernel's feature map (a fixed two-wire,
one-layer MuTA circuit with the paper's Eq. 5 angle assignment) produce the
same output density matrices -- and therefore the same Gram (fidelity)
matrix -- as an independently executed MentPy simulation of the identical
semantic circuit and angles?

Theory/equations: MuTAKernel.features assigns
{alpha.w0.c0:x0, alpha.w1.c0:x1, alpha.w1.c1:cos(x0)cos(x1), alpha.w0.c2:x0,
alpha.w1.c2:x1} (kernels.py) to a MuTA(2,1,one_column=True) circuit; the
kernel value is k(x,y) = |<psi(x)|psi(y)>|^2 = Tr(rho(x) rho(y)) for pure
states.

Functionality tested: MuTAKernel.features / gram_matrix (photographiqml.kernels)
vs. mp.PatternSimulator executing the semantically mapped identical circuit
(mentpy), via validation.mentpy_reference's mapping (a small, independent
simulator-construction helper is written fresh in this script since
validation.compare_mentpy only returns a scalar error, not the density
matrix needed to build an independent Gram matrix).

Oracle and independence class: B (independent external implementation).

Exact/approximate/statistical status: exact, up to floating-point roundoff.

Primary metric: max per-sample density-matrix error between PhotoGraphiQML
and MentPy; max entrywise Gram-matrix error between kernel.gram_matrix(X) and
the independently assembled MentPy Gram matrix Tr(rho_i rho_j).

Declared acceptance condition: both max errors < tol
(tol = declare_tolerance(scale=1, safety_factor=1000)).

Expected cost: light (small fixed 2-wire kernel circuit; <=12 samples).

Manuscript destination: Main text (Fig. 4, kernel/learning section).

Scientific limitations: MentPy validates the ideal logical kernel only; no
raw-CV MentPy oracle exists for any future physical kernel (see
docs/research/muta-mapping.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTAKernel
from photographiqml.validation import MENTPY_COMMIT, mentpy_reference

EXPERIMENT_ID = "R11"


def mentpy_density_matrix(model, input_state, parameters):
    """Independently execute the semantically mapped MentPy circuit and
    return its output density matrix (mirrors validation.compare_mentpy's
    internal simulator construction, which only exposes a scalar error)."""
    import mentpy as mp

    reference, mapping = mentpy_reference(model, fix_measurements=True)
    angles = model._parameters.bind(parameters)
    values = [angles[model.parameter_name(mapping[v])] for v in reference.trainable_nodes]
    reverse = {v: k for k, v in mapping.items()}
    schedule = [reverse[v] for v in model.measurement_order + model.output_nodes]
    positions = {v: i for i, v in enumerate(schedule)}
    window = max(abs(positions[u] - positions[v]) + 1 for u, v in reference.graph.edges)
    window = max(window, model.n_wires + 1)
    simulator = mp.PatternSimulator(
        reference,
        input_state=np.asarray(input_state, complex),
        backend="numpy-sv",
        schedule=schedule,
        window_size=window,
    )
    return simulator.run(values, output_form="dm")


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1000.0)
    kernel = MuTAKernel()
    generator = common.rng(0)
    X = generator.uniform(-2 * np.pi, 2 * np.pi, size=(8, 2))

    state_rows = []
    mentpy_densities = []
    for i, (x0, x1) in enumerate(X):
        angles = {
            "alpha.w0.c0": x0,
            "alpha.w1.c0": x1,
            "alpha.w1.c1": np.cos(x0) * np.cos(x1),
            "alpha.w0.c2": x0,
            "alpha.w1.c2": x1,
        }
        pqml_state = kernel.model.run([1, 0, 0, 0], angles).state
        pqml_density = np.outer(pqml_state, pqml_state.conj())
        mp_density = mentpy_density_matrix(kernel.model, [1, 0, 0, 0], angles)
        mentpy_densities.append(mp_density)
        error = common.frobenius_error(pqml_density, mp_density)
        state_rows.append(
            {"sample": i, "x0": float(x0), "x1": float(x1), "density_matrix_error": error}
        )

    pqml_gram = kernel.gram_matrix(X)
    mentpy_gram = np.array(
        [
            [
                float(np.real(np.trace(mentpy_densities[i] @ mentpy_densities[j])))
                for j in range(len(X))
            ]
            for i in range(len(X))
        ]
    )
    gram_error = common.frobenius_error(pqml_gram, mentpy_gram)

    max_state_error = max(r["density_matrix_error"] for r in state_rows)
    status = "pass" if max(max_state_error, gram_error) < tol else "fail"

    common.save_result(
        state_rows,
        "R11_kernel_gram_agreement",
        extra={
            "protocol": "MuTAKernel.features/gram_matrix vs. independently executed MentPy PatternSimulator",
            "oracle_class": "B",
            "status_category": "exact",
            "mentpy_commit": MENTPY_COMMIT,
            "tolerance": tol,
            "acceptance_condition": f"max density-matrix error and max Gram-matrix error < {tol:.3e}",
            "max_state_error": max_state_error,
            "gram_matrix_error": gram_error,
            "photographiqml_gram": pqml_gram.tolist(),
            "mentpy_gram": mentpy_gram.tolist(),
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "B", "status": status},
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].semilogy(
        [r["density_matrix_error"] for r in state_rows], "o-", color=common.COLORS["photographiqml"]
    )
    axes[0].axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    axes[0].set(title="Per-sample state error", xlabel="sample", ylabel="density-matrix error")
    axes[0].legend()
    im1 = axes[1].imshow(pqml_gram, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("PhotoGraphiQML Gram matrix")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(abs(pqml_gram - mentpy_gram), cmap="magma")
    axes[2].set_title(f"|Gram diff| (max={gram_error:.2e})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f"R11: Eq. 5 kernel vs. MentPy (status={status})")
    common.save_figure(fig, "R11_kernel_gram_agreement")
    plt.close(fig)

    common.print_summary(
        "R11 kernel/Gram agreement",
        n_samples=len(X),
        tol=tol,
        max_state_error=max_state_error,
        gram_matrix_error=gram_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R11 failed: state={max_state_error} gram={gram_error}")


if __name__ == "__main__":
    main()
