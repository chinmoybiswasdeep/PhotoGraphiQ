"""R35: Fock-space dimension and memory model (analytical, cheap).

Uses the actual PhotoGraphiQ total-photon Fock dimension formula
D = binom(modes+cutoff-1, modes) (``PiquassoFockBackend.dimension``, cross-
checked against ``common.fock_dimension``) over a modes x cutoff grid, and
estimates pure-vector and dense-density-matrix byte costs. The default
``max_dimension`` allocation guard is read directly from the backend class.
Dense-density-matrix bytes are a rough dense upper-style estimate; the actual
mixed-Fock implementation may differ (see docs/performance.md).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from photographiq.backends.fock import PiquassoFockBackend  # noqa: E402

MODES = list(range(1, 21))
CUTOFFS = list(range(2, 41))


def default_max_dimension() -> int:
    return inspect.signature(PiquassoFockBackend.__init__).parameters["max_dimension"].default


def main():
    plt = common.setup_style()
    max_dimension = default_max_dimension()

    backend = PiquassoFockBackend(cutoff=CUTOFFS[0])
    rows = []
    for modes in MODES:
        for cutoff in CUTOFFS:
            backend.cutoff = cutoff
            dimension_native = backend.dimension(modes)
            dimension_independent = common.fock_dimension(modes, cutoff)
            if dimension_native != dimension_independent:
                raise AssertionError(f"Dimension formulas disagree at modes={modes}, cutoff={cutoff}")
            rows.append(
                {
                    "modes": modes,
                    "cutoff": cutoff,
                    "dimension": dimension_native,
                    "pure_vector_bytes": common.vector_bytes(dimension_native),
                    "dense_density_matrix_bytes_rough_estimate": common.density_bytes(dimension_native),
                    "exceeds_default_max_dimension": dimension_native > max_dimension,
                }
            )

    common.save_result(rows, "R35_fock_dimension_scaling", extra={"default_max_dimension": max_dimension})

    dims = np.array([[common.fock_dimension(m, c) for c in CUTOFFS] for m in MODES])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mesh = axes[0].imshow(
        np.log10(dims), aspect="auto", origin="lower", cmap="viridis",
        extent=[CUTOFFS[0], CUTOFFS[-1], MODES[0], MODES[-1]],
    )
    axes[0].set_xlabel("Cutoff")
    axes[0].set_ylabel("Modes")
    axes[0].set_title(r"$\log_{10}$(Fock dimension)")
    fig.colorbar(mesh, ax=axes[0])

    for modes in (1, 2, 4, 8, 16, 20):
        axes[1].semilogy(CUTOFFS, [common.vector_bytes(common.fock_dimension(modes, c)) for c in CUTOFFS], "o-", label=f"{modes} modes", markersize=3)
    axes[1].axhline(common.vector_bytes(max_dimension), color="red", linestyle="--", label="default max_dimension")
    axes[1].set_xlabel("Cutoff")
    axes[1].set_ylabel("Pure state-vector bytes")
    axes[1].legend(fontsize=6)
    fig.suptitle("Fock-space dimension and memory scaling (D = binom(modes+cutoff-1, modes))")
    common.save_figure(fig, "R35_fock_dimension_scaling")
    plt.close(fig)

    common.print_summary(
        "R35 Fock-space dimension and memory model",
        grid_points=len(rows),
        default_max_dimension=max_dimension,
        max_dimension_in_grid=max(r["dimension"] for r in rows),
        n_exceeding_default_guard=sum(r["exceeds_default_max_dimension"] for r in rows),
    )


if __name__ == "__main__":
    main()
