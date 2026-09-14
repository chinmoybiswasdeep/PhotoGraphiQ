"""R4: Squeezing compiler validation.

Compiles ``Circuit(1).squeeze(0, r_logical)`` for a sweep of logical squeezing
parameters, extracts the exact unconditional Gaussian channel and compares it
to an independently written NumPy squeezing matrix diag(exp(-r), exp(r))
(``photographiq.gaussian.squeezing`` uses the identical formula but is the
function the compiler is built from, so an inline copy is used as the
independent oracle here). Numerical stability is monitored via the condition
number of the compiled map and the growth of the added noise as |r| grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

LOGICAL_SQUEEZING = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5]
RESOURCE_SQUEEZING = 1.2


def independent_squeezing_matrix(r: float) -> np.ndarray:
    """Independent NumPy squeezing matrix; not photographiq.gaussian.squeezing."""
    return np.array([[np.exp(-r), 0.0], [0.0, np.exp(r)]])


def main():
    plt = common.setup_style()
    rows = []
    for r_logical in LOGICAL_SQUEEZING:
        circuit = pg.Circuit(1).squeeze(0, r_logical)
        pattern = circuit.compile(squeezing=RESOURCE_SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        target = independent_squeezing_matrix(r_logical)
        map_error = common.frobenius_error(channel.matrix, target)
        condition_number = float(np.linalg.cond(channel.matrix))
        rows.append(
            {
                "logical_squeezing_r": r_logical,
                "resource_squeezing": RESOURCE_SQUEEZING,
                "map_frobenius_error": map_error,
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_trace": float(np.trace(channel.noise)),
                "matrix_condition_number": condition_number,
                "matrix_finite": bool(np.isfinite(channel.matrix).all()),
                "noise_finite": bool(np.isfinite(channel.noise).all()),
            }
        )

    tolerance = 1e-9
    failures = [r for r in rows if r["map_frobenius_error"] > tolerance or not r["matrix_finite"] or not r["noise_finite"]]
    if failures:
        common.save_csv(failures, "R4_squeezing_compile_FAILURES")
        raise AssertionError("Compiled squeezing map disagreed with the analytic target; see FAILURES csv")

    common.save_result(rows, "R4_squeezing_compile", extra={"resource_squeezing": RESOURCE_SQUEEZING})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    rs = [r["logical_squeezing_r"] for r in rows]
    axes[0].semilogy(rs, [r["noise_spectral_norm"] for r in rows], "o-", color="#24677b")
    axes[0].set_xlabel("Logical squeezing r")
    axes[0].set_ylabel("Noise spectral norm ||N||")
    axes[1].semilogy(rs, [r["matrix_condition_number"] for r in rows], "o-", color="#c66d27")
    axes[1].set_xlabel("Logical squeezing r")
    axes[1].set_ylabel("Compiled map condition number")
    fig.suptitle(f"Squeezing gate compilation, resource squeezing r={RESOURCE_SQUEEZING}")
    common.save_figure(fig, "R4_squeezing_compile")
    plt.close(fig)

    common.print_summary(
        "R4 squeezing compiler validation",
        points=len(rows),
        max_map_error=max(r["map_frobenius_error"] for r in rows),
        max_condition_number=max(r["matrix_condition_number"] for r in rows),
    )


if __name__ == "__main__":
    main()
