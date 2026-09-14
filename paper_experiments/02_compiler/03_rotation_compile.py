"""R3: Rotation compiler validation.

For each rotation angle theta, compiles ``Circuit(1).rotate(0, theta)`` to a
finite-squeezing MBQC pattern, extracts its exact unconditional Gaussian
channel, and compares the resulting linear map against an independently
written NumPy rotation matrix (not ``photographiq.gaussian.rotation``, which
is the function the compiler itself is built from).

Exact functionality: in the r -> infinity limit the compiled channel matrix
converges to the ideal rotation; finite r contributes physical additive noise
N, not a truncation artifact. A second sweep fixes theta and varies resource
squeezing to show the noise floor separately from the angle dependence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

THETAS = np.linspace(-np.pi, np.pi, 25)
SQUEEZING_SWEEP = [0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5]
FIXED_THETA_FOR_SQUEEZING_SWEEP = 0.7
FIXED_SQUEEZING_FOR_THETA_SWEEP = 1.2


def independent_rotation_matrix(theta: float) -> np.ndarray:
    """Independent NumPy rotation matrix; not photographiq.gaussian.rotation."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]])


def main():
    plt = common.setup_style()

    theta_rows = []
    for theta in THETAS:
        circuit = pg.Circuit(1).rotate(0, float(theta))
        pattern = circuit.compile(squeezing=FIXED_SQUEEZING_FOR_THETA_SWEEP)
        channel = pg.gaussian_channel(pattern)
        target = independent_rotation_matrix(float(theta))
        map_error = common.frobenius_error(channel.matrix, target)
        noise_norm = float(np.linalg.norm(channel.noise, ord=2))
        theta_rows.append(
            {
                "theta": float(theta),
                "squeezing": FIXED_SQUEEZING_FOR_THETA_SWEEP,
                "map_frobenius_error": map_error,
                "noise_spectral_norm": noise_norm,
                "noise_trace": float(np.trace(channel.noise)),
                "commands": len(pattern.commands),
            }
        )

    squeezing_rows = []
    target = independent_rotation_matrix(FIXED_THETA_FOR_SQUEEZING_SWEEP)
    for r in SQUEEZING_SWEEP:
        circuit = pg.Circuit(1).rotate(0, FIXED_THETA_FOR_SQUEEZING_SWEEP)
        pattern = circuit.compile(squeezing=r)
        channel = pg.gaussian_channel(pattern)
        squeezing_rows.append(
            {
                "theta": FIXED_THETA_FOR_SQUEEZING_SWEEP,
                "squeezing": r,
                "map_frobenius_error": common.frobenius_error(channel.matrix, target),
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_trace": float(np.trace(channel.noise)),
            }
        )

    map_tolerance = 1e-9  # squeezing does not affect the ideal linear map, only noise
    failures = [r for r in theta_rows + squeezing_rows if r["map_frobenius_error"] > map_tolerance]
    if failures:
        common.save_csv(failures, "R3_rotation_compile_FAILURES")
        raise AssertionError("Compiled rotation map disagreed with the analytic target; see FAILURES csv")

    common.save_result(theta_rows, "R3_rotation_compile_theta_sweep")
    common.save_result(squeezing_rows, "R3_rotation_compile_squeezing_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].semilogy(
        [r["theta"] for r in theta_rows], [r["noise_spectral_norm"] for r in theta_rows], "o-", color="#24677b"
    )
    axes[0].set_xlabel(r"Rotation angle $\theta$ (rad)")
    axes[0].set_ylabel("Noise spectral norm ||N||")
    axes[0].set_title(f"Squeezing r={FIXED_SQUEEZING_FOR_THETA_SWEEP} fixed")

    axes[1].semilogy(
        [r["squeezing"] for r in squeezing_rows], [r["noise_spectral_norm"] for r in squeezing_rows], "o-", color="#c66d27"
    )
    axes[1].set_xlabel("Resource squeezing r")
    axes[1].set_ylabel("Noise spectral norm ||N||")
    axes[1].set_title(rf"$\theta$={FIXED_THETA_FOR_SQUEEZING_SWEEP:.2f} fixed")
    fig.suptitle("Rotation gate compilation: finite-resource noise")
    common.save_figure(fig, "R3_rotation_compile")
    plt.close(fig)

    common.print_summary(
        "R3 rotation compiler validation",
        theta_points=len(theta_rows),
        max_map_error=max(r["map_frobenius_error"] for r in theta_rows + squeezing_rows),
        max_noise_norm=max(r["noise_spectral_norm"] for r in theta_rows),
    )


if __name__ == "__main__":
    main()
