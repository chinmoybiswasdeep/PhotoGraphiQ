"""R29: Finite squeezing noise scan (main-text result).

Fixes a compound logical circuit (single-mode rotate+squeeze, and a two-mode
CZ-entangled circuit) and sweeps the MBQC RESOURCE squeezing used to compile
it. For each resource squeezing value, extracts the exact unconditional
Gaussian channel (S, N, d) and reports: the logical-map error against an
independently built NumPy target matrix (expected to stay near machine
precision, since squeezing does not change the ideal map), Tr(N), the
spectral norm/max eigenvalue of the added-noise matrix N, and the resulting
output variance for a fixed coherent input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

RESOURCE_SQUEEZING = np.geomspace(0.1, 3.0, 20)
LOGICAL_ANGLE = 0.5
LOGICAL_SQUEEZE = 0.3
CZ_WEIGHT = 0.6
INPUT_ALPHA = 0.3 + 0.2j


def independent_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def independent_squeezing(r):
    return np.array([[np.exp(-r), 0.0], [0.0, np.exp(r)]])


def main():
    plt = common.setup_style()

    # Circuit.rotate() is applied before Circuit.squeeze() in construction order,
    # so the composed map is S(squeeze) @ R(angle) (squeeze acts on the already-
    # rotated state), not R @ S.
    single_mode_target = independent_squeezing(LOGICAL_SQUEEZE) @ independent_rotation(LOGICAL_ANGLE)
    two_mode_target = np.eye(4)
    two_mode_target[1, 2] = CZ_WEIGHT
    two_mode_target[3, 0] = CZ_WEIGHT

    single_rows, two_rows = [], []
    for r in RESOURCE_SQUEEZING:
        r = float(r)
        circuit = pg.Circuit(1).rotate(0, LOGICAL_ANGLE).squeeze(0, LOGICAL_SQUEEZE)
        pattern = circuit.compile(squeezing=r)
        channel = pg.gaussian_channel(pattern)
        input_state = pg.GaussianInput.coherent(INPUT_ALPHA)
        applied = channel.apply(input_state.state(0))
        single_rows.append(
            {
                "resource_squeezing": r,
                "map_frobenius_error": common.frobenius_error(channel.matrix, single_mode_target),
                "noise_trace": float(np.trace(channel.noise)),
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_max_eigenvalue": float(np.linalg.eigvalsh(channel.noise).max()),
                "output_q_variance": float(applied.covariance[0, 0]),
                "output_p_variance": float(applied.covariance[1, 1]),
            }
        )

        circuit2 = pg.Circuit(2).cz(0, 1, CZ_WEIGHT)
        pattern2 = circuit2.compile(squeezing=r)
        channel2 = pg.gaussian_channel(pattern2)
        two_rows.append(
            {
                "resource_squeezing": r,
                "map_frobenius_error": common.frobenius_error(channel2.matrix, two_mode_target),
                "noise_trace": float(np.trace(channel2.noise)),
                "noise_spectral_norm": float(np.linalg.norm(channel2.noise, ord=2)),
                "noise_max_eigenvalue": float(np.linalg.eigvalsh(channel2.noise).max()),
            }
        )

    tolerance = 1e-7
    failures = [r for r in single_rows + two_rows if r["map_frobenius_error"] > tolerance]
    if failures:
        common.save_csv(failures, "R29_finite_squeezing_FAILURES")
        raise AssertionError("Resource squeezing altered the ideal logical map beyond tolerance; see FAILURES csv")

    common.save_result(single_rows, "R29_finite_squeezing_single_mode", extra={"angle": LOGICAL_ANGLE, "squeeze": LOGICAL_SQUEEZE})
    common.save_result(two_rows, "R29_finite_squeezing_cz", extra={"weight": CZ_WEIGHT})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    rs = [row["resource_squeezing"] for row in single_rows]
    axes[0].loglog(rs, [row["noise_trace"] for row in single_rows], "o-", label="Rotate+Squeeze circuit", color="#24677b")
    axes[0].loglog(rs, [row["noise_trace"] for row in two_rows], "s-", label="CZ circuit", color="#c66d27")
    axes[0].set_xlabel("Resource squeezing r")
    axes[0].set_ylabel("Tr(N): total added variance")
    axes[0].legend(fontsize=8)

    axes[1].semilogx(rs, [row["output_q_variance"] for row in single_rows], "o-", label="Output q variance", color="#627a36")
    axes[1].semilogx(rs, [row["output_p_variance"] for row in single_rows], "s-", label="Output p variance", color="#a83279")
    axes[1].set_xlabel("Resource squeezing r")
    axes[1].set_ylabel("Output variance (rotate+squeeze circuit)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Finite resource squeezing: MBQC compilation noise floor")
    common.save_figure(fig, "R29_finite_squeezing")
    plt.close(fig)

    common.print_summary(
        "R29 finite squeezing noise scan (main-text)",
        resource_squeezing_points=len(RESOURCE_SQUEEZING),
        max_map_error=max(r["map_frobenius_error"] for r in single_rows + two_rows),
        noise_trace_range=(min(r["noise_trace"] for r in single_rows), max(r["noise_trace"] for r in single_rows)),
    )


if __name__ == "__main__":
    main()
