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

Upgrade migration (2026-11): representative compiled patterns are also
EXECUTED with ``run_shots`` on NumPy ``gaussian`` and C++ ``native-gaussian``
(see ``execute_compiled``). The sampled ensembles must match the analytic
channel (map AND added noise) within 5 shot-noise standard errors and the two
engines must agree trajectory by trajectory. The map error (compilation) and
the added noise N (finite squeezing) stay in separate columns.
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
EXEC_THETAS = [-2.0, 0.7, 2.5]
EXEC_ALPHA = 0.3 - 0.2j


def independent_rotation_matrix(theta: float) -> np.ndarray:
    """Independent NumPy rotation matrix; not photographiq.gaussian.rotation."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]])


# --------------------------------------------------------------------------- #
# Upgrade migration: execution of the compiled pattern on the trajectory engines
# --------------------------------------------------------------------------- #
EXEC_BACKENDS = ("gaussian", "native-gaussian")
EXEC_SHOTS = 600
EXEC_SEED = 20261113
SEM_FACTOR = 5.0  # ensemble moments within 5 shot-noise standard errors of the channel
NATIVE_TRAJECTORY_TOLERANCE = 1e-9  # native-gaussian vs gaussian records, same seed


def execute_compiled(pattern, channel, alphas, *, shots=EXEC_SHOTS, seed=EXEC_SEED):
    """Run ``pattern`` on both Gaussian engines and compare with ``channel``.

    Returns the worst z-scores of the ensemble mean/covariance against
    ``channel.apply`` (shot-noise standard errors from the analytic spread of
    the conditional means), the native-vs-NumPy trajectory difference, and the
    planner's backend choice. The compilation map error and the added noise N
    are reported separately by the caller; N is never counted as map error.
    """
    inputs = {node: pg.GaussianInput.coherent(a) for node, a in zip(pattern.inputs, alphas, strict=True)}
    mean = np.concatenate([[2 * a.real, 2 * a.imag] for a in alphas])
    ideal = channel.apply(pg.GaussianState(mean, np.eye(2 * len(alphas)), pattern.inputs))
    conditional = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=seed).state.covariance
    spread = ideal.covariance - conditional
    d = np.clip(np.diag(spread), 0, None)
    sem_mean = np.sqrt(d / shots)
    sem_cov = np.sqrt((np.outer(d, d) + spread**2) / shots)
    out, runs = {}, {}
    for backend in EXEC_BACKENDS:
        result = pg.run_shots(pattern, shots, backend=backend, inputs=inputs, seed=seed)
        runs[backend] = result.trajectories
        ensemble = result.ensemble_state()
        out[f"{backend}_max_z_mean"] = float((np.abs(ensemble.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)).max())
        out[f"{backend}_max_z_covariance"] = float(
            (np.abs(ensemble.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)).max()
        )
    out["native_vs_gaussian_max_record_difference"] = max(
        abs(a.outcomes[k] - b.outcomes[k])
        for a, b in zip(runs["gaussian"], runs["native-gaussian"], strict=True)
        for k in a.outcomes
    )
    out["planner_backend"] = pg.select_backend(pattern, inputs).name
    out["shots"] = shots
    out["measurements"] = sum(isinstance(c, pg.Measure) for c in pattern.commands)
    return out


def execution_failed(row) -> bool:
    return (
        any(row[f"{b}_max_z_{m}"] > SEM_FACTOR for b in EXEC_BACKENDS for m in ("mean", "covariance"))
        or row["native_vs_gaussian_max_record_difference"] > NATIVE_TRAJECTORY_TOLERANCE
    )


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

    exec_rows = []
    for theta in EXEC_THETAS:
        pattern = pg.Circuit(1).rotate(0, theta).compile(squeezing=FIXED_SQUEEZING_FOR_THETA_SWEEP)
        channel = pg.gaussian_channel(pattern)
        exec_rows.append(
            {
                "theta": theta,
                "squeezing": FIXED_SQUEEZING_FOR_THETA_SWEEP,
                "map_frobenius_error": common.frobenius_error(channel.matrix, independent_rotation_matrix(theta)),
                "noise_trace": float(np.trace(channel.noise)),
                **execute_compiled(pattern, channel, [EXEC_ALPHA]),
            }
        )
    common.save_result(
        exec_rows, "R3_rotation_compile_execution",
        extra={"backends": list(EXEC_BACKENDS), "seed": EXEC_SEED, "alpha": [EXEC_ALPHA.real, EXEC_ALPHA.imag],
               "sem_factor": SEM_FACTOR, "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE},
    )

    map_tolerance = 1e-9  # squeezing does not affect the ideal linear map, only noise
    failures = [r for r in theta_rows + squeezing_rows if r["map_frobenius_error"] > map_tolerance]
    failures += [r for r in exec_rows if execution_failed(r)]
    if failures:
        common.save_csv(failures, "R3_rotation_compile_FAILURES")
        raise AssertionError("Compiled rotation map or its execution failed a check; see FAILURES csv")

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
        execution_max_z=max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS),
        execution_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in exec_rows),
        planner=sorted({r["planner_backend"] for r in exec_rows}),
    )


if __name__ == "__main__":
    main()
