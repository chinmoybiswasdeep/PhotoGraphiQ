"""R4: Squeezing compiler validation.

Compiles ``Circuit(1).squeeze(0, r_logical)`` for a sweep of logical squeezing
parameters, extracts the exact unconditional Gaussian channel and compares it
to an independently written NumPy squeezing matrix diag(exp(-r), exp(r))
(``photographiq.gaussian.squeezing`` uses the identical formula but is the
function the compiler is built from, so an inline copy is used as the
independent oracle here). Numerical stability is monitored via the condition
number of the compiled map and the growth of the added noise as |r| grows.

Upgrade migration (2026-11): three representative compiled patterns are also
EXECUTED on NumPy ``gaussian`` and C++ ``native-gaussian`` (``execute_compiled``):
sampled ensembles vs the analytic channel within 5 shot-noise standard
errors, native vs NumPy trajectory by trajectory. Map error and added noise
stay in separate columns.
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
EXEC_LOGICAL_SQUEEZING = [-1.0, 0.2, 1.5]
EXEC_ALPHA = 0.2 + 0.3j


def independent_squeezing_matrix(r: float) -> np.ndarray:
    """Independent NumPy squeezing matrix; not photographiq.gaussian.squeezing."""
    return np.array([[np.exp(-r), 0.0], [0.0, np.exp(r)]])


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

    exec_rows = []
    for r_logical in EXEC_LOGICAL_SQUEEZING:
        pattern = pg.Circuit(1).squeeze(0, r_logical).compile(squeezing=RESOURCE_SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        exec_rows.append(
            {
                "logical_squeezing_r": r_logical,
                "resource_squeezing": RESOURCE_SQUEEZING,
                "map_frobenius_error": common.frobenius_error(channel.matrix, independent_squeezing_matrix(r_logical)),
                "noise_trace": float(np.trace(channel.noise)),
                **execute_compiled(pattern, channel, [EXEC_ALPHA]),
            }
        )
    common.save_result(
        exec_rows, "R4_squeezing_compile_execution",
        extra={"backends": list(EXEC_BACKENDS), "seed": EXEC_SEED, "alpha": [EXEC_ALPHA.real, EXEC_ALPHA.imag],
               "sem_factor": SEM_FACTOR, "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE},
    )

    tolerance = 1e-9
    failures = [r for r in rows if r["map_frobenius_error"] > tolerance or not r["matrix_finite"] or not r["noise_finite"]]
    failures += [r for r in exec_rows if execution_failed(r)]
    if failures:
        common.save_csv(failures, "R4_squeezing_compile_FAILURES")
        raise AssertionError("Compiled squeezing map or its execution failed a check; see FAILURES csv")

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
        execution_max_z=max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS),
        execution_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in exec_rows),
    )


if __name__ == "__main__":
    main()
