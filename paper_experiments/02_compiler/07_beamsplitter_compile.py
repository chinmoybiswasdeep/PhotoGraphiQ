"""R7: Beamsplitter compilation.

Sweeps the beamsplitter mixing angle theta in ``Circuit(2).beamsplitter``,
compiles the (deliberately resource-expensive, per docs/theory.md) SUM-gate
decomposition, and compares the exact unconditional Gaussian channel to an
independently written analytic beamsplitter symplectic matrix (the package's
own convention: c=cos, s=sin on (q0,p0,q1,p1)). Records the MBQC resource
overhead (nodes/measurements/commands), which the docs flag as especially
costly for this gate.

Upgrade migration (2026-11): representative compiled patterns are also
EXECUTED on NumPy ``gaussian`` and C++ ``native-gaussian`` (``execute_compiled``):
sampled ensembles vs the analytic channel (map and added noise) within 5
shot-noise standard errors; native vs NumPy trajectory by trajectory. The
added noise is reported in its own column, never as compilation error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

THETAS = np.linspace(-np.pi + 0.05, np.pi - 0.05, 17)  # avoid the singular +-pi boundary
SQUEEZING = 1.2
EXEC_THETAS = [0.3]
EXEC_ALPHAS = [0.3 + 0.1j, -0.2 + 0.2j]
EXEC_BS_SHOTS = 300  # ~96 homodynes per shot, so fewer shots than the single-mode gates


def independent_beamsplitter(theta: float) -> np.ndarray:
    """Independent beamsplitter symplectic matrix (package sign convention)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, -s, 0],
            [0, c, 0, -s],
            [s, 0, c, 0],
            [0, s, 0, c],
        ]
    )


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
    for theta in THETAS:
        circuit = pg.Circuit(2).beamsplitter(0, 1, float(theta))
        pattern, trace = circuit.compile(squeezing=SQUEEZING, return_trace=True)
        channel = pg.gaussian_channel(pattern)
        target = independent_beamsplitter(float(theta))
        map_error = common.frobenius_error(channel.matrix, target)
        step = trace.steps[0]
        rows.append(
            {
                "theta": float(theta),
                "squeezing": SQUEEZING,
                "map_frobenius_error": map_error,
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_trace": float(np.trace(channel.noise)),
                "resource_nodes": len(step.resource_nodes),
                "measurements": len(step.measurements),
                "commands": len(pattern.commands),
            }
        )

    exec_rows = []
    for theta in EXEC_THETAS:
        pattern = pg.Circuit(2).beamsplitter(0, 1, theta).compile(squeezing=SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        exec_rows.append(
            {
                "theta": theta,
                "squeezing": SQUEEZING,
                "map_frobenius_error": common.frobenius_error(channel.matrix, independent_beamsplitter(theta)),
                "noise_trace": float(np.trace(channel.noise)),
                **execute_compiled(pattern, channel, EXEC_ALPHAS, shots=EXEC_BS_SHOTS),
            }
        )
    common.save_result(
        exec_rows, "R7_beamsplitter_compile_execution",
        extra={"backends": list(EXEC_BACKENDS), "seed": EXEC_SEED, "alphas": [[a.real, a.imag] for a in EXEC_ALPHAS],
               "sem_factor": SEM_FACTOR, "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE},
    )

    tolerance = 1e-6
    failures = [r for r in rows if r["map_frobenius_error"] > tolerance]
    failures += [r for r in exec_rows if execution_failed(r)]
    if failures:
        common.save_csv(failures, "R7_beamsplitter_compile_FAILURES")
        raise AssertionError("Compiled beamsplitter map or its execution failed a check; see FAILURES csv")

    common.save_result(rows, "R7_beamsplitter_compile", extra={"squeezing": SQUEEZING})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ts = [r["theta"] for r in rows]
    axes[0].semilogy(ts, [r["map_frobenius_error"] for r in rows], "o-", color="#24677b")
    axes[0].set_xlabel(r"Beamsplitter angle $\theta$ (rad)")
    axes[0].set_ylabel("Compiled map Frobenius error")
    axes[1].plot(ts, [r["resource_nodes"] for r in rows], "o-", color="#c66d27", label="Resource nodes")
    axes[1].plot(ts, [r["measurements"] for r in rows], "s--", color="#627a36", label="Measurements")
    axes[1].set_xlabel(r"Beamsplitter angle $\theta$ (rad)")
    axes[1].set_ylabel("MBQC resource count")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Beamsplitter compilation, resource squeezing r={SQUEEZING}")
    common.save_figure(fig, "R7_beamsplitter_compile")
    plt.close(fig)

    common.print_summary(
        "R7 beamsplitter compilation",
        thetas=len(rows),
        max_map_error=max(r["map_frobenius_error"] for r in rows),
        max_resource_nodes=max(r["resource_nodes"] for r in rows),
        max_commands=max(r["commands"] for r in rows),
        execution_max_z=max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS),
        execution_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in exec_rows),
    )


if __name__ == "__main__":
    main()
