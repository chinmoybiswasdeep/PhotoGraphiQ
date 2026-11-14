"""R6: Two-mode CZ compilation.

Sweeps the logical controlled-Z weight in ``Circuit(2).cz(0, 1, weight)``,
compiles each circuit, and compares the exact unconditional two-mode Gaussian
channel to an independently written analytic CZ symplectic matrix
CZ(g): p_i -> p_i + g*q_j (photographiq.gaussian.cz implements the identical
convention; the copy here is written independently as the cross-check
target). For representative coherent inputs, mean/covariance and the
generated cross-mode correlations are also compared.

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
from photographiq.gaussian import wire_channel  # noqa: E402

WEIGHTS = [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
SQUEEZING = 1.0
INPUT_ALPHAS = [(0.3 + 0.1j, -0.2 + 0.4j), (0.0, 0.5 + 0.0j)]
EXEC_WEIGHTS = [-1.0, 0.5, 2.0]
EXEC_ALPHAS = [0.3 + 0.1j, -0.2 + 0.4j]


def independent_cz(weight: float) -> np.ndarray:
    """Independent CZ(g) symplectic matrix on (q0,p0,q1,p1); p_i -> p_i+g*q_j."""
    matrix = np.eye(4)
    matrix[1, 2] = weight  # p0 += g q1
    matrix[3, 0] = weight  # p1 += g q0
    return matrix


def expected_noise(squeezing: float) -> np.ndarray:
    """Independent noise oracle: logical CZ = instantaneous Entangle followed by
    an identity wire (four k=0 teleportation steps) on each output mode; the
    entangling step itself is a direct physical transform and adds no noise,
    so the total noise is block-diagonal in the per-mode wire noise.
    """
    _, per_mode_noise = wire_channel([0.0] * 4, squeezing)
    noise = np.zeros((4, 4))
    noise[:2, :2] = per_mode_noise
    noise[2:, 2:] = per_mode_noise
    return noise


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
    for weight in WEIGHTS:
        circuit = pg.Circuit(2).cz(0, 1, weight)
        pattern = circuit.compile(squeezing=SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        target = independent_cz(weight)
        map_error = common.frobenius_error(channel.matrix, target)
        noise_oracle = expected_noise(SQUEEZING)
        noise_error = common.frobenius_error(channel.noise, noise_oracle)

        state_errors = []
        cross_correlations = []
        for alpha0, alpha1 in INPUT_ALPHAS:
            inputs = pg.GaussianState(
                np.array(
                    [2 * alpha0.real, 2 * alpha0.imag, 2 * alpha1.real, 2 * alpha1.imag]
                ),
                np.eye(4),
                (0, 1),
            )
            actual = channel.apply(inputs)
            ideal_mean = target @ inputs.mean
            ideal_cov = target @ inputs.covariance @ target.T + noise_oracle
            state_errors.append(
                {
                    "mean_error": common.frobenius_error(actual.mean, ideal_mean),
                    "covariance_error": common.frobenius_error(actual.covariance, ideal_cov),
                }
            )
            cross_correlations.append(float(actual.covariance[0, 2]))  # q0-q1 correlation

        rows.append(
            {
                "weight": weight,
                "squeezing": SQUEEZING,
                "map_frobenius_error": map_error,
                "noise_oracle_frobenius_error": noise_error,
                "noise_spectral_norm": float(np.linalg.norm(channel.noise, ord=2)),
                "noise_trace": float(np.trace(channel.noise)),
                "max_state_mean_error": max(s["mean_error"] for s in state_errors),
                "max_state_covariance_error": max(s["covariance_error"] for s in state_errors),
                "cross_correlations": cross_correlations,
            }
        )

    exec_rows = []
    for weight in EXEC_WEIGHTS:
        pattern = pg.Circuit(2).cz(0, 1, weight).compile(squeezing=SQUEEZING)
        channel = pg.gaussian_channel(pattern)
        exec_rows.append(
            {
                "weight": weight,
                "squeezing": SQUEEZING,
                "map_frobenius_error": common.frobenius_error(channel.matrix, independent_cz(weight)),
                "noise_oracle_frobenius_error": common.frobenius_error(channel.noise, expected_noise(SQUEEZING)),
                **execute_compiled(pattern, channel, EXEC_ALPHAS),
            }
        )
    common.save_result(
        exec_rows, "R6_cz_compile_execution",
        extra={"backends": list(EXEC_BACKENDS), "seed": EXEC_SEED, "alphas": [[a.real, a.imag] for a in EXEC_ALPHAS],
               "sem_factor": SEM_FACTOR, "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE},
    )

    tolerance = 1e-8
    failures = [r for r in rows if r["map_frobenius_error"] > tolerance or r["noise_oracle_frobenius_error"] > tolerance
                or r["max_state_mean_error"] > tolerance or r["max_state_covariance_error"] > tolerance]
    failures += [r for r in exec_rows if execution_failed(r)]
    if failures:
        common.save_csv(failures, "R6_cz_compile_FAILURES")
        raise AssertionError("Compiled CZ map or its execution failed a check; see FAILURES csv")

    common.save_result(rows, "R6_cz_compile", extra={"squeezing": SQUEEZING})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ws = [r["weight"] for r in rows]
    axes[0].plot(ws, [r["map_frobenius_error"] for r in rows], "o-", color="#24677b")
    axes[0].set_xlabel("CZ weight g")
    axes[0].set_ylabel("Compiled map Frobenius error")
    axes[1].semilogy(ws, [r["noise_spectral_norm"] for r in rows], "o-", color="#c66d27")
    axes[1].set_xlabel("CZ weight g")
    axes[1].set_ylabel("Noise spectral norm ||N||")
    fig.suptitle(f"CZ gate compilation, resource squeezing r={SQUEEZING}")
    common.save_figure(fig, "R6_cz_compile")
    plt.close(fig)

    common.print_summary(
        "R6 CZ compilation",
        weights=len(rows),
        max_map_error=max(r["map_frobenius_error"] for r in rows),
        max_noise_norm=max(r["noise_spectral_norm"] for r in rows),
        execution_max_z=max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS),
        execution_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in exec_rows),
    )


if __name__ == "__main__":
    main()
