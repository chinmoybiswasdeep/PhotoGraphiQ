"""R29: Finite squeezing noise scan (main-text result).

Fixes a compound logical circuit (single-mode rotate+squeeze, and a two-mode
CZ-entangled circuit) and sweeps the MBQC RESOURCE squeezing used to compile
it. For each resource squeezing value, extracts the exact unconditional
Gaussian channel (S, N, d) and reports: the logical-map error against an
independently built NumPy target matrix (expected to stay near machine
precision, since squeezing does not change the ideal map), Tr(N), the
spectral norm/max eigenvalue of the added-noise matrix N, and the resulting
output variance for a fixed coherent input.

Upgrade migration (2026-11): (1) the CZ circuit's added noise N is compared
with an independent closed form (four identity-wire steps per output mode,
``gaussian.wire_channel``, block diagonal; the instantaneous Entangle adds no
noise); (2) at EXEC_SQUEEZING the compiled patterns are EXECUTED on NumPy
``gaussian`` and C++ ``native-gaussian`` (``execute_compiled``): sampled
ensembles must reproduce the analytic channel -- ideal map AND added noise --
within 5 shot-noise standard errors, and the engines must agree trajectory by
trajectory. The logical-map error and the added noise stay separate columns:
finite squeezing is never reported as compilation error.
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
EXEC_SQUEEZING = [0.3, 1.0, 2.0]
NOISE_ORACLE_TOLERANCE = 1e-9


def cz_noise_oracle(r: float) -> np.ndarray:
    from photographiq.gaussian import wire_channel

    _, per_mode = wire_channel([0.0] * 4, r)
    noise = np.zeros((4, 4))
    noise[:2, :2] = per_mode
    noise[2:, 2:] = per_mode
    return noise
def independent_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def independent_squeezing(r):
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
                "noise_oracle_error": common.frobenius_error(channel2.noise, cz_noise_oracle(r)),
            }
        )

    exec_rows = []
    for r in EXEC_SQUEEZING:
        single = pg.Circuit(1).rotate(0, LOGICAL_ANGLE).squeeze(0, LOGICAL_SQUEEZE).compile(squeezing=r)
        exec_rows.append({"circuit": "rotate+squeeze", "resource_squeezing": r,
                          **execute_compiled(single, pg.gaussian_channel(single), [INPUT_ALPHA])})
        two = pg.Circuit(2).cz(0, 1, CZ_WEIGHT).compile(squeezing=r)
        exec_rows.append({"circuit": "cz", "resource_squeezing": r,
                          **execute_compiled(two, pg.gaussian_channel(two), [INPUT_ALPHA, -0.1 + 0.25j])})
    common.save_csv(exec_rows, "R29_finite_squeezing_execution")

    tolerance = 1e-7
    failures = [r for r in single_rows + two_rows if r["map_frobenius_error"] > tolerance]
    failures += [r for r in two_rows if r["noise_oracle_error"] > NOISE_ORACLE_TOLERANCE]
    failures += [r for r in exec_rows if execution_failed(r)]
    if failures:
        common.save_csv(failures, "R29_finite_squeezing_FAILURES")
        raise AssertionError("Logical map, CZ noise oracle or execution check failed; see FAILURES csv")

    common.save_result(single_rows, "R29_finite_squeezing_single_mode", extra={"angle": LOGICAL_ANGLE, "squeeze": LOGICAL_SQUEEZE})
    common.save_result(two_rows, "R29_finite_squeezing_cz", extra={
        "weight": CZ_WEIGHT, "noise_oracle": "4 identity-wire steps per output mode (gaussian.wire_channel)",
        "noise_oracle_tolerance": NOISE_ORACLE_TOLERANCE, "execution": exec_rows,
        "execution_backends": list(EXEC_BACKENDS), "sem_factor": SEM_FACTOR})

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
    max_z = max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS)
    common.save_caption(
        "R29_finite_squeezing",
        f"""
**R29: finite resource squeezing sets the MBQC noise floor.** Left: total added variance Tr(N) of the unconditional
channel of two compiled circuits -- R({LOGICAL_ANGLE}) followed by S({LOGICAL_SQUEEZE}) on one mode, and
CZ({CZ_WEIGHT}) on two modes -- versus resource squeezing r = {RESOURCE_SQUEEZING[0]:.1f}..{RESOURCE_SQUEEZING[-1]:.1f}
({len(RESOURCE_SQUEEZING)} values); right: output q and p variances of the single-mode circuit for coherent input
alpha = {INPUT_ALPHA}. The logical map equals the independent NumPy target to
{max(r['map_frobenius_error'] for r in single_rows + two_rows):.1e} at every r: squeezing changes only the added noise,
which falls from Tr N = {max(r['noise_trace'] for r in single_rows):.3g} to {min(r['noise_trace'] for r in single_rows):.3g}
(single mode). The CZ noise equals the closed form of four identity-wire steps per output mode to
{max(r['noise_oracle_error'] for r in two_rows):.1e}. At r = {EXEC_SQUEEZING} both circuits were executed on `gaussian` and
`native-gaussian` ({EXEC_SHOTS} trajectories): sampled ensembles reproduce map and noise within {max_z:.2f} shot-noise
standard errors. Model: hbar = 2, vacuum covariance I; the noise is physical (finite squeezing), not a truncation or
compilation error. Source: `paper_experiments/09_noise/29_finite_squeezing.py`.
""",
    )

    common.print_summary(
        "R29 finite squeezing noise scan (main-text)",
        resource_squeezing_points=len(RESOURCE_SQUEEZING),
        max_map_error=max(r["map_frobenius_error"] for r in single_rows + two_rows),
        execution_max_z=max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in exec_rows for b in EXEC_BACKENDS),
        max_cz_noise_oracle_error=max(r["noise_oracle_error"] for r in two_rows),
        noise_trace_range=(min(r["noise_trace"] for r in single_rows), max(r["noise_trace"] for r in single_rows)),
    )


if __name__ == "__main__":
    main()
