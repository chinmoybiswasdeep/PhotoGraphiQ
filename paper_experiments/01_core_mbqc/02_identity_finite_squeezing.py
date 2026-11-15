"""R2: Identity wire versus finite squeezing.

Sweeps resource squeezing r for the package's actual four-step identity wire
(``photographiq.protocols.identity``, i.e. four Fourier teleportation steps
whose ideal composed map is identity) and measures how the output mean error
and added covariance noise shrink as r grows.

Exact functionality: ``gaussian_channel`` computes the unconditional affine
map (S, N, d) analytically for the fixed-angle pattern; per docs/theory.md an
L-step wire accumulates noise recursively and four k=0 steps implement
identity with noise 2*exp(-2r)*I. That closed form is reproduced here
independently (via ``photographiq.gaussian.wire_channel``, a separate
function from the pattern-level analyzer) as a second oracle.

Upgrade migration (2026-11): at each r the pattern is also EXECUTED on the
NumPy ``gaussian`` and C++ ``native-gaussian`` trajectory engines
(``run_shots``, same seeds). The sampled ensemble mean/covariance must agree
with the analytic channel within 5 shot-noise standard errors, and the two
engines must agree trajectory by trajectory (documented RNG contract). The
identity map and the added noise are reported separately: the noise is the
physical cost of finite squeezing, never a compilation error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.gaussian import wire_channel  # noqa: E402

SQUEEZING_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
INPUT_ALPHA = 0.4 + 0.3j
SHOTS = 1500
SEED = 20261113
EXEC_BACKENDS = ("gaussian", "native-gaussian")

# Acceptance criteria (declared before evaluation).
CHANNEL_TOLERANCE = 1e-9  # original tolerance, unchanged
SEM_FACTOR = 5.0
NATIVE_TRAJECTORY_TOLERANCE = 1e-10


def ensemble_z_scores(ensemble, ideal, conditional_cov, shots):
    """Max |ensemble - analytic| in units of shot-noise standard errors (mean, covariance)."""
    spread = ideal.covariance - conditional_cov
    d = np.clip(np.diag(spread), 0, None)
    sem_mean = np.sqrt(d / shots)
    sem_cov = np.sqrt((np.outer(d, d) + spread**2) / shots)
    z_mean = np.abs(ensemble.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)
    z_cov = np.abs(ensemble.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)
    return float(z_mean.max()), float(z_cov.max())


def main():
    plt = common.setup_style()
    rows = []
    for r in SQUEEZING_VALUES:
        pattern = pg.protocols.identity(squeezing=r)
        output_node = pattern.outputs[0]
        channel = pg.gaussian_channel(pattern)

        # Independent closed-form oracle: four k=0 wire steps.
        closed_form_matrix, closed_form_noise = wire_channel([0.0] * 4, r)

        input_state = pg.GaussianInput.coherent(INPUT_ALPHA)
        applied = channel.apply(input_state.state(0))

        matrix_error = common.frobenius_error(channel.matrix, np.eye(2))
        oracle_matrix_error = common.frobenius_error(channel.matrix, closed_form_matrix)
        oracle_noise_error = common.frobenius_error(channel.noise, closed_form_noise)

        mean_error = common.frobenius_error(applied.mean, np.array([2 * INPUT_ALPHA.real, 2 * INPUT_ALPHA.imag]))
        added_noise_trace = float(np.trace(channel.noise))

        single = pg.simulate(pattern, backend="gaussian", inputs={0: input_state}, seed=7)
        _, q_var = single.state.quadrature(output_node)
        _, p_var = single.state.quadrature(output_node, np.pi / 2)

        execution, trajectories = {}, {}
        for backend in EXEC_BACKENDS:
            shots = pg.run_shots(pattern, SHOTS, backend=backend, inputs={0: input_state}, seed=SEED)
            trajectories[backend] = shots.trajectories
            ensemble = shots.ensemble_state()
            z_mean, z_cov = ensemble_z_scores(ensemble, applied, single.state.covariance, SHOTS)
            execution[f"{backend}_ensemble_q_variance"] = float(ensemble.covariance[0, 0])
            execution[f"{backend}_ensemble_p_variance"] = float(ensemble.covariance[1, 1])
            execution[f"{backend}_max_z_mean"] = z_mean
            execution[f"{backend}_max_z_covariance"] = z_cov
        execution["native_vs_gaussian_max_record_difference"] = max(
            abs(a.outcomes[k] - b.outcomes[k])
            for a, b in zip(trajectories["gaussian"], trajectories["native-gaussian"], strict=True)
            for k in a.outcomes
        )

        rows.append(
            {
                "squeezing_r": r,
                "identity_matrix_error": matrix_error,
                "channel_vs_closed_form_matrix_error": oracle_matrix_error,
                "channel_vs_closed_form_noise_error": oracle_noise_error,
                "output_mean_error": mean_error,
                "added_variance_trace": added_noise_trace,
                # Per-quadrature closed form (N = 2 e^{-2r} I); the legacy figure plotted this
                # against Tr(N), a factor-2 mislabel (ISSUES_FOUND.md, upgrade section).
                "predicted_added_variance": 2 * np.exp(-2 * r),
                "predicted_noise_trace": 4 * np.exp(-2 * r),
                "noise_trace_error_vs_4exp": abs(added_noise_trace - 4 * np.exp(-2 * r)),
                "output_q_variance": q_var,
                "output_p_variance": p_var,
                "log10_added_variance": common.safe_log10(added_noise_trace),
                **execution,
            }
        )

    failures = [
        r for r in rows
        if r["channel_vs_closed_form_matrix_error"] > CHANNEL_TOLERANCE
        or r["channel_vs_closed_form_noise_error"] > CHANNEL_TOLERANCE
        or r["identity_matrix_error"] > CHANNEL_TOLERANCE
        or r["noise_trace_error_vs_4exp"] > CHANNEL_TOLERANCE
        or any(r[f"{b}_max_z_{m}"] > SEM_FACTOR for b in EXEC_BACKENDS for m in ("mean", "covariance"))
        or r["native_vs_gaussian_max_record_difference"] > NATIVE_TRAJECTORY_TOLERANCE
    ]
    common.save_result(
        rows,
        "R2_identity_finite_squeezing",
        extra={
            "input_alpha": [INPUT_ALPHA.real, INPUT_ALPHA.imag],
            "protocol": "four-step identity wire",
            "execution_backends": list(EXEC_BACKENDS),
            "shots": SHOTS,
            "seed": SEED,
            "acceptance": {
                "channel_tolerance": CHANNEL_TOLERANCE,
                "sem_factor": SEM_FACTOR,
                "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE,
            },
        },
    )
    if failures:
        common.save_csv(failures, "R2_identity_finite_squeezing_FAILURES")
        raise AssertionError("Identity wire channel or its executed ensembles failed a check; see FAILURES csv")

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    rs = np.asarray(SQUEEZING_VALUES)
    ax.semilogy(rs, [r["added_variance_trace"] for r in rows], "o-", label="Tr(N), gaussian_channel", color="#24677b")
    ax.semilogy(rs, [r["predicted_noise_trace"] for r in rows], "--", label=r"$4e^{-2r}$ (closed form, N = $2e^{-2r}I$)", color="#c66d27")
    for backend, marker, color in (("gaussian", "s", "#627a36"), ("native-gaussian", "x", "#7a3e9d")):
        sampled = [r[f"{backend}_ensemble_q_variance"] + r[f"{backend}_ensemble_p_variance"] - 2.0 for r in rows]
        ax.semilogy(rs, np.clip(sampled, 1e-4, None), marker, label=f"{backend} ensemble (Tr V - 2)", color=color)
    ax.set_xlabel("Resource squeezing r")
    ax.set_ylabel("Added output variance (vacuum units)")
    ax.legend(fontsize=7)
    ax.set_title("Four-step identity wire: finite-squeezing noise")
    common.save_figure(fig, "R2_identity_finite_squeezing")
    plt.close(fig)

    max_z = max(r[f"{b}_max_z_{m}"] for r in rows for b in EXEC_BACKENDS for m in ("mean", "covariance"))
    common.save_caption(
        "R2_identity_finite_squeezing",
        f"""
**R2: four-step identity wire versus finite squeezing.** Total added output variance Tr(N) of
`protocols.identity` (four k = 0 teleportation steps, coherent input alpha = {INPUT_ALPHA}) versus
resource squeezing r = {SQUEEZING_VALUES[0]}..{SQUEEZING_VALUES[-1]}. Circles: unconditional channel from
`gaussian_channel`; dashed: closed form Tr(N) = 4 e^(-2r), i.e. N = 2 e^(-2r) I (the independent `gaussian.wire_channel` recursion
agrees to {max(r['channel_vs_closed_form_noise_error'] for r in rows):.1e}); squares/crosses: sampled
ensembles of {SHOTS} trajectories on `gaussian` and `native-gaussian` (Tr V - 2, clipped at 1e-4 where shot
noise dominates; both engines draw the same random numbers for equal seeds, so their markers coincide).
The logical map is the identity to {max(r['identity_matrix_error'] for r in rows):.1e} at every r; the added
noise is reported separately and is the physical cost of finite squeezing, not a compilation error. All
sampled moments lie within {max_z:.2f} shot-noise standard errors of the analytic channel (acceptance
<= {SEM_FACTOR:g}); the native engine reproduces the NumPy trajectories seed by seed (max record difference
{max(r['native_vs_gaussian_max_record_difference'] for r in rows):.1e}). Model: hbar = 2, vacuum covariance
I. Source: `paper_experiments/01_core_mbqc/02_identity_finite_squeezing.py`.
""",
    )

    common.print_summary(
        "R2 identity vs finite squeezing",
        squeezing_values=SQUEEZING_VALUES,
        max_channel_vs_closed_form_error=max(r["channel_vs_closed_form_matrix_error"] for r in rows),
        max_added_variance=max(r["added_variance_trace"] for r in rows),
        min_added_variance=min(r["added_variance_trace"] for r in rows),
        max_z_score=max_z,
        max_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in rows),
    )


if __name__ == "__main__":
    main()
