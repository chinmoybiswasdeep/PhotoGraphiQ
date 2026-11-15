"""R1: Basic CV wire / one-step teleportation.

Verifies the elementary CV-MBQC transport primitive: a single p-homodyne
measurement on a two-mode momentum-squeezed resource teleports a Fourier
transform of the input mode (``photographiq.protocols.wire`` with one shear).

Exact functionality: the unconditional affine channel (S, N, d) is computed
independently by ``photographiq.analysis.gaussian_channel``, which propagates
Wigner variables through the pattern and is a separate code path from the
conditional trajectory simulator (``simulate``/``run_shots``). Agreement
between the two is evidence that finite-squeezing ensemble simulation matches
the analytic finite-resource channel derived in docs/theory.md.

Finite-resource approximation: resource squeezing r is finite (r=1.0 here);
noise diag(0, exp(-2r)) is physical, not a truncation artifact.

Upgrade migration (2026-11): the ensemble is now sampled on three Gaussian
trajectory engines -- NumPy ``gaussian``, the C++ ``native-gaussian`` and the
Piquasso-backed ``piquasso`` (kept as an external reference). In addition to
the original fixed tolerance, every first and second moment of each ensemble
is checked against the analytic channel within 5 shot-noise standard errors,
the corrected output channel (S, N) is checked against an independently
written Fourier map F and noise diag(0, e^{-2r}), and the native engine is
checked against the NumPy engine trajectory by trajectory (documented RNG
contract: same Generator calls, equal seeds give equal trajectories up to
floating-point summation order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SHOTS = 4000
SQUEEZING = 1.0
SEED = 20260913
BACKENDS = ("gaussian", "native-gaussian", "piquasso")
NOISE_SWEEP = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

COHERENT_INPUTS = [0.0 + 0.0j, 0.3 + 0.2j, -0.5 + 0.1j, 1.0 + 0.0j, 0.2 - 0.4j, -0.8 - 0.6j]

# Acceptance criteria (declared before evaluation).
LEGACY_TOLERANCE = 0.35  # original fixed tolerance, kept unchanged
SEM_FACTOR = 5.0  # |ensemble moment - analytic| <= 5 standard errors
CHANNEL_TOLERANCE = 1e-12  # (S, N) vs independent Fourier map / diag(0, e^-2r)
NATIVE_TRAJECTORY_TOLERANCE = 1e-10  # native-gaussian vs gaussian, same seed


def fourier_map() -> np.ndarray:
    """Independent one-step wire target: (q, p) -> (-p, q)."""
    return np.array([[0.0, -1.0], [1.0, 0.0]])


def wire_noise(r: float) -> np.ndarray:
    """Independent finite-squeezing noise of one step: diag(0, e^{-2r})."""
    return np.diag([0.0, np.exp(-2 * r)])


def moment_standard_errors(spread: np.ndarray, shots: int):
    """Standard errors of the ensemble mean and covariance.

    ``spread`` is the covariance of the conditional output means (analytic
    total covariance minus the record-independent conditional covariance);
    only that part is estimated from samples. For Gaussian samples
    Var(cov_ij) = (W_ii W_jj + W_ij^2)/N.
    """
    sem_mean = np.sqrt(np.clip(np.diag(spread), 0, None) / shots)
    d = np.clip(np.diag(spread), 0, None)
    sem_cov = np.sqrt((np.outer(d, d) + spread**2) / shots)
    return sem_mean, sem_cov


def main():
    plt = common.setup_style()
    pattern = pg.protocols.wire([0.0], squeezing=SQUEEZING)  # one-step Fourier wire
    channel = pg.gaussian_channel(pattern)
    output_node = pattern.outputs[0]

    channel_map_error = common.frobenius_error(channel.matrix, fourier_map())
    channel_noise_error = common.frobenius_error(channel.noise, wire_noise(SQUEEZING))

    rows, failures = [], []
    for index, alpha in enumerate(COHERENT_INPUTS):
        input_state = pg.GaussianInput.coherent(alpha)
        ideal = channel.apply(input_state.state(0))

        single = pg.simulate(pattern, backend="gaussian", inputs={0: input_state}, seed=SEED + index)
        conditional_cov = single.state.covariance
        spread = ideal.covariance - conditional_cov
        sem_mean, sem_cov = moment_standard_errors(spread, SHOTS)

        q_mean_single, q_var_single = single.state.quadrature(output_node)
        p_mean_single, p_var_single = single.state.quadrature(output_node, np.pi / 2)
        row = {
            "alpha_real": alpha.real,
            "alpha_imag": alpha.imag,
            "ideal_q_mean": ideal.mean[0],
            "ideal_p_mean": ideal.mean[1],
            "ideal_q_var": ideal.covariance[0, 0],
            "ideal_p_var": ideal.covariance[1, 1],
            "single_trajectory_q_mean": q_mean_single,
            "single_trajectory_p_mean": p_mean_single,
            "single_trajectory_q_var": q_var_single,
            "single_trajectory_p_var": p_var_single,
            "sem_q_mean": sem_mean[0],
            "sem_p_mean": sem_mean[1],
            "sem_q_var": sem_cov[0, 0],
            "sem_p_var": sem_cov[1, 1],
        }
        shot_results = {}
        for backend in BACKENDS:
            shots = pg.run_shots(pattern, SHOTS, backend=backend, inputs={0: input_state}, seed=SEED + 97 * index)
            shot_results[backend] = shots
            state = shots.ensemble_state()
            row[f"{backend}_ensemble_q_mean"] = state.mean[0]
            row[f"{backend}_ensemble_p_mean"] = state.mean[1]
            row[f"{backend}_ensemble_q_var"] = state.covariance[0, 0]
            row[f"{backend}_ensemble_p_var"] = state.covariance[1, 1]
            row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, ideal.mean)
            row[f"{backend}_covariance_error"] = common.frobenius_error(state.covariance, ideal.covariance)
            # Largest deviation in units of its own standard error (floor avoids 0/0 for
            # entries with no sampled spread, which must then agree to rounding).
            z_mean = np.abs(state.mean - ideal.mean) / np.maximum(sem_mean, 1e-12)
            z_cov = np.abs(state.covariance - ideal.covariance) / np.maximum(sem_cov, 1e-12)
            row[f"{backend}_max_z_mean"] = float(z_mean.max())
            row[f"{backend}_max_z_covariance"] = float(z_cov.max())
            if row[f"{backend}_mean_error"] > LEGACY_TOLERANCE or row[f"{backend}_covariance_error"] > LEGACY_TOLERANCE:
                failures.append({**row, "failed_check": f"{backend} legacy tolerance"})
            if max(row[f"{backend}_max_z_mean"], row[f"{backend}_max_z_covariance"]) > SEM_FACTOR:
                failures.append({**row, "failed_check": f"{backend} {SEM_FACTOR} SEM"})

        reference = shot_results["gaussian"].trajectories
        native = shot_results["native-gaussian"].trajectories
        records = max(
            abs(a.outcomes[k] - b.outcomes[k]) for a, b in zip(reference, native, strict=True) for k in a.outcomes
        )
        means = max(float(np.abs(a.state.mean - b.state.mean).max()) for a, b in zip(reference, native, strict=True))
        row["native_vs_gaussian_max_record_difference"] = records
        row["native_vs_gaussian_max_output_mean_difference"] = means
        if max(records, means) > NATIVE_TRAJECTORY_TOLERANCE:
            failures.append({**row, "failed_check": "native trajectory contract"})
        rows.append(row)

    noise_rows = []
    for r in NOISE_SWEEP:
        sweep_channel = pg.gaussian_channel(pg.protocols.wire([0.0], squeezing=r))
        noise_rows.append(
            {
                "squeezing_r": r,
                "map_error_vs_fourier": common.frobenius_error(sweep_channel.matrix, fourier_map()),
                "noise_error_vs_closed_form": common.frobenius_error(sweep_channel.noise, wire_noise(r)),
                "added_p_variance": float(sweep_channel.noise[1, 1]),
                "added_q_variance": float(sweep_channel.noise[0, 0]),
                "noise_trace": float(np.trace(sweep_channel.noise)),
            }
        )
    channel_errors = [channel_map_error, channel_noise_error] + [
        e for r in noise_rows for e in (r["map_error_vs_fourier"], r["noise_error_vs_closed_form"])
    ]
    if max(channel_errors) > CHANNEL_TOLERANCE:
        failures.append({"failed_check": "corrected output channel", "max_channel_error": max(channel_errors)})

    extra = {
        "protocol": "one-step Fourier wire (k=0)",
        "squeezing": SQUEEZING,
        "shots": SHOTS,
        "seed_base": SEED,
        "backends": list(BACKENDS),
        "channel_matrix": channel.matrix.tolist(),
        "channel_noise": channel.noise.tolist(),
        "channel_map_error_vs_independent_fourier": channel_map_error,
        "channel_noise_error_vs_closed_form": channel_noise_error,
        "noise_sweep": noise_rows,
        "acceptance": {
            "legacy_tolerance": LEGACY_TOLERANCE,
            "sem_factor": SEM_FACTOR,
            "channel_tolerance": CHANNEL_TOLERANCE,
            "native_trajectory_tolerance": NATIVE_TRAJECTORY_TOLERANCE,
        },
    }
    common.save_result(rows, "R1_wire_teleportation", extra=extra)
    common.save_csv(noise_rows, "R1_wire_teleportation_noise_sweep")
    if failures:
        common.save_csv(failures, "R1_wire_teleportation_FAILURES")
        raise AssertionError(f"{len(failures)} R1 checks failed; see FAILURES csv")

    colors = {"analytic": "#24677b", "gaussian": "#c66d27", "native-gaussian": "#7a3e9d", "piquasso": "#627a36"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    x = np.arange(len(rows))
    labels = [f"{r['alpha_real']:.1f}{r['alpha_imag']:+.1f}i" for r in rows]
    markers = {"gaussian": "o", "native-gaussian": "x", "piquasso": "+"}
    for ax, kind in zip(axes[:2], ("mean", "var"), strict=True):
        for quad, offset in (("q", -0.12), ("p", 0.12)):
            for backend in BACKENDS:
                key = f"{backend}_ensemble_{quad}_{'mean' if kind == 'mean' else 'var'}"
                z = [
                    (r[key] - r[f"ideal_{quad}_{kind}"]) / max(r[f"sem_{quad}_{kind}"], 1e-12) for r in rows
                ]
                ax.plot(x + offset, z, markers[backend], color=colors[backend], markersize=6,
                        label=f"{backend}, {quad}")
        ax.axhspan(-SEM_FACTOR, SEM_FACTOR, color="grey", alpha=0.12, label=f"acceptance $\\pm{SEM_FACTOR:g}$ SE")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(r"Coherent input $\alpha$ (left marker q, right marker p)")
        ax.set_ylabel(f"(ensemble - analytic) / SE, {'mean' if kind == 'mean' else 'variance'}")
        ax.set_ylim(-SEM_FACTOR - 1, SEM_FACTOR + 1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    keep = [i for i, lab in enumerate(legend_labels) if lab.endswith(", q") or "acceptance" in lab]
    axes[0].legend([handles[i] for i in keep], [legend_labels[i].replace(", q", "") for i in keep],
                   fontsize=6.5, loc="lower left", ncol=2)
    rs = [r["squeezing_r"] for r in noise_rows]
    axes[2].semilogy(rs, [r["added_p_variance"] for r in noise_rows], "o", color=colors["analytic"], label="gaussian_channel N_pp")
    fine = np.linspace(min(rs), max(rs), 100)
    axes[2].semilogy(fine, np.exp(-2 * fine), "--", color="grey", label=r"$e^{-2r}$ (closed form)")
    axes[2].set_xlabel("Resource squeezing r")
    axes[2].set_ylabel("Added output variance")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Finite-squeezing noise (N_qq = 0)")
    fig.suptitle(f"One-step Fourier wire, resource squeezing r={SQUEEZING}, {SHOTS} shots per backend")
    common.save_figure(fig, "R1_wire_teleportation")
    plt.close(fig)

    max_z = max(max(r[f"{b}_max_z_mean"], r[f"{b}_max_z_covariance"]) for r in rows for b in BACKENDS)
    common.save_caption(
        "R1_wire_teleportation",
        f"""
**R1: one-step CV wire teleportation.** Left/centre: deviation of the sampled ensemble mean and
variance of the output q (left marker) and p (right marker) quadratures from the analytic
unconditional channel (`gaussian_channel`), in units of the shot-noise standard error, for the
one-step Fourier wire (`protocols.wire([0.0])`, resource squeezing r = {SQUEEZING}) and {len(rows)}
coherent inputs; {SHOTS} homodyne trajectories with displacement feed-forward per input on the NumPy
`gaussian`, C++ `native-gaussian` and Piquasso-backed `piquasso` engines. The three engines consume the
same random numbers for equal seeds, so their markers coincide: they are one statistical sample, not
three independent ones. Every ensemble first and second moment lies within {max_z:.2f} standard errors
of the analytic value (acceptance <= {SEM_FACTOR:g}). The corrected output channel equals the independent Fourier map
(q, p) -> (-p, q) to {channel_map_error:.1e} and its added noise equals diag(0, e^(-2r)) to
{channel_noise_error:.1e}. `native-gaussian` reproduces the NumPy trajectories seed by seed (max record
difference {max(r['native_vs_gaussian_max_record_difference'] for r in rows):.1e}). Right: added p
variance versus r (q receives none). Model: hbar = 2, vacuum covariance I; the noise is the physical
cost of finite squeezing, not a truncation artifact, and vanishes only as r -> infinity. Source:
`paper_experiments/01_core_mbqc/01_wire_teleportation.py`.
""",
    )

    common.print_summary(
        "R1 wire teleportation",
        inputs=len(rows),
        **{f"max_{b}_mean_error": max(r[f"{b}_mean_error"] for r in rows) for b in BACKENDS},
        **{f"max_{b}_covariance_error": max(r[f"{b}_covariance_error"] for r in rows) for b in BACKENDS},
        max_z_score=max_z,
        channel_map_error=channel_map_error,
        channel_noise_error=channel_noise_error,
        max_native_record_difference=max(r["native_vs_gaussian_max_record_difference"] for r in rows),
    )


if __name__ == "__main__":
    main()
