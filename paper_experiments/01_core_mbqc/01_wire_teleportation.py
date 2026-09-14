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

COHERENT_INPUTS = [0.0 + 0.0j, 0.3 + 0.2j, -0.5 + 0.1j, 1.0 + 0.0j, 0.2 - 0.4j, -0.8 - 0.6j]


def main():
    plt = common.setup_style()
    pattern = pg.protocols.wire([0.0], squeezing=SQUEEZING)  # one-step Fourier wire
    channel = pg.gaussian_channel(pattern)
    output_node = pattern.outputs[0]

    rows = []
    for index, alpha in enumerate(COHERENT_INPUTS):
        input_state = pg.GaussianInput.coherent(alpha)
        ideal = channel.apply(input_state.state(0))

        ensemble = {}
        for backend in ("gaussian", "piquasso"):
            shots = pg.run_shots(
                pattern,
                SHOTS,
                backend=backend,
                inputs={0: input_state},
                seed=SEED + 97 * index,
            )
            ensemble[backend] = shots.ensemble_state()

        single = pg.simulate(pattern, backend="gaussian", inputs={0: input_state}, seed=SEED + index)
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
        }
        for backend, state in ensemble.items():
            row[f"{backend}_ensemble_q_mean"] = state.mean[0]
            row[f"{backend}_ensemble_p_mean"] = state.mean[1]
            row[f"{backend}_ensemble_q_var"] = state.covariance[0, 0]
            row[f"{backend}_ensemble_p_var"] = state.covariance[1, 1]
            row[f"{backend}_mean_error"] = common.frobenius_error(state.mean, ideal.mean)
            row[f"{backend}_covariance_error"] = common.frobenius_error(state.covariance, ideal.covariance)
        rows.append(row)

    # Shot-noise-scaled tolerance: SEM of a variance estimate from SHOTS draws
    # scales roughly like sqrt(2/SHOTS) times the variance itself; use a
    # generous multiple to avoid a flaky assertion while still catching bugs.
    tolerance = 0.35
    failures = [
        r for r in rows if r["gaussian_mean_error"] > tolerance or r["gaussian_covariance_error"] > tolerance
    ]
    if failures:
        common.save_csv(failures, "R1_wire_teleportation_FAILURES")
        raise AssertionError(f"{len(failures)} coherent inputs exceeded tolerance {tolerance}; see FAILURES csv")

    common.save_result(
        rows,
        "R1_wire_teleportation",
        extra={
            "protocol": "one-step Fourier wire (k=0)",
            "squeezing": SQUEEZING,
            "shots": SHOTS,
            "seed_base": SEED,
            "channel_matrix": channel.matrix.tolist(),
            "channel_noise": channel.noise.tolist(),
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    x = np.arange(len(rows))
    labels = [f"{r['alpha_real']:.1f}{r['alpha_imag']:+.1f}i" for r in rows]
    width = 0.25
    axes[0].bar(x - width, [r["ideal_q_var"] for r in rows], width, label="Analytic channel", color="#24677b")
    axes[0].bar(x, [r["gaussian_ensemble_q_var"] for r in rows], width, label="Gaussian ensemble", color="#c66d27")
    axes[0].bar(x + width, [r["piquasso_ensemble_q_var"] for r in rows], width, label="Piquasso ensemble", color="#627a36")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Output q variance")
    axes[0].legend(fontsize=7)

    axes[1].bar(x - width, [r["ideal_p_var"] for r in rows], width, label="Analytic channel", color="#24677b")
    axes[1].bar(x, [r["gaussian_ensemble_p_var"] for r in rows], width, label="Gaussian ensemble", color="#c66d27")
    axes[1].bar(x + width, [r["piquasso_ensemble_p_var"] for r in rows], width, label="Piquasso ensemble", color="#627a36")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Output p variance")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"One-step Fourier wire, resource squeezing r={SQUEEZING}")
    common.save_figure(fig, "R1_wire_teleportation")
    plt.close(fig)

    common.print_summary(
        "R1 wire teleportation",
        inputs=len(rows),
        max_gaussian_mean_error=max(r["gaussian_mean_error"] for r in rows),
        max_gaussian_covariance_error=max(r["gaussian_covariance_error"] for r in rows),
        max_piquasso_mean_error=max(r["piquasso_mean_error"] for r in rows),
        max_piquasso_covariance_error=max(r["piquasso_covariance_error"] for r in rows),
    )


if __name__ == "__main__":
    main()
