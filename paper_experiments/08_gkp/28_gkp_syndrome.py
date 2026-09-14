"""R28: GKP syndrome extraction (finite-energy/Fock approximation).

Prepares a finite-energy GKP |0>_L codeword, physically displaces it by a
known small q-shift, then runs the package's actual one-ancilla SUM syndrome
extraction (``photographiq.gkp.correction_pattern``) with a finite "logical
plus" ancilla. A single trajectory's raw ancilla homodyne outcome samples one
of many comb peaks of the (also finite-energy) ancilla, so the decoded
residual has substantial single-shot noise set by the resource's intrinsic
peak width; this script instead reports the ENSEMBLE (circular) mean and
spread over repeated seeded trajectories, which is the physically meaningful
way to test whether extraction is centered on the true displacement. This is
a finite-ancilla, finite-energy approximation, not a fault-tolerant claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.gkp import SPACING, decode_shift  # noqa: E402

CUTOFF = 80  # two GKP-scale modes share one total-photon cutoff (input + ancilla)
PEAK_WIDTH = ENVELOPE = 0.35
DELTAS = np.linspace(-0.6, 0.6, 9)
N_SEEDS = 16


def displaced_fock_vector(fock_input, delta_q, cutoff):
    pattern = pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=delta_q)).append(pg.Output((0,))).validate()
    state = pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff, inputs={0: fock_input}, seed=0).state
    amplitudes = dict(zip(state.basis, state.state_vector, strict=True))
    vector = np.array([amplitudes.get((n,), 0j) for n in range(cutoff)])
    vector /= np.linalg.norm(vector)
    return pg.FockInput(tuple(vector))


def circular_stats(residuals: np.ndarray, spacing: float):
    angles = residuals / spacing * 2 * np.pi
    mean_vector = np.mean(np.exp(1j * angles))
    circular_mean = float(np.angle(mean_vector) * spacing / (2 * np.pi))
    circular_std = float(np.sqrt(max(0.0, -2 * np.log(abs(mean_vector)))) * spacing / (2 * np.pi))
    return circular_mean, circular_std


def main():
    plt = common.setup_style()
    code_zero = pg.GKPResource(0, peak_width=PEAK_WIDTH, envelope=ENVELOPE)
    fock_zero, _ = code_zero.project(CUTOFF)
    ancilla_plus = pg.gkp.superposition(1, 1, cutoff=CUTOFF, peak_width=PEAK_WIDTH, envelope=ENVELOPE)
    pattern = pg.gkp.correction_pattern(quadrature="q", resource=ancilla_plus)

    rows = []
    for delta in DELTAS:
        delta = float(delta)
        shifted_input = displaced_fock_vector(fock_zero, delta, CUTOFF)

        residuals, corrected_q, ok_seeds = [], [], 0
        for seed in range(N_SEEDS):
            failure = common.safe_cutoff_run(
                lambda seed=seed: pg.simulate(
                    pattern, backend="piquasso-fock", cutoff=CUTOFF, inputs={"in": shifted_input}, seed=3000 + 97 * seed
                ),
                label=f"delta={delta} seed={seed}",
            )
            if not failure["ok"]:
                continue
            result = failure["value"]
            residual, _ = decode_shift(result.records["syndrome"])
            residuals.append(residual)
            corrected_q.append(result.state.quadrature("in")[0])
            ok_seeds += 1

        expected_residual = float(((delta + SPACING / 2) % SPACING) - SPACING / 2)
        if residuals:
            circular_mean, circular_std = circular_stats(np.array(residuals), SPACING)
        else:
            circular_mean = circular_std = None
        rows.append(
            {
                "delta": delta,
                "ok_seeds": ok_seeds,
                "expected_residual": expected_residual,
                "circular_mean_decoded_residual": circular_mean,
                "circular_std_decoded_residual": circular_std,
                "circular_mean_error": None if circular_mean is None else abs(circular_mean - expected_residual),
                "mean_abs_corrected_q": float(np.mean(np.abs(corrected_q))) if corrected_q else None,
                "abs_uncorrected_q": abs(delta),
            }
        )

    ok_rows = [r for r in rows if r["ok_seeds"] > 0]
    # Statistical tolerance: circular_std / sqrt(N) sets the expected standard
    # error of the ensemble mean; a factor of 5 gives headroom against a
    # single flaky run while still catching a genuinely biased estimator.
    failures = [
        r for r in ok_rows
        if r["circular_mean_error"] > 5 * r["circular_std_decoded_residual"] / np.sqrt(max(1, r["ok_seeds"])) + 0.05
    ]
    if failures:
        common.save_csv(failures, "R28_gkp_syndrome_FAILURES")
        raise AssertionError(f"{len(failures)} deltas show a biased syndrome estimate beyond statistical tolerance; see FAILURES csv")

    common.save_result(rows, "R28_gkp_syndrome", extra={"cutoff": CUTOFF, "peak_width": PEAK_WIDTH, "envelope": ENVELOPE, "n_seeds": N_SEEDS})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].errorbar(
        [r["delta"] for r in ok_rows], [r["circular_mean_decoded_residual"] for r in ok_rows],
        yerr=[r["circular_std_decoded_residual"] / np.sqrt(r["ok_seeds"]) for r in ok_rows],
        fmt="o-", label=f"Ensemble mean decoded residual (n={N_SEEDS})", color="#24677b", capsize=3,
    )
    axes[0].plot([r["delta"] for r in ok_rows], [r["expected_residual"] for r in ok_rows], "--", label="Ideal (delta mod spacing)", color="grey")
    axes[0].set_xlabel("Imposed displacement delta")
    axes[0].set_ylabel("Decoded residual (ensemble mean +/- SEM)")
    axes[0].legend(fontsize=7)

    axes[1].plot([r["delta"] for r in ok_rows], [r["mean_abs_corrected_q"] for r in ok_rows], "o-", label="<|corrected mean q|>", color="#c66d27")
    axes[1].plot([r["delta"] for r in ok_rows], [r["abs_uncorrected_q"] for r in ok_rows], "--", label="|uncorrected q| = |delta|", color="grey")
    axes[1].set_xlabel("Imposed displacement delta")
    axes[1].set_ylabel("Residual |mean q| after correction")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"GKP q-syndrome extraction, finite ancilla (cutoff={CUTOFF}, width={PEAK_WIDTH})")
    common.save_figure(fig, "R28_gkp_syndrome")
    plt.close(fig)

    common.print_summary(
        "R28 GKP syndrome extraction (ensemble statistics)",
        deltas=len(rows),
        seeds_per_delta=N_SEEDS,
        max_circular_mean_error=max((r["circular_mean_error"] for r in ok_rows), default=None),
        median_circular_std=float(np.median([r["circular_std_decoded_residual"] for r in ok_rows])) if ok_rows else None,
    )


if __name__ == "__main__":
    main()
