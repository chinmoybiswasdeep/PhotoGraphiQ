"""R18: Gaussian backend cross-check.

Runs the same random, low-energy, purely physical (non-adaptive) two-mode
Gaussian workloads through the NumPy Gaussian backend and the Piquasso
Gaussian backend (both exact) and compares state mean/covariance/photon
number/quadrature exactly. It additionally runs the same preparation through
the experimental ``piquasso-mixed-fock`` backend at a modest cutoff and
compares photon number/quadrature -- this third comparison is a *finite Fock
truncation approximation*, not an exact-backend agreement, and uses a looser,
explicitly reported tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SEED = 20260913
N_RANDOM = 15
FOCK_CUTOFF = 14
GATE_KINDS = ("rotate", "squeeze", "beamsplitter", "cz", "displace")


def random_pattern(generator, n_gates=4):
    pattern = pg.Pattern(inputs=(0, 1))
    for _ in range(n_gates):
        kind = generator.choice(GATE_KINDS)
        if kind == "rotate":
            pattern.append(pg.Rotate(int(generator.integers(0, 2)), float(generator.uniform(-np.pi, np.pi))))
        elif kind == "squeeze":
            pattern.append(pg.Squeeze(int(generator.integers(0, 2)), float(generator.uniform(-0.4, 0.4))))
        elif kind == "beamsplitter":
            pattern.append(pg.BeamSplitter(0, 1, float(generator.uniform(-np.pi / 2, np.pi / 2))))
        elif kind == "cz":
            pattern.append(pg.Entangle(0, 1, float(generator.uniform(-1, 1))))
        else:
            pattern.append(
                pg.Displace(int(generator.integers(0, 2)), q=float(generator.uniform(-0.3, 0.3)), p=float(generator.uniform(-0.3, 0.3)))
            )
    pattern.append(pg.Output((0, 1)))
    return pattern.validate()


def main():
    plt = common.setup_style()
    generator = common.rng(SEED)
    rows = []
    for i in range(N_RANDOM):
        pattern = random_pattern(generator)
        alpha0 = generator.uniform(-0.25, 0.25) + 1j * generator.uniform(-0.25, 0.25)
        alpha1 = generator.uniform(-0.25, 0.25) + 1j * generator.uniform(-0.25, 0.25)
        inputs = {0: pg.GaussianInput.coherent(alpha0), 1: pg.GaussianInput.coherent(alpha1)}

        gaussian = pg.simulate(pattern, backend="gaussian", inputs=inputs, seed=0).state
        piquasso = pg.simulate(pattern, backend="piquasso", inputs=inputs, seed=0).state

        mean_error = common.frobenius_error(piquasso.mean, gaussian.mean)
        cov_error = common.frobenius_error(piquasso.covariance, gaussian.covariance)
        photon_error = max(abs(piquasso.photon_number(n) - gaussian.photon_number(n)) for n in (0, 1))

        row = {
            "sample": i,
            "n_gates": len(pattern.commands) - 1,
            "gaussian_vs_piquasso_mean_error": mean_error,
            "gaussian_vs_piquasso_covariance_error": cov_error,
            "gaussian_vs_piquasso_photon_number_error": photon_error,
        }

        fock_failure = common.safe_cutoff_run(
            lambda: pg.simulate(pattern, backend="piquasso-mixed-fock", cutoff=FOCK_CUTOFF, inputs=inputs, seed=0).state,
            label=f"sample {i} mixed-fock",
        )
        if fock_failure["ok"]:
            fock_state = fock_failure["value"]
            row["mixed_fock_photon_number_error"] = max(
                abs(fock_state.photon_number(n) - gaussian.photon_number(n)) for n in (0, 1)
            )
            row["mixed_fock_q_error"] = max(
                abs(fock_state.quadrature(n)[0] - gaussian.quadrature(n)[0]) for n in (0, 1)
            )
            row["mixed_fock_cutoff"] = FOCK_CUTOFF
            row["mixed_fock_ok"] = True
        else:
            row.update(
                mixed_fock_photon_number_error=None, mixed_fock_q_error=None,
                mixed_fock_cutoff=FOCK_CUTOFF, mixed_fock_ok=False, mixed_fock_error=fock_failure["error"],
            )
        rows.append(row)

    exact_tolerance = 1e-8
    exact_failures = [
        r for r in rows
        if r["gaussian_vs_piquasso_mean_error"] > exact_tolerance
        or r["gaussian_vs_piquasso_covariance_error"] > exact_tolerance
        or r["gaussian_vs_piquasso_photon_number_error"] > exact_tolerance
    ]
    if exact_failures:
        common.save_csv(exact_failures, "R18_gaussian_backends_FAILURES")
        raise AssertionError("NumPy Gaussian and Piquasso Gaussian backends disagreed beyond tolerance; see FAILURES csv")

    common.save_result(rows, "R18_gaussian_backends", extra={"seed": SEED, "fock_cutoff": FOCK_CUTOFF})

    ok_rows = [r for r in rows if r["mixed_fock_ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist([common.safe_log10(r["gaussian_vs_piquasso_covariance_error"]) for r in rows], bins=15, color="#24677b")
    axes[0].set_xlabel(r"$\log_{10}$(covariance error): NumPy vs. Piquasso (exact)")
    axes[0].set_ylabel("Count")
    if ok_rows:
        axes[1].hist([common.safe_log10(r["mixed_fock_photon_number_error"]) for r in ok_rows], bins=15, color="#c66d27")
    axes[1].set_xlabel(rf"$\log_{{10}}$(photon-number error): Gaussian vs. mixed-Fock ($c={FOCK_CUTOFF}$)")
    fig.suptitle("Backend cross-checks: exact (left) vs. finite-cutoff approximation (right)")
    common.save_figure(fig, "R18_gaussian_backends")
    plt.close(fig)

    common.print_summary(
        "R18 Gaussian backend cross-check",
        samples=len(rows),
        max_exact_covariance_error=max(r["gaussian_vs_piquasso_covariance_error"] for r in rows),
        mixed_fock_successes=len(ok_rows),
        median_mixed_fock_photon_error=float(np.median([r["mixed_fock_photon_number_error"] for r in ok_rows])) if ok_rows else None,
    )


if __name__ == "__main__":
    main()
