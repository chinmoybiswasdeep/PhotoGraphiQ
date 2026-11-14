"""R18: Gaussian backend cross-check.

Runs the same random, low-energy, purely physical (non-adaptive) two-mode
Gaussian workloads through the NumPy Gaussian backend and the Piquasso
Gaussian backend (both exact) and compares state mean/covariance/photon
number/quadrature exactly. It additionally runs the same preparation through
the experimental ``piquasso-mixed-fock`` backend at a modest cutoff and
compares photon number/quadrature -- this third comparison is a *finite Fock
truncation approximation*, not an exact-backend agreement, and uses a looser,
explicitly reported tolerance.

Upgrade migration (2026-11): the C++ ``native-gaussian`` engine joins the
exact comparison (same 1e-8 tolerance against NumPy ``gaussian``). The
finite-cutoff comparison now covers all four Fock engines --
``native-fock``, ``piquasso-fock``, ``native-mixed-fock`` and the historical
``piquasso-mixed-fock`` (columns ``mixed_fock_*`` keep their old meaning) --
at the same cutoff, reporting photon-number and quadrature-mean errors
against the exact Gaussian state and the minimum retained Fock norm of each
run. Engines that hit the norm guard are recorded, never silently dropped.
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
FOCK_ENGINES = ("native-fock", "piquasso-fock", "native-mixed-fock")
# Finite-cutoff agreement criterion for the Fock engines (declared before evaluation):
# |photon number - Gaussian| and |<q> - Gaussian| below 1e-3 whenever the engine's minimum
# retained norm is at least 1 - 1e-6. Below that the comparison is reported, not asserted.
FOCK_TOLERANCE = 1e-3


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
        native = pg.simulate(pattern, backend="native-gaussian", inputs=inputs, seed=0).state

        mean_error = common.frobenius_error(piquasso.mean, gaussian.mean)
        cov_error = common.frobenius_error(piquasso.covariance, gaussian.covariance)
        photon_error = max(abs(piquasso.photon_number(n) - gaussian.photon_number(n)) for n in (0, 1))

        row = {
            "sample": i,
            "n_gates": len(pattern.commands) - 1,
            "gaussian_vs_piquasso_mean_error": mean_error,
            "gaussian_vs_piquasso_covariance_error": cov_error,
            "gaussian_vs_piquasso_photon_number_error": photon_error,
            "gaussian_vs_native_mean_error": common.frobenius_error(native.mean, gaussian.mean),
            "gaussian_vs_native_covariance_error": common.frobenius_error(native.covariance, gaussian.covariance),
        }
        for engine in FOCK_ENGINES:
            run = common.safe_cutoff_run(
                lambda engine=engine: pg.simulate(pattern, backend=engine, cutoff=FOCK_CUTOFF, inputs=inputs, seed=0).state,
                label=f"sample {i} {engine}",
            )
            if run["ok"]:
                st = run["value"]
                row[f"{engine}_photon_number_error"] = max(abs(st.photon_number(n) - gaussian.photon_number(n)) for n in (0, 1))
                row[f"{engine}_q_error"] = max(abs(st.quadrature(n)[0] - gaussian.quadrature(n)[0]) for n in (0, 1))
                row[f"{engine}_min_retained_norm"] = min(st.retained_norms)
            else:
                row[f"{engine}_photon_number_error"] = row[f"{engine}_q_error"] = None
                row[f"{engine}_min_retained_norm"] = None
                row[f"{engine}_error"] = run["error"]

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
            row["piquasso-mixed-fock_min_retained_norm"] = min(fock_state.retained_norms)
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
        or r["gaussian_vs_native_mean_error"] > exact_tolerance
        or r["gaussian_vs_native_covariance_error"] > exact_tolerance
    ]
    fock_failures = [
        r for r in rows for engine in FOCK_ENGINES
        if r[f"{engine}_min_retained_norm"] is not None and r[f"{engine}_min_retained_norm"] >= 1 - 1e-6
        and max(r[f"{engine}_photon_number_error"], r[f"{engine}_q_error"]) > FOCK_TOLERANCE
    ]
    if fock_failures:
        common.save_csv(fock_failures, "R18_gaussian_backends_FAILURES")
        raise AssertionError("A Fock engine disagreed with the exact Gaussian state at retained norm ~1; see FAILURES csv")
    if exact_failures:
        common.save_csv(exact_failures, "R18_gaussian_backends_FAILURES")
        raise AssertionError("NumPy Gaussian and Piquasso Gaussian backends disagreed beyond tolerance; see FAILURES csv")

    common.save_result(rows, "R18_gaussian_backends",
                       extra={"seed": SEED, "fock_cutoff": FOCK_CUTOFF, "fock_engines": list(FOCK_ENGINES) + ["piquasso-mixed-fock"],
                              "exact_tolerance": exact_tolerance, "fock_tolerance": FOCK_TOLERANCE})

    ok_rows = [r for r in rows if r["mixed_fock_ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(len(rows))
    floor = -17.0  # exact zeros are drawn at the floor
    for key, label, marker, color in (
        ("gaussian_vs_piquasso_covariance_error", "piquasso", "o", "#24677b"),
        ("gaussian_vs_native_covariance_error", "native-gaussian", "x", "#7a3e9d"),
    ):
        axes[0].plot(x, [max(floor, common.safe_log10(r[key])) for r in rows], marker, color=color, label=label, linestyle="none")
    axes[0].set_xlabel("Random two-mode workload (sample index)")
    axes[0].set_ylabel(r"$\log_{10}$(cov. error vs. NumPy)")
    axes[0].set_ylim(floor - 0.5, -10)
    axes[0].legend(fontsize=7)
    series = [(f"{e}_photon_number_error", e) for e in FOCK_ENGINES] + [("mixed_fock_photon_number_error", "piquasso-mixed-fock")]
    for (key, label), marker, offset in zip(series, ("o", "x", "s", "+"), (-0.24, -0.08, 0.08, 0.24), strict=True):
        pts = [(i + offset, max(floor, common.safe_log10(r[key]))) for i, r in enumerate(rows) if r.get(key) is not None]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker, linestyle="none", label=f"{label} ({len(pts)}/{len(rows)} ok)")
    axes[1].axhline(np.log10(FOCK_TOLERANCE), color="grey", linestyle=":", label="tolerance")
    axes[1].legend(fontsize=6.5)
    axes[1].set_xlabel("Random two-mode workload (sample index)")
    axes[1].set_ylabel(rf"$\log_{{10}}$(photon-number error), $c={FOCK_CUTOFF}$")
    fig.suptitle("Backend cross-checks: exact (left) vs. finite-cutoff approximation (right)")
    common.save_figure(fig, "R18_gaussian_backends")
    plt.close(fig)

    common.print_summary(
        "R18 Gaussian backend cross-check",
        samples=len(rows),
        max_exact_covariance_error=max(r["gaussian_vs_piquasso_covariance_error"] for r in rows),
        mixed_fock_successes=len(ok_rows),
        max_native_gaussian_covariance_error=max(r["gaussian_vs_native_covariance_error"] for r in rows),
        **{f"{e}_successes": sum(r[f"{e}_min_retained_norm"] is not None for r in rows) for e in FOCK_ENGINES},
        **{f"{e}_max_photon_error": max((r[f"{e}_photon_number_error"] for r in rows if r[f"{e}_photon_number_error"] is not None), default=None) for e in FOCK_ENGINES},
        median_mixed_fock_photon_error=float(np.median([r["mixed_fock_photon_number_error"] for r in ok_rows])) if ok_rows else None,
    )


if __name__ == "__main__":
    main()
