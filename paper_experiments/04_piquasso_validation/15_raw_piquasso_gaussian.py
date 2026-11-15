"""R15: Raw Piquasso Gaussian reference (independent, not through PhotoGraphiQ).

For displacement, rotation, squeezing, beamsplitter, CZ (native
``pq.ControlledZ``, the correct Gaussian-regime representation; Piquasso's
``GaussianTransform`` is instead used by PhotoGraphiQ's *Fock* backend) and a
short compound circuit, this script:

1. Builds a raw ``piquasso`` program directly with the Piquasso public API
   (no PhotoGraphiQ wrapper call in the numerical path) starting from vacuum,
   preparing the same coherent input and applying the same native gate.
2. Runs the identical physical operation through PhotoGraphiQ's
   ``backend="piquasso"`` path.
3. Compares mean-vector norm error, covariance Frobenius error and mean
   photon number (via the documented analytic Gaussian formula, applied
   identically to both sides) across many random parameter draws.

Upgrade migration (2026-11): step 2 now runs the identical pattern on all
three PhotoGraphiQ Gaussian engines -- ``piquasso`` (historical columns, names
unchanged), NumPy ``gaussian`` and C++ ``native-gaussian`` -- and every engine
must match raw Piquasso to the same tolerance. Raw Piquasso stays the
independent reference (no PhotoGraphiQ code in its numerical path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import piquasso as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

SEED = 20260913
N_RANDOM = 20
HBAR = 2.0
PG_BACKENDS = ("piquasso", "gaussian", "native-gaussian")


def mean_photon_number(mean: np.ndarray, covariance: np.ndarray) -> float:
    """Independent analytic single-mode formula: (Tr(V)+mu.mu-2)/4 (docs/theory.md)."""
    return float((np.trace(covariance) + mean @ mean - 2) / 4)


def raw_single_mode(alpha: complex, gate_builder):
    """Prepare a coherent state and apply one native gate via a fresh raw program."""
    with pq.Program() as program:
        pq.Q(0) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=abs(alpha), phi=np.angle(alpha))
        gate_builder(program)
    state = pq.GaussianSimulator(d=1, config=pq.Config(hbar=HBAR)).execute(program).state
    return np.array(state.xpxp_mean_vector), np.array(state.xpxp_covariance_matrix) / 2


def raw_two_mode(alpha0: complex, alpha1: complex, gate_builder):
    with pq.Program() as program:
        pq.Q(0, 1) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=abs(alpha0), phi=np.angle(alpha0))
        pq.Q(1) | pq.Displacement(r=abs(alpha1), phi=np.angle(alpha1))
        gate_builder(program)
    state = pq.GaussianSimulator(d=2, config=pq.Config(hbar=HBAR)).execute(program).state
    return np.array(state.xpxp_mean_vector), np.array(state.xpxp_covariance_matrix) / 2


def pg_single_mode(alpha: complex, command):
    pattern = pg.Pattern(inputs=(0,)).append(command).append(pg.Output((0,)))
    out = {}
    for backend in PG_BACKENDS:
        result = pg.simulate(pattern, backend=backend, inputs={0: pg.GaussianInput.coherent(alpha)}, seed=0)
        out[backend] = (result.state.mean, result.state.covariance)
    return out


def pg_two_mode(alpha0: complex, alpha1: complex, commands):
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.extend(commands)
    pattern.append(pg.Output((0, 1)))
    inputs = {0: pg.GaussianInput.coherent(alpha0), 1: pg.GaussianInput.coherent(alpha1)}
    out = {}
    for backend in PG_BACKENDS:
        result = pg.simulate(pattern, backend=backend, inputs=inputs, seed=0)
        out[backend] = (result.state.mean, result.state.covariance)
    return out


def compare(name, alpha_desc, raw_mean, raw_cov, pg_results):
    pg_mean, pg_cov = pg_results["piquasso"]
    extra = {}
    for backend in PG_BACKENDS[1:]:
        extra[f"{backend}_mean_vector_norm_error"] = common.frobenius_error(pg_results[backend][0], raw_mean)
        extra[f"{backend}_covariance_frobenius_error"] = common.frobenius_error(pg_results[backend][1], raw_cov)
    mean_error = common.frobenius_error(pg_mean, raw_mean)
    cov_error = common.frobenius_error(pg_cov, raw_cov)
    raw_n = mean_photon_number(raw_mean[:2], raw_cov[:2, :2]) if len(raw_mean) == 2 else None
    pg_n = mean_photon_number(pg_mean[:2], pg_cov[:2, :2]) if len(pg_mean) == 2 else None
    return {
        "gate": name,
        "params": alpha_desc,
        "mean_vector_norm_error": mean_error,
        "covariance_frobenius_error": cov_error,
        "photon_number_error": None if raw_n is None else abs(raw_n - pg_n),
        **extra,
    }


def main():
    plt = common.setup_style()
    generator = common.rng(SEED)
    rows = []

    for _ in range(N_RANDOM):
        alpha = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        q, p = generator.uniform(-1, 1), generator.uniform(-1, 1)
        shift = complex(q, p) / 2

        def gate_builder(program, shift=shift):
            pq.Q(0) | pq.Displacement(r=abs(shift), phi=np.angle(shift))

        raw_mean, raw_cov = raw_single_mode(alpha, gate_builder)
        pg_results = pg_single_mode(alpha, pg.Displace(0, q=q, p=p))
        rows.append(compare("displacement", {"alpha": [alpha.real, alpha.imag], "q": q, "p": p}, raw_mean, raw_cov, pg_results))

    for _ in range(N_RANDOM):
        alpha = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        angle = generator.uniform(-np.pi, np.pi)
        raw_mean, raw_cov = raw_single_mode(alpha, lambda program, angle=angle: (pq.Q(0) | pq.Phaseshifter(phi=angle)))
        pg_results = pg_single_mode(alpha, pg.Rotate(0, angle))
        rows.append(compare("rotation", {"alpha": [alpha.real, alpha.imag], "angle": angle}, raw_mean, raw_cov, pg_results))

    for _ in range(N_RANDOM):
        alpha = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        r = generator.uniform(-1.2, 1.2)
        raw_mean, raw_cov = raw_single_mode(alpha, lambda program, r=r: (pq.Q(0) | pq.Squeezing(r=r)))
        pg_results = pg_single_mode(alpha, pg.Squeeze(0, r))
        rows.append(compare("squeezing", {"alpha": [alpha.real, alpha.imag], "r": r}, raw_mean, raw_cov, pg_results))

    for _ in range(N_RANDOM):
        alpha0 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        alpha1 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        theta = generator.uniform(-np.pi / 2, np.pi / 2)
        raw_mean, raw_cov = raw_two_mode(alpha0, alpha1, lambda program, theta=theta: (pq.Q(0, 1) | pq.Beamsplitter(theta=theta, phi=0.0)))
        pg_results = pg_two_mode(alpha0, alpha1, [pg.BeamSplitter(0, 1, theta)])
        rows.append(
            compare(
                "beamsplitter",
                {"alpha0": [alpha0.real, alpha0.imag], "alpha1": [alpha1.real, alpha1.imag], "theta": theta},
                raw_mean, raw_cov, pg_results,
            )
        )

    for _ in range(N_RANDOM):
        alpha0 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        alpha1 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        weight = generator.uniform(-2, 2)
        raw_mean, raw_cov = raw_two_mode(alpha0, alpha1, lambda program, weight=weight: (pq.Q(0, 1) | pq.ControlledZ(s=weight)))
        pg_results = pg_two_mode(alpha0, alpha1, [pg.Entangle(0, 1, weight)])
        rows.append(
            compare(
                "cz",
                {"alpha0": [alpha0.real, alpha0.imag], "alpha1": [alpha1.real, alpha1.imag], "weight": weight},
                raw_mean, raw_cov, pg_results,
            )
        )

    for _ in range(N_RANDOM):
        alpha0 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        alpha1 = generator.uniform(-1, 1) + 1j * generator.uniform(-1, 1)
        r, angle, weight, q, p = (
            generator.uniform(-0.6, 0.6),
            generator.uniform(-np.pi, np.pi),
            generator.uniform(-1, 1),
            generator.uniform(-1, 1),
            generator.uniform(-1, 1),
        )

        def raw_compound(program, r=r, angle=angle, weight=weight, q=q, p=p):
            shift = complex(q, p) / 2
            pq.Q(0) | pq.Squeezing(r=r)
            pq.Q(0) | pq.Phaseshifter(phi=angle)
            pq.Q(0, 1) | pq.ControlledZ(s=weight)
            pq.Q(1) | pq.Displacement(r=abs(shift), phi=np.angle(shift))

        raw_mean, raw_cov = raw_two_mode(alpha0, alpha1, raw_compound)
        pg_results = pg_two_mode(
            alpha0, alpha1, [pg.Squeeze(0, r), pg.Rotate(0, angle), pg.Entangle(0, 1, weight), pg.Displace(1, q=q, p=p)]
        )
        rows.append(
            compare(
                "compound(squeeze+rotate+cz+displace)",
                {"r": r, "angle": angle, "weight": weight, "q": q, "p": p},
                raw_mean, raw_cov, pg_results,
            )
        )

    tolerance = 1e-9
    error_keys = [k for k in rows[0] if k.endswith(("mean_vector_norm_error", "covariance_frobenius_error"))]
    failures = [r for r in rows if any(r[k] > tolerance for k in error_keys)]
    if failures:
        common.save_csv(failures, "R15_raw_piquasso_gaussian_FAILURES")
        raise AssertionError(f"{len(failures)} cases disagreed with raw Piquasso beyond tolerance; see FAILURES csv")

    common.save_result(rows, "R15_raw_piquasso_gaussian", extra={"n_random_per_gate": N_RANDOM, "seed": SEED})

    gates = sorted(set(r["gate"] for r in rows))
    summary_table = [
        {
            "gate": gate,
            "n": len([r for r in rows if r["gate"] == gate]),
            "max_mean_error": max(r["mean_vector_norm_error"] for r in rows if r["gate"] == gate),
            "max_covariance_error": max(r["covariance_frobenius_error"] for r in rows if r["gate"] == gate),
            **{
                f"{b}_max_{kind}_error": max(r[f"{b}_{key}"] for r in rows if r["gate"] == gate)
                for b in PG_BACKENDS[1:]
                for kind, key in (("mean", "mean_vector_norm_error"), ("covariance", "covariance_frobenius_error"))
            },
        }
        for gate in gates
    ]
    common.save_csv(summary_table, "R15_raw_piquasso_gaussian_summary_table")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(gates))
    width = 0.27
    ax.bar(x - width, [common.safe_log10(t["max_covariance_error"]) for t in summary_table], width,
           color="#627a36", label="piquasso backend")
    ax.bar(x, [common.safe_log10(t["gaussian_max_covariance_error"]) for t in summary_table], width,
           color="#c66d27", label="gaussian (NumPy)")
    ax.bar(x + width, [common.safe_log10(t["native-gaussian_max_covariance_error"]) for t in summary_table], width,
           color="#7a3e9d", label="native-gaussian (C++)")
    ax.legend(fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$\log_{10}$(max covariance error)")
    ax.set_title("PhotoGraphiQ Gaussian engines vs. raw Piquasso (exact zero plotted at -300)", fontsize=9)
    common.save_figure(fig, "R15_raw_piquasso_gaussian")
    plt.close(fig)

    common.print_summary(
        "R15 raw Piquasso Gaussian reference",
        total_cases=len(rows),
        max_mean_error=max(r["mean_vector_norm_error"] for r in rows),
        max_covariance_error=max(r["covariance_frobenius_error"] for r in rows),
        max_error_any_engine=max(r[k] for r in rows for k in error_keys),
    )


if __name__ == "__main__":
    main()
