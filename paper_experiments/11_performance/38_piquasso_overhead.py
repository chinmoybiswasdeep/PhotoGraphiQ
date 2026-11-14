"""R38: PhotoGraphiQ overhead versus raw Piquasso.

Builds matched N-mode Gaussian workloads (per-mode squeeze+rotate, plus a CZ
chain) and times two independent execution paths with warm-up and repeated
measurements: (1) one single raw ``piquasso`` program executed once with one
``GaussianSimulator`` call (the natural, most efficient native-engine usage),
and (2) the identical physical operations through
``photographiq.simulate(..., backend="piquasso")``, which executes one native
Program/Simulator per gate (see ``backends/piquasso.py``). Raw Piquasso is
the numerical engine reference; PhotoGraphiQ adds graph/pattern/MBQC
abstractions on top, and the ratio T_PG/T_PQ quantifies that added cost on
this machine (not a comparative benchmark claim across machines).

Upgrade migration (2026-11): the same workload is also timed on NumPy
``gaussian`` and C++ ``native-gaussian`` (ratios to raw Piquasso reported as
additional columns; the original two columns are unchanged). A correctness
gate precedes timing: every engine's output covariance must match the raw
Piquasso state (CORRECTNESS_TOLERANCE). ``--quick`` runs a short sweep; the
default is the publication sweep; timings are provisional unless
PGQ_IDLE_MACHINE=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import piquasso as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import metadata  # noqa: E402

import photographiq as pg  # noqa: E402

MODE_COUNTS = [5, 10, 20, 40]
HBAR = 2.0


QUICK = "--quick" in sys.argv[1:]


def timing_environment() -> dict:
    """Thread/hardware context for timings (the metadata companion adds CPU/GPU/packages)."""
    import os

    from photographiq import native

    return {
        "configuration": "quick" if QUICK else "full",
        # Timings are provisional unless the run was declared to be on an idle machine.
        "timing_status": "final (idle machine declared via PGQ_IDLE_MACHINE=1)"
        if os.environ.get("PGQ_IDLE_MACHINE") == "1"
        else "provisional (shared machine)",
        "os_cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "native_build_info": native.build_info(),
        "native_components": list(native.components()),
    }

if QUICK:
    MODE_COUNTS = [5, 10]
EXTRA_BACKENDS = ("gaussian", "native-gaussian")
CORRECTNESS_TOLERANCE = 1e-9
def raw_piquasso_run(n_modes: int):
    with pq.Program() as program:
        pq.Q(*range(n_modes)) | pq.Vacuum()
        for i in range(n_modes):
            pq.Q(i) | pq.Squeezing(r=0.1 + 0.01 * i)
            pq.Q(i) | pq.Phaseshifter(phi=0.2 + 0.01 * i)
        for i in range(n_modes - 1):
            pq.Q(i, i + 1) | pq.ControlledZ(s=0.3)
    return pq.GaussianSimulator(d=n_modes, config=pq.Config(hbar=HBAR)).execute(program).state


def build_pattern(n_modes: int) -> pg.Pattern:
    pattern = pg.Pattern(inputs=tuple(range(n_modes)))
    for i in range(n_modes):
        pattern.append(pg.Squeeze(i, 0.1 + 0.01 * i))
        pattern.append(pg.Rotate(i, 0.2 + 0.01 * i))
    for i in range(n_modes - 1):
        pattern.append(pg.Entangle(i, i + 1, 0.3))
    pattern.append(pg.Output(tuple(range(n_modes))))
    return pattern.validate()


def main():
    plt = common.setup_style()
    rows = []
    for n_modes in MODE_COUNTS:
        pattern = build_pattern(n_modes)
        warmup, repeats = (2, 9) if n_modes <= 10 else (1, 5)
        if QUICK:
            warmup, repeats = 1, 3
        raw_state = raw_piquasso_run(n_modes)
        raw_cov = np.array(raw_state.xpxp_covariance_matrix) / 2
        for backend in ("piquasso", *EXTRA_BACKENDS):
            cov = pg.simulate(pattern, backend=backend, seed=0).state.covariance
            if common.frobenius_error(cov, raw_cov) > CORRECTNESS_TOLERANCE:
                raise AssertionError(f"{backend} disagrees with raw Piquasso at n={n_modes}")
        extra_timing = {
            b: common.benchmark(lambda pattern=pattern, b=b: pg.simulate(pattern, backend=b, seed=0), warmup=warmup, repeats=repeats)
            for b in EXTRA_BACKENDS
        }

        raw_timing = common.benchmark(lambda n=n_modes: raw_piquasso_run(n), warmup=warmup, repeats=repeats)
        pg_timing = common.benchmark(
            lambda pattern=pattern: pg.simulate(pattern, backend="piquasso", seed=0), warmup=warmup, repeats=repeats
        )

        rows.append(
            {
                "modes": n_modes,
                "gates": 3 * n_modes - 1,
                "raw_piquasso_median_seconds": raw_timing["median_seconds"],
                "raw_piquasso_iqr_seconds": raw_timing["iqr_seconds"],
                "photographiq_median_seconds": pg_timing["median_seconds"],
                "photographiq_iqr_seconds": pg_timing["iqr_seconds"],
                "overhead_ratio": pg_timing["median_seconds"] / raw_timing["median_seconds"],
                **{f"{b}_median_seconds": extra_timing[b]["median_seconds"] for b in EXTRA_BACKENDS},
                **{f"{b}_ratio_to_raw_piquasso": extra_timing[b]["median_seconds"] / raw_timing["median_seconds"] for b in EXTRA_BACKENDS},
            }
        )
        print(f"  modes={n_modes}: raw={raw_timing['median_seconds']:.5f}s pg={pg_timing['median_seconds']:.5f}s ratio={rows[-1]['overhead_ratio']:.2f}", flush=True)

    common.save_result(rows, "R38_piquasso_overhead", extra={
        "environment": metadata.collect(), "timing_environment": timing_environment(),
        "correctness_tolerance": CORRECTNESS_TOLERANCE})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    modes = [r["modes"] for r in rows]
    axes[0].loglog(modes, [r["raw_piquasso_median_seconds"] for r in rows], "o-", label="Raw Piquasso (1 program)", color="#24677b")
    axes[0].loglog(modes, [r["photographiq_median_seconds"] for r in rows], "s-", label="PhotoGraphiQ (piquasso backend)", color="#c66d27")
    axes[0].loglog(modes, [r["gaussian_median_seconds"] for r in rows], "^-", label="PhotoGraphiQ (gaussian, NumPy)", color="#627a36")
    axes[0].loglog(modes, [r["native-gaussian_median_seconds"] for r in rows], "D-", label="PhotoGraphiQ (native-gaussian)", color="#7a3e9d")
    axes[0].set_xlabel("Modes")
    axes[0].set_ylabel("Median runtime (s)")
    axes[0].legend(fontsize=7)

    axes[1].plot(modes, [r["overhead_ratio"] for r in rows], "s-", color="#c66d27", label="piquasso backend")
    axes[1].plot(modes, [r["gaussian_ratio_to_raw_piquasso"] for r in rows], "^-", color="#627a36", label="gaussian")
    axes[1].plot(modes, [r["native-gaussian_ratio_to_raw_piquasso"] for r in rows], "D-", color="#7a3e9d", label="native-gaussian")
    axes[1].axhline(1.0, color="grey", linestyle=":")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("Modes")
    axes[1].set_ylabel(r"Overhead ratio $T_{PG}/T_{PQ}$")
    fig.suptitle(f"PhotoGraphiQ engines vs. one raw Piquasso program -- timings {timing_environment()['timing_status']}")
    common.save_figure(fig, "R38_piquasso_overhead")
    plt.close(fig)
    env = timing_environment()
    last = rows[-1]
    common.save_caption(
        "R38_piquasso_overhead",
        f"""
**R38: PhotoGraphiQ Gaussian engines versus raw Piquasso.** Left: median wall time of an N-mode Gaussian workload
(per-mode squeeze and rotation, nearest-neighbour CZ chain; N = {MODE_COUNTS}) as one raw Piquasso program with a single
`GaussianSimulator` call, and through `photographiq.simulate` on the `piquasso` backend (one Piquasso program per gate),
NumPy `gaussian` and C++ `native-gaussian`; right: ratio to raw Piquasso (dotted line: parity). Every engine first
reproduced the raw Piquasso covariance to {CORRECTNESS_TOLERANCE:g}. At N = {last['modes']}: raw Piquasso
{last['raw_piquasso_median_seconds']:.3g} s, piquasso backend {last['photographiq_median_seconds']:.3g} s (x{last['overhead_ratio']:.2f}),
gaussian {last['gaussian_median_seconds']:.3g} s (x{last['gaussian_ratio_to_raw_piquasso']:.2f}), native-gaussian
{last['native-gaussian_median_seconds']:.3g} s (x{last['native-gaussian_ratio_to_raw_piquasso']:.2f}). Timings are
{env['timing_status']} ({env['configuration']} configuration, {env['os_cpu_count']} logical CPUs, native build
{env['native_build_info'].get('compiler')}, OpenMP {env['native_build_info'].get('openmp')}); local observations, not cross-machine
claims. Source: `paper_experiments/11_performance/38_piquasso_overhead.py`.
""",
    )

    common.print_summary("R38 PhotoGraphiQ overhead vs. raw Piquasso", mode_counts=MODE_COUNTS)


if __name__ == "__main__":
    main()
