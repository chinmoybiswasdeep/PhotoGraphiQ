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
            }
        )
        print(f"  modes={n_modes}: raw={raw_timing['median_seconds']:.5f}s pg={pg_timing['median_seconds']:.5f}s ratio={rows[-1]['overhead_ratio']:.2f}", flush=True)

    common.save_result(rows, "R38_piquasso_overhead", extra={"environment": metadata.collect()})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    modes = [r["modes"] for r in rows]
    axes[0].loglog(modes, [r["raw_piquasso_median_seconds"] for r in rows], "o-", label="Raw Piquasso (1 program)", color="#24677b")
    axes[0].loglog(modes, [r["photographiq_median_seconds"] for r in rows], "s-", label="PhotoGraphiQ (piquasso backend)", color="#c66d27")
    axes[0].set_xlabel("Modes")
    axes[0].set_ylabel("Median runtime (s)")
    axes[0].legend(fontsize=7)

    axes[1].plot(modes, [r["overhead_ratio"] for r in rows], "o-", color="#627a36")
    axes[1].set_xlabel("Modes")
    axes[1].set_ylabel(r"Overhead ratio $T_{PG}/T_{PQ}$")
    fig.suptitle("PhotoGraphiQ overhead over raw Piquasso (this machine only)")
    common.save_figure(fig, "R38_piquasso_overhead")
    plt.close(fig)

    common.print_summary("R38 PhotoGraphiQ overhead vs. raw Piquasso", mode_counts=MODE_COUNTS)


if __name__ == "__main__":
    main()
