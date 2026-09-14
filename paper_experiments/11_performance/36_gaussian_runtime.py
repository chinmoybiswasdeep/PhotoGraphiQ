"""R36: Gaussian runtime scaling.

Benchmarks preparing and entangling an N-mode line resource
(``CVGraph.line(N)``) through the NumPy Gaussian backend and the Piquasso
Gaussian backend, with warm-up runs and repeated timing measurements
(median, IQR, std). Timing is a local observation of this machine, not a
comparative performance claim (docs/performance.md, docs/validation.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import metadata  # noqa: E402

import photographiq as pg  # noqa: E402

MODE_COUNTS = [10, 20, 50, 100, 200]


def build_pattern(n_modes: int) -> pg.Pattern:
    graph = pg.CVGraph.line(n_modes, squeezing=1.0, inputs=(0,))
    return pg.Pattern(graph)


def main():
    plt = common.setup_style()
    rows = []
    for n_modes in MODE_COUNTS:
        pattern = build_pattern(n_modes)
        for backend in ("gaussian", "piquasso"):
            warmup, repeats = (2, 9) if n_modes <= 50 else (1, 5)
            timing = common.benchmark(
                lambda pattern=pattern, backend=backend: pg.simulate(pattern, backend=backend, seed=0),
                warmup=warmup, repeats=repeats,
            )
            rows.append(
                {
                    "modes": n_modes,
                    "commands": len(pattern.commands),
                    "backend": backend,
                    "median_seconds": timing["median_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "iqr_seconds": timing["iqr_seconds"],
                    "min_seconds": timing["min_seconds"],
                    "max_seconds": timing["max_seconds"],
                    "repeats": timing["repeats"],
                    "warmup": timing["warmup"],
                }
            )
            print(f"  modes={n_modes} backend={backend}: median={timing['median_seconds']:.5f}s", flush=True)

    common.save_result(rows, "R36_gaussian_runtime", extra={"environment": metadata.collect()})

    fig, ax = plt.subplots(figsize=(6, 4))
    for backend, color in (("gaussian", "#24677b"), ("piquasso", "#c66d27")):
        subset = [r for r in rows if r["backend"] == backend]
        ax.errorbar(
            [r["modes"] for r in subset], [r["median_seconds"] for r in subset],
            yerr=[r["iqr_seconds"] / 2 for r in subset], fmt="o-", label=backend, color=color, capsize=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of modes (line resource)")
    ax.set_ylabel("Median runtime (s)")
    ax.legend(fontsize=8)
    ax.set_title("Gaussian backend runtime scaling (this machine only)")
    common.save_figure(fig, "R36_gaussian_runtime")
    plt.close(fig)

    common.print_summary("R36 Gaussian runtime scaling", mode_counts=MODE_COUNTS)


if __name__ == "__main__":
    main()
