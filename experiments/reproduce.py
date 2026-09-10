"""Reproduce paper figures/data without modifying source or fetching resources."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg
from photographiq.visualization import draw_dependencies, draw_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "paper" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, r in enumerate([0.3, 0.6, 0.9, 1.2, 1.5]):
        pattern = pg.protocols.identity(squeezing=r)
        channel = pg.gaussian_channel(pattern)
        expected = channel.apply(pg.GaussianInput().state(0))
        start = time.perf_counter()
        ensemble = pg.run_shots(
            pattern, args.shots, backend="gaussian", seed=20260910 + index
        ).ensemble_state()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "r": r,
                "squeezing_db": 20 * r / np.log(10),
                "predicted_variance": float(expected.covariance[0, 0]),
                "observed_q_variance": float(ensemble.covariance[0, 0]),
                "observed_p_variance": float(ensemble.covariance[1, 1]),
                "max_covariance_error": float(
                    np.max(np.abs(ensemble.covariance - expected.covariance))
                ),
                "coherent_fidelity": float(1 / (1 + np.exp(-2 * r))),
                "shots": args.shots,
                "seconds": elapsed,
            }
        )
    with (directory / "finite_squeezing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, fidelity) = plt.subplots(1, 2, figsize=(9, 3.4))
    rs = np.linspace(0.2, 1.6, 150)
    ax.plot(rs, 1 + 2 * np.exp(-2 * rs), label="Exact ensemble variance", color="#24677b")
    ax.scatter(
        [r["r"] for r in rows],
        [r["observed_q_variance"] for r in rows],
        label="Sampled q variance",
        color="#c66d27",
    )
    ax.scatter(
        [r["r"] for r in rows],
        [r["observed_p_variance"] for r in rows],
        label="Sampled p variance",
        marker="x",
        color="#627a36",
    )
    ax.set(xlabel="Resource squeezing r", ylabel="Output variance (vacuum = 1)")
    ax.legend(fontsize=8)
    fidelity.plot(rs, 1 / (1 + np.exp(-2 * rs)), color="#24677b")
    fidelity.set(xlabel="Resource squeezing r", ylabel="Coherent-state fidelity", ylim=(0.5, 1))
    fig.tight_layout()
    fig.savefig(directory / "finite_squeezing.pdf")
    fig.savefig(directory / "finite_squeezing.png", dpi=160)
    plt.close(fig)
    graph = pg.CVGraph.rectangular(
        2, 3, squeezing=1.0, inputs=((0, 0), (1, 0)), outputs=((0, 2), (1, 2))
    )
    ax = draw_graph(graph, positions={n: (n[1], -n[0]) for n in graph.nodes})
    ax.figure.set_size_inches(4.6, 2.8)
    ax.figure.savefig(directory / "resource.pdf", bbox_inches="tight")
    plt.close(ax.figure)
    ax = draw_dependencies(pg.protocols.wire([0, 0.3], squeezing=1.0))
    ax.figure.savefig(directory / "dependencies.pdf", bbox_inches="tight")
    plt.close(ax.figure)
    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "piquasso", "graphix", "networkx", "matplotlib"]
        },
        "seed_base": 20260910,
        "rows": rows,
    }
    (directory / "environment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = "\n".join(
        f"{r['r']:.1f} & {r['predicted_variance']:.4f} & {r['observed_q_variance']:.4f} & {r['observed_p_variance']:.4f} & {r['coherent_fidelity']:.4f} \\\\"
        for r in rows
    )
    (directory / "noise_table.tex").write_text(table + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(directory),
                "shots_per_point": args.shots,
                "max_covariance_error": max(r["max_covariance_error"] for r in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
