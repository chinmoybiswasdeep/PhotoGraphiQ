"""Regenerate the small SVG gallery from executable PhotoGraphiQ experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg
from photographiq.visualization import draw_dependencies, draw_graph


def main():
    output = Path(__file__).resolve().parents[1] / "docs/assets"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "photographiq"
    circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.15)
    pattern = circuit.compile(squeezing=0.8)
    fig = pg.visualize_compilation(circuit, pattern, layout="grid")
    fig.savefig(output / "circuit-to-mbqc.svg", bbox_inches="tight", metadata={"Date": None})
    circuit.draw().figure.savefig(
        output / "circuit.svg", bbox_inches="tight", metadata={"Date": None}
    )
    draw_graph(pg.CVGraph.square(3, squeezing=0.7), layout="grid").figure.savefig(
        output / "cluster.svg", bbox_inches="tight", metadata={"Date": None}
    )
    draw_dependencies(pg.protocols.adaptive()).figure.savefig(
        output / "dependencies.svg", bbox_inches="tight", metadata={"Date": None}
    )
    state = pg.simulate(
        pg.Pattern(inputs=(0,)),
        inputs={0: pg.FockInput.number(1)},
        backend="piquasso-fock",
        cutoff=8,
    ).state
    state.wigner(np.linspace(-5, 5, 61), np.linspace(-5, 5, 61)).plot()[0].figure.savefig(
        output / "wigner.svg", bbox_inches="tight", metadata={"Date": None}
    )
    fig, ax = plt.subplots()
    rs = np.linspace(0.2, 1.4, 15)
    ax.plot(
        rs,
        [
            np.trace(pg.gaussian_channel(pg.protocols.identity(squeezing=float(r))).noise)
            for r in rs
        ],
    )
    ax.set(
        xlabel="Resource squeezing r",
        ylabel="Added covariance trace",
        title="Finite-squeezing wire noise",
    )
    fig.savefig(output / "finite-squeezing.svg", bbox_inches="tight", metadata={"Date": None})
    study = pg.cutoff_convergence(
        pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.05)),
        [12, 18, 24, 32],
        high_order_moments=True,
    )
    fig, ax = plt.subplots()
    ax.plot(
        [r["cutoff"] for r in study.rows],
        [r["high_order_moments"][0]["p4"] for r in study.rows],
        marker="o",
    )
    ax.set(
        xlabel="Total-photon cutoff",
        ylabel="Fourth momentum moment",
        title="Cubic-phase convergence",
    )
    fig.savefig(output / "cutoff-convergence.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close("all")
    print("Generated seven SVG figures in", output)


if __name__ == "__main__":
    main()
