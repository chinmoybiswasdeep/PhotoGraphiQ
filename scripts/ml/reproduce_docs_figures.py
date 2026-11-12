"""Generate graph, entanglement, resource and scaling data from executable models."""

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photographiqml import GKPBridge, MuTA, TriangleNeuron
from photographiqml.diagnostics import concurrence
from photographiqml.models import haar_states

# Repository-root anchored, so running from any cwd only touches the PhotoGraphiQML trees.
ROOT = Path(__file__).resolve().parents[2]


def main():
    destination = ROOT / "docs" / "ml" / "assets"
    destination.mkdir(parents=True, exist_ok=True)
    for name, model in (("triangle", TriangleNeuron()), ("muta", MuTA(2))):
        ax = model.draw()
        ax.figure.savefig(destination / f"{name}.svg", bbox_inches="tight")
        plt.close(ax.figure)
    model = MuTA(2, one_column=True)
    angles = np.linspace(0, np.pi, 101)
    values = [concurrence(model.run([1, 0, 0, 0], {"alpha.w1.c1": a}).state) for a in angles]
    fig, ax = plt.subplots()
    ax.plot(angles, values)
    ax.set(xlabel="Logical base-center angle", ylabel="Pure-state concurrence")
    fig.savefig(destination / "entanglement.svg", bbox_inches="tight")
    plt.close(fig)
    resources = [GKPBridge(cutoff=c).diagnostics() for c in (8, 16, 24, 32, 48)]
    fig, ax = plt.subplots()
    for bit in (0, 1):
        ax.plot(
            [r["cutoff"] for r in resources],
            [r["captured_weights"][bit] for r in resources],
            label=f"logical basis {bit}",
        )
    ax.set(
        xlabel="One-mode Fock cutoff",
        ylabel="Captured projection weight",
        title="Resource diagnostic; not MuTA convergence",
    )
    ax.legend()
    fig.savefig(destination / "gkp_resource.svg", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for wires in (1, 2, 3, 4, 6):
        for layers in (1, 2, 4):
            model = MuTA(wires, layers)
            state = haar_states(wires, 1, 42)[0]
            start = perf_counter()
            for _ in range(3):
                model.run(state)
            rows.append(
                {
                    "wires": wires,
                    "layers": layers,
                    "parameters": model.n_parameters,
                    "nodes": len(model.graph),
                    "seconds_per_run": (perf_counter() - start) / 3,
                    "statevector_bytes": state.nbytes,
                }
            )
    (destination / "resource_and_scaling.json").write_text(
        json.dumps({"gkp_resources": resources, "logical_scaling": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
