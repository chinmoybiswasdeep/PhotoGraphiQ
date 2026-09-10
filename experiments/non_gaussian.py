"""Reproduce non-Gaussian manuscript figures, convergence data and local timings."""

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg


def main():
    directory = Path(__file__).resolve().parents[1] / "paper" / "results"
    directory.mkdir(exist_ok=True)
    studies = {
        "cubic_gate": pg.cutoff_convergence(
            pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, 0.65)]), [12, 24, 48, 80]
        ),
        "cubic_injection": pg.cutoff_convergence(
            pg.non_gaussian.cubic_injection(0.3, 0.2), [36, 48, 64], measurement_outcomes={"m": 0.4}
        ),
        "cat": pg.cutoff_convergence(
            pg.Pattern().append(pg.Prepare(0, state=pg.CatResource(1.5))), [12, 16, 24]
        ),
    }
    fields = [
        "experiment",
        "cutoff",
        "seconds",
        "norm",
        "minimum_retained_norm",
        "maximum_boundary_population",
        "peak_dimension",
        "fidelity_to_previous",
        "trace_distance_to_previous",
        "probability_l1_to_previous",
        "photon_number",
    ]
    with (directory / "non_gaussian.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, study in studies.items():
            for row in study.rows:
                data = {key: row[key] for key in fields if key in row}
                data.update(
                    experiment=name, photon_number=next(iter(row["photon_numbers"].values()))
                )
                writer.writerow(data)
    (directory / "non_gaussian_environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: version(name) for name in ("photographiq", "piquasso", "numpy", "scipy")
                },
                "seed": 0,
                "gate_gamma": 0.65,
                "injection_gamma": 0.3,
                "injection_squeezing": 0.2,
                "fixed_homodyne_outcome": 0.4,
                "timings": "Local wall-clock observations, including first-call overhead; not comparative benchmarks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for name in ("cubic_gate", "cubic_injection"):
        rows = studies[name].rows[1:]
        axes[0].semilogy(
            [r["cutoff"] for r in rows],
            [max(1e-15, 1 - r["fidelity_to_previous"]) for r in rows],
            "o-",
            label=name.replace("_", " "),
        )
        axes[1].plot(
            [r["cutoff"] for r in studies[name].rows],
            [r["minimum_retained_norm"] for r in studies[name].rows],
            "o-",
        )
    axes[0].set(xlabel="Total-photon cutoff", ylabel="1 - adjacent-cutoff fidelity")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Total-photon cutoff", ylabel="Minimum retained norm")
    axes[1].ticklabel_format(useOffset=False, axis="y")
    fig.tight_layout()
    fig.savefig(directory / "non_gaussian_convergence.pdf")
    plt.close(fig)
    axis = np.linspace(-6, 6, 161)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    odd = pg.simulate(
        pg.Pattern().append(pg.Prepare(0, state=pg.CatResource(1.2, -1))),
        backend="piquasso-fock",
        cutoff=24,
    ).state
    states = [odd, studies["cubic_injection"].results[-1].state]
    diagnostics = []
    for ax, state, title in zip(
        axes, states, ["Odd cat", "Conditional cubic injection"], strict=True
    ):
        grid = state.wigner(axis, axis)
        _, artist = grid.plot(ax)
        fig.colorbar(artist, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=10)
        diagnostics.append(
            {
                "state": title,
                "captured_mass": grid.captured_mass,
                "negative_volume": grid.negative_volume,
                "grid": [-6, 6, 161],
            }
        )
    fig.tight_layout()
    fig.savefig(directory / "non_gaussian_wigner.pdf")
    plt.close(fig)
    (directory / "non_gaussian_wigner.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Experiment & $c$ & $1-F$ & min. norm \\",
        r"\midrule",
    ]
    for name, study in studies.items():
        for row in study.rows[1:]:
            lines.append(
                f"{name.replace('_', ' ')} & {row['cutoff']} & "
                f"{1 - row['fidelity_to_previous']:.2e} & {row['minimum_retained_norm']:.6f} "
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (directory / "non_gaussian_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote non-Gaussian data, figures and environment to", directory)


if __name__ == "__main__":
    main()
