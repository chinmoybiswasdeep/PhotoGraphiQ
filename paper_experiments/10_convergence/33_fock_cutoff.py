"""R33: Fock cutoff convergence.

Runs representative non-Gaussian workloads (cat state, photon subtraction,
cubic phase, GKP resource) at strictly increasing Fock cutoffs using
``photographiq.convergence.cutoff_convergence``, and reports retained norm,
boundary population, photon number, adjacent-cutoff fidelity/trace distance
(from the study itself) plus fidelity to the HIGHEST tested cutoff (computed
separately here). The highest tested cutoff is a reference point, not an
exact/infinite-cutoff ground truth -- convergence must still be judged from
the trend, not assumed from the last value's smallness alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import photographiq as pg  # noqa: E402

WORKLOADS = {
    "cat_state": dict(
        pattern=pg.Pattern(inputs=()).append(pg.Prepare(0, state=pg.CatResource(1.5, parity=1))).append(pg.Output((0,))),
        cutoffs=[16, 24, 32, 48, 64],
        node=0,
        kwargs={},
    ),
    "photon_subtraction": dict(
        pattern=pg.non_gaussian.photon_subtraction(0.3),
        cutoffs=[8, 12, 16, 24, 32],
        node="in",
        kwargs={"inputs": {"in": pg.FockInput.number(2)}, "measurement_outcomes": {"count": 1}},
    ),
    "cubic_phase": dict(
        pattern=pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.15)).append(pg.Output((0,))),
        cutoffs=[16, 24, 32, 48, 64],
        node=0,
        kwargs={"inputs": {0: pg.GaussianInput.coherent(0.3 + 0.0j)}},
    ),
    "gkp_resource": dict(
        pattern=pg.Pattern(inputs=()).append(pg.Prepare(0, state=pg.GKPResource(0, peak_width=0.4, envelope=0.4))).append(pg.Output((0,))),
        cutoffs=[24, 32, 48, 64, 96],
        node=0,
        kwargs={},
    ),
}


def main():
    plt = common.setup_style()
    all_rows = {}
    for name, spec in WORKLOADS.items():
        study = pg.cutoff_convergence(spec["pattern"], spec["cutoffs"], **spec["kwargs"])
        final_state = study.results[-1].state
        rows = []
        for row, result in zip(study.rows, study.results, strict=True):
            fidelity_to_highest = pg.fidelity(result.state, final_state) if row["cutoff"] != spec["cutoffs"][-1] else 1.0
            trace_distance_to_highest = pg.trace_distance(result.state, final_state) if row["cutoff"] != spec["cutoffs"][-1] else 0.0
            rows.append(
                {
                    "workload": name,
                    "cutoff": row["cutoff"],
                    "dimension": common.fock_dimension(len(result.state.nodes), row["cutoff"]),
                    "seconds": row["seconds"],
                    "norm": row["norm"],
                    "minimum_retained_norm": row["minimum_retained_norm"],
                    "maximum_boundary_population": row["maximum_boundary_population"],
                    "photon_number": row["photon_numbers"].get(spec["node"]),
                    "fidelity_to_previous": row["fidelity_to_previous"],
                    "trace_distance_to_previous": row["trace_distance_to_previous"],
                    "fidelity_to_highest_cutoff": fidelity_to_highest,
                    "trace_distance_to_highest_cutoff": trace_distance_to_highest,
                }
            )
        all_rows[name] = rows
        print(f"  {name}: " + ", ".join(f"c={r['cutoff']}:F={r['fidelity_to_highest_cutoff']:.6f}" for r in rows), flush=True)

    flat_rows = [row for rows in all_rows.values() for row in rows]
    common.save_result(flat_rows, "R33_fock_cutoff", extra={"workloads": list(WORKLOADS)})

    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(4.2 * len(WORKLOADS), 3.6))
    for ax, (name, rows) in zip(axes, all_rows.items(), strict=True):
        infidelities = [max(1e-16, 1 - r["fidelity_to_highest_cutoff"]) for r in rows[:-1]]
        ax.semilogy([r["cutoff"] for r in rows[:-1]], infidelities, "o-", color="#24677b")
        ax.set_xlabel("Fock cutoff")
        ax.set_title(name)
    axes[0].set_ylabel(f"1 - fidelity to cutoff={list(WORKLOADS.values())[0]['cutoffs'][-1]}")
    fig.suptitle("Fock cutoff convergence (highest cutoff is a reference, not exact truth)")
    common.save_figure(fig, "R33_fock_cutoff")
    plt.close(fig)

    common.print_summary(
        "R33 Fock cutoff convergence",
        workloads=list(WORKLOADS),
        min_final_fidelity=min(rows[-2]["fidelity_to_highest_cutoff"] for rows in all_rows.values() if len(rows) > 1),
    )


if __name__ == "__main__":
    main()
