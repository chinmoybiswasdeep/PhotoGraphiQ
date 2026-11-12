"""Aggregate performance figure: resource and runtime scaling across the suite.

Scientific question: none directly -- this script does not run a new
experiment. It assembles one multi-panel summary figure from already-saved
JSON produced by R13 (matched-workload runtime/resource scaling), R36
(lowering preservation + Fock dimension), R42 (abstraction overhead), and
R47 (resource-axis convergence), as required by the task's performance
panels (section 6: "Include performance panels within R13, R36, R42, and
R47, and generate a final aggregate performance figure").

Functionality tested: none (pure reporting/aggregation of prior results).

Oracle and independence class: N/A (reporting only).

Exact/approximate/statistical status: n/a.

Primary metric: n/a -- this script reproduces the aggregate figure from
saved CSV/JSON without rerunning any simulation, satisfying the
reproduce-without-rerunning-expensive-simulations requirement.

Declared acceptance condition: all four source JSON files must exist and be
readable; if any is missing, that panel is drawn with an explicit "missing:
run R<N> first" placeholder rather than silently omitted.

Expected cost: negligible (JSON loading + plotting only).

Manuscript destination: Main text (Fig. 13, aggregate performance figure).

Scientific limitations: Aggregates local, single-machine timings already
reported by the source experiments; adds no new measurement of its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import common

EXPERIMENT_ID = "R_PERF"


def load(name):
    path = common.JSON_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def placeholder(ax, source_id):
    ax.text(
        0.5,
        0.5,
        f"missing: run {source_id} first",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=common.COLORS["unsupported"],
    )
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    plt = common.setup_style()
    r13 = load("R13_runtime_scaling")
    r36 = load("R36_lowering_preservation")
    r42 = load("R42_abstraction_overhead")
    r47 = load("R47_resource_axis_convergence")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    if r13:
        rows = r13["rows"]
        wires = [r["n_wires"] for r in rows]
        ax.errorbar(
            wires,
            [r["photographiqml_median_seconds"] for r in rows],
            yerr=[r["photographiqml_iqr_seconds"] / 2 for r in rows],
            marker="o",
            color=common.COLORS["photographiqml"],
            label="PhotoGraphiQML",
        )
        ax.errorbar(
            wires,
            [r["mentpy_median_seconds"] for r in rows],
            yerr=[r["mentpy_iqr_seconds"] / 2 for r in rows],
            marker="s",
            color=common.COLORS["mentpy"],
            label="MentPy",
        )
        ax.set_yscale("log")
        ax.set(
            title="(a) Matched-workload logical runtime vs. wires (R13)",
            xlabel="n_wires",
            ylabel="seconds",
        )
        ax.legend(fontsize=7)
        common.panel_label(ax, "a")
    else:
        placeholder(ax, "R13")

    ax = axes[0, 1]
    if r36:
        rows = r36["rows"]
        labels = [f"n{r['n_wires']}L{r['n_layers']}" for r in rows]
        ax.bar(
            range(len(rows)),
            [r["reported_fock_dimension"] for r in rows],
            color=common.COLORS["piquasso"],
        )
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set(title="(b) Total-photon Fock dimension D=C(K+m-1,m) (R36)", ylabel="dimension")
        common.panel_label(ax, "b")
    else:
        placeholder(ax, "R36")

    ax = axes[1, 0]
    if r42:
        rows = r42["rows"]
        ax.bar(
            range(len(rows)),
            [r["median_seconds"] for r in rows],
            yerr=[r["iqr_seconds"] / 2 for r in rows],
            color=common.COLORS["photographiqml"],
            capsize=4,
        )
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["pipeline"] for r in rows], rotation=25, ha="right", fontsize=6.5)
        ax.set(
            title=f"(c) Abstraction overhead, matched 1-wire workload (R42)\nwrapper overhead={r42['wrapper_overhead_seconds']:.3f}s",
            ylabel="median seconds",
        )
        common.panel_label(ax, "c")
    else:
        placeholder(ax, "R42")

    ax = axes[1, 1]
    if r47:
        rows = r47["rows"]
        # cutoff (~20-28) and grid_points (~513-2049) live on incompatible
        # x-scales; sharing one linear x-axis collapses the cutoff series
        # into an invisible sliver, so each gets its own x-axis (twiny),
        # sharing only the log-scale y-axis. The first point of each study
        # has max_probability_delta=None (no previous point yet) and is
        # skipped rather than plotted as a fake zero on a log axis.
        ax_cutoff, ax_grid = ax, ax.twiny()
        for axis, plot_ax, color in (
            ("cutoff", ax_cutoff, common.COLORS["photographiqml"]),
            ("grid_points", ax_grid, common.COLORS["mentpy"]),
        ):
            axis_rows = [
                r for r in rows if r["axis"] == axis and r["max_probability_delta"] is not None
            ]
            plot_ax.plot(
                [r["value"] for r in axis_rows],
                [r["max_probability_delta"] for r in axis_rows],
                "o-",
                color=color,
                label=axis,
            )
            plot_ax.set_xlabel(axis, color=color, fontsize=8)
            plot_ax.tick_params(axis="x", colors=color, labelsize=7)
        ax_cutoff.set_yscale("log")
        ax_cutoff.set(
            title="(d) Numerical-refinement convergence (R47)", ylabel="max probability delta"
        )
        handles = ax_cutoff.get_lines() + ax_grid.get_lines()
        ax_cutoff.legend(handles, [h.get_label() for h in handles], fontsize=7)
        common.panel_label(ax, "d")
    else:
        placeholder(ax, "R47")

    fig.suptitle(
        "Aggregate performance and resource scaling (reproduced from saved R13/R36/R42/R47 results)"
    )
    common.save_figure(fig, "R_PERF_aggregate_performance")
    plt.close(fig)

    sources_present = sum(x is not None for x in (r13, r36, r42, r47))
    common.save_result(
        [{"sources_present": sources_present, "sources_total": 4}],
        "R_PERF_aggregate_performance",
        extra={
            "protocol": "Aggregate R13/R36/R42/R47 performance figure from saved evidence",
            "oracle_class": "N/A",
            "status_category": "descriptive",
            "acceptance_condition": "all four source JSON files exist and are readable",
            "status": "pass" if sources_present == 4 else "fail",
            "claim_supported": "the figure faithfully aggregates its available sources",
            "claim_not_supported": "no cross-machine performance conclusion",
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "N/A"},
    )
    common.print_summary(
        "R_PERF aggregate performance figure", sources_present=sources_present, sources_total=4
    )


if __name__ == "__main__":
    main()
