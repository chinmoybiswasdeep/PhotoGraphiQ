"""R13: Matched-workload runtime and scaling observations for PhotoGraphiQML
and raw MentPy.

Scientific question: How does wall-clock execution time for one logical
MuTA evaluation scale with wire count for PhotoGraphiQML's frontier-
translation execute() versus an independently constructed, matched-workload
raw MentPy PatternSimulator run, on this local machine?

Theory/equations: none (descriptive performance measurement only).

Functionality tested: MuTA.run (photographiqml) vs. mp.PatternSimulator.run
(mentpy), matched at the same semantic circuit/angles via
validation.mentpy_reference.

Oracle and independence class: N/A (descriptive performance measurement, no
correctness oracle -- MentPy is used here as a comparison workload, not as a
ground truth; correctness agreement is R7-R12's job).

Exact/approximate/statistical status: statistical (repeated timings with
warmup; median/IQR/min/max reported, not a single measurement).

Primary metric: median wall-clock seconds per evaluation, vs. n_wires, for
each implementation; also graph vertices/edges/parameters (structural
resource counts feeding the suite's aggregate performance figure, R_PERF in
12_performance/).

Declared acceptance condition: none (N/A oracle class; this experiment
always "passes" once it completes and produces finite, positive timings --
there is no correctness threshold to fail).

Expected cost: light-to-moderate (up to 3 wires; 5 warmups + 11 repeats per
point).

Manuscript destination: Appendix (performance panel; also feeds the final
aggregate performance figure in 12_performance/).

Scientific limitations: Single-machine, single-process timings (this
Windows host only); not a comparison of algorithmic complexity classes, and
not a claim that either implementation is "faster" in general -- MentPy's
PatternSimulator was not designed or tuned for this specific workload shape,
and neither was PhotoGraphiQML tuned against it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.models import haar_states
from photographiqml.validation import mentpy_reference

EXPERIMENT_ID = "R13"


def build_mentpy_runner(model, state, angles):
    """PatternSimulator instances are single-use (internal measurement state
    is consumed by run()), so a fresh instance is built inside the returned
    closure -- matching how a workload of "one full evaluation" is actually
    used, since MuTA.run also rebinds/traverses fresh per call."""
    import mentpy as mp

    reference, mapping = mentpy_reference(model, fix_measurements=True)
    values = [angles[model.parameter_name(mapping[v])] for v in reference.trainable_nodes]
    reverse = {v: k for k, v in mapping.items()}
    schedule = [reverse[v] for v in model.measurement_order + model.output_nodes]
    positions = {v: i for i, v in enumerate(schedule)}
    window = max(abs(positions[u] - positions[v]) + 1 for u, v in reference.graph.edges)
    window = max(window, model.n_wires + 1)
    state_array = np.asarray(state, complex)

    def run_once():
        simulator = mp.PatternSimulator(
            reference,
            input_state=state_array,
            backend="numpy-sv",
            schedule=schedule,
            window_size=window,
        )
        return simulator.run(values, output_form="dm")

    return run_once


def main():
    plt = common.setup_style()
    rows = []
    for n_wires in (1, 2, 3):
        model = MuTA(n_wires, 1, one_column=True)
        state = haar_states(n_wires, 1, seed=n_wires)[0]
        angles = dict(
            zip(
                model.trainable_parameters(),
                common.rng(n_wires).uniform(-np.pi, np.pi, model.n_parameters),
            )
        )
        pqml_bench = common.benchmark(lambda: model.run(state, angles), warmup=5, repeats=11)
        mentpy_runner = build_mentpy_runner(model, state, angles)
        mentpy_bench = common.benchmark(mentpy_runner, warmup=5, repeats=11)
        rows.append(
            {
                "n_wires": n_wires,
                "graph_vertices": len(model.graph),
                "graph_edges": model.graph.number_of_edges(),
                "n_parameters": model.n_parameters,
                "photographiqml_median_seconds": pqml_bench["median_seconds"],
                "photographiqml_iqr_seconds": pqml_bench["iqr_seconds"],
                "photographiqml_min_seconds": pqml_bench["min_seconds"],
                "photographiqml_max_seconds": pqml_bench["max_seconds"],
                "mentpy_median_seconds": mentpy_bench["median_seconds"],
                "mentpy_iqr_seconds": mentpy_bench["iqr_seconds"],
                "mentpy_min_seconds": mentpy_bench["min_seconds"],
                "mentpy_max_seconds": mentpy_bench["max_seconds"],
            }
        )

    common.save_result(
        rows,
        "R13_runtime_scaling",
        extra={
            "protocol": "Matched-workload PhotoGraphiQML.run vs. raw MentPy PatternSimulator.run, repeated timings",
            "oracle_class": "N/A",
            "status_category": "statistical",
            "acceptance_condition": "N/A (descriptive performance measurement)",
            "status": "pass",
            "machine_note": "Single Windows host, single process; local observation only, not a general performance claim",
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "N/A", "status": "pass"},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    wires = [r["n_wires"] for r in rows]
    axes[0].errorbar(
        wires,
        [r["photographiqml_median_seconds"] for r in rows],
        yerr=[r["photographiqml_iqr_seconds"] / 2 for r in rows],
        marker="o",
        color=common.COLORS["photographiqml"],
        label="PhotoGraphiQML",
    )
    axes[0].errorbar(
        wires,
        [r["mentpy_median_seconds"] for r in rows],
        yerr=[r["mentpy_iqr_seconds"] / 2 for r in rows],
        marker="s",
        color=common.COLORS["mentpy"],
        label="MentPy",
    )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Matched-workload runtime (median +/- IQR/2)", xlabel="n_wires", ylabel="seconds"
    )
    axes[0].set_xticks(wires)
    axes[0].legend()
    axes[1].plot(
        wires,
        [r["graph_vertices"] for r in rows],
        "o-",
        color=common.COLORS["analytic"],
        label="graph vertices",
    )
    axes[1].plot(
        wires,
        [r["graph_edges"] for r in rows],
        "s--",
        color=common.COLORS["classical_baseline"],
        label="graph edges",
    )
    axes[1].plot(
        wires,
        [r["n_parameters"] for r in rows],
        "^:",
        color=common.COLORS["piquasso"],
        label="parameters",
    )
    axes[1].set(title="Structural resource scaling", xlabel="n_wires", ylabel="count")
    axes[1].set_xticks(wires)
    axes[1].legend(fontsize=7)
    fig.suptitle("R13: matched-workload runtime and resource scaling (N/A oracle; descriptive)")
    common.save_figure(fig, "R13_runtime_scaling")
    plt.close(fig)

    common.print_summary("R13 runtime scaling", n_configs=len(rows), status="pass")


if __name__ == "__main__":
    main()
