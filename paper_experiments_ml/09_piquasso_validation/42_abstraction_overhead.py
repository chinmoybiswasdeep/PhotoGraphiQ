"""R42: Abstraction overhead for matched physical workloads: PhotoGraphiQML,
direct PhotoGraphiQ, and raw Piquasso.

Scientific question: How much wall-clock overhead does each layer of
abstraction add for the identical one-wire signed-X physical-conditional
workload -- PhysicalMuTA.run's full pipeline (audit + lowering + simulate +
decode), calling pg.simulate directly on an already-lowered Pattern
(construction + simulate only, no decode bookkeeping), and the fully
independent raw Piquasso Pattern built in R38/R41 (independent
construction + simulate)?

Theory/equations: none (descriptive performance measurement).

Functionality tested: PhysicalMuTA.run vs. lower_muta_to_gkp+pg.simulate
(direct) vs. the R38/R41-style independently assembled Pattern, timed
separately for construction/lowering and simulation phases.

Oracle and independence class: N/A (descriptive performance measurement, no
correctness oracle -- correctness of all three paths is already established
by R38/R41; this experiment only measures their relative cost).

Exact/approximate/statistical status: statistical (repeated timings with
warmup; median/IQR/min/max reported).

Primary metric: median wall-clock seconds for each phase (construction/
lowering, simulation, decoding+readout) and pipeline, plus wrapper overhead
= PhysicalMuTA.run total - (lowering + direct simulate) for the matched
workload.

Declared acceptance condition: none (N/A oracle class; always "passes" once
all timings are finite and positive).

Expected cost: moderate (repeated one-wire Fock simulations, cutoff=24).

Manuscript destination: Appendix (performance panel feeding the final
aggregate performance figure in 12_performance/).

Scientific limitations: Single-machine, single-process timings on this
Windows host; not a claim about relative performance on other hardware or
at other resource scales (see R13, R36 for companion performance panels).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import photographiq as pg

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA
from photographiqml.lowering import lower_muta_to_gkp

EXPERIMENT_ID = "R42"


def build_independent_pattern_and_run(config, seed=11):
    code = pg.GKPCode(
        cutoff=config.cutoff,
        peak_width=config.peak_width,
        envelope=config.envelope,
        peaks=config.peaks,
        grid_points=config.grid_points,
    )
    decoder = pg.NearestCellDecoder()
    readout = code.logical_measurement("XY", alpha=0.0, decoder=decoder)
    plus = code.plus()
    input_node, columns = (0, 0), [(0, c) for c in range(5)]
    pattern = pg.Pattern(inputs=[input_node])

    def make_interpreted(node, z_source):
        def interpreted(records, node=node, z_source=z_source):
            raw_bit = int(records[node].bit)
            z_correction = int(records[("bit", z_source)]) if z_source is not None else 0
            return raw_bit ^ z_correction

        declared = frozenset({node} | ({("bit", z_source)} if z_source is not None else set()))
        return interpreted, declared

    for c in range(4):
        pattern.append(pg.Prepare(columns[c + 1], state=plus))
        pattern.append(code.logical_cz(columns[c], columns[c + 1]))
        pattern.append(pg.Measure(columns[c], readout, columns[c]))
        z_source = columns[c - 2] if c - 2 >= 0 else None
        fn, deps = make_interpreted(columns[c], z_source)
        pattern.append(pg.Signal(("bit", columns[c]), pg.CallableExpression(fn, deps)))
    pattern.append(pg.Output((columns[4],)))
    pattern.validate()
    encoded_input = code.encode(1, 0)
    outcomes = {columns[c]: 0.0 for c in range(4)}
    return pg.simulate(
        pattern,
        seed=seed,
        inputs={input_node: encoded_input},
        backend=config.backend,
        cutoff=config.cutoff,
        measurement_outcomes=outcomes,
    )


def main():
    plt = common.setup_style()
    config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    model = MuTA(1, 1, one_column=True)
    physical_model = PhysicalMuTA(1, physical_config=config)
    outcomes = dict.fromkeys(model.measurement_order, 0.0)

    full_run = common.benchmark(
        lambda: physical_model.run([1, 0], mode="physical-conditional", analog_outcomes=outcomes),
        warmup=1,
        repeats=5,
    )
    lowering_only = common.benchmark(
        lambda: lower_muta_to_gkp(model, config=config), warmup=1, repeats=5
    )
    lowered = lower_muta_to_gkp(model, config=config)
    encoded_input = lowered.code.encode(1, 0)
    direct_simulate = common.benchmark(
        lambda: pg.simulate(
            lowered.pattern,
            seed=11,
            inputs={model.input_nodes[0]: encoded_input},
            backend=config.backend,
            cutoff=config.cutoff,
            measurement_outcomes={
                lowered.measurement_keys[n]: 0.0 for n in model.measurement_order
            },
        ),
        warmup=1,
        repeats=5,
    )
    raw_independent = common.benchmark(
        lambda: build_independent_pattern_and_run(config), warmup=1, repeats=5
    )

    pipelines = {
        "photographiqml_full_run": full_run,
        "photographiqml_lowering_only": lowering_only,
        "photographiq_direct_simulate": direct_simulate,
        "raw_piquasso_independent_construct_and_simulate": raw_independent,
    }
    rows = [
        {
            "pipeline": name,
            "median_seconds": b["median_seconds"],
            "iqr_seconds": b["iqr_seconds"],
            "min_seconds": b["min_seconds"],
            "max_seconds": b["max_seconds"],
        }
        for name, b in pipelines.items()
    ]
    wrapper_overhead = full_run["median_seconds"] - (
        lowering_only["median_seconds"] + direct_simulate["median_seconds"]
    )
    decoded_result = full_run["result"]
    last_diagnostics = {
        "reported_simulation_seconds": decoded_result.diagnostics["simulation_seconds"],
        "reported_readout_seconds": decoded_result.diagnostics["readout_seconds"],
    }

    all_finite = all(r["median_seconds"] > 0 for r in rows)
    status = "pass" if all_finite else "fail"

    common.save_result(
        rows,
        "R42_abstraction_overhead",
        extra={
            "protocol": "Matched one-wire physical-conditional workload timed across PhotoGraphiQML/direct PhotoGraphiQ/raw Piquasso",
            "oracle_class": "N/A",
            "status_category": "statistical",
            "wrapper_overhead_seconds": wrapper_overhead,
            "internal_diagnostics_breakdown": last_diagnostics,
            "acceptance_condition": "N/A (descriptive performance measurement); all timings finite and positive",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "N/A", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [r["pipeline"] for r in rows]
    ax.bar(
        range(len(rows)),
        [r["median_seconds"] for r in rows],
        yerr=[r["iqr_seconds"] / 2 for r in rows],
        color=common.COLORS["photographiqml"],
        capsize=4,
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set(
        title=f"R42: abstraction overhead, matched 1-wire workload (wrapper overhead={wrapper_overhead:.3f}s)",
        ylabel="median seconds",
    )
    common.save_figure(fig, "R42_abstraction_overhead")
    plt.close(fig)

    common.print_summary(
        "R42 abstraction overhead", wrapper_overhead_seconds=wrapper_overhead, status=status
    )
    if status != "pass":
        raise AssertionError("R42 failed: some pipeline reported a non-positive median timing")


if __name__ == "__main__":
    main()
