"""R36: Lowering preservation of logical vertices, graph edges, I/O mapping,
measurement keys, schedule and the analytic Fock dimension.

Scientific question: Does lower_muta_to_gkp preserve MuTA's logical graph
exactly (identity node map, same input/output node sets, one measurement
key per measured node, one frame dependency entry per node consistent with
model.corrections), does the resulting Pattern validate, and does the
reported Hilbert dimension match the analytic total-photon combinatorial
formula comb(cutoff+peak-1, peak)?

Theory/equations: for cutoff K and m simultaneously live modes, the
exclusive total-photon Fock-space dimension is D = C(K+m-1, m)
(docs/physical/lowering.md); this experiment recomputes that binomial
coefficient independently (math.comb, common.fock_dimension) and compares
to the audit's own reported hilbert_dimension.

Functionality tested: photographiqml.lowering.lower_muta_to_gkp
(PhysicalLoweringResult: node_map, input_modes/output_modes,
measurement_keys, frame_dependencies, audit, pattern.validate()).

Oracle and independence class: A for the Fock-dimension formula
(independent combinatorial computation); E for the structural
preservation checks (self-consistency against the model's own public
graph/corrections attributes, which R1/R7/R8 already independently validate
as correct).

Exact/approximate/statistical status: exact.

Primary metric: node-map identity violations; input/output set mismatches;
measurement-key coverage mismatches; frame-dependency mismatches vs.
model.corrections; |Fock dimension formula - audit report|; pattern
validation success.

Declared acceptance condition: 0 violations/mismatches in every category;
Fock dimension exact match; pattern.validate() raises nothing.

Expected cost: light-to-moderate (lowering allocates finite GKP codewords
for small 1-2 wire signed-X models, no Fock-space simulation).

Manuscript destination: Appendix (Fig. 9 supporting lowering-preservation
table).

Scientific limitations: Only signed-X-supported (0/pi) angle configurations
are tested, since lowering itself requires the capability audit to pass
first (see R35).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import numpy as np

from photographiqml import MuTA
from photographiqml.lowering import GKPPhysicalConfig, lower_muta_to_gkp

EXPERIMENT_ID = "R36"


def main():
    plt = common.setup_style()
    rows = []
    config = GKPPhysicalConfig(cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513)

    for n_wires, n_layers in [(1, 1), (1, 2), (2, 1)]:
        model = MuTA(n_wires, n_layers, one_column=True)
        angles = dict(
            zip(
                model.trainable_parameters(),
                common.rng(n_wires * 10 + n_layers).choice([0.0, np.pi], model.n_parameters),
            )
        )
        lowered = lower_muta_to_gkp(model, angles, config=config)

        node_map_identity_violations = sum(
            1 for v in model.graph.nodes if lowered.node_map.get(v) != v
        )
        io_ok = tuple(lowered.input_modes) == tuple(model.input_nodes) and tuple(
            lowered.output_modes
        ) == tuple(model.output_nodes)
        measurement_key_coverage_ok = set(lowered.measurement_keys.keys()) == set(
            model.measurement_order
        )

        frame_dependency_mismatches = 0
        for node in model.graph.nodes:
            expected_sources = {
                source
                for source, (successor, z) in model.corrections.items()
                if node == successor or node in z
            }
            actual_sources = set(lowered.frame_dependencies.get(node, ()))
            if expected_sources != actual_sources:
                frame_dependency_mismatches += 1

        try:
            lowered.pattern.validate()
            pattern_valid = True
        except Exception:  # noqa: BLE001 -- validation failure is itself the observation
            pattern_valid = False

        peak = lowered.audit["peak_live_modes"]
        analytic_dimension = common.fock_dimension(peak, config.cutoff)
        reported_dimension = lowered.audit["hilbert_dimension"]

        rows.append(
            {
                "n_wires": n_wires,
                "n_layers": n_layers,
                "node_map_identity_violations": node_map_identity_violations,
                "io_ok": io_ok,
                "measurement_key_coverage_ok": measurement_key_coverage_ok,
                "frame_dependency_mismatches": frame_dependency_mismatches,
                "pattern_valid": pattern_valid,
                "peak_live_modes": peak,
                "peak_at_least_n_wires": peak >= n_wires,
                "analytic_fock_dimension": analytic_dimension,
                "reported_fock_dimension": reported_dimension,
                "fock_dimension_error": abs(analytic_dimension - reported_dimension),
                "cz_edges_match_graph": lowered.audit["cz_operations"]
                == model.graph.number_of_edges(),
            }
        )

    all_ok = all(
        r["node_map_identity_violations"] == 0
        and r["io_ok"]
        and r["measurement_key_coverage_ok"]
        and r["frame_dependency_mismatches"] == 0
        and r["pattern_valid"]
        and r["peak_at_least_n_wires"]
        and r["fock_dimension_error"] == 0
        and r["cz_edges_match_graph"]
        for r in rows
    )
    status = "pass" if all_ok else "fail"

    common.save_result(
        rows,
        "R36_lowering_preservation",
        extra={
            "protocol": "lower_muta_to_gkp structural preservation + analytic Fock-dimension cross-check",
            "oracle_class": "A/E",
            "status_category": "exact",
            "acceptance_condition": "0 violations in every category; Fock dimension exact match; pattern validates",
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "A/E", "status": status},
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"n{r['n_wires']}L{r['n_layers']}" for r in rows]
    checks = ["node_map_identity_violations", "frame_dependency_mismatches", "fock_dimension_error"]
    x = np.arange(len(rows))
    # All-zero values render invisibly as bars; use markers so every check
    # is visibly plotted, not just an empty axes.
    for i, check in enumerate(checks):
        ax.scatter(x + i * 0.25, [r[check] for r in rows], label=check, zorder=3)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.5, 1.0)
    ax.set(
        title=f"R36: lowering preservation violations, {len(rows)} configurations (status={status})",
        ylabel="violation count",
    )
    ax.legend(fontsize=6.5)
    common.save_figure(fig, "R36_lowering_preservation")
    plt.close(fig)

    common.print_summary(
        "R36 lowering preservation", n_configs=len(rows), all_ok=all_ok, status=status
    )
    if status != "pass":
        raise AssertionError(
            f"R36 failed: {[r for r in rows if not (r['node_map_identity_violations'] == 0)]}"
        )


if __name__ == "__main__":
    main()
