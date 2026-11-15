"""R41: PhotoGraphiQML versus an independently assembled public
PhotoGraphiQ pattern for fixed-outcome conditional execution, across
several conditioning vectors and signed-X angle patterns.

Scientific question: Does PhysicalMuTA(1).run in physical-conditional mode
agree with the same independently assembled one-wire Pattern used in R38
(built directly from public pg.Prepare/Measure/Signal/Output commands, not
calling lower_muta_to_gkp), across several different fixed analog
postselection vectors and both members of the signed-X angle family
(alpha=0 and alpha=pi)?

Theory/equations: same one-wire linear-chain correction recursion as R38
(interpreted(c) = raw_bit(c) XOR interpreted(c-2)), independently derived
and reused here without calling photographiqml.lowering.node_frame.

Functionality tested: PhysicalMuTA.run(mode="physical-conditional") vs. the
independent Pattern builder, generalized to accept an arbitrary per-column
angle and arbitrary fixed analog outcome vector.

Oracle and independence class: C (shared finite-GKP resource inputs,
independent Pattern orchestration -- same classification rationale as R38).

Exact/approximate/statistical status: exact (both sides simulate the
identical finite resource and identical fixed conditioning outcomes).

Primary metric: max absolute difference between decoded joint probability
distributions, across all declared cases.

Declared acceptance condition: max difference < tol
(tol = declare_tolerance(scale=1, safety_factor=1e6), matching R38).

Expected cost: moderate (several one-wire Fock-space simulations, cutoff=24,
just-in-time live-mode schedule).

Manuscript destination: Appendix (Fig. 10 supporting table, broader
fixed-outcome coverage complementing R38's single baseline case).

Scientific limitations: One-wire only (matching R38's tractable scope);
larger models are not attempted here (see R42 for abstraction-overhead
timing at matched workloads instead).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import photographiq as pg

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA

EXPERIMENT_ID = "R41"


def build_and_run_independent_pattern(config, angles, outcomes, seed=11):
    code = pg.GKPCode(
        cutoff=config.cutoff,
        peak_width=config.peak_width,
        envelope=config.envelope,
        peaks=config.peaks,
        grid_points=config.grid_points,
    )
    decoder = pg.NearestCellDecoder()
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
        readout = code.logical_measurement("XY", alpha=angles[c], decoder=decoder)
        pattern.append(pg.Measure(columns[c], readout, columns[c]))
        z_source = columns[c - 2] if c - 2 >= 0 else None
        fn, deps = make_interpreted(columns[c], z_source)
        pattern.append(pg.Signal(("bit", columns[c]), pg.CallableExpression(fn, deps)))
    pattern.append(pg.Output((columns[4],)))
    pattern.validate()

    encoded_input = code.encode(1, 0)
    outcome_map = {columns[c]: float(outcomes[c]) for c in range(4)}
    raw_result = pg.simulate(
        pattern,
        seed=seed,
        inputs={input_node: encoded_input},
        backend=config.backend,
        cutoff=config.cutoff,
        measurement_outcomes=outcome_map,
    )

    x_bit = int(raw_result.records[("bit", columns[3])])
    z_bit = int(raw_result.records[("bit", columns[2])])
    frame = {columns[4]: pg.LogicalPauliFrame(x_bit, z_bit)}
    readout_result = pg.multimode_readout(raw_result.state, {columns[4]: "Z"}, frames=frame)
    return readout_result["joint_probabilities"]


def main():
    plt = common.setup_style()
    tol = common.declare_tolerance(scale=1.0, safety_factor=1e6)
    config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    model = MuTA(1, 1, one_column=True)
    physical_model = PhysicalMuTA(1, physical_config=config)

    cases = [
        {"name": "zero_angles_zero_outcomes", "angles": [0.0] * 4, "outcomes": [0.0] * 4},
        {"name": "zero_angles_uniform_outcomes", "angles": [0.0] * 4, "outcomes": [0.3] * 4},
        {
            "name": "zero_angles_mixed_outcomes",
            "angles": [0.0] * 4,
            "outcomes": [0.2, -0.1, 0.15, 0.05],
        },
        {
            "name": "pi_first_angle",
            "angles": [3.141592653589793, 0.0, 0.0, 0.0],
            "outcomes": [0.0] * 4,
        },
    ]

    rows = []
    for case in cases:
        angle_map = dict(zip(model.measurement_order, case["angles"]))
        analog_outcomes = dict(zip(model.measurement_order, case["outcomes"]))
        parameter_dict = {model.parameter_name(node): angle for node, angle in angle_map.items()}
        pqml_result = physical_model.run(
            [1, 0],
            parameters=parameter_dict,
            mode="physical-conditional",
            analog_outcomes=analog_outcomes,
        )
        independent_probabilities = build_and_run_independent_pattern(
            config, case["angles"], case["outcomes"]
        )
        for label in pqml_result.decoded_joint_probabilities:
            pqml_p = pqml_result.decoded_joint_probabilities[label]
            independent_p = independent_probabilities.get(label, 0.0)
            rows.append(
                {
                    "case": case["name"],
                    "label": str(label),
                    "photographiqml": pqml_p,
                    "independent_pattern": independent_p,
                    "error": abs(pqml_p - independent_p),
                }
            )

    max_error = max(r["error"] for r in rows)
    status = "pass" if max_error < tol else "fail"

    common.save_result(
        rows,
        "R41_conditional_execution_comparison",
        extra={
            "protocol": "PhysicalMuTA(1) physical-conditional vs. independently assembled Pattern, several conditioning vectors/angles",
            "oracle_class": "C",
            "status_category": "exact",
            "tolerance": tol,
            "acceptance_condition": f"max joint-probability difference < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "C", "status": status},
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{r['case']}\n{r['label']}" for r in rows]
    ax.semilogy(
        range(len(rows)),
        [max(r["error"], 1e-18) for r in rows],
        "o",
        color=common.COLORS["photographiq"],
    )
    ax.axhline(tol, color=common.COLORS["acceptance"], linestyle="--", label=f"tol={tol:.1e}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=5.5)
    ax.set(
        title=f"R41: conditional execution agreement (status={status})", ylabel="probability error"
    )
    ax.legend()
    common.save_figure(fig, "R41_conditional_execution_comparison")
    plt.close(fig)

    common.print_summary(
        "R41 conditional execution comparison",
        n_cases=len(cases),
        max_error=max_error,
        status=status,
    )
    if status != "pass":
        raise AssertionError(f"R41 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
