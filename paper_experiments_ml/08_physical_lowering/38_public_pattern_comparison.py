"""R38: Physical one-wire result versus an independently assembled public
PhotoGraphiQ pattern that does not call the PhotoGraphiQML lowering helper.

Scientific question: Does an independently assembled public PhotoGraphiQ
Pattern -- built directly from pg.Prepare/Entangle-via-GKPCode.logical_cz/
Measure/Signal/Output commands and simulated with pg.simulate, with its own
freshly derived (not reused) Pauli-frame arithmetic for this specific
one-wire linear-chain topology -- reproduce the same decoded output
probabilities as PhysicalMuTA(1).run in physical-conditional mode?

Theory/equations: for a one-wire, one-layer linear chain (columns 0..4,
column 4 = output), the correction map is corrections[(0,c)] =
(successor=(0,c+1), z_targets={(0,c+2)} if c+2<=4 else {}); this experiment
derives the resulting interpreted-bit recursion
interpreted(c) = raw_bit(c) XOR interpreted(c-2) (with interpreted(negative)=0)
directly from that chain topology, independently of
photographiqml.lowering.node_frame (which is not called anywhere in this
script).

Functionality tested: PhysicalMuTA.run (photographiqml.physical) vs. a
hand-assembled pg.Pattern using only public PhotoGraphiQ classes
(pg.Prepare, pg.Measure, pg.Signal, pg.CallableExpression, pg.Output,
pg.simulate, pg.GKPCode.plus/encode/logical_cz/logical_measurement,
pg.NearestCellDecoder, pg.LogicalPauliFrame, pg.multimode_readout).

Oracle and independence class: C -- the finite-GKP resource (GKPCode with
the same cutoff/width/envelope/peaks/grid_points) and the input encoding
(code.encode) are shared inputs with PhysicalMuTA (both must describe the
same physical resource to be a meaningful comparison), while the Pattern
*construction*, schedule and Pauli-frame arithmetic are independently
written here, not calling lower_muta_to_gkp. This is therefore explicitly
classified C (shared resource inputs, independent orchestration), not B.

Exact/approximate/statistical status: exact (single physical-conditional
trajectory; both sides use identical analog postselection outcomes and the
identical finite resource, so any nonzero difference indicates an
orchestration or frame-convention disagreement, not sampling noise).

Primary metric: max absolute difference between the two decoded joint
probability distributions.

Declared acceptance condition: max difference < tol
(tol = declare_tolerance(scale=1, safety_factor=1e6), loosened to absorb
two independently constructed Fock-space contractions of the same finite
resource, which need not agree to machine precision).

Expected cost: moderate (one Fock-space simulation of a 5-node one-wire
pattern per side, cutoff=24).

Manuscript destination: Main text (Fig. 10, independent public-Pattern
cross-check of restricted physical execution).

Scientific limitations: One-wire, all-zero-angle, physical-conditional case
only; this is the largest genuinely matched subproblem practical to hand-
assemble here, not a general-purpose independent physical backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import photographiq as pg

from photographiqml import GKPPhysicalConfig, MuTA, PhysicalMuTA

EXPERIMENT_ID = "R38"


def build_and_run_independent_pattern(config, angle=0.0, seed=11):
    code = pg.GKPCode(
        cutoff=config.cutoff,
        peak_width=config.peak_width,
        envelope=config.envelope,
        peaks=config.peaks,
        grid_points=config.grid_points,
    )
    decoder = pg.NearestCellDecoder()
    readout = code.logical_measurement("XY", alpha=angle, decoder=decoder)
    plus = code.plus()

    input_node, columns = (0, 0), [(0, c) for c in range(5)]
    pattern = pg.Pattern(inputs=[input_node])

    def make_interpreted(node, z_source):
        def interpreted(records, node=node, z_source=z_source):
            # Signal registers evaluate to floats even for a 0/1 bit value
            # (matches photographiq.lowering's own node_frame, which casts
            # with int(...) before XOR-ing); cast explicitly here too.
            raw_bit = int(records[node].bit)
            z_correction = int(records[("bit", z_source)]) if z_source is not None else 0
            return raw_bit ^ z_correction

        declared = frozenset({node} | ({("bit", z_source)} if z_source is not None else set()))
        return interpreted, declared

    # Just-in-time schedule (prepare/CZ/measure interleaved) keeps at most 2
    # modes simultaneously live, matching a linear-chain topology's minimal
    # Fock-dimension footprint; preparing all ancillas upfront instead would
    # make comb(cutoff+5-1,5) modes live at once and is intractably slow.
    for c in range(4):
        pattern.append(pg.Prepare(columns[c + 1], state=plus))
        pattern.append(code.logical_cz(columns[c], columns[c + 1]))
        pattern.append(pg.Measure(columns[c], readout, columns[c]))
        z_source = columns[c - 2] if c - 2 >= 0 else None
        fn, deps = make_interpreted(columns[c], z_source)
        pattern.append(pg.Signal(("bit", columns[c]), pg.CallableExpression(fn, deps)))
    pattern.append(pg.Output((columns[4],)))
    pattern.validate()

    encoded_input = code.encode(1, 0)  # logical |0>, matches PhysicalMuTA.run([1,0], ...)
    outcomes = {columns[c]: 0.0 for c in range(4)}
    raw_result = pg.simulate(
        pattern,
        seed=seed,
        inputs={input_node: encoded_input},
        backend=config.backend,
        cutoff=config.cutoff,
        measurement_outcomes=outcomes,
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

    model = MuTA(1, 1, one_column=True)  # semantic labels only, not executed
    physical_model = PhysicalMuTA(1, physical_config=config)
    pqml_result = physical_model.run(
        [1, 0],
        mode="physical-conditional",
        analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
    )
    independent_probabilities = build_and_run_independent_pattern(config, angle=0.0)

    labels = list(pqml_result.decoded_joint_probabilities)
    rows = []
    for label in labels:
        pqml_p = pqml_result.decoded_joint_probabilities[label]
        independent_p = independent_probabilities.get(label, 0.0)
        rows.append(
            {
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
        "R38_public_pattern_comparison",
        extra={
            "protocol": "PhysicalMuTA(1) physical-conditional vs. independently assembled public Pattern",
            "oracle_class": "C",
            "status_category": "exact",
            "resource_config": config.to_dict(),
            "tolerance": tol,
            "acceptance_condition": f"max joint-probability difference < {tol:.3e}",
            "max_error": max_error,
            "status": status,
        },
        meta_extra={"experiment_id": EXPERIMENT_ID, "oracle_class": "C", "status": status},
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(rows))
    ax.bar(
        [i - 0.15 for i in x],
        [r["photographiqml"] for r in rows],
        width=0.3,
        label="PhysicalMuTA",
        color=common.COLORS["photographiqml"],
    )
    ax.bar(
        [i + 0.15 for i in x],
        [r["independent_pattern"] for r in rows],
        width=0.3,
        label="independent Pattern",
        color=common.COLORS["photographiq"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set(
        title=f"R38: decoded joint probabilities (status={status}, max_err={max_error:.2e})",
        ylabel="probability",
    )
    ax.legend(fontsize=7)
    common.save_figure(fig, "R38_public_pattern_comparison")
    plt.close(fig)

    common.print_summary(
        "R38 public pattern comparison", n_labels=len(rows), max_error=max_error, status=status
    )
    if status != "pass":
        raise AssertionError(f"R38 failed: max_error={max_error} tol={tol}")


if __name__ == "__main__":
    main()
