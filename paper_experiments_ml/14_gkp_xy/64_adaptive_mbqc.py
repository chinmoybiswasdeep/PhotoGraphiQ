"""R64: Adaptive arbitrary-angle MBQC with physical XY measurements (Experiment E).

Scientific question: does a previous logical outcome correctly change a later physical
measurement instruction? With frame tracking (no physical Pauli corrections), an X byproduct
turns M(a) into M(-a) and a Z byproduct into -M(a); the lowering therefore measures
(-1)^x a and flips the decoded bit by z.

Part 1 (ideal, exhaustive): for a one-wire, one-layer one-column MuTA chain (4 measured
nodes, generic input, non-Clifford angles) every raw-outcome branch of the frame-tracked
pattern is simulated on qubits and compared with photographiqml.validation.contract_branch
(an independent contraction that applies physical corrections). The adaptive rule must
reproduce every branch exactly; the non-adaptive rule (always +a) must fail on some branch.

Part 2 (physical, postselected): the same chain is lowered to finite-energy GKP (broad
resource, cutoff 56) and executed in physical-conditional mode with analog records chosen so
that node (0,0) decodes to 0 or to 1. The instrument record of node (0,1) must show angle +a or
-a respectively, and the exact conditional decoded output is compared with the logical target
for the adaptive lowering and for a deliberately non-adaptive variant.

Acceptance: part 1 adaptive fidelity >= 1 - 1e-12 and equal branch probabilities on all 16
branches, non-adaptive fidelity < 0.99 on at least one branch; part 2 recorded angle equals
(-1)^x a exactly, and in the x = 1 branch the adaptive TVD is below the non-adaptive TVD.
"""

import dataclasses
from itertools import product

import numpy as np
from xy_support import GENERIC, pg, plt, save

from photographiqml.logical import X, Z, cz, execute, local_gate, statevector
from photographiqml.lowering import GKPPhysicalConfig, lower_muta_to_gkp, node_frame
from photographiqml.physical import PhysicalMuTA
from photographiqml.validation import contract_branch

NAME = "R64_adaptive_mbqc"
ANGLES = (0.0, 1.0, 0.6, 0.0)
SPACING = float(np.sqrt(2 * np.pi))


def model():
    config = GKPPhysicalConfig(cutoff=56, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049,
                               backend="native-fock", xy_protocol="auto", max_dimension=200_000)
    return PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)


def frame_tracked(chain, input_state, angles, raw, adaptive=True):
    """Ideal frame-tracked execution: returns (corrected output, probability, interpreted bits)."""
    nodes = list(chain.input_nodes) + [v for v in chain.graph if v not in chain.input_nodes]
    state = statevector(input_state, chain.n_wires)
    for _ in range(len(nodes) - chain.n_wires):
        state = np.kron(state, np.ones(2) / np.sqrt(2))
    for u, v in chain.graph.edges:
        state = cz(state, nodes.index(u), nodes.index(v), len(nodes))
    bits, probability = {}, 1.0
    for node, r in zip(chain.measurement_order, raw):
        frame = node_frame(chain, node, {("bit", k): v for k, v in bits.items()})
        alpha = angles[chain.parameter_name(node)]
        measured = (-1) ** frame.x_bit * alpha if adaptive else alpha
        bra = np.array([1, (-1) ** r * np.exp(-1j * measured)]) / np.sqrt(2)
        tensor = np.moveaxis(state.reshape([2] * len(nodes)), nodes.index(node), 0)
        state = (bra @ tensor.reshape(2, -1)).reshape(-1)
        mass = float(np.vdot(state, state).real)
        probability *= mass
        state /= np.sqrt(mass)
        nodes.remove(node)
        bits[node] = r ^ frame.correction("X")
    for output in chain.output_nodes:
        frame = node_frame(chain, output, {("bit", k): v for k, v in bits.items()})
        if frame.z_bit:
            state = local_gate(state, Z, nodes.index(output), len(nodes))
        if frame.x_bit:
            state = local_gate(state, X, nodes.index(output), len(nodes))
    return state, probability, [bits[v] for v in chain.measurement_order]


def physical_branch(chain, parameters, first_bit, adaptive):
    lowering = lower_muta_to_gkp(chain, parameters)
    commands = []
    for command in lowering.pattern.commands:
        if isinstance(command, pg.Measure) and command.node == (0, 1) and not adaptive:
            command = dataclasses.replace(
                command, measurement=dataclasses.replace(command.measurement, angle=float(ANGLES[1]))
            )
        commands.append(command)
    pattern = pg.Pattern(inputs=chain.input_nodes)
    for command in commands:
        pattern.append(command)
    records = {
        ("raw", (0, 0)): {"ancilla": [], "data": first_bit * SPACING},  # alpha 0: X readout only
        ("raw", (0, 1)): {"ancilla": [0.0], "data": 0.0},  # +-1.0: injection succeeds in round 1
        ("raw", (0, 2)): {"ancilla": [0.0], "data": 0.0},  # 0.6: injection succeeds in round 1
        ("raw", (0, 3)): {"ancilla": [], "data": 0.0},
    }
    gkp = lowering.code
    encoded = {chain.input_nodes[0]: gkp.encode(*GENERIC)}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pg.simulate(pattern, inputs=encoded, backend="native-fock", cutoff=gkp.cutoff,
                             measurement_outcomes=records)
    frame = {v: node_frame(chain, v, result.records) for v in chain.output_nodes}
    readout = pg.multimode_readout(result.state, dict.fromkeys(chain.output_nodes, "Z"), frames=frame)
    probabilities = np.array([readout["joint_probabilities"][(b,)] for b in (0, 1)])
    return probabilities, result.records[("raw", (0, 1))].angle, result.records[("bit", (0, 0))]


def main():
    chain = model()
    parameters = dict(zip(chain.parameters(), ANGLES))
    angles = chain._parameters.bind(parameters)
    input_state = np.array(GENERIC, complex)
    rows = []
    worst_adaptive, worst_static = 1.0, 1.0
    for raw in product((0, 1), repeat=len(chain.measurement_order)):
        for adaptive in (True, False):
            output, probability, bits = frame_tracked(chain, input_state, angles, raw, adaptive)
            reference, ref_probability = contract_branch(chain, input_state, parameters, bits)
            fidelity = float(abs(np.vdot(reference, output)) ** 2)
            rows.append({"part": "ideal", "raw": list(raw), "interpreted": bits, "adaptive": adaptive,
                         "fidelity": fidelity, "probability": probability, "reference_probability": ref_probability})
            if adaptive:
                worst_adaptive = min(worst_adaptive, fidelity)
                assert abs(probability - ref_probability) < 1e-12, (raw, probability, ref_probability)
            else:
                worst_static = min(worst_static, fidelity)
    ideal_ok = worst_adaptive >= 1 - 1e-12 and worst_static < 0.99
    print(f"ideal: worst adaptive fidelity {worst_adaptive:.15f}, worst non-adaptive {worst_static:.4f}", flush=True)

    logical = np.abs(execute(chain, input_state, angles).state) ** 2
    physical = {}
    angle_ok = True
    for first_bit in (0, 1):
        for adaptive in (True, False):
            probabilities, angle, bit = physical_branch(chain, parameters, first_bit, adaptive)
            expected = (-1) ** bit * ANGLES[1] if adaptive else ANGLES[1]
            angle_ok &= abs(angle - expected) < 1e-15
            tvd = float(np.abs(probabilities - logical).sum() / 2)
            physical[(first_bit, adaptive)] = tvd
            rows.append({"part": "physical", "first_bit": first_bit, "interpreted_first_bit": int(bit),
                         "adaptive": adaptive, "recorded_angle_node_0_1": angle, "expected_angle": expected,
                         "decoded_output": probabilities.tolist(), "logical_target": logical.tolist(), "tvd": tvd})
            print(f"physical first bit {first_bit} adaptive={adaptive}: angle={angle:+.4f} tvd={tvd:.4f}", flush=True)
    flipped = [k for k in physical if k[0] == 1]
    branch_ok = physical[(1, True)] < physical[(1, False)]
    status = "pass" if ideal_ok and angle_ok and branch_ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(9.5, 3.8))
    ideal_rows = [r for r in rows if r["part"] == "ideal"]
    for adaptive, marker in ((True, "o"), (False, "x")):
        sub = [r for r in ideal_rows if r["adaptive"] == adaptive]
        axes[0].plot(range(len(sub)), [1 - r["fidelity"] + 1e-17 for r in sub], marker,
                     label="adaptive (-1)$^x\\alpha$" if adaptive else "non-adaptive $\\alpha$")
    axes[0].set(yscale="log", xlabel="Raw-outcome branch (4 measured nodes)", ylabel="1 - branch fidelity",
                title="Ideal frame-tracked MBQC vs corrected contraction")
    axes[0].legend(fontsize=7)
    labels = [f"node(0,0)={b}\n{'adaptive' if a else 'non-adaptive'}" for b, a in physical]
    axes[1].bar(labels, list(physical.values()), color=["C0" if a else "C3" for _, a in physical])
    axes[1].set(ylabel="Decoded-output TVD to logical target", title="Physical GKP (broad, postselected records)")
    axes[1].tick_params(axis="x", labelsize=6.5)
    save(rows, NAME, {"oracle_class": "A", "status": status, "ideal_ok": ideal_ok, "angle_ok": angle_ok,
                      "x1_branch_adaptive_better": branch_ok, "angles": ANGLES, "flipped_branches": flipped}, fig)
    print(f"R64 status={status}")
    if status != "pass":
        raise AssertionError("R64 acceptance failed")


if __name__ == "__main__":
    main()
