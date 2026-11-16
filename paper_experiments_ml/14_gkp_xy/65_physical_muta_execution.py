"""R65: Physical multilayer MuTA with non-Pauli equatorial angles (Experiment F).

Scientific question: can a multilayer PhysicalMuTA whose measurement angles include non-Pauli
values be executed end to end on finite-energy GKP (Gaussian shears, magic-state injection,
frame-adaptive angles, decoded Pauli frames), and how close is its decoded output to the
logical MuTA target at two resource qualities?

Model: one wire, two layers, one column (8 measured nodes), generic logical input,
angles (0, pi/4, 0, pi/2, 0.7, 0, pi, 0): two injected non-Pauli nodes, one shear, Pauli
nodes. Baseline: the same chain with every angle rounded to the signed-X family {0, pi}
(the previous physical capability), which shows the finite-energy floor of the CZ chain
itself. Physical-shots mode; the decoded output probabilities are the conditional joint POVM
averaged over trajectories (Rao-Blackwell), with shot-to-shot SE.

Convergence study (deterministic): the one-layer chain with a Clifford node angle +-pi/2 at
node (0,1) (Gaussian shear on a CZ-entangled GKP mode, frame-adaptive), postselected on three
analog branches at cell centres, compared with photographiqml.validation.contract_branch for
widths 0.6 / 0.45 / 0.35 (cutoffs 48 / 72 / 104; two live modes). This isolates the lowering's
correctness from sampling and shows the finite-energy error of the chain itself.

Acceptance: all probabilities valid; heralded fraction >= 0.8 for the injected chains; in the
convergence study the worst branch error decreases with resource quality and is <= 0.05 at
width 0.35. The injected two-layer TVDs are descriptive: injected three-mode chains at
width <= 0.45 need cutoffs beyond the practical Fock budget (R66, R67), so their ideal limit
is not numerically accessible here.

Revision note: the first design asserted "medium TVD <= broad TVD + 2 SE" for the injected
chain. It passed only because of large SE (medium 0.61 +- 0.10 vs broad 0.43 +- 0.08 with 8-16
shots) and did not resolve a trend; a postselected study then showed the chain error is
finite-energy dominated at width >= 0.45 and converges by width 0.35, so that deterministic
study replaced the unresolved trend criterion.

Revision note: a first run at cutoffs 56/72 stopped on the retained-norm guard (three live
modes: two chain modes plus the magic ancilla, after a shear on an entangled mode); cutoffs were
raised to 72/88 and shots reduced to 16/8 to bound runtime. Criteria unchanged.
"""

import warnings

import numpy as np
from xy_support import GENERIC, plt, save

from photographiqml.lowering import GKPPhysicalConfig
from photographiqml.physical import PhysicalMuTA, compare_logical_physical

NAME = "R65_physical_muta_execution"
ANGLES = (0.0, np.pi / 4, 0.0, np.pi / 2, 0.7, 0.0, np.pi, 0.0)
SETTINGS = {
    "broad": {"peak_width": 0.6, "envelope": 0.6, "cutoff": 72, "shots": 16},
    "medium": {"peak_width": 0.45, "envelope": 0.45, "cutoff": 88, "shots": 8},
}


def run(resource, family, angles):
    s = SETTINGS[resource]
    config = GKPPhysicalConfig(cutoff=s["cutoff"], peak_width=s["peak_width"], envelope=s["envelope"], peaks=6,
                               grid_points=2049, backend="native-fock", max_dimension=200_000,
                               xy_protocol="auto" if family == "XY" else "signed-x")
    model = PhysicalMuTA(1, 2, one_column=True, measurement_family=family, physical_config=config)
    parameters = dict(zip(model.parameters(), angles))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.run(np.array(GENERIC, complex), parameters, mode="physical-shots", shots=s["shots"], seed=65)
    comparison = compare_logical_physical(result)
    ideal = np.abs(result.logical_target.state) ** 2
    # One output bit: TVD = |P(0) - P_ideal(0)|, so its SE is the shot SE of P(0).
    tvd_se = float(result.standard_errors[(0,)])
    records = [r for shot in result.measurement_records for r in shot.values()]
    heralded = [r.heralded_success for r in records if hasattr(r, "heralded_success")]
    return {
        "resource": resource,
        "family": family,
        "angles": list(angles),
        "decoded": list(result.decoded_joint_probabilities.values()),
        "logical_target": ideal.tolist(),
        "tvd": comparison["total_variation_distance"],
        "tvd_shot_se": tvd_se,
        "heralded_fraction": float(np.mean(heralded)) if heralded else 1.0,
        "mean_injection_rounds": float(np.mean([len(getattr(r, "rounds", ())) for r in records])),
        "seconds_per_shot": result.diagnostics["simulation_seconds"] / SETTINGS[resource]["shots"],
        "hilbert_dimension": result.diagnostics["resource_accounting"]["hilbert_dimension"],
        "shots": SETTINGS[resource]["shots"],
    }


def chain_convergence():
    """Postselected Clifford-node branches vs contract_branch at three resource qualities."""
    import photographiq as pg
    from photographiqml.lowering import lower_muta_to_gkp, node_frame
    from photographiqml.validation import contract_branch

    spacing = float(np.sqrt(2 * np.pi))
    rows = []
    for width, cutoff in ((0.6, 48), (0.45, 72), (0.35, 104)):
        config = GKPPhysicalConfig(cutoff=cutoff, peak_width=width, envelope=width, peaks=6, grid_points=2049,
                                   backend="native-fock", xy_protocol="clifford", max_dimension=200_000)
        model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
        for sign in (1, -1):
            parameters = dict(zip(model.parameters(), (0.0, sign * np.pi / 2, 0.0, 0.0)))
            lowering = lower_muta_to_gkp(model, parameters)
            for raw in ((0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)):
                records = {("raw", v): {"ancilla": [], "data": b * spacing}
                           for v, b in zip(model.measurement_order, raw)}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = pg.simulate(lowering.pattern,
                                         inputs={model.input_nodes[0]: lowering.code.encode(*GENERIC)},
                                         backend="native-fock", cutoff=cutoff, measurement_outcomes=records)
                frame = {v: node_frame(model, v, result.records) for v in model.output_nodes}
                readout = pg.multimode_readout(result.state, dict.fromkeys(model.output_nodes, "Z"), frames=frame)
                interpreted = [int(result.records[("bit", v)]) for v in model.measurement_order]
                target = abs(contract_branch(model, np.array(GENERIC, complex), parameters, interpreted)[0][0]) ** 2
                p0 = readout["joint_probabilities"][(0,)]
                rows.append({"study": "convergence", "width": width, "cutoff": cutoff, "angle": sign * np.pi / 2,
                             "raw": list(raw), "interpreted": interpreted, "p0": p0,
                             "target_p0": float(target), "error": abs(p0 - target)})
        worst = max(r["error"] for r in rows if r["width"] == width)
        print(f"convergence width {width}: worst branch error {worst:.4f}", flush=True)
    return rows


def main():
    rows = chain_convergence()
    widths = (0.6, 0.45, 0.35)
    worst = [max(r["error"] for r in rows if r["width"] == w) for w in widths]
    convergence_ok = all(b < a for a, b in zip(worst, worst[1:])) and worst[-1] <= 0.05
    for resource in SETTINGS:
        for family, angles in (("XY", ANGLES), ("X", tuple(0.0 if abs(a) < np.pi / 2 else np.pi for a in ANGLES))):
            row = run(resource, family, angles)
            rows.append(row)
            print(f"{resource} {family}: tvd={row['tvd']:.4f}+-{row['tvd_shot_se']:.4f} "
                  f"heralded={row['heralded_fraction']:.3f} {row['seconds_per_shot']:.1f}s/shot", flush=True)
    executed = [r for r in rows if "family" in r]
    xy = {r["resource"]: r for r in executed if r["family"] == "XY"}
    valid = all(min(r["decoded"]) >= -1e-9 and abs(sum(r["decoded"]) - 1) < 1e-9 for r in executed)
    heralded = all(r["heralded_fraction"] >= 0.8 for r in xy.values())
    status = "pass" if valid and heralded and convergence_ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(13, 3.8))
    for sign, marker in ((1, "o"), (-1, "s")):
        for raw in ((0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)):
            sub = [r for r in rows if r.get("study") == "convergence" and r["angle"] == sign * np.pi / 2
                   and r["raw"] == list(raw)]
            axes[2].semilogy([r["width"] for r in sub], [max(r["error"], 1e-4) for r in sub], marker + "-", lw=0.8,
                             label=f"{'+' if sign > 0 else '-'}pi/2, raw {''.join(map(str, raw))}")
    axes[2].invert_xaxis()
    axes[2].set(xlabel="GKP width = envelope", ylabel="|P(0) - target| per branch",
                title="Chain with a shear node (postselected)")
    axes[2].legend(fontsize=5.5, ncol=2)
    labels = ["0", "1"]
    for k, row in enumerate(r for r in executed if r["family"] == "XY"):
        axes[0].bar(np.arange(2) + (k - 0.5) * 0.35, row["decoded"], 0.35, label=f"physical XY ({row['resource']})")
    axes[0].plot(np.arange(2), executed[0]["logical_target"], "ko", mfc="none", ms=9, label="logical target")
    axes[0].set(xticks=range(2), xticklabels=labels, xlabel="Decoded output bit", ylabel="Probability",
                title="Two-layer MuTA, angles incl. $\\pi/4$ and 0.7")
    axes[0].legend(fontsize=7)
    names = [f"{r['resource']}\n{'XY' if r['family'] == 'XY' else 'signed-X baseline'}" for r in executed]
    axes[1].bar(names, [r["tvd"] for r in executed], yerr=[r["tvd_shot_se"] for r in executed],
                color=["C0" if r["family"] == "XY" else "0.6" for r in executed])
    axes[1].set(ylabel="TVD to logical target", title="Finite-energy error vs resource quality")
    axes[1].tick_params(axis="x", labelsize=7)
    save(rows, NAME, {"oracle_class": "N/A", "status": status, "valid": valid, "heralded": heralded,
                      "convergence_ok": convergence_ok, "worst_branch_error": dict(zip(map(str, widths), worst)),
                      "settings": SETTINGS}, fig)
    print(f"R65 status={status}")
    if status != "pass":
        raise AssertionError("R65 acceptance failed")


if __name__ == "__main__":
    main()
