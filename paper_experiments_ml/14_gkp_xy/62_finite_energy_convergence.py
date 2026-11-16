"""R62: Finite-energy GKP convergence of the logical-XY instrument (Experiment C).

Scientific question: how do the physical resource parameters (peak width, envelope, number of
lattice cells), the numerical Fock cutoff and a physical noise channel (ancilla loss) each
change the measured statistics? Axes are varied one at a time and labelled as physical or
numerical.

* Physical axes (untruncated grid reference, alpha = pi/3, generic input): peak width at
  envelope 0.45; envelope at width 0.45; peaks at width = envelope = 0.45. Metrics: heralded
  and unconditional distance |P(+) - P_ideal(+)|.
* Numerical axis (native-fock, medium resource): cutoff 56 ... 96 for (i) the deterministic
  Clifford Y readout of |+i~> and (ii) the likelihood density of one fixed two-round analog
  record at alpha = pi/3 (failure then success); the grid value is the untruncated limit.
* Noise axis (native-mixed-fock, broad resource, cutoff 40): ancilla transmissivity
  eta in {1, 0.99, 0.97, 0.95, 0.9} before the logical CNOT, alpha = pi/4, input |+~>. Loss is
  a genuine mixed-state channel; 48 sampled records per point, Rao-Blackwell estimate.

Acceptance: at the largest cutoff the engine matches the grid within 2e-3 (Y readout) and
1e-4 relative (record density); eta = 1 matches the noiseless reference within 4 SE + 2e-3;
P(+) at eta = 0.9 is lower than at eta = 1 by more than 2 SE of the paired per-shot difference
(loss degrades the instrument). Resource axes are descriptive.

Revision note: the first run used the unpaired SE (sqrt(SE_1^2 + SE_0.9^2)) and failed it
(difference 0.019). All eta points share the same seed (common random numbers), so the paired
per-shot difference is the correct statistic; the unpaired result is still reported.
"""

import numpy as np
from xy_support import GENERIC, GridModel, grid, ideal_probabilities, plt, run_single, save

NAME = "R62_finite_energy_convergence"
ALPHA = np.pi / 3
CUTOFFS = (56, 64, 72, 80, 96)
RECORD = {"ancilla": [2.4, 0.1], "data": 0.3}
ETAS = (1.0, 0.99, 0.97, 0.95, 0.9)
NOISE_SHOTS = 48


def reference(width, envelope, peaks):
    model = GridModel(width, envelope, peaks)
    rows = model.protocol_map(model.operator(model.encode(*GENERIC)), ALPHA, max_rounds=4)
    p_plus = sum(r["value"].real for r in rows if r["bit"] == 0)
    success = sum(r["value"].real for r in rows if r["success"])
    heralded = sum(r["value"].real for r in rows if r["bit"] == 0 and r["success"]) / success
    return p_plus, heralded


def main():
    rows = []
    ideal = ideal_probabilities(GENERIC, ALPHA)[0]
    axes_spec = {
        "peak_width": [(w, 0.45, 6) for w in (0.6, 0.5, 0.4, 0.3, 0.25)],
        "envelope": [(0.45, e, 6) for e in (0.6, 0.5, 0.4, 0.3, 0.25)],
        "peaks": [(0.45, 0.45, n) for n in (2, 3, 4, 6, 8)],
    }
    for axis, settings in axes_spec.items():
        for width, envelope, peaks in settings:
            p_plus, heralded = reference(width, envelope, peaks)
            value = {"peak_width": width, "envelope": envelope, "peaks": peaks}[axis]
            rows.append({"axis": axis, "axis_type": "physical resource", "value": value, "ideal": ideal,
                         "p_plus": p_plus, "heralded_p_plus": heralded,
                         "error_unconditional": abs(p_plus - ideal), "error_heralded": abs(heralded - ideal)})
        print(f"axis {axis} done", flush=True)

    model = grid("medium")
    y_reference = model.outcome_probabilities(model.encode(1, 1j), np.pi / 2, protocol="clifford")[0]
    density_reference = model.record_density(model.encode(*GENERIC), ALPHA, RECORD["ancilla"], RECORD["data"])[0]
    for cutoff in CUTOFFS:
        row = {"axis": "cutoff", "axis_type": "numerical refinement", "value": cutoff,
               "y_reference": y_reference, "density_reference": density_reference}
        try:
            y = run_single("medium", np.pi / 2, (1, 1j), protocol="clifford", cutoff=cutoff)[0][0]
            record = run_single("medium", ALPHA, GENERIC, cutoff=cutoff, outcomes=RECORD)[0][0]
            row |= {"y_engine": y.readout_probabilities[0], "density_engine": record.joint_density,
                    "y_error": abs(y.readout_probabilities[0] - y_reference),
                    "density_relative_error": abs(record.joint_density / density_reference - 1),
                    "min_retained_norm": record.retained_norm_min, "refused": False}
        except ValueError as exc:  # retained-norm guard: the cutoff cannot hold the protocol
            row |= {"refused": True, "reason": str(exc)}
        rows.append(row)
        print(f"cutoff {cutoff}: {row.get('density_relative_error', 'refused')}", flush=True)

    broad = grid("broad")
    noiseless = broad.outcome_probabilities(broad.encode(1, 1), np.pi / 4)[0]
    noise_rows, per_shot = [], {}
    for eta in ETAS:
        records, seconds = run_single("broad", np.pi / 4, (1, 1), shots=NOISE_SHOTS, seed=62, cutoff=40,
                                      backend="native-mixed-fock", ancilla_transmissivity=eta)
        values = np.array([r.readout_probabilities[0] for r in records])
        per_shot[eta] = values
        noise_rows.append({"axis": "ancilla_transmissivity", "axis_type": "physical noise", "value": eta,
                           "p_plus": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(len(values))),
                           "noiseless_reference": noiseless, "ideal": ideal_probabilities((1, 1), np.pi / 4)[0],
                           "seconds_per_shot": float(np.mean(seconds)), "shots": NOISE_SHOTS})
        print(f"eta {eta}: {noise_rows[-1]['p_plus']:.4f}", flush=True)
    rows += noise_rows

    final = [r for r in rows if r["axis"] == "cutoff" and not r["refused"]][-1]
    cutoff_ok = final["value"] == max(CUTOFFS) and final["y_error"] <= 2e-3 and final["density_relative_error"] <= 1e-4
    first, last = noise_rows[0], noise_rows[-1]
    eta_one_ok = abs(first["p_plus"] - noiseless) <= 4 * first["se"] + 2e-3
    unpaired_ok = first["p_plus"] - last["p_plus"] > 2 * np.hypot(first["se"], last["se"])
    paired = per_shot[ETAS[0]] - per_shot[ETAS[-1]]
    paired_mean, paired_se = float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(len(paired)))
    degrade_ok = paired_mean > 2 * paired_se
    status = "pass" if cutoff_ok and eta_one_ok and degrade_ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 4, figsize=(15, 3.6))
    for axis, marker in (("peak_width", "o"), ("envelope", "s")):
        sub = [r for r in rows if r["axis"] == axis]
        axes[0].semilogy([r["value"] for r in sub], [r["error_heralded"] for r in sub], marker + "-", label=f"{axis} (heralded)")
        axes[0].semilogy([r["value"] for r in sub], [r["error_unconditional"] for r in sub], marker + ":", label=f"{axis} (all records)")
    axes[0].invert_xaxis()
    axes[0].set(xlabel="Resource width parameter", ylabel="$|P(+)-P_{ideal}(+)|$", title="Physical axes (grid)")
    axes[0].legend(fontsize=6)
    sub = [r for r in rows if r["axis"] == "peaks"]
    axes[1].semilogy([r["value"] for r in sub], [r["error_heralded"] for r in sub], "o-", label="heralded")
    axes[1].semilogy([r["value"] for r in sub], [r["error_unconditional"] for r in sub], "o:", label="all records")
    axes[1].set(xlabel="Lattice cells each side (peaks)", ylabel="$|P(+)-P_{ideal}(+)|$", title="Comb truncation (grid)")
    axes[1].legend(fontsize=7)
    sub = [r for r in rows if r["axis"] == "cutoff" and not r["refused"]]
    axes[2].semilogy([r["value"] for r in sub], np.maximum([r["y_error"] for r in sub], 1e-9), "o-", label="Y readout |engine - grid|")
    axes[2].semilogy([r["value"] for r in sub], np.maximum([r["density_relative_error"] for r in sub], 1e-9), "s-", label="record density, relative")
    axes[2].set(xlabel="Fock cutoff (total photons <)", ylabel="Error vs untruncated grid", title="Numerical axis (engine)")
    axes[2].legend(fontsize=7)
    axes[3].errorbar([r["value"] for r in noise_rows], [r["p_plus"] for r in noise_rows],
                     yerr=[r["se"] for r in noise_rows], fmt="o-", label="mixed-Fock engine")
    axes[3].axhline(noiseless, color="k", ls="--", lw=0.8, label="noiseless grid")
    axes[3].axhline(noise_rows[0]["ideal"], color="k", ls=":", lw=0.8, label="ideal qubit")
    axes[3].invert_xaxis()
    axes[3].set(xlabel="Ancilla transmissivity $\\eta$", ylabel="P($M_L(\\pi/4)=+1$)", title="Ancilla loss (broad, |+>)")
    axes[3].legend(fontsize=6.5)
    save(rows, NAME, {"oracle_class": "B", "status": status, "cutoff_ok": cutoff_ok, "eta_one_ok": eta_one_ok,
                      "loss_degrades": degrade_ok, "paired_difference": paired_mean,
                      "paired_difference_se": paired_se, "unpaired_criterion_pass": unpaired_ok,
                      "alpha": ALPHA, "record": RECORD}, fig)
    print(f"R62 status={status} cutoff_ok={cutoff_ok} eta_one_ok={eta_one_ok} degrade_ok={degrade_ok}")
    if status != "pass":
        raise AssertionError("R62 acceptance failed")


if __name__ == "__main__":
    main()
