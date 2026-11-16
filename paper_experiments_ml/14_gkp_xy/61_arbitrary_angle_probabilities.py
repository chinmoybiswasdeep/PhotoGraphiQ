"""R61: Arbitrary-angle physical XY probabilities over 0 <= alpha < 2 pi (Experiment B).

Scientific question: does the physical instrument measure M_L(alpha) = cos(alpha) X_L +
sin(alpha) Y_L for every angle, approaching the ideal qubit as the GKP resource improves?

Input: the generic logical state cos(0.4)|0> + e^{0.9i} sin(0.4)|1> (no alpha -> -alpha
symmetry, so a sign error cannot hide). Revision note: a first run at cutoff 80 stopped on
the retained-norm guard (a four-failure trajectory; each failed round adds ~1.4 photons to the
data mode), so the engine cutoff was raised to 96. A second run then failed a pointwise
criterion (heralded TVD decreasing at every angle): at 7 of 66 angles the broad resource's
signed error passes through zero, so its |error| is accidentally tiny there. The criterion was
replaced by the worst-case and mean error over angles; reference values are unchanged. Reference: untruncated grid model, every repeat-
until-success branch enumerated exactly, dense sweep of 64 angles for three resources.
Engine: native-fock at the eight required angles (medium resource, cutoff 96), averaging the
exact conditional readout probability over 32 sampled records (Rao-Blackwell).

Statistics are reported both unconditionally (all records, including heralded failures that
measured the wrong angle) and conditioned on heralded success. Error types are separate:
ideal-vs-reference (finite energy), reference-vs-engine (Fock truncation), SE (sampling).

Acceptance: engine vs reference |difference| <= 4 SE + 2e-3 at every required angle;
the maximum and the mean over angles of the heralded TVD to the ideal decrease broad ->
medium -> sharp, and the sharp maximum is <= 0.03; heralded-success
probability equals 1 - 2^-4 for angles that never become Clifford within four rounds.
"""

import numpy as np
from xy_support import (
    GENERIC,
    REQUIRED_ANGLES,
    RESOURCES,
    grid,
    ideal_probabilities,
    plt,
    run_single,
    save,
)

NAME = "R61_arbitrary_angle_probabilities"
SWEEP = np.linspace(0, 2 * np.pi, 64, endpoint=False)
SHOTS = 32


def reference_row(resource, alpha):
    model = grid(resource)
    rows = model.protocol_map(model.operator(model.encode(*GENERIC)), alpha, max_rounds=4)
    p_plus = sum(r["value"].real for r in rows if r["bit"] == 0)
    success = sum(r["value"].real for r in rows if r["success"])
    heralded = sum(r["value"].real for r in rows if r["bit"] == 0 and r["success"]) / success
    return p_plus, heralded, success


def main():
    rows = []
    for resource in RESOURCES:
        for alpha in sorted(set(SWEEP) | set(REQUIRED_ANGLES)):
            ideal = ideal_probabilities(GENERIC, alpha)[0]
            p_plus, heralded, success = reference_row(resource, alpha)
            rows.append({"kind": "reference", "resource": resource, "alpha": float(alpha), "ideal_p_plus": ideal,
                         "p_plus": p_plus, "heralded_p_plus": heralded, "heralded_success": success,
                         "tvd_unconditional": abs(p_plus - ideal), "tvd_heralded": abs(heralded - ideal)})
        print(f"reference {resource} done", flush=True)
    failures = []
    for alpha in REQUIRED_ANGLES:
        ref = next(r for r in rows if r["kind"] == "reference" and r["resource"] == "medium" and r["alpha"] == alpha)
        records, seconds = run_single("medium", alpha, GENERIC, shots=SHOTS, seed=61, cutoff=96)
        values = np.array([r.readout_probabilities[0] for r in records])
        engine, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
        ok = abs(engine - ref["p_plus"]) <= 4 * se + 2e-3
        if not ok:
            failures.append(float(alpha))
        rows.append({"kind": "engine", "resource": "medium", "alpha": float(alpha), "ideal_p_plus": ref["ideal_p_plus"],
                     "p_plus": engine, "se": se, "reference_p_plus": ref["p_plus"], "shots": SHOTS,
                     "heralded_fraction": float(np.mean([r.heralded_success for r in records])),
                     "mean_rounds": float(np.mean([len(r.rounds) for r in records])),
                     "seconds_per_shot": float(np.mean(seconds)), "pass": ok})
        print(f"engine a={alpha:.4f} eng={engine:.4f}+-{se:.4f} ref={ref['p_plus']:.4f} ideal={ref['ideal_p_plus']:.4f}", flush=True)
    sharp = [r for r in rows if r["kind"] == "reference" and r["resource"] == "sharp"]
    sharp_ok = max(r["tvd_heralded"] for r in sharp) <= 0.03
    summary = {}
    for name in RESOURCES:
        errors = [r["tvd_heralded"] for r in rows if r["kind"] == "reference" and r["resource"] == name]
        summary[name] = {"max": float(max(errors)), "mean": float(np.mean(errors))}
    for key in ("max", "mean"):
        values = [summary[name][key] for name in RESOURCES]
        sharp_ok &= all(b < a for a, b in zip(values, values[1:]))
    print("heralded TVD summary", summary, flush=True)
    generic = [r for r in rows if r["kind"] == "reference" and r["alpha"] in (np.pi / 3, 2 * np.pi / 3)]
    herald_ok = all(abs(r["heralded_success"] - (1 - 2.0**-4)) < 1e-3 for r in generic)
    status = "pass" if not failures and sharp_ok and herald_ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(13, 3.9))
    dense = np.linspace(0, 2 * np.pi, 400)
    axes[0].plot(dense, [ideal_probabilities(GENERIC, a)[0] for a in dense], "k-", lw=1, label="ideal qubit")
    for resource in RESOURCES:
        sub = [r for r in rows if r["kind"] == "reference" and r["resource"] == resource]
        axes[0].plot([r["alpha"] for r in sub], [r["heralded_p_plus"] for r in sub], "-", label=f"{resource} (grid, heralded)")
    engine_rows = [r for r in rows if r["kind"] == "engine"]
    axes[0].errorbar([r["alpha"] for r in engine_rows], [r["p_plus"] for r in engine_rows],
                     yerr=[r["se"] for r in engine_rows], fmt="o", color="k", mfc="white", label="engine, medium (all records)")
    axes[0].set(xlabel="Measurement angle $\\alpha$", ylabel="P($M_L(\\alpha)=+1$)", title="Outcome probability")
    axes[0].legend(fontsize=6.5)
    for resource in RESOURCES:
        sub = [r for r in rows if r["kind"] == "reference" and r["resource"] == resource]
        axes[1].semilogy([r["alpha"] for r in sub], np.maximum([r["tvd_heralded"] for r in sub], 1e-6), label=resource)
    axes[1].set(xlabel="Measurement angle $\\alpha$", ylabel="TVD to ideal (heralded)", title="Finite-energy error")
    axes[1].legend(fontsize=7)
    sub = [r for r in rows if r["kind"] == "reference" and r["resource"] == "medium"]
    axes[2].plot([r["alpha"] for r in sub], [r["heralded_success"] for r in sub], ".-")
    axes[2].axhline(1 - 2.0**-4, color="k", ls="--", lw=0.8, label="$1-2^{-4}$")
    axes[2].set(xlabel="Measurement angle $\\alpha$", ylabel="P(heralded success)", title="Repeat-until-success (4 rounds)")
    axes[2].legend(fontsize=7)
    save(rows, NAME, {"oracle_class": "B", "status": status, "engine_failures": failures, "sharp_heralded_ok": sharp_ok,
                      "herald_probability_ok": herald_ok, "heralded_tvd_summary": summary, "input_state": [str(c) for c in GENERIC], "shots": SHOTS}, fig)
    print(f"R61 status={status} engine_failures={failures} sharp_ok={sharp_ok} herald_ok={herald_ok}")
    if status != "pass":
        raise AssertionError("R61 acceptance failed")


if __name__ == "__main__":
    main()
