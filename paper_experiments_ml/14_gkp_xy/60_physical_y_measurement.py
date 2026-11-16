"""R60: Physical logical-Y measurement on finite-energy GKP (Experiment A).

Scientific question: does pg.GKPLogicalXYMeasurement at alpha = pi/2 measure the encoded Y on
all six cardinal logical inputs, both through the Gaussian Clifford shear (S^dagger followed
by the p-homodyne X readout) and through Y-eigenstate injection (finite GKP |+-i~> ancilla,
logical CNOT, ancilla q homodyne; failure leaves an exact Z frame flip)?

Levels kept separate: ideal qubit (logical reference), untruncated position-grid model
(independent finite-energy reference, oracle class B/D: separate gates, no Fock truncation),
native Fock engine (system under test). Clifford readouts are deterministic, so the engine
value is the exact readout probability of the pre-readout state; injection readouts average
the exact conditional readout probability over sampled ancilla records (Rao-Blackwell).

Acceptance: Clifford |engine - reference| <= 2e-3 for every state and resource;
injection |engine - reference| <= 4 SE + 2e-3; for the Y eigenstates the finite-energy error
|reference - ideal| decreases monotonically from broad to sharp resources.
"""

import numpy as np
from xy_support import CARDINAL, RESOURCES, grid, ideal_probabilities, plt, run_single, save

NAME = "R60_physical_y_measurement"
SHOTS = 16
INJECTION_RESOURCES = ("broad", "medium")


def main():
    rows, failures = [], []
    for resource in RESOURCES:
        model = grid(resource)
        for label, logical in CARDINAL.items():
            ideal = ideal_probabilities(logical, np.pi / 2)[0]
            state = model.encode(*logical)
            for protocol in ("clifford", "injection"):
                if protocol == "injection" and resource not in INJECTION_RESOURCES:
                    continue
                reference = model.outcome_probabilities(state, np.pi / 2, protocol=protocol, max_rounds=4)[0]
                shots = 1 if protocol == "clifford" else SHOTS
                records, seconds = run_single(resource, np.pi / 2, logical, protocol=protocol, shots=shots, seed=60)
                values = np.array([r.readout_probabilities[0] for r in records])
                engine = float(values.mean())
                se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                tolerance = 2e-3 + 4 * se
                ok = abs(engine - reference) <= tolerance
                if not ok:
                    failures.append((resource, label, protocol))
                rows.append(
                    {
                        "resource": resource,
                        "state": label,
                        "protocol": protocol,
                        "ideal_p_plus": ideal,
                        "reference_p_plus": reference,
                        "engine_p_plus": engine,
                        "engine_se": se,
                        "shots": shots,
                        "truncation_error": abs(engine - reference),
                        "finite_energy_error": abs(reference - ideal),
                        "mean_rounds": float(np.mean([len(r.rounds) for r in records])),
                        "seconds_per_shot": float(np.mean(seconds)),
                        "min_retained_norm": min(
                            (r.retained_norm_min for r in records if r.retained_norm_min is not None), default=None
                        ),
                        "tolerance": tolerance,
                        "pass": ok,
                    }
                )
                print(f"{resource:6s} {label:2s} {protocol:9s} ideal={ideal:.3f} ref={reference:.4f} "
                      f"eng={engine:.4f}+-{se:.4f}", flush=True)
    # Finite-energy error on the Y eigenstates must shrink with resource quality.
    monotone = True
    for label in ("+i", "-i"):
        errors = [r["finite_energy_error"] for r in rows if r["state"] == label and r["protocol"] == "clifford"]
        monotone &= all(b < a for a, b in zip(errors, errors[1:]))
    status = "pass" if not failures and monotone else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 3.8))
    labels = list(CARDINAL)
    width = 0.25
    for k, resource in enumerate(RESOURCES):
        sub = [r for r in rows if r["resource"] == resource and r["protocol"] == "clifford"]
        xs = np.arange(len(labels)) + (k - 1) * width
        axes[0].bar(xs, [r["engine_p_plus"] for r in sub], width, label=f"{resource} (engine)")
        axes[0].plot(xs, [r["reference_p_plus"] for r in sub], "k_", markersize=9)
    axes[0].plot(np.arange(len(labels)), [ideal_probabilities(CARDINAL[s], np.pi / 2)[0] for s in labels],
                 "o", mfc="none", color="k", label="ideal qubit")
    axes[0].set(xticks=np.arange(len(labels)), xticklabels=[f"|{s}>" for s in labels],
                ylabel="P(Y = +1)", title="Clifford Y readout (bars: engine, dashes: grid reference)")
    axes[0].legend(fontsize=7)
    for protocol, marker in (("clifford", "o"), ("injection", "s")):
        sub = [r for r in rows if r["protocol"] == protocol and r["state"] in ("+i", "-i")]
        for kind, color in (("finite_energy_error", "C0"), ("truncation_error", "C3")):
            xs = [list(RESOURCES).index(r["resource"]) for r in sub]
            axes[1].semilogy(xs, np.maximum([r[kind] for r in sub], 1e-7), marker, color=color,
                             mfc="none" if protocol == "injection" else color,
                             label=f"{kind.replace('_', ' ')} ({protocol})")
    axes[1].set(xticks=range(len(RESOURCES)), xticklabels=list(RESOURCES), ylabel="|difference| in P(Y = +1)",
                title="Y eigenstates: finite-energy vs Fock-truncation error")
    axes[1].legend(fontsize=6.5)
    save(rows, NAME, {"oracle_class": "B", "status": status, "monotone_finite_energy": monotone,
                      "failures": failures, "resources": RESOURCES, "shots": SHOTS}, fig)
    print(f"R60 status={status} failures={failures} monotone={monotone}")
    if status != "pass":
        raise AssertionError(f"R60 failed: {failures}, monotone={monotone}")


if __name__ == "__main__":
    main()
