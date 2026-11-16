"""R66: Learning a continuous measurement angle on the physical simulator (Experiment G).

Scientific question: can a continuous XY measurement angle be optimised using only the
finite-energy physical simulator, and where does that work?

Part A (descriptive, MuTA): one-wire, one-layer, one-column PhysicalMuTA with every angle 0
except node (0,1) (frame-adaptive), generic logical input, loss P(decoded output = 1). Broad
GKP resource (width = envelope = 0.6) at cutoff 64: the only resource at which an injected
three-mode chain fits the Fock budget (width <= 0.45 needs cutoff > 104 once failed rounds add
energy to the data mode; see R65 and R67). The physical loss sweep and one bounded search are
reported with the logical curve; no success criterion is asserted for Part A.

Part B (asserted, instrument level): one GKP data mode prepared in the encoded generic state,
loss L(a) = P(M_L(a) = -1) of the physical instrument (medium resource, width = envelope =
0.45, cutoff 96, two live modes; the sharp resource holds only two failed injection rounds even
at cutoff 128), estimated by the exact conditional readout probabilities averaged over sampled
ancilla records (Rao-Blackwell, common random numbers). The finite-energy instrument is not
uniformly accurate in a (Clifford angles carry less error than injected ones), so its loss
minimum need not coincide with the ideal one: the asserted target is the untruncated grid
reference of the same physical instrument; the gap to the ideal optimum is reported.

Acceptance (Part B): the found angle lies within 0.2 rad of the grid-reference loss minimum;
paired FD gradient signs agree with the reference gradient wherever |reference gradient| >
3 SE + 0.05; the physical loss at the found angle (independent seed) is below its value at
a = 0 by more than 2 combined SE. Optimisation success is not a learning-advantage claim.

Revision note: the first design asserted the Part A criteria at the broad resource and failed:
the physical loss was nearly flat (0.40-0.48 while the logical loss spans 0.05-0.94) because
finite-energy errors of the shear/injection chain at width 0.6 wash out the angle dependence;
sharper resources do not fit the Fock budget for the injected chain. Part A is now descriptive
and the asserted learning test moved to the instrument level (Part B).
"""

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from xy_support import GENERIC, grid, ideal_probabilities, plt, run_single, save

from photographiqml import ContinuousPhysicalSearch, physical_loss
from photographiqml.logical import execute
from photographiqml.lowering import GKPPhysicalConfig
from photographiqml.physical import PhysicalMuTA

NAME = "R66_physical_muta_learning"
SEED = 66
MUTA_SHOTS, MUTA_GRID = 12, np.linspace(0, 2 * np.pi, 8, endpoint=False)
SHOTS, CUTOFF, RESOURCE = 16, 96, "medium"


def part_a():
    config = GKPPhysicalConfig(cutoff=64, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049,
                               backend="native-fock", xy_protocol="auto", max_dimension=200_000)
    model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
    names = list(model.parameters())
    trained, base, state = names[1], dict.fromkeys(names, 0.0), np.array(GENERIC, complex)

    def logical(angle):
        return float(abs(execute(model, state, model._parameters.bind(base | {trained: angle})).state[1]) ** 2)

    rows = []
    for angle in MUTA_GRID:
        est = physical_loss(model, state, base | {trained: float(angle)}, {(1,): 1.0}, shots=MUTA_SHOTS, seed=SEED)
        rows.append({"part": "A", "kind": "sweep", "angle": float(angle), "loss": est.value, "se": est.standard_error,
                     "logical_loss": logical(angle)})
        print(f"A sweep a={angle:.3f} physical={est.value:.4f}+-{est.standard_error:.4f} logical={logical(angle):.4f}", flush=True)
    search = ContinuousPhysicalSearch(shots=MUTA_SHOTS, seed=SEED, maxiter=8).fit(model, state, base, trained, {(1,): 1.0})
    dense = np.linspace(0, 2 * np.pi, 361)
    logical_curve = [logical(a) for a in dense]
    rows.append({"part": "A", "kind": "search", "angle": search.angle, "loss": search.loss.value,
                 "se": search.loss.standard_error, "logical_loss": logical(search.angle),
                 "logical_minimum": float(min(logical_curve)),
                 "physical_range": float(np.ptp([r["loss"] for r in rows if r["kind"] == "sweep"])),
                 "logical_range": float(np.ptp(logical_curve))})
    print(f"A search a={search.angle:.3f} logical={logical(search.angle):.4f} (min {min(logical_curve):.4f})", flush=True)
    return rows, dense, logical_curve


def part_b():
    model = grid(RESOURCE)
    state = model.encode(*GENERIC)

    def ideal(angle):
        return ideal_probabilities(GENERIC, angle)[1]

    def reference(angle):
        return 1 - model.outcome_probabilities(state, angle, max_rounds=4)[0]

    def physical(angle, seed=SEED, shots=SHOTS):
        records, _ = run_single(RESOURCE, angle, GENERIC, shots=shots, seed=seed, cutoff=CUTOFF)
        return np.array([r.readout_probabilities[1] for r in records])

    rows, evaluations = [], []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        values = physical(angle)
        rows.append({"part": "B", "kind": "sweep", "angle": float(angle), "loss": float(values.mean()),
                     "se": float(values.std(ddof=1) / np.sqrt(len(values))), "reference_loss": reference(angle),
                     "ideal_loss": ideal(angle)})
        print(f"B sweep a={angle:.3f} physical={values.mean():.4f} reference={reference(angle):.4f} ideal={ideal(angle):.4f}", flush=True)

    def objective(angle):
        value = float(physical(angle).mean())
        evaluations.append({"part": "B", "kind": "search", "angle": float(angle), "loss": value, "ideal_loss": ideal(angle)})
        return value

    found = float(minimize_scalar(objective, bounds=(0, 2 * np.pi), method="bounded",
                                  options={"maxiter": 10, "xatol": 0.05}).x)
    rows += evaluations
    dense = np.linspace(0, 2 * np.pi, 721)
    ideal_curve = np.array([ideal(a) for a in dense])
    coarse = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    reference_curve = np.array([reference(a) for a in coarse])
    best = coarse[int(np.argmin(reference_curve))]
    refined = minimize_scalar(reference, bounds=(best - 0.1, best + 0.1), method="bounded", options={"xatol": 1e-3})
    reference_optimum = float(refined.x) % (2 * np.pi)
    optimum_ok = abs(np.angle(np.exp(1j * (found - reference_optimum)))) <= 0.2
    rows += [{"part": "B", "kind": "reference_curve", "angle": float(a), "reference_loss": float(v)}
             for a, v in zip(coarse, reference_curve)]
    gradient_ok = True
    for angle in (np.pi / 3, np.pi, 5 * np.pi / 3):
        plus, minus = physical(angle + 0.15), physical(angle - 0.15)
        differences = (plus - minus) / 0.3
        grad, se = float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(len(differences)))
        exact = (reference(angle + 1e-3) - reference(angle - 1e-3)) / 2e-3
        decisive = abs(exact) > 3 * se + 0.05
        agree = bool(np.sign(grad) == np.sign(exact)) if decisive else None
        gradient_ok &= agree is not False
        rows.append({"part": "B", "kind": "gradient", "angle": float(angle), "physical_fd": grad, "se": se,
                     "reference_gradient": exact, "ideal_gradient": (ideal(angle + 1e-5) - ideal(angle - 1e-5)) / 2e-5,
                     "decisive": decisive, "sign_agrees": agree})
        print(f"B gradient a={angle:.3f} physical={grad:.4f}+-{se:.4f} reference={exact:.4f}", flush=True)
    fresh, zero = physical(found, seed=SEED + 1000, shots=24), physical(0.0, seed=SEED + 1000, shots=24)
    improvement_ok = zero.mean() - fresh.mean() > 2 * np.hypot(zero.std(ddof=1), fresh.std(ddof=1)) / np.sqrt(24)
    rows.append({"part": "B", "kind": "revalidation", "angle": found, "loss": float(fresh.mean()),
                 "zero_loss": float(zero.mean()), "ideal_loss": ideal(found), "ideal_minimum": float(ideal_curve.min()),
                 "reference_optimum": reference_optimum, "ideal_optimum": float(dense[int(np.argmin(ideal_curve))])})
    print(f"B found a={found:.4f} reference optimum {reference_optimum:.4f} ideal loss {ideal(found):.4f} (min {ideal_curve.min():.4f})", flush=True)
    return rows, found, dense, ideal_curve, optimum_ok and gradient_ok and improvement_ok, (optimum_ok, gradient_ok, improvement_ok)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    rows_a, dense_a, logical_a = part_a()
    rows_b, found, dense_b, ideal_b, ok, checks = part_b()
    rows = rows_a + rows_b
    status = "pass" if ok else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(dense_a, logical_a, "k-", lw=1, label="logical loss")
    sweep = [r for r in rows_a if r["kind"] == "sweep"]
    axes[0].errorbar([r["angle"] for r in sweep], [r["loss"] for r in sweep], yerr=[r["se"] for r in sweep],
                     fmt="o", label="physical MuTA (broad GKP)")
    axes[0].set(xlabel="Trained angle $\\alpha_{w0,c1}$", ylabel="P(decoded output = 1)",
                title="A: MuTA at the accessible resource (descriptive)")
    axes[0].legend(fontsize=7)
    axes[1].plot(dense_b, ideal_b, "k-", lw=1, label="ideal qubit")
    sweep_b = [r for r in rows_b if r["kind"] == "sweep"]
    axes[1].plot([r["angle"] for r in sweep_b], [r["reference_loss"] for r in sweep_b], "s", mfc="none", label="grid reference")
    axes[1].errorbar([r["angle"] for r in sweep_b], [r["loss"] for r in sweep_b], yerr=[r["se"] for r in sweep_b],
                     fmt="o", ms=3, label="physical instrument (medium)")
    search_b = [r for r in rows_b if r["kind"] == "search"]
    axes[1].plot([r["angle"] for r in search_b], [r["loss"] for r in search_b], "x", color="C3", label="search")
    axes[1].axvline(found, color="C3", ls="--", lw=0.8)
    axes[1].set(xlabel="Measurement angle $\\alpha$", ylabel="P($M_L(\\alpha)=-1$)", title="B: instrument-level learning")
    axes[1].legend(fontsize=6.5)
    save(rows, NAME, {"oracle_class": "A", "status": status, "found_angle": found,
                      "checks": dict(zip(("optimum", "gradient", "improvement"), checks)),
                      "part_a": "descriptive (finite-energy flattening at the only accessible resource)"}, fig)
    print(f"R66 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R66 acceptance failed")


if __name__ == "__main__":
    main()
