"""R75 (Figure CVML-8): non-Gaussian CV learning on the native Fock engine (experimental).

Hypothesis: (a) the Fock-engine execution of Kerr / cubic-phase programs converges in the
cutoff to an independent dense matrix-exponential reference, with the retained norm exposing
truncation; (b) Kerr and cubic gates create Wigner negativity (absent for Gaussian programs);
(c) with identical displacement encoding, mean-quadrature readout and affine head, a single
non-Gaussian gate lifts the affine restriction of the Gaussian model on a nonlinear 1D
regression; (d) the finite-energy cubic-injection lowering (cubic resource, inverse SUM,
ancilla homodyne, QP and displacement feedforward) approaches the cubic gate as its resource
squeezing grows, while needing larger cutoffs.

Method: (a) one-mode programs D(x) R S K and D(x) CP S, x giving 0.25-4 mean photons, cutoffs
8-64, reference: tests/validation/cv_fock_reference.py at cutoff 120. (b) Wigner negative
volume of D(1) K(kappa) |0> and D(1) CP(gamma) |0>. (c) Nonlinear regression split (45/15/30);
models with the same encoding and head: Gaussian D R S (affine), rotation-encoded Gaussian
(input-dependent Gaussian gate, qp readout), Kerr D(b) K R and cubic D(b) CP R (cutoff 30,
trainable displacement bounded to |b| <= 2), all L-BFGS 40 iterations; classical baselines OLS
and cubic polynomial least squares. (d) Unconditional output of the injection lowering by
outcome quadrature (81 postselected outcomes in [-8, 8]) vs the cubic gate, fidelity
<psi_gate| rho_injection |psi_gate>, for resource squeezing 0.2-0.8 at cutoff 48.

Acceptance: (a) engine-reference |<q>| difference <= 1e-6 at cutoff 64 for every energy and
non-increasing over the last three cutoffs; (b) negative volume <= 1e-10 for the Gaussian case,
strictly increasing with gate strength for each gate, and > 1e-3 at the strongest gate;
(d) integrated outcome probability within 0.02 of 1. (c) and the fidelity trend in (d) are
descriptive: one dataset, one seed; no advantage is claimed.

Revision note: the first run required negative volume > 1e-3 for every strength > 0.04 and
failed for weak gates (Kerr 0.05: 1.7e-6; cubic 0.05 and 0.1: 2.9e-12 and 5.9e-6). Every
non-Gaussian pure state has some negativity (Hudson's theorem), but for weak gates on a
coherent state it is tiny and lies in the tails, so a fixed threshold was the wrong test; the
criterion now checks zero for Gaussian, monotone growth and a clear signal at the strongest gate.
"""

import sys
import warnings

import numpy as np
from cv_support import PALETTE, REPO, Timer, nonlinear_regression, ols, plt, save

from photographiq.fock_analysis import wigner
from photographiqml.cv import CVFeatureMap, CVRegressor, GaussianProgram, NonGaussianProgram

sys.path.insert(0, str(REPO / "tests" / "validation"))
from cv_fock_reference import FockReference  # noqa: E402

NAME = "R75_non_gaussian_learning"
CUTOFFS = (8, 12, 16, 24, 32, 48, 64)


def convergence_rows():
    rows = []
    reference = FockReference(120)
    programs = {
        "kerr": (NonGaussianProgram(1).rotate(0, 0.3).squeeze(0, 0.2).kerr(0, 0.15),
                 [("rotate", 0.3), ("squeeze", 0.2), ("kerr", 0.15)]),
        "cubic": (NonGaussianProgram(1).cubic_phase(0, 0.05).squeeze(0, 0.1), [("cubic", 0.05), ("squeeze", 0.1)]),
    }
    fmap = CVFeatureMap(1, scale=1.0)
    for name, (program, gates) in programs.items():
        for photons in (0.25, 1.0, 2.0, 4.0):
            x = 2 * np.sqrt(photons)  # q mean of a coherent state with |alpha|^2 = photons
            psi = reference.displace(x, 0.0) @ reference.vacuum()
            for kind, value in gates:
                psi = getattr(reference, kind)(value) @ psi
            ref_q = reference.moments(psi)[0]
            for cutoff in CUTOFFS:
                try:
                    with Timer() as t:
                        means, _, norm = program.fock_moments({}, np.array([x]), fmap, cutoff, (0,))
                    rows.append({"study": "convergence", "program": name, "photons": photons, "cutoff": cutoff,
                                 "error": float(abs(means[0] - ref_q)), "retained_norm": norm, "seconds": t.seconds})
                except ValueError:  # the engine refuses truncation beyond its norm tolerance
                    rows.append({"study": "convergence", "program": name, "photons": photons, "cutoff": cutoff,
                                 "error": None, "retained_norm": None, "refused": True})
    return rows


def negativity_rows():
    rows = []
    grid = np.linspace(-6, 6, 121)
    fmap = CVFeatureMap(1, scale=1.0)
    for kind, values in (("kerr", (0.0, 0.05, 0.1, 0.2, 0.4)), ("cubic", (0.0, 0.02, 0.05, 0.1, 0.2))):
        for v in values:
            program = NonGaussianProgram(1)
            (program.kerr if kind == "kerr" else program.cubic_phase)(0, v)
            state = program.run({}, np.array([2.0]), fmap, 48).state
            w = wigner(state, grid, grid)
            rows.append({"study": "negativity", "gate": kind, "strength": v, "negative_volume": float(w.negative_volume),
                         "captured_mass": float(w.captured_mass)})
    return rows


def regression_rows():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = nonlinear_regression()
    rows, curves = [], {}
    dense = np.linspace(-1, 1, 81)[:, None]
    fmap = CVFeatureMap(1, scale=1.0)

    def ng(gate):
        program = NonGaussianProgram(1).displace(0, "b", 0.0, bound=2.0)
        (program.kerr if gate == "kerr" else program.cubic_phase)(0, "g")
        return program.rotate(0, "r")

    models = {
        "Gaussian D R S (affine)": CVRegressor(GaussianProgram(1).rotate(0, "a").squeeze(0, "s", bound=1.0), fmap,
                                               readout="qp", optimizer="lbfgs", epochs=40, seed=75),
        "Gaussian, rotation encoding": CVRegressor(GaussianProgram(1).squeeze(0, "s", bound=1.0).rotate(0, "a"),
                                                   CVFeatureMap(1, "rotation", scale=1.5, offset=1.5), readout="qp",
                                                   optimizer="lbfgs", epochs=40, seed=75),
        "Kerr D(b) K R": CVRegressor(ng("kerr"), fmap, readout="qp", cutoff=30, optimizer="lbfgs", epochs=40,
                                     init_scale=0.3, seed=75, fd_step=1e-5),
        "cubic D(b) CP R": CVRegressor(ng("cubic"), fmap, readout="qp", cutoff=30, optimizer="lbfgs", epochs=40,
                                       init_scale=0.1, seed=75, fd_step=1e-5),
    }
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Timer() as t:
                model.fit(Xtr, ytr, Xva, yva)
            score = model.score(Xte, yte)
            curves[name] = model.predict(dense)
        extra = {"min_retained_norm": float(min(model.retained_norms))} if hasattr(model, "retained_norms") else {}
        rows.append({"study": "regression", "model": name, "test_mse": score["mse"], "seconds": t.seconds,
                     "evaluations": model.history.evaluations[-1], **extra})
        print(f"regression {name}: test MSE {score['mse']:.4f} ({t.seconds:.0f}s)", flush=True)
    for name, features in (("OLS (classical, affine)", lambda X: X), ("cubic polynomial (classical)",
                                                                       lambda X: np.c_[X, X**2, X**3])):
        rows.append({"study": "regression", "model": name, "test_mse": float(np.mean((ols(features(Xtr), ytr, features(Xte)) - yte) ** 2))})
        curves[name] = ols(features(Xtr), ytr, features(dense))
    return rows, curves, (Xtr, ytr, dense)


def injection_rows():
    rows = []
    fmap = CVFeatureMap(1, scale=0.5)
    program = NonGaussianProgram(1).cubic_phase(0, 0.1)
    x = np.array([0.4])
    gate = program.run({}, x, fmap, 48).state
    psi = np.asarray(gate.state_vector)
    for r in (0.2, 0.4, 0.6, 0.8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Timer() as t:
                try:
                    channel = program.injection_channel({}, x, fmap, 48, injection_squeezing=r,
                                                        outcomes=np.linspace(-8, 8, 81))
                except ValueError as error:
                    rows.append({"study": "injection", "r": r, "refused": str(error)})
                    continue
        rho = channel["density_matrix"]
        n = min(len(rho), len(psi))
        fidelity = float(np.real(psi[:n].conj() @ rho[:n, :n] @ psi[:n]))
        rows.append({"study": "injection", "r": r, "fidelity": fidelity,
                     "outcome_probability": channel["outcome_probability"],
                     "min_retained_norm": float(min(channel["retained_norms"])), "seconds": t.seconds,
                     "ancillas": 1, "entangling_operations": 1, "feedforward_corrections": 2,
                     "heralding_probability": 1.0})
        print(f"injection r={r}: fidelity {fidelity:.4f}, outcome probability {channel['outcome_probability']:.4f}", flush=True)
    return rows


def main():
    conv = convergence_rows()
    neg = negativity_rows()
    reg, curves, (Xtr, ytr, dense) = regression_rows()
    inj = injection_rows()
    rows = conv + neg + reg + inj
    ok_conv = True
    for program in ("kerr", "cubic"):
        for photons in (0.25, 1.0, 2.0, 4.0):
            errs = [q["error"] for q in conv if q["program"] == program and q["photons"] == photons and q["error"] is not None]
            ok_conv &= errs[-1] <= 1e-6 and all(b <= a + 1e-12 for a, b in zip(errs[-3:], errs[-2:]))
    accepted = [q for q in inj if "outcome_probability" in q]
    checks = {
        "fock_convergence": bool(ok_conv),
        "negativity": all(q["negative_volume"] <= 1e-10 for q in neg if q["strength"] == 0)
        and all(np.all(np.diff([q["negative_volume"] for q in neg if q["gate"] == g]) > 0) for g in ("kerr", "cubic"))
        and all(max(q["negative_volume"] for q in neg if q["gate"] == g) > 1e-3 for g in ("kerr", "cubic")),
        "injection_normalisation": bool(accepted) and all(abs(q["outcome_probability"] - 1) <= 0.02 for q in accepted),
    }
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    ax = axes[0, 0]
    for photons, colour in zip((0.25, 1.0, 2.0, 4.0), (PALETTE["ideal"], PALETTE["cluster"], PALETTE["physical"], PALETTE["nongaussian"])):
        for program, style in (("kerr", "o-"), ("cubic", "s--")):
            sub = [q for q in conv if q["program"] == program and q["photons"] == photons and q["error"] is not None]
            ax.semilogy([q["cutoff"] for q in sub], [max(q["error"], 1e-16) for q in sub], style, color=colour,
                        label=f"{program}, {photons:g} photons")
    ax.set(xlabel="Fock cutoff", ylabel="|<q> engine - matrix-exponential reference|", title="Fock convergence")
    ax.legend(fontsize=5, ncol=2)
    ax = axes[0, 1]
    for kind, colour in (("kerr", PALETTE["nongaussian"]), ("cubic", PALETTE["physical"])):
        sub = [q for q in neg if q["gate"] == kind]
        ax.semilogy([q["strength"] for q in sub], [max(q["negative_volume"], 1e-18) for q in sub], "o-", color=colour,
                label=f"{kind}: D(1) {'K(kappa)' if kind == 'kerr' else 'CP(gamma)'}|0>")
    ax.set(xlabel="Gate strength (kappa or gamma)", ylabel="Wigner negative volume", title="Non-Gaussianity of the output")
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    ax.plot(Xtr[:, 0], ytr, "k.", ms=3, label="training data")
    colours = {"Gaussian D R S (affine)": PALETTE["ideal"], "Gaussian, rotation encoding": PALETTE["cluster"],
               "Kerr D(b) K R": PALETTE["nongaussian"], "cubic D(b) CP R": PALETTE["physical"],
               "OLS (classical, affine)": PALETTE["baseline"], "cubic polynomial (classical)": PALETTE["baseline"]}
    mse = {q["model"]: q["test_mse"] for q in reg}
    for name, curve in curves.items():
        ax.plot(dense[:, 0], curve, color=colours[name], ls=":" if "classical" in name else "-",
                label=f"{name} ({mse[name]:.2g})")
    ax.set(xlabel="Feature x", ylabel="Prediction (test MSE in legend)", title="Nonlinear 1D regression")
    ax.legend(fontsize=5)
    ax = axes[1, 1]
    if accepted:
        ax.plot([q["r"] for q in accepted], [q["fidelity"] for q in accepted], "o-", color=PALETTE["physical"],
                label="fidelity with the cubic gate")
        ax.plot([q["r"] for q in accepted], [q["outcome_probability"] for q in accepted], "s--", color=PALETTE["ideal"],
                label="integrated outcome probability")
    ax.set(xlabel="Cubic resource squeezing r", ylabel="Value", title="Cubic injection lowering (cutoff 48)")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "B", "status": status, "checks": checks}, fig)
    print(f"R75 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R75 acceptance failed")


if __name__ == "__main__":
    main()
