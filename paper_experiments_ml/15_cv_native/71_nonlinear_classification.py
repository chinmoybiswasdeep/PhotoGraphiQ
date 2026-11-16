"""R71 (Figure CVML-4): where does nonlinear classification come from in CV models?

Hypothesis: a Gaussian program with displacement encoding and a linear mean readout is affine
in the features, so with a logistic head it draws a linear boundary and must match classical
linear logistic regression; the circle task becomes learnable only through an identified
nonlinearity: (1) classical nonlinear feature encoding, (2) input-dependent Gaussian gates
(rotation encoding: means offset (cos sx, sin sx)), (3) covariance-derived observables
(squeezing encoding read out through q variances exp(+-2 s x)), (4) a classical nonlinear
(quadratic) head on the quantum features, (5) genuinely non-Gaussian processing (per-mode
displacement + Kerr on the native Fock engine, mean readout, linear logistic head).

Method: deterministic circles split (100/30/70, radius 0.62) and linearly separable blobs
(60/20/40). Gaussian models are 2x3 CVCluster clusters (physical, r = 1, exact moments); the
non-Gaussian model is a two-mode Kerr program at Fock cutoff 24, its trainable
displacements bounded to |b| <= 2 (energy budget) (retained norm reported).
Baselines: linear and quadratic logistic regression. One initialisation seed (0) per model.

Acceptance (structural): Gaussian displacement + linear readout matches linear logistic test
accuracy within 0.05 on both tasks (it is affine); on blobs, every Gaussian head (logistic,
physical homodyne) is within 0.05 of the classical baseline. Accuracies of (1)-(5) are
reported descriptively: they attribute the nonlinearity, they do not claim an advantage.

Revision note: the first run trained every model with Adam (Kerr: 40 epochs, lr 0.15) and the
Kerr classifier stayed at the majority-class accuracy (0.63). A one-dimensional probe showed
under-training rather than missing expressivity (the same Kerr program fits x^2 to MSE 4e-4
with L-BFGS, 1e-3 of Adam's), so all models in this study now use L-BFGS (40 iterations) for
a like-for-like comparison. The L-BFGS line search then drove the Kerr program's trainable
displacement beyond the Fock cutoff (the engine refused truncation at cutoffs 16 and 28), so the
displacement is bounded, b = 2 tanh(theta), like the squeezing budget of the Gaussian ansatz.
"""

import numpy as np
from cv_support import (
    PALETTE,
    Timer,
    accuracy,
    blobs,
    circles,
    logistic,
    plt,
    quadratic_features,
    save,
)

from photographiqml.cv import CVClassifier, CVCluster, CVFeatureMap, NonGaussianProgram

NAME = "R71_nonlinear_classification"
EPOCHS = 40


def cluster():
    return CVCluster(2, 3, squeezing=1.0)


def gaussian(fmap=None, readout="q", head="logistic"):
    return CVClassifier(cluster(), fmap, readout=readout, head=head, mode="physical", epochs=EPOCHS,
                        optimizer="lbfgs", seed=0)


def kerr_model():
    program = NonGaussianProgram(2)
    for j in range(2):
        program.displace(j, f"b{j}", 0.0, bound=2.0).kerr(j, f"k{j}").rotate(j, f"r{j}")
    return CVClassifier(program, CVFeatureMap(2, scale=1.0), readout="q", cutoff=24, epochs=EPOCHS,
                        optimizer="lbfgs", init_scale=0.3, seed=0, fd_step=1e-5)


def square(X):
    return 2 * X**2 - 1  # classical nonlinear preprocessing, rescaled to [-1, 1]


def main():
    rows, maps = [], {}
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41)), -1).reshape(-1, 2)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = circles()
    baselines = {"linear logistic (classical)": logistic(Xtr, ytr, Xte),
                 "quadratic logistic (classical)": logistic(quadratic_features(Xtr), ytr, quadratic_features(Xte))}
    for name, proba in baselines.items():
        rows.append({"task": "circles", "model": name, "source": "classical baseline", "test_accuracy": accuracy(proba, yte)})
    candidates = {
        "Gaussian, displacement, linear readout": (gaussian(), lambda X: X, "affine (no nonlinearity)"),
        "(1) classical preprocessing x^2": (gaussian(), square, "classical feature encoding"),
        "(2) rotation encoding": (gaussian(CVFeatureMap(2, "rotation", scale=2.0, offset=1.5), "qp"), lambda X: X,
                                  "input-dependent Gaussian gates"),
        "(3) squeezing encoding, variances": (gaussian(CVFeatureMap(2, "squeezing", scale=0.8, max_squeezing=1.0), "moments"),
                                              lambda X: X, "covariance observables"),
        "(5) Kerr (non-Gaussian)": (kerr_model(), lambda X: X, "non-Gaussian quantum"),
    }
    fitted = {}
    for name, (model, prep, source) in candidates.items():
        with Timer() as timer:
            model.fit(prep(Xtr), ytr, prep(Xva), yva)
        score = model.score(prep(Xte), yte)
        fitted[name] = (model, prep)
        extra = {"min_retained_norm": float(min(model.retained_norms))} if hasattr(model, "retained_norms") else {}
        rows.append({"task": "circles", "model": name, "source": source, "test_accuracy": score["accuracy"],
                     "test_log_loss": score["log_loss"], "seconds": timer.seconds, **extra})
        maps[name] = model.predict_proba(prep(grid))[:, 1]
        print(f"circles {name}: test accuracy {score['accuracy']:.3f} ({timer.seconds:.0f}s)", flush=True)
    # (4) classical quadratic head on the affine quantum features of the Gaussian model
    base, _ = fitted["Gaussian, displacement, linear readout"]
    features = lambda X: quadratic_features(np.real(base.features(base.params, X)))  # noqa: E731
    proba = logistic(features(Xtr), ytr, features(Xte))
    rows.append({"task": "circles", "model": "(4) quadratic classical head", "source": "classical nonlinear head",
                 "test_accuracy": accuracy(proba, yte)})
    maps["(4) quadratic classical head"] = logistic(features(Xtr), ytr, features(grid))

    (Btr, btr), (Bva, bva), (Bte, bte) = blobs()
    blob_acc = {"linear logistic (classical)": accuracy(logistic(Btr, btr, Bte), bte)}
    for name, model in (("Gaussian cluster, logistic head", gaussian()),
                        ("Gaussian, physical homodyne head", CVClassifier(cluster(), head="homodyne", mode="physical",
                                                                          epochs=EPOCHS, optimizer="lbfgs", seed=0))):
        blob_acc[name] = model.fit(Btr, btr, Bva, bva).score(Bte, bte)["accuracy"]
    for name, value in blob_acc.items():
        rows.append({"task": "blobs", "model": name, "test_accuracy": value})

    acc = {r["model"]: r["test_accuracy"] for r in rows if r["task"] == "circles"}
    checks = {
        "affine_matches_linear_logistic": abs(acc["Gaussian, displacement, linear readout"]
                                              - acc["linear logistic (classical)"]) <= 0.05,
        "blobs_gaussian_heads": all(abs(v - blob_acc["linear logistic (classical)"]) <= 0.05
                                    for v in blob_acc.values()),
    }
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig = pyplot.figure(figsize=(7.0, 5.6))
    names = ["Gaussian, displacement, linear readout", "(1) classical preprocessing x^2", "(2) rotation encoding",
             "(3) squeezing encoding, variances", "(4) quadratic classical head", "(5) Kerr (non-Gaussian)"]
    for i, name in enumerate(names):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.contourf(grid[:, 0].reshape(41, 41), grid[:, 1].reshape(41, 41), maps[name].reshape(41, 41),
                    levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.75)
        ax.contour(grid[:, 0].reshape(41, 41), grid[:, 1].reshape(41, 41), maps[name].reshape(41, 41),
                   levels=[0.5], colors="k", linewidths=0.8)
        ax.scatter(*Xte.T, c=yte, cmap="RdBu_r", s=5, edgecolors="k", linewidths=0.2)
        ax.set_title(f"{name}\nacc {acc[name]:.2f}", fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    ax = fig.add_subplot(3, 1, 3)
    order = ["linear logistic (classical)", "quadratic logistic (classical)"] + names
    colours = [PALETTE["baseline"]] * 2 + [PALETTE["physical"], PALETTE["baseline"], PALETTE["physical"],
                                           PALETTE["physical"], PALETTE["baseline"], PALETTE["nongaussian"]]
    ax.barh(range(len(order)), [acc[n] for n in order], color=colours)
    ax.set_yticks(range(len(order)), order, fontsize=6)
    ax.axvline(0.5, color="k", ls=":", lw=0.6)
    ax.set_xlim(0.4, 1.0)
    ax.set(xlabel="Circles test accuracy (grey: classical, orange: Gaussian quantum, red: non-Gaussian)",
           title="Attribution of the nonlinearity")
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks, "blobs": blob_acc,
                      "circles": acc, "seed": 0, "epochs": EPOCHS}, fig, )
    print(f"R71 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R71 acceptance failed")


if __name__ == "__main__":
    main()
