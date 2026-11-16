"""R72 (Figure CVML-5): learning under finite squeezing and finite measurement shots.

Hypothesis. For flow-corrected Gaussian CV-MBQC the unconditional output mean equals the
ideal one at every resource squeezing r (R69), so finite squeezing enters a mean-readout model
only through the output covariance X V X^T + Y(r): a prediction estimated from S homodyne shots
has expected squared error (mu - y)^2 + w^T C_qq(r) w / S. Hence (i) the shot-limited test error
of a fixed model falls monotonically with squeezing and with S; (ii) training on the physical
shot estimator (common random numbers) can trade bias for lower output noise at weak squeezing;
(iii) an explicit noise penalty (mean added noise per quadrature, vacuum units, dimensionless
task loss MSE / Var y) lowers the channel noise the optimiser settles on.

Method: linear-regression split; 2x4 CVCluster cluster (q readout of both outputs, affine head).
A: ideal training, evaluated at r in [0.25, 2] with S in {10, 100, 1000}. B: physical training
with S = 100 shots at each r. C: exact-moment physical training with the noise penalty
(lambda = 0.05) at each r. The exact expected test MSE is computed from the moments and checked
against 200 sampled shot evaluations. A physical homodyne-head classifier on blobs reports
single-shot accuracy E[P(correct)] vs r (Born probability of the binned outcome), for a model
trained once in the ideal limit and evaluated at each r, and for a model retrained at each r.
The single-shot accuracy is bounded by the coherent-state encoding: a Gaussian unitary
amplifies signal and input vacuum noise together, so retraining can only suppress the
resource noise (by gain), not the encoding's own shot noise; output gain is recorded.

Acceptance: sampled vs exact expected MSE within 4 SE everywhere; A decreasing in r for each S
and decreasing in S for each r; C's added noise <= A's at every r. (ii) is reported, not
asserted.
"""

import numpy as np
from cv_support import PALETTE, blobs, linear_regression, plt, save

from photographiqml.cv import CVClassifier, CVCluster, CVRegressor

NAME = "R72_finite_squeezing_learning"
SQUEEZING = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
SHOTS = (10, 100, 1000)
DRAWS = 200
EPOCHS = 80


def expected_mse(model, X, y, shots):
    means, covs = model.moments(model.params, X, mode="physical")
    w, b = np.real(model.head[:-1]), np.real(model.head[-1])
    mu = np.real(means[:, 0::2]) @ w + b
    var = np.einsum("j,njk,k->n", w, np.real(covs[:, 0::2, 0::2]), w) / shots
    return float(np.mean((mu - y) ** 2 + var)), float(np.mean((mu - y) ** 2)), float(np.mean(var * shots))


def added_noise(model, r):
    Y = model.program.channel(model.params, mode="physical", squeezing=r)[2]
    return float(np.trace(np.real(Y)) / len(Y))


def variance_split(clf, r):
    """Output q variance of the first readout mode: amplified encoding noise vs resource noise."""
    X, _, Y, _ = clf.program.channel(clf.params, mode="physical", squeezing=float(r))
    encoded = np.real(X @ X.T)[0, 0]  # coherent inputs: V_in = I
    return {"encoding_variance": float(encoded), "resource_variance": float(np.real(Y)[0, 0])}


def build(r, **kw):
    return CVRegressor(CVCluster(2, 4, squeezing=r), mode="physical", epochs=EPOCHS, seed=72, **kw)


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = linear_regression()
    rows = []
    ideal = CVRegressor(CVCluster(2, 4, squeezing=1.0), epochs=EPOCHS, seed=72).fit(Xtr, ytr)
    z_scores = []
    for r in SQUEEZING:
        ideal.squeezing, ideal.mode = float(r), "physical"
        variants = {"A ideal-trained": ideal,
                    "B shot-trained (S=100)": build(r, shots=100).fit(Xtr, ytr),
                    "C noise-penalised": build(r, penalty="noise", penalty_weight=0.05).fit(Xtr, ytr)}
        for label, model in variants.items():
            for shots in SHOTS:
                mse, bias, variance = expected_mse(model, Xte, yte, shots)
                row = {"study": "regression", "variant": label, "r": float(r), "db": float(8.6859 * r),
                       "shots": shots, "expected_mse": mse, "bias_mse": bias, "output_variance": variance,
                       "added_noise": added_noise(model, r)}
                if label.startswith("A") and shots == 100:
                    draws = np.array([np.mean((model.predict(Xte, shots=shots, seed=s) - yte) ** 2)
                                      for s in range(DRAWS)])
                    row.update(sampled_mse=float(draws.mean()), sampled_se=float(draws.std(ddof=1) / np.sqrt(DRAWS)))
                    z_scores.append(abs(row["sampled_mse"] - mse) / row["sampled_se"])
                rows.append(row)
        print(f"r={r}: " + ", ".join(f"{k} E[MSE|S=100]={next(q['expected_mse'] for q in rows if q['variant'] == k and q['r'] == r and q['shots'] == 100):.4f}"
                                      for k in variants), flush=True)
    (Btr, btr), _, (Bte, bte) = blobs()
    fixed = CVClassifier(CVCluster(2, 3, squeezing=1.0), head="homodyne", epochs=EPOCHS, seed=72).fit(Btr, btr)
    for r in SQUEEZING:
        fixed.mode, fixed.squeezing = "physical", float(r)
        retrained = CVClassifier(CVCluster(2, 3, squeezing=float(r)), head="homodyne", mode="physical",
                                 epochs=EPOCHS, seed=72).fit(Btr, btr)
        for label, clf in (("ideal-trained, fixed", fixed), ("retrained at r", retrained)):
            p = clf.predict_proba(Bte)[np.arange(len(bte)), bte]
            means, covs = clf.moments(clf.params, Bte)
            rows.append({"study": "homodyne", "variant": label, "r": float(r), "db": float(8.6859 * r),
                         "single_shot_accuracy": float(p.mean()), "majority_accuracy": float(np.mean(p > 0.5)),
                         "mean_abs_signal": float(np.mean(np.abs(np.real(means[:, 0]) - np.real(clf.head[0])))),
                         "output_std": float(np.sqrt(np.mean(np.real(covs[:, 0, 0])))),
                         **variance_split(clf, r)})

    reg = [q for q in rows if q["study"] == "regression"]

    def curve(variant, shots):
        return [next(q["expected_mse"] for q in reg if q["variant"] == variant and q["shots"] == shots and q["r"] == r)
                for r in SQUEEZING]

    a = {s: curve("A ideal-trained", s) for s in SHOTS}
    noise_a = [next(q["added_noise"] for q in reg if q["variant"].startswith("A") and q["r"] == r) for r in SQUEEZING]
    noise_c = [next(q["added_noise"] for q in reg if q["variant"].startswith("C") and q["r"] == r) for r in SQUEEZING]
    checks = {
        "sampled_matches_exact": max(z_scores) <= 4,
        "monotone_in_squeezing": all(np.all(np.diff(v) < 0) for v in a.values()),
        "monotone_in_shots": all(a[s1][i] > a[s2][i] for s1, s2 in zip(SHOTS, SHOTS[1:]) for i in range(len(SQUEEZING))),
        "penalty_lowers_noise": all(c <= n for c, n in zip(noise_c, noise_a)),
    }
    b100, a100 = curve("B shot-trained (S=100)", 100), a[100]
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    db = 8.6859 * SQUEEZING
    ax = axes[0, 0]
    for s, colour in zip(SHOTS, (PALETTE["physical"], PALETTE["cluster"], PALETTE["ideal"])):
        ax.semilogy(db, a[s], "o-", color=colour, label=f"S = {s}")
    sampled = [q for q in reg if "sampled_mse" in q]
    ax.errorbar([q["db"] for q in sampled], [q["sampled_mse"] for q in sampled],
                yerr=[2 * q["sampled_se"] for q in sampled], fmt="x", color="k", ms=4, label="sampled (S=100)")
    ax.axhline(np.mean((ideal.predict(Xte, mode="ideal") - yte) ** 2), color=PALETTE["baseline"], ls=":",
               label="exact moments")
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Expected test MSE", title="Ideal-trained model, shot-limited")
    ax.legend(fontsize=6)
    ax = axes[0, 1]
    for variant, colour in zip(("A ideal-trained", "B shot-trained (S=100)", "C noise-penalised"),
                               (PALETTE["ideal"], PALETTE["physical"], PALETTE["cluster"])):
        ax.semilogy(db, curve(variant, 100), "o-", color=colour, label=variant)
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Expected test MSE (S = 100)", title="Training strategies")
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    for variant, colour, marker in (("A ideal-trained", PALETTE["ideal"], "o"), ("C noise-penalised", PALETTE["cluster"], "s"),
                                    ("B shot-trained (S=100)", PALETTE["physical"], "^")):
        bias = [next(q["bias_mse"] for q in reg if q["variant"] == variant and q["r"] == r and q["shots"] == 100) for r in SQUEEZING]
        var = [next(q["output_variance"] for q in reg if q["variant"] == variant and q["r"] == r and q["shots"] == 100) / 100
               for r in SQUEEZING]
        ax.semilogy(db, bias, marker + "-", color=colour, label=f"{variant.split()[0]} bias")
        ax.semilogy(db, var, marker + "--", color=colour, mfc="none", label=f"{variant.split()[0]} shot variance")
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Contribution to test MSE (S = 100)", title="Bias / variance split")
    ax.legend(fontsize=5.5, ncol=2)
    ax = axes[1, 1]
    for variant, colour in (("ideal-trained, fixed", PALETTE["ideal"]), ("retrained at r", PALETTE["physical"])):
        hom = [q for q in rows if q["study"] == "homodyne" and q["variant"] == variant]
        ax.semilogy([q["db"] for q in hom], [q["encoding_variance"] for q in hom], "o-", color=colour,
                    label=f"{variant}: amplified encoding noise")
        ax.semilogy([q["db"] for q in hom], [q["resource_variance"] for q in hom], "s--", color=colour, mfc="none",
                    label=f"{variant}: resource noise")
    acc = [q for q in rows if q["study"] == "homodyne"]
    ax.set(xlabel="Resource squeezing (dB)", ylabel="Output q variance (vacuum units)",
           title=f"Homodyne classifier (blobs): accuracy {min(q['single_shot_accuracy'] for q in acc):.2f} single shot, "
                 f"{min(q['majority_accuracy'] for q in acc):.3f} many shots at every r")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks, "max_z": float(max(z_scores)),
                      "shot_training_vs_ideal_S100": {"ideal": a100, "shot_trained": b100},
                      "penalty_weight": 0.05, "draws": DRAWS, "epochs": EPOCHS}, fig)
    print(f"R72 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R72 acceptance failed")


if __name__ == "__main__":
    main()
