"""R70 (Figure CVML-3): Gaussian CV regression on affine targets, held-out evaluation.

Hypothesis: with displacement encoding and a linear mean-quadrature readout the Gaussian
model is affine in the features (R73 checks this structurally), so it can represent affine
targets exactly; training (Adam, complex-step gradients) should reach the ordinary-least-
squares (OLS) test error, up to optimisation tolerance. Finite resource squeezing does not
change the unconditional mean (R69), so physical execution with exact moments is expected
to match ideal execution; shot noise is studied in R72.

Method: deterministic linear-regression split (2 features, 60/20/40, label noise 0.02) and
a 3-feature multivariate split. Models: compiled two-mode layered program (ideal and physical,
r = 1), directly trainable 2x3 CVCluster cluster (physical, r = 1) and a 3x3 cluster for the
multivariate task; five initialisation seeds each; Adam, SGD and L-BFGS on the cluster.
Baseline: OLS (closed form) on the same split.

Acceptance: median test MSE of every quantum model <= 2 x OLS test MSE + 1e-4. Demonstrated:
trainability and generalisation on affine targets. Not demonstrated: any advantage (an affine
classical model is optimal here by construction).
"""

import numpy as np
from cv_support import PALETTE, Timer, linear_regression, multivariate, ols, plt, save

from photographiqml.cv import CVCluster, CVRegressor, layered_program

NAME = "R70_gaussian_regression"
SEEDS = range(5)
EPOCHS = 120


def models(task):
    if task == "linear":
        return {
            "layered (ideal)": lambda s: CVRegressor(layered_program(2, 1), epochs=EPOCHS, seed=s),
            "layered (compiled, r=1)": lambda s: CVRegressor(layered_program(2, 1), mode="physical",
                                                             squeezing=1.0, epochs=EPOCHS, seed=s),
            "2x3 cluster (r=1)": lambda s: CVRegressor(CVCluster(2, 3, squeezing=1.0), mode="physical",
                                                       epochs=EPOCHS, learning_rate=0.05, seed=s),
        }
    return {"3x3 cluster (r=1)": lambda s: CVRegressor(CVCluster(3, 3, squeezing=1.0), mode="physical",
                                                       epochs=EPOCHS, seed=s)}


def main():
    rows, curves = [], {}
    tasks = {"linear": linear_regression(), "multivariate": multivariate()}
    ols_mse = {}
    for task, ((Xtr, ytr), (Xva, yva), (Xte, yte)) in tasks.items():
        ols_mse[task] = float(np.mean((ols(Xtr, ytr, Xte) - yte) ** 2))
        rows.append({"task": task, "model": "OLS baseline", "seed": -1, "test_mse": ols_mse[task],
                     "test_mae": float(np.mean(np.abs(ols(Xtr, ytr, Xte) - yte)))})
        for label, build in models(task).items():
            for seed in SEEDS:
                with Timer() as timer:
                    model = build(seed).fit(Xtr, ytr, Xva, yva)
                score = model.score(Xte, yte)
                rows.append({"task": task, "model": label, "seed": seed, "test_mse": score["mse"],
                             "test_mae": score["mae"], "final_train_loss": model.history.losses[-1],
                             "final_validation_loss": model.history.validation_losses[-1],
                             "seconds": timer.seconds, "evaluations": model.history.evaluations[-1],
                             "parameters": len(model.history.settings["parameter_names"])})
                if seed == 0:
                    curves[(task, label)] = (model.history.losses, model.history.validation_losses,
                                             model.predict(Xte), yte)
            med = np.median([r["test_mse"] for r in rows if r["model"] == label])
            print(f"{task} {label}: median test MSE {med:.2e} (OLS {ols_mse[task]:.2e})", flush=True)
    (Xtr, ytr), (Xva, yva), _ = tasks["linear"]
    optimisers = {}
    for name, lr in (("adam", 0.05), ("sgd", 0.05), ("lbfgs", 0.05)):
        model = CVRegressor(CVCluster(2, 3, squeezing=1.0), mode="physical", optimizer=name, epochs=EPOCHS,
                            learning_rate=lr, seed=0).fit(Xtr, ytr)
        optimisers[name] = (model.history.evaluations, model.history.losses)
        rows.append({"task": "optimiser", "model": name, "seed": 0, "final_train_loss": model.history.losses[-1],
                     "evaluations": model.history.evaluations[-1]})

    medians = {(r["task"], r["model"]): np.median([q["test_mse"] for q in rows if q["model"] == r["model"]
                                                   and q["task"] == r["task"]])
               for r in rows if r["task"] in tasks and r["model"] != "OLS baseline"}
    checks = {f"{t}:{m}": bool(v <= 2 * ols_mse[t] + 1e-4) for (t, m), v in medians.items()}
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    colours = [PALETTE["ideal"], PALETTE["physical"], PALETTE["cluster"], PALETTE["nongaussian"]]
    ax = axes[0, 0]
    for (key, (train, val, _, _)), colour in zip(curves.items(), colours):
        ax.semilogy(train, color=colour, label=f"{key[1]} train")
        ax.semilogy(val, color=colour, ls="--", lw=0.9)
    ax.axhline(ols_mse["linear"], color=PALETTE["baseline"], ls=":", label="OLS test MSE")
    ax.set(xlabel="Epoch", ylabel="MSE (solid: train, dashed: validation)", title="Training curves")
    ax.legend(fontsize=5.5)
    ax = axes[0, 1]
    for (key, (_, _, pred, truth)), colour in zip(curves.items(), colours):
        ax.plot(truth, pred, "o", ms=2.5, color=colour, label=f"{key[1]} ({key[0]})")
    lim = [min(min(c[3]) for c in curves.values()) - 0.1, max(max(c[3]) for c in curves.values()) + 0.1]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set(xlabel="Held-out target", ylabel="Prediction", title="Held-out predictions (seed 0)")
    ax.legend(fontsize=5.5)
    ax = axes[1, 0]
    labels = [f"{t}\n{m}" for (t, m) in medians] + ["linear\nOLS", "multivariate\nOLS"]
    values = [[r["test_mse"] for r in rows if r["task"] == t and r["model"] == m] for (t, m) in medians]
    values += [[ols_mse["linear"]], [ols_mse["multivariate"]]]
    for i, v in enumerate(values):
        colour = PALETTE["baseline"] if i >= len(medians) else colours[i % len(colours)]
        ax.scatter(np.full(len(v), i), v, s=10, color=colour)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=colour)
    ax.set_xticks(range(len(labels)), labels, fontsize=5.5, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set(ylabel="Test MSE (5 initialisation seeds)", title="Held-out error vs OLS")
    ax = axes[1, 1]
    for (name, (evals, losses)), colour in zip(optimisers.items(), colours):
        ax.loglog(np.maximum(evals, 1), losses, color=colour, label=name)
    ax.set(xlabel="Simulator evaluations", ylabel="Training MSE", title="Optimisers on the 2x3 cluster")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks, "ols_test_mse": ols_mse,
                      "median_test_mse": {f"{t}:{m}": float(v) for (t, m), v in medians.items()},
                      "seeds": list(SEEDS), "epochs": EPOCHS}, fig)
    print(f"R70 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R70 acceptance failed")


if __name__ == "__main__":
    main()
