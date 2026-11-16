"""R73 (Figure CVML-6): gradient verification and gradient statistics vs model depth.

Hypothesis: every deterministic Gaussian-moment objective of photographiqml.cv is real-analytic
in its parameters, so complex-step derivatives Im f(x + ih)/h are exact to rounding (no
subtractive cancellation), agree with central differences to their O(h^2) truncation, and with
closed forms where those exist; they stay finite up to the homodyne-angle margin. For shot-based
objectives the reparameterised estimator with common random numbers (CRN) has far lower
variance than finite differences with independent shot noise.

Method: (1) closed form: a chain of L rotations on a q-displaced coherent input has
<q> = d cos(sum theta), d<q>/dtheta_k = -d sin(sum theta). (2) Physical homodyne-head
classification losses (Born probabilities Phi((mu - b) / sigma), so the loss depends on the output
covariance and hence on r; a mean-readout MSE would not) of compiled layered programs
(2 modes, L = 1..6 layers) at r in {0.25, 1, 2} and of 2 x (L+1) clusters,
five random parameter draws each: complex-step vs central differences (h = 1e-6), and an
h-sweep. (3) A 1 x 3 cluster with angle parameter phi -> theta approaching the margin.
(4) Gradient statistics: variance over 40 random initialisations of the loss gradient
component of the first trainable parameter, vs depth. (5) Shot gradients (S = 100): CRN
pathwise estimate vs central differences with fresh shot noise per evaluation, 40 repeats.

Acceptance: (1) <= 1e-12 absolute; (2) relative |CS - FD| / |CS| <= 1e-6 for every case;
(3) finite and within 1e-4 relative of FD with h scaled to the local curvature; (5) CRN
standard deviation below the independent-noise one for every component. (4) is descriptive.
"""

import numpy as np
from cv_support import PALETTE, linear_regression, plt, save

from photographiqml.cv import (
    CVClassifier,
    CVCluster,
    CVRegressor,
    GaussianProgram,
    central_difference,
    complex_step_gradient,
    layered_program,
)

NAME = "R73_gradients"


def loss_function(model, X, y, shots=None, noise=None):
    names = model.quantum_parameters
    nf = model._n_features()

    def loss(v):
        params = dict(zip(names, v[: len(names)]))
        head = v[len(names):]
        return np.mean((model._predict(model.features(params, X, shots=shots, noise=noise), head) - y) ** 2)

    return loss, len(names) + nf + 1


def homodyne_loss(model, X, labels):
    """Cross-entropy of the physical homodyne head: depends on the output covariance, hence on r."""
    names = model.quantum_parameters

    def loss(v):
        return model._loss(model.features(dict(zip(names, v[:-1])), X), v[-1:], labels)

    return loss, len(names) + 1


def main():
    (Xtr, ytr), _, _ = linear_regression()
    X, y = Xtr[:20], ytr[:20]
    rows = []
    # (1) closed form
    for L in range(1, 9):
        prog = GaussianProgram(1)
        for k in range(L):
            prog.rotate(0, f"t{k}")
        model = CVRegressor(prog)
        rng = np.random.default_rng(L)
        theta = rng.uniform(-np.pi, np.pi, L)
        f = lambda v: model.features(dict(zip(prog.parameters, v)), np.array([[0.7]]))[0, 0]  # noqa: E731
        exact = np.full(L, -0.7 * np.sin(theta.sum()))
        cs = complex_step_gradient(f, theta)
        rows.append({"study": "closed_form", "depth": L, "error": float(np.abs(cs - exact).max())})
    # (2) CS vs FD
    for family in ("compiled", "cluster"):
        for r in (0.25, 1.0, 2.0):
            for L in range(1, 7):
                for seed in range(5):
                    if family == "compiled":
                        model = CVClassifier(layered_program(2, L), head="homodyne", mode="physical", squeezing=r)
                    else:
                        model = CVClassifier(CVCluster(2, L + 1, squeezing=r), head="homodyne", mode="physical")
                    loss, n = homodyne_loss(model, X, (y > np.median(y)).astype(int))
                    v = np.random.default_rng(100 * L + seed).normal(scale=0.5, size=n)
                    cs, fd = complex_step_gradient(loss, v), central_difference(loss, v, 1e-6)
                    rows.append({"study": "cs_vs_fd", "family": family, "r": r, "depth": L, "seed": seed,
                                 "parameters": n, "relative_error": float(np.linalg.norm(cs - fd) / np.linalg.norm(cs)),
                                 "gradient_norm": float(np.linalg.norm(cs))})
            print(f"{family} r={r}: worst rel {max(q['relative_error'] for q in rows if q.get('family') == family and q.get('r') == r):.1e}", flush=True)
    model = CVRegressor(layered_program(2, 3), mode="physical", squeezing=1.0)
    loss, n = loss_function(model, X, y)
    v = np.random.default_rng(7).normal(scale=0.5, size=n)
    cs = complex_step_gradient(loss, v)
    for h in 10.0 ** -np.arange(1, 13):
        rows.append({"study": "step_sweep", "h": float(h),
                     "fd_error": float(np.linalg.norm(central_difference(loss, v, h) - cs) / np.linalg.norm(cs)),
                     "cs_error": float(np.linalg.norm(complex_step_gradient(loss, v, h) - cs) / np.linalg.norm(cs))})
    # (3) near the homodyne margin
    wire = CVCluster(1, 3, margin=0.05)
    reg = CVRegressor(wire, mode="physical")
    for phi in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        f = lambda w: reg.features(dict(zip(wire.parameters, w)), np.array([[0.5]]))[0, 0]  # noqa: E731
        point = np.array([phi, -0.3])
        cs = complex_step_gradient(f, point)
        h = 1e-6 / np.cosh(phi) ** 2  # dtheta/dphi ~ sech^2 phi
        fd = central_difference(f, point, max(h, 1e-9))
        theta = wire.angles(dict(zip(wire.parameters, point)))[0]
        rows.append({"study": "margin", "phi": phi, "theta": float(theta), "distance_to_singular": float(min(theta, np.pi - theta)),
                     "cs": cs.tolist(), "fd": fd.tolist(), "finite": bool(np.all(np.isfinite(cs))),
                     "relative_error": float(np.linalg.norm(cs - fd) / max(np.linalg.norm(cs), 1e-300))})
    # (4) gradient statistics vs depth
    for family in ("compiled", "cluster"):
        for L in range(1, 7):
            model = (CVRegressor(layered_program(2, L), mode="physical", squeezing=1.0) if family == "compiled"
                     else CVRegressor(CVCluster(2, L + 1, squeezing=1.0), mode="physical"))
            loss, n = loss_function(model, X, y)
            grads = []
            for seed in range(40):
                v = np.random.default_rng(seed).normal(scale=1.0, size=n)
                e = np.zeros(n)
                e[0] = 1e-20
                grads.append(np.imag(loss(v + 1j * e)) / 1e-20)
            rows.append({"study": "statistics", "family": family, "depth": L, "variance": float(np.var(grads)),
                         "mean_abs": float(np.mean(np.abs(grads)))})
    # (5) shot gradients: CRN pathwise vs independent-noise finite differences
    model = CVRegressor(CVCluster(2, 3, squeezing=0.5), mode="physical", shots=100)
    names = model.quantum_parameters
    v = np.random.default_rng(3).normal(scale=0.4, size=len(names) + 3)
    crn, independent = [], []
    for rep in range(40):
        noise = np.random.default_rng(rep).standard_normal((len(X), 1, 2))
        loss, _ = loss_function(model, X, y, shots=100, noise=noise)
        crn.append(complex_step_gradient(loss, v))
        rng = np.random.default_rng(10_000 + rep)

        def fresh(w):
            return np.real(loss_function(model, X, y, shots=100, noise=rng.standard_normal((len(X), 1, 2)))[0](w))

        independent.append(central_difference(fresh, v, 1e-2))
    crn, independent = np.array(crn), np.array(independent)
    exact_loss, _ = loss_function(model, X, y)
    for j in range(len(v)):
        rows.append({"study": "shots", "component": j, "crn_mean": float(crn[:, j].mean()), "crn_std": float(crn[:, j].std(ddof=1)),
                     "independent_mean": float(independent[:, j].mean()), "independent_std": float(independent[:, j].std(ddof=1))})

    checks = {
        "closed_form": max(q["error"] for q in rows if q["study"] == "closed_form") <= 1e-12,
        "cs_vs_fd": max(q["relative_error"] for q in rows if q["study"] == "cs_vs_fd") <= 1e-6,
        "margin": all(q["finite"] and q["relative_error"] <= 1e-4 for q in rows if q["study"] == "margin"),
        "crn_variance": all(q["crn_std"] < q["independent_std"] for q in rows if q["study"] == "shots"),
    }
    status = "pass" if all(checks.values()) else "fail"

    pyplot = plt()
    fig, axes = pyplot.subplots(2, 2, figsize=(7.0, 5.4))
    ax = axes[0, 0]
    for family, marker in (("compiled", "o"), ("cluster", "s")):
        for r, colour in zip((0.25, 1.0, 2.0), (PALETTE["physical"], PALETTE["cluster"], PALETTE["ideal"])):
            sub = [q for q in rows if q["study"] == "cs_vs_fd" and q["family"] == family and q["r"] == r]
            depths = sorted({q["depth"] for q in sub})
            ax.semilogy(depths, [max(q["relative_error"] for q in sub if q["depth"] == d) for d in depths],
                        marker + ("-" if family == "compiled" else "--"), color=colour, label=f"{family}, r={r}")
    cf = [q for q in rows if q["study"] == "closed_form"]
    ax.semilogy([q["depth"] for q in cf], [max(q["error"], 1e-17) for q in cf], "k^-", label="CS vs closed form")
    ax.axhline(1e-6, color="k", ls=":", lw=0.7)
    ax.set(xlabel="Depth (layers / cluster columns - 1)", ylabel="Worst relative gradient difference",
           title="Complex step vs central differences")
    ax.legend(fontsize=5, ncol=2)
    ax = axes[0, 1]
    sweep = [q for q in rows if q["study"] == "step_sweep"]
    ax.loglog([q["h"] for q in sweep], [max(q["fd_error"], 1e-17) for q in sweep], "o-", color=PALETTE["baseline"], label="central difference")
    ax.loglog([q["h"] for q in sweep], [max(q["cs_error"], 1e-17) for q in sweep], "s-", color=PALETTE["physical"], label="complex step")
    ax.set(xlabel="Step h", ylabel="Relative error vs complex step at h = 1e-20", title="Step-size dependence")
    ax.invert_xaxis()
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    for family, colour in (("compiled", PALETTE["physical"]), ("cluster", PALETTE["cluster"])):
        sub = [q for q in rows if q["study"] == "statistics" and q["family"] == family]
        ax.semilogy([q["depth"] for q in sub], [q["variance"] for q in sub], "o-", color=colour, label=family)
    ax.set(xlabel="Depth", ylabel="Var of dL/d(first parameter), 40 inits", title="Gradient statistics vs depth")
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    sh = [q for q in rows if q["study"] == "shots"]
    idx = np.arange(len(sh))
    ax.bar(idx - 0.2, [q["independent_std"] for q in sh], 0.4, color=PALETTE["baseline"], label="independent shots, FD")
    ax.bar(idx + 0.2, [q["crn_std"] for q in sh], 0.4, color=PALETTE["physical"], label="CRN pathwise")
    ax.set_yscale("log")
    ax.set(xlabel="Parameter index (2x3 cluster + head)", ylabel="Std of gradient estimate (S = 100)",
           title="Shot-based gradient estimators")
    ax.legend(fontsize=6)
    save(rows, NAME, {"oracle_class": "A", "status": status, "checks": checks}, fig)
    print(f"R73 status={status} checks={checks}")
    if status != "pass":
        raise AssertionError("R73 acceptance failed")


if __name__ == "__main__":
    main()
