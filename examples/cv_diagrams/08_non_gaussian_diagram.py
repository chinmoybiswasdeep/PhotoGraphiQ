"""Tutorial 8: non-Gaussian diagrams on the truncated Fock engines.

The restricted non-Gaussian layer: 1 -> 1 q-spiders exp(i f(q)) (``PhasePolynomial``, the
q-spider of Nagayoshi et al., PRR 7, 033141 (2025), restated with hbar = 2), Kerr, Fock
preparation, photon counting and post-selection. Their semantics is numerical (``run_fock``,
exclusive total-photon cutoff, retained norms reported); agreement at a finite cutoff is
evidence for an operator identity, never a proof.

Key claims checked:
* phase-polynomial fusion and moving a q displacement through exp(i f(q)) give the same state
  as a dense matrix-exponential reference, with infidelity decreasing in the cutoff;
* a *false* rule (commuting exp(i f(q)) with a p displacement) stays wrong at every cutoff;
* heralded photon subtraction has herald probability mu e^{-mu}, mu = (alpha sin theta)^2,
  reported separately from the normalised output state;
* non-Gaussian diagrams have no Gaussian semantics (NotImplementedError).
"""

import warnings
from time import perf_counter

import numpy as np
from scipy.linalg import expm

import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    BeamSplitter,
    CVDiagram,
    Displacement,
    FockPrepare,
    Kerr,
    PhasePolynomial,
    PhotonCount,
    Postselect,
    apply_rewrite,
    cutoff_study,
    evaluate,
    run_fock,
)

warnings.simplefilter("ignore", RuntimeWarning)  # truncation is reported as retained norms
start = perf_counter()
F, G = (0.0, 0.1, 0.05, 0.03), (0.0, 0.0, -0.02, 0.01)  # f(q) = sum_k c_k q^k
original = CVDiagram(
    (0,), (Displacement(0, 0.4, 0.0), PhasePolynomial(0, F), PhasePolynomial(0, G)), (0,)
)
fused = apply_rewrite(original, "phase-polynomial-fusion")
moved = apply_rewrite(fused, "displacement-through-phase-polynomial")
false_rule = CVDiagram((0,), (PhasePolynomial(0, F), Displacement(0, 0.0, 0.5)), (0,))
print("fused coefficients:", fused.boxes[-1].coefficients)
print(
    "after propagation:", [(type(b).__name__, b.params().get("coefficients")) for b in moved.boxes]
)

# Dense reference (hbar = 2: q = a + a^dag, p = -i(a - a^dag)), large cutoff.
N = 150
a = np.diag(np.sqrt(np.arange(1, N)), 1)
q, p = a + a.T, -1j * (a - a.T)
vacuum = np.zeros(N)
vacuum[0] = 1.0


def D(dq, dp):
    return expm(0.5j * (dp * q - dq * p))


def phase(c):
    return expm(1j * sum(ck * np.linalg.matrix_power(q, k) for k, ck in enumerate(c)))


x0 = (0.5, 0.2)
psi = D(*x0) @ vacuum
target = phase(G) @ phase(F) @ D(0.4, 0.0) @ psi
wrong_target = D(0.0, 0.5) @ phase(F) @ psi  # the true order of false_rule's boxes


def infidelity(reference):
    def observable(run):
        vec = np.asarray(run.state.state_vector)
        return float(1 - abs(np.vdot(reference[: len(vec)], vec)) ** 2)

    return observable


cutoffs = (20, 30, 45, 60)
inputs = {0: pg.GaussianInput(x0)}
table = {
    "original": cutoff_study(original, cutoffs, infidelity(target), inputs=inputs),
    "fused": cutoff_study(fused, cutoffs, infidelity(target), inputs=inputs),
    "fused+moved": cutoff_study(moved, cutoffs, infidelity(target), inputs=inputs),
    "false rule": cutoff_study(false_rule, cutoffs, infidelity(target), inputs=inputs),
}
print("infidelity vs dense reference at cutoffs", cutoffs)
for label, rows in table.items():
    print(f"  {label:>12}: " + "  ".join(f"{row['value']:.2e}" for row in rows))
print("  retained norm: " + "  ".join(f"{row['retained_norm']:.7f}" for row in table["original"]))
own = cutoff_study(false_rule, (60,), infidelity(wrong_target), inputs=inputs)[0]["value"]
print(f"  false-rule diagram vs its own true order at cutoff 60: {own:.2e}")

# Heralded photon subtraction from a coherent state.
alpha, theta = 0.8, 0.3
sub = CVDiagram(
    (0,),
    (FockPrepare("a", 0), BeamSplitter(0, "a", theta), PhotonCount("a", "n"), Postselect("n", 1)),
    (0,),
)
mu = (alpha * np.sin(theta)) ** 2
rows = cutoff_study(
    sub,
    (8, 16, 24),
    lambda run: run.probability,
    inputs={0: pg.GaussianInput.coherent(alpha)},
    outcomes={"n": 1},
)
print(
    "herald probability:",
    [(r["cutoff"], round(r["value"], 12)) for r in rows],
    f"closed form {mu * np.exp(-mu):.12f}",
)
try:
    run_fock(sub, 12, inputs={0: pg.GaussianInput.coherent(alpha)})
except ValueError as exc:
    print("missing post-selection outcome:", exc)
try:
    evaluate(CVDiagram((0,), (Kerr(0, 0.2),), (0,)))
except NotImplementedError as exc:
    print("Gaussian semantics refused:", exc)
print(f"runtime: {perf_counter() - start:.1f} s")

for label in ("original", "fused", "fused+moved"):
    values = [row["value"] for row in table[label]]
    assert values[-1] < 1e-9 and values[-1] < values[0], label
assert all(row["value"] > 1e-3 for row in table["false rule"]) and own < 1e-9
assert abs(rows[-1]["value"] - mu * np.exp(-mu)) < 1e-10
