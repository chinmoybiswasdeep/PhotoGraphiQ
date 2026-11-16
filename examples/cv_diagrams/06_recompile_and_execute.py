"""Tutorial 6: optimise, recompile with ``to_pattern`` and execute on the Gaussian backend.

A logically optimised diagram is a new executable Pattern. It implements the same ideal map
(X, d) as the original but has its *own* finite-squeezing noise, so it is validated against
its own channel, never against the original's covariance.

Key claims checked:
* ``analysis.gaussian_channel`` of the recompiled pattern equals the optimised diagram's
  channel, and its X and d equal the original's;
* a postselected Gaussian-backend trajectory of the recompiled pattern equals the diagram's
  conditional semantics (state and record density);
* trajectory ensembles of both patterns agree with their own channel predictions (|z| <= 4.5);
* a ClassicalSource recompiles to one ancilla prepared with the source variance and
  measured in q, with an identical averaged channel.
"""

from time import perf_counter

import numpy as np

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import (
    CZ,
    CVDiagram,
    Feedforward,
    Homodyne,
    Prepare,
    apply_rewrite,
    evaluate,
    optimize_cv_pattern,
    resource_cost,
    to_pattern,
)

start = perf_counter()
SHOTS = 400
circuit = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.25).rotate(0, -0.9)
original = circuit.compile(squeezing=1.0)
result = optimize_cv_pattern(original, equivalence="logical")
recompiled = to_pattern(result.best)
print(
    f"resource modes {result.original_cost['resource_modes']} -> "
    f"{result.best_cost['resource_modes']}; commands {len(original.commands)} -> "
    f"{len(recompiled.commands)}"
)

# 1. Executed channel of the recompiled pattern.
X, Y, d = evaluate(result.best).channel()
X0, Y0, d0 = evaluate(result.original).channel()
executed = gaussian_channel(recompiled)
exec_gap = max(np.abs(executed.matrix - X).max(), np.abs(executed.noise - Y).max())
print(f"|gaussian_channel(recompiled) - own diagram channel| = {exec_gap:.1e}")
print(
    f"|X - X_original| = {np.abs(X - X0).max():.1e}, |d - d_original| = "
    f"{np.abs(d - d0).max():.1e}, noise tr Y/2: {np.trace(Y0) / 2:.4f} -> {np.trace(Y) / 2:.4f}"
)

# 2. One postselected trajectory vs the conditional instrument.
rng = np.random.default_rng(6)
mean_in = np.array([0.6, -0.3])
outcomes = {k: float(rng.normal()) for k in evaluate(result.best).records}
mean, cov, density = evaluate(result.best).conditional(outcomes, mean_in, np.eye(2))
run = pg.simulate(
    recompiled,
    backend="gaussian",
    measurement_outcomes=outcomes,
    initial_state=pg.GaussianState(mean_in, np.eye(2), recompiled.inputs),
)
cond_gap = max(
    np.abs(run.state.mean - mean).max(),
    np.abs(run.state.covariance - cov).max(),
    abs(np.exp(run.log_likelihood) - density) / density,
)
print(f"conditional: |trajectory - diagram| = {cond_gap:.1e} (density {density:.4g})")


# 3. Ensembles: each pattern against its own channel prediction.
def ensemble_z(pattern, X, Y, d):
    inputs = {pattern.inputs[0]: pg.GaussianInput(tuple(mean_in))}
    runs = pg.run_shots(pattern, SHOTS, seed=6, inputs=inputs).trajectories
    states = [r.state.reduced(pattern.outputs) for r in runs]
    means = np.array([s.mean for s in states])
    mean_meas, cov_meas = means.mean(0), states[0].covariance + np.cov(means.T)
    mean_pred, cov_pred = X @ mean_in + d, X @ X.T + Y
    diag = np.diag(cov_pred)
    z_mean = (mean_meas - mean_pred) / np.sqrt(diag / SHOTS)
    z_cov = (cov_meas - cov_pred) / np.sqrt((cov_pred**2 + np.outer(diag, diag)) / SHOTS)
    return float(max(np.abs(z_mean).max(), np.abs(z_cov).max())), mean_meas


z_orig, m_orig = ensemble_z(original, X0, Y0, d0)
z_new, m_new = ensemble_z(recompiled, X, Y, d)
print(
    f"{SHOTS} shots each: max|z| original {z_orig:.2f}, recompiled {z_new:.2f}; "
    f"ensemble means {np.round(m_orig, 3).tolist()} vs {np.round(m_new, 3).tolist()} "
    f"(both predict {np.round(X @ mean_in + d, 3).tolist()})"
)

# 4. ClassicalSource -> ancilla measured in q (one optical mode per source).
base = CVDiagram(
    (0,),
    (
        Prepare.squeezed("a", 0.4),
        CZ(0, "a", 0.8),
        Homodyne("a", 0.0, "m"),
        Feedforward(0, (("m", (0.3, 0.0)),)),
    ),
    (0,),
)
eliminated = apply_rewrite(
    apply_rewrite(base, "cz-q-measurement-to-feedforward"), "prepare-measure-to-source"
)
source_pattern = to_pattern(eliminated)
src = gaussian_channel(source_pattern)
Xb, Yb, _ = evaluate(base).channel()
src_gap = max(np.abs(src.matrix - Xb).max(), np.abs(src.noise - Yb).max())
print(
    f"ClassicalSource: diagram modes {resource_cost(eliminated)['resource_modes']}, "
    f"recompiled preparations {sum(isinstance(c, pg.Prepare) for c in source_pattern.commands)}, "
    f"channel residual {src_gap:.1e}"
)
print(f"runtime: {perf_counter() - start:.1f} s")

assert exec_gap < 1e-12 and np.abs(X - X0).max() < 1e-10 and np.abs(d - d0).max() < 1e-10
assert cond_gap < 1e-8
assert z_orig <= 4.5 and z_new <= 4.5
assert result.best_cost["resource_modes"] < result.original_cost["resource_modes"]
assert src_gap < 1e-12
