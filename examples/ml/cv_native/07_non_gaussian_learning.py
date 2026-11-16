"""Tutorial 7 (experimental): a Kerr program fits an even target that a Gaussian model cannot.

Same encoding (coherent D(x)|0>), same readout (<q>) and the same affine head for both
models. The Gaussian model is affine in x, so it fits y = x^2 no better than a constant.
The non-Gaussian program D(b) K(kappa) R(phi) runs on PhotoGraphiQ's native Fock engine
(cutoff 30, retained norm reported). Kerr, rotation and squeezing commute with parity, so
without the trainable displacement b the output <q> would be odd in x; b breaks that.
Gradients are central differences (the Fock engine is real-valued); L-BFGS, 15 iterations.
"""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from photographiqml.cv import CVFeatureMap, CVRegressor, NonGaussianProgram, layered_program

start = perf_counter()
X = np.linspace(-1, 1, 7)[:, None]
y = X[:, 0] ** 2
fmap = CVFeatureMap(1, scale=1.0)

gaussian = CVRegressor(layered_program(1, 1), fmap, optimizer="lbfgs", epochs=15).fit(X, y)

program = NonGaussianProgram(1).displace(0, "b", 0.0, bound=2.0).kerr(0, "k").rotate(0, "r")
kerr = CVRegressor(
    program, fmap, cutoff=30, optimizer="lbfgs", epochs=15, init_scale=0.3, seed=1
).fit(X, y)
mse_g, mse_k = gaussian.score(X, y)["mse"], kerr.score(X, y)["mse"]
print(f"constant-predictor MSE (Var y): {np.var(y):.4f}")
print(f"Gaussian (affine) MSE:          {mse_g:.4f}")
print(f"Kerr program MSE:               {mse_k:.4f}  gradient: {kerr.history.settings['gradient']}")
print(
    f"trained b = 2 tanh(theta_b) = {2 * np.tanh(kerr.params['b']):.3f}, "
    f"kappa = {kerr.params['k']:.3f}"
)
print(f"1 - minimum retained Fock norm: {1 - min(kerr.retained_norms):.1e}")

# Parity: without the displacement, <q> is odd in x.
odd = NonGaussianProgram(1).kerr(0, 0.3).rotate(0, 0.7)
q_plus = odd.fock_moments({}, np.array([0.8]), fmap, 30, (0,))[0][0]
q_minus = odd.fock_moments({}, np.array([-0.8]), fmap, 30, (0,))[0][0]
print(f"no displacement: <q>(+0.8) = {q_plus:.6f}, <q>(-0.8) = {q_minus:.6f}")

# Independent dense matrix-exponential reference (shares no code with photographiq).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "validation"))
from cv_fock_reference import FockReference  # noqa: E402

ref = FockReference(90)
psi = ref.rotate(0.7) @ ref.kerr(0.3) @ ref.displace(0.8, 0.0) @ ref.vacuum()
ref_q = ref.moments(psi)[0]
print(f"engine vs expm reference <q>: |difference| = {abs(q_plus - ref_q):.1e}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert mse_k < 0.5 * mse_g
assert min(kerr.retained_norms) > 1 - 1e-6
assert abs(q_plus + q_minus) < 1e-8
assert abs(q_plus - ref_q) < 1e-6
