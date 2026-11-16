"""Tutorial 5: noise-aware training under finite squeezing and finite shots.

Flow-corrected CV-MBQC leaves the unconditional output means at their ideal values for every
resource squeezing r; finite squeezing only adds output covariance Y(r). A mean-readout
prediction estimated from S homodyne shots therefore has expected squared error
(mu - y)^2 + w^T C_qq w / S, where C = X V X^T + Y(r) and w is the affine head.

Three 2 x 4 CVCluster regressors at r = 0.5:
  A exact moments, no penalty (the loss never sees the noise);
  B trained on the S = 100 shot estimator (fixed normals, pathwise gradients);
  C exact moments + noise penalty (mean added noise per quadrature, task MSE / Var y).
"""

from time import perf_counter

import numpy as np

from photographiqml.cv import CVCluster, CVRegressor

start = perf_counter()
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (80, 2))
y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2 + 0.02 * rng.standard_normal(80)
X_train, y_train, X_test, y_test = X[:50], y[:50], X[50:], y[50:]
r, S = 0.5, 100


def model(**kw):
    return CVRegressor(CVCluster(2, 4, squeezing=r), mode="physical", epochs=80, seed=0, **kw)


def expected_mse(m, shots):
    """Exact expected test MSE of the S-shot sample-mean prediction, from the moments."""
    mu, cov = m.moments(m.params, X_test)
    w = m.head[:-1]
    bias = np.real(mu[:, 0::2] @ w + m.head[-1] - y_test) ** 2
    var = np.einsum("j,njk,k->n", w, np.real(cov[:, 0::2, 0::2]), w) / shots
    return float(np.mean(bias + var)), float(np.mean(bias))


def added_noise(m):
    """Mean added noise per quadrature, tr(Y) / 2n in vacuum units (the penalty quantity)."""
    Y = m.program.channel(m.params, mode="physical")[2]
    return float(np.trace(np.real(Y)) / len(Y))


models = {
    "A exact": model().fit(X_train, y_train),
    "B shots": model(shots=S).fit(X_train, y_train),
    "C penalty": model(penalty="noise", penalty_weight=0.05).fit(X_train, y_train),
}
print(f"r = {r} ({r * 20 / np.log(10):.2f} dB), S = {S} shots per prediction")
print(f"{'model':<10} {'added noise':>12} {'bias MSE':>10} {'E[MSE] S=100':>13}")
results = {}
for name, m in models.items():
    total, bias = expected_mse(m, S)
    results[name] = (added_noise(m), total)
    print(f"{name:<10} {results[name][0]:>12.4f} {bias:>10.5f} {total:>13.5f}")

# Check the closed form against sampled shot predictions of model A.
a = models["A exact"]
draws = np.array([np.mean((a.predict(X_test, shots=S, seed=s) - y_test) ** 2) for s in range(200)])
z = (draws.mean() - results["A exact"][1]) / (draws.std(ddof=1) / np.sqrt(len(draws)))
print(f"sampled mean MSE over 200 draws: {draws.mean():.5f} (z = {z:.2f} vs exact)")
ideal_mu = a.moments(a.params, X_test, mode="ideal")[0]
phys_mu = a.moments(a.params, X_test, mode="physical")[0]
print(f"max |ideal - physical| output mean: {np.abs(ideal_mu - phys_mu).max():.1e}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert abs(z) < 4
assert np.abs(ideal_mu - phys_mu).max() < 1e-12
assert results["C penalty"][0] <= results["A exact"][0]
