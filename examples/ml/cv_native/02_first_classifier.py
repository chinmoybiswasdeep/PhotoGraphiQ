"""Tutorial 2: a first CV classifier, and where its nonlinearity comes from.

Task: points inside / outside a circle of radius 0.62 (not linearly separable).

Model A: displacement encoding + Gaussian cluster + mean-q readout + logistic head. The
quantum part is affine in x, so this model can only draw a straight boundary.
Model B: squeezing encoding S(0.8 x_j)|0> + the same cluster + readout of means and q
variances ("moments"). The variances depend on exp(+-1.6 x_j), a nonlinear function of x
that comes from the covariance observable, not from a non-Gaussian gate.

Quantum: encoding, 2 x 3 CVCluster at resource squeezing r = 1 (physical channel,
exact moments). Classical: logistic head and L-BFGS.
"""

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from photographiqml.cv import CVClassifier, CVCluster, CVFeatureMap

start = perf_counter()
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (160, 2))
y = (np.hypot(*X.T) < 0.62).astype(int)
X_train, y_train, X_test, y_test = X[:100], y[:100], X[100:], y[100:]


def linear_logistic_accuracy():
    A, B = np.c_[X_train, np.ones(100)], np.c_[X_test, np.ones(60)]
    loss = lambda w: np.mean(np.logaddexp(0, -(2 * y_train - 1) * (A @ w)))  # noqa: E731
    w = minimize(loss, np.zeros(3)).x
    return float(np.mean((B @ w > 0) == y_test))


def classifier(feature_map, readout):
    cluster = CVCluster(2, 3, squeezing=1.0)  # 4 trainable homodyne angles
    return CVClassifier(
        cluster, feature_map, readout=readout, mode="physical", optimizer="lbfgs", epochs=40, seed=0
    )


affine = classifier(CVFeatureMap(2, "displacement"), "q").fit(X_train, y_train)
squeezed = classifier(CVFeatureMap(2, "squeezing", scale=0.8, max_squeezing=1.0), "moments").fit(
    X_train, y_train
)

baseline = linear_logistic_accuracy()
acc_affine = affine.score(X_test, y_test)["accuracy"]
acc_squeezed = squeezed.score(X_test, y_test)["accuracy"]
print(f"majority class fraction (test): {max(y_test.mean(), 1 - y_test.mean()):.3f}")
print(f"classical linear logistic:                  {baseline:.3f}")
print(f"A displacement encoding, mean readout:      {acc_affine:.3f}")
print(f"B squeezing encoding, means + q variances:  {acc_squeezed:.3f}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert abs(acc_affine - baseline) <= 0.05  # affine quantum features = linear model
assert acc_squeezed >= 0.9
