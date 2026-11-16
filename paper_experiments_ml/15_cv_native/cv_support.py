"""Shared helpers for the CV-native learning studies R68-R75 (extension catalogue).

Deterministic datasets with explicit train/validation/test splits, classical baselines
(ordinary least squares, linear and quadratic logistic regression; NumPy/SciPy only), and the
suite's save/plot conventions (``common.save_result``, Times New Roman manuscript style).
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "tests" / "validation"))

import common  # noqa: E402

COLORS = common.COLORS
PALETTE = {
    "ideal": COLORS["analytic"],
    "physical": COLORS["photographiqml"],
    "cluster": COLORS["photographiq"],
    "baseline": COLORS["classical_baseline"],
    "nongaussian": COLORS["piquasso"],
    "reference": COLORS["mentpy"],
}


def plt():
    return common.setup_style()


def provenance():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    import scipy

    return {"revision": revision, "working_tree_modified": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def save(rows, name, extra, figure=None):
    extra = dict(extra)
    extra.setdefault("provenance", provenance())
    common.save_result(rows, name, extra=extra, meta_extra={"experiment_id": name.split("_")[0]})
    if figure is not None:
        common.save_figure(figure, name)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


# ----------------------------------------------------------------------------- datasets
def split(X, y, n_train, n_val, seed):
    order = np.random.default_rng(seed).permutation(len(X))
    a, b = order[:n_train], order[n_train:n_train + n_val]
    c = order[n_train + n_val:]
    return (X[a], y[a]), (X[b], y[b]), (X[c], y[c])


def linear_regression(seed=0, n=120, noise=0.02):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2 + noise * rng.standard_normal(n)
    return split(X, y, 60, 20, seed + 1)


def nonlinear_regression(seed=0, n=90, noise=0.01):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 1))
    y = np.sin(2.5 * X[:, 0]) * 0.5 + 0.4 * X[:, 0] ** 2 + noise * rng.standard_normal(n)
    return split(X, y, 45, 15, seed + 1)


def blobs(seed=0, n=120):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    centres = np.array([[-0.45, -0.3], [0.45, 0.3]])
    X = np.clip(centres[y] + 0.3 * rng.standard_normal((n, 2)), -1.2, 1.2)
    return split(X, y, 60, 20, seed + 1)


def circles(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = (np.hypot(*X.T) < 0.62).astype(int)
    return split(X, y, 100, 30, seed + 1)


def multivariate(seed=0, n=120, noise=0.02):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 3))
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + noise * rng.standard_normal(n)
    return split(X, y, 60, 20, seed + 1)


# ----------------------------------------------------------------------------- baselines
def ols(Xtr, ytr, Xte):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[Xte, np.ones(len(Xte))] @ w


def quadratic_features(X):
    cols = [X] + [X[:, [i]] * X[:, [j]] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
    return np.hstack(cols)


def logistic(Xtr, ytr, Xte, l2=1e-3):
    A, B = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]

    def loss(w):
        z = A @ w
        return np.mean(np.logaddexp(0, -z) * ytr + np.logaddexp(0, z) * (1 - ytr)) + l2 * w @ w

    w = minimize(loss, np.zeros(A.shape[1]), method="L-BFGS-B").x
    return 1 / (1 + np.exp(-B @ w))


def accuracy(proba, y):
    return float(np.mean((proba > 0.5).astype(int) == y))
