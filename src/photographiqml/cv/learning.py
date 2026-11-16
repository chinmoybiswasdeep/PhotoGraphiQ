"""Supervised learning on CV Gaussian programs: readouts, heads, gradients and resources.

What is quantum and what is classical:

* quantum: the encoded input state, the Gaussian program (direct circuit, compiled CV-MBQC
  channel or trainable cluster) and the output moments / homodyne statistics it produces;
* classical: the affine (regression), logistic or softmax (classification) head applied to
  the readout features, and the optimiser. The ``homodyne`` binary head is the exception: its
  probability is the Born probability P(q_out > b) of a binned physical homodyne outcome.

Readout features are output means (``q``, ``p``, ``qp``) or means plus q variances
(``moments``). With ``shots = S`` a mean feature is the sample mean of S homodyne outcomes,
drawn exactly as mu + chol(C / S) z with fixed standard normals z (common random numbers);
``qp`` splits the shots between q and p settings because one homodyne cannot read both.

Gradients: ``complex-step`` (exact to rounding for these deterministic moment objectives,
including the fixed-noise shot estimator), ``finite-difference`` (central) or a user-supplied
callable; ``auto`` picks complex-step for Gaussian programs and central differences for
non-Gaussian (Fock-engine) programs, whose native execution is real-valued. Gaussian moment predictions are analytic in every parameter; the
sampled outcomes themselves are not differentiated, only the reparameterised estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..training import Trainer
from .gaussian import CVFeatureMap, GaussianProgram
from .mbqc import CVCluster

READOUTS = ("q", "p", "qp", "moments")
PENALTIES = (None, "noise", "squeezing")
STEP = 1e-20  # complex-step size


def _chol(c):
    """Unpivoted Cholesky with plain arithmetic (analytic, so complex-step safe)."""
    n = len(c)
    out = np.zeros_like(c)
    for i in range(n):
        for j in range(i + 1):
            s = c[i, j] - out[i, :j] @ out[j, :j]
            out[i, j] = np.sqrt(s) if i == j else s / out[j, j]
    return out


def _softplus(z):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(np.real(z) > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))


def complex_step_gradient(objective, x, step=STEP):
    """d objective / dx by complex steps: Im f(x + i h e_j) / h (no subtractive cancellation)."""
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for j in range(len(x)):
        z = x.astype(complex)
        z[j] += 1j * step
        grad[j] = np.imag(objective(z)) / step
    return grad


def central_difference(objective, x, step=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = step
        grad[j] = (np.real(objective(x + e)) - np.real(objective(x - e))) / (2 * step)
    return grad


@dataclass
class CVTrainingHistory:
    """Per-epoch record of a fit (the Trainer history plus provenance)."""

    losses: list = field(default_factory=list)
    validation_losses: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class _CVModel:
    """Shared quantum pipeline; subclasses define the classical head and the task loss."""

    def _targets(self, y):
        raise NotImplementedError

    def _scale(self, y):
        raise NotImplementedError

    def _head_names(self, nf):
        raise NotImplementedError

    def _head_init(self, nf):
        raise NotImplementedError

    def _loss(self, f, head, y):
        raise NotImplementedError

    def __init__(
        self,
        program,
        feature_map=None,
        *,
        readout="q",
        readout_modes=None,
        mode="ideal",
        squeezing=None,
        shots=None,
        penalty=None,
        penalty_weight=0.0,
        optimizer="adam",
        epochs=100,
        learning_rate=0.05,
        gradient="auto",
        init_scale=0.1,
        seed=0,
        cutoff=None,
        fd_step=1e-6,
    ):
        if not isinstance(program, (GaussianProgram, CVCluster)):
            raise TypeError("program must be a GaussianProgram or a CVCluster")
        feature_map = feature_map or CVFeatureMap(program.modes)
        if feature_map.modes != program.modes:
            raise ValueError("Feature map and program must act on the same number of modes")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if mode not in ("ideal", "physical"):
            raise ValueError("mode must be 'ideal' or 'physical'")
        if mode == "physical" and isinstance(program, GaussianProgram) and squeezing is None:
            raise ValueError("Physical (compiled) execution needs the resource squeezing")
        if shots is not None and (
            isinstance(shots, bool) or not isinstance(shots, int) or shots < 2
        ):
            raise ValueError("shots must be an integer >= 2 or None")
        if shots is not None and readout == "moments":
            raise ValueError("Shot-estimated variances are not supported; use a mean readout")
        if penalty not in PENALTIES or not np.isfinite(penalty_weight) or penalty_weight < 0:
            raise ValueError(f"penalty must be one of {PENALTIES} with a finite weight >= 0")
        if penalty == "noise" and mode != "physical":
            raise ValueError("The noise penalty needs physical execution")
        if penalty == "squeezing" and isinstance(program, CVCluster):
            raise ValueError("A CVCluster has no squeeze gates; use penalty='noise'")
        non_gaussian = getattr(program, "non_gaussian", False)
        # Flow certificates are solved for real weights, so trainable rung weights have no
        # complex-step derivative: they need central differences.
        real_only = non_gaussian or getattr(program, "trainable_weights", False)
        if gradient == "auto":
            gradient = "finite-difference" if real_only else "complex-step"
        if gradient == "complex-step" and getattr(program, "trainable_weights", False):
            raise ValueError("Trainable rung weights need gradient='finite-difference'")
        if gradient not in ("complex-step", "finite-difference") and not callable(gradient):
            raise ValueError("gradient must be auto, complex-step, finite-difference or a callable")
        if non_gaussian:
            if gradient == "complex-step":
                raise ValueError("Fock-engine execution is real-valued; use finite differences")
            if cutoff is None or shots is not None or mode != "ideal":
                raise ValueError(
                    "Non-Gaussian programs need a Fock cutoff, no training shots "
                    "(sample means are not Gaussian) and mode='ideal' (gate level)"
                )
        self.cutoff, self.fd_step = cutoff, float(fd_step)
        self.program, self.feature_map = program, feature_map
        self.readout = readout
        self.readout_modes = (
            tuple(range(program.modes)) if readout_modes is None else tuple(readout_modes)
        )
        if not self.readout_modes or not set(self.readout_modes) <= set(range(program.modes)):
            raise ValueError("readout_modes must be a nonempty subset of the program modes")
        self.mode, self.squeezing, self.shots = mode, squeezing, shots
        self.penalty, self.penalty_weight = penalty, float(penalty_weight)
        self.trainer = Trainer(optimizer, epochs, learning_rate)
        self.gradient, self.init_scale, self.seed = gradient, float(init_scale), seed
        self.params: dict | None = None
        self.history: CVTrainingHistory | None = None
        self._evaluations = 0

    # ------------------------------------------------------------------ quantum part
    @property
    def quantum_parameters(self):
        return self.program.parameters

    def _channel(self, params, x, mode):
        return self.program.channel(params, x, mode=mode, squeezing=self.squeezing)

    def moments(self, params, X, *, mode=None):
        """Output (means (N, 2n), covariances (N, 2n, 2n)) of the readout-mode marginal."""
        mode = self.mode if mode is None else mode
        X = self.feature_map.check(X)
        self._evaluations += 1
        idx = [j for m in self.readout_modes for j in (2 * m, 2 * m + 1)]
        if getattr(self.program, "non_gaussian", False):  # Fock engine; diagonal variances only
            means, covs, self.retained_norms = [], [], []
            for x in X:
                program: Any = self.program
                mu, var, norm = program.fock_moments(
                    {k: float(np.real(v)) for k, v in params.items()},
                    x,
                    self.feature_map,
                    self.cutoff,
                    self.readout_modes,
                )
                cov = np.full((len(idx), len(idx)), np.nan)
                np.fill_diagonal(cov, var)
                means.append(mu)
                covs.append(cov)
                self.retained_norms.append(norm)
            return np.array(means), np.array(covs)
        static = (
            None
            if getattr(self.program, "data_dependent", False)
            else self._channel(params, (), mode)
        )
        means, covs = [], []
        for x in X:
            A, t, Y, _ = static if static is not None else self._channel(params, x, mode)
            d, V = self.feature_map.moments(x)
            means.append((A @ d + t)[idx])
            covs.append((A @ V @ A.T + Y)[np.ix_(idx, idx)])
        return np.array(means), np.array(covs)

    def features(self, params, X, *, mode=None, shots=None, noise=None):
        """Readout features; ``shots`` uses sample means with fixed normals ``noise``."""
        if shots is not None and getattr(self.program, "non_gaussian", False):
            raise ValueError("Shot sample means of non-Gaussian outputs are not Gaussian")
        means, covs = self.moments(params, X, mode=mode)
        q, p = means[:, 0::2], means[:, 1::2]
        if shots is not None:
            k = len(self.readout_modes)
            settings = ("q", "p") if self.readout == "qp" else (self.readout,)
            per = shots // len(settings)
            if noise is None:
                noise = np.random.default_rng(self.seed).standard_normal((len(X), len(settings), k))
            out = []
            for s, setting in enumerate(settings):
                block = covs[:, 0::2, 0::2] if setting == "q" else covs[:, 1::2, 1::2]
                mu = q if setting == "q" else p
                out.append(
                    np.array([m + _chol(c / per) @ z for m, c, z in zip(mu, block, noise[:, s])])
                )
            return np.concatenate(out, axis=1)
        if self.readout == "q":
            return q
        if self.readout == "p":
            return p
        if self.readout == "qp":
            return np.concatenate([q, p], axis=1)
        return np.concatenate([q, np.einsum("nii->ni", covs[:, 0::2, 0::2])], axis=1)

    def resource_penalty(self, params, X):
        if self.penalty == "noise":  # mean added noise per quadrature, vacuum units
            Y = self._channel(params, X[0] if len(X) else (), "physical")[2]
            return np.trace(Y) / len(Y)
        if self.penalty == "squeezing":  # mean photons a squeeze gate adds to vacuum
            r = (
                self.program.squeezing_values(params, X[0] if len(X) else ())
                if isinstance(self.program, GaussianProgram)
                else []
            )
            return sum(np.sinh(v) ** 2 for v in r)
        return 0.0

    # ------------------------------------------------------------------ parameters
    def _names(self, n_features):
        return self.quantum_parameters + self._head_names(n_features)

    def _split(self, vector, n_features):
        names = self._names(n_features)
        values = dict(zip(names, vector))
        head = np.array([values[n] for n in names[len(self.quantum_parameters) :]])
        return {n: values[n] for n in self.quantum_parameters}, head

    def _n_features(self):
        k = len(self.readout_modes)
        return 2 * k if self.readout in ("qp", "moments") else k

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None):
        X, y = self.feature_map.check(X), self._targets(y)
        if len(X) != len(y) or not len(X):
            raise ValueError("X and y must be nonempty with the same number of samples")
        rng = np.random.default_rng(self.seed)
        nf = self._n_features()
        defaults = getattr(self.program, "defaults", {})  # e.g. CVCluster rung weights
        centre = np.array([defaults.get(n, 0.0) for n in self.quantum_parameters])
        init = np.r_[
            centre + rng.normal(0.0, self.init_scale, len(self.quantum_parameters)),
            self._head_init(nf),
        ]
        noise = None
        if self.shots is not None:
            settings = 2 if self.readout == "qp" else 1
            noise = rng.standard_normal((len(X), settings, len(self.readout_modes)))
        self._task_scale = self._scale(y)
        self._evaluations = 0

        def objective(vector):
            params, head = self._split(vector, nf)
            f = self.features(params, X, shots=self.shots, noise=noise)
            loss = self._loss(f, head, y) / self._task_scale
            if self.penalty is not None:
                loss = loss + self.penalty_weight * self.resource_penalty(params, X)
            return loss

        def validation(vector):
            params, head = self._split(np.asarray(vector, float), nf)
            return float(
                np.real(
                    self._loss(self.features(params, X_val), head, self._targets(y_val))
                    / self._task_scale
                )
            )

        if callable(self.gradient):
            grad = self.gradient
        elif self.gradient == "complex-step":
            grad = lambda v: complex_step_gradient(objective, v)  # noqa: E731
        else:
            grad = lambda v: central_difference(objective, v, self.fd_step)  # noqa: E731
        history = CVTrainingHistory(
            settings={
                "mode": self.mode,
                "squeezing": self.squeezing,
                "shots": self.shots,
                "gradient": self.gradient if isinstance(self.gradient, str) else "user",
                "complex_step": STEP,
                "fd_step": self.fd_step,
                "cutoff": self.cutoff,
                "optimizer": self.trainer.optimizer,
                "epochs": self.trainer.epochs,
                "learning_rate": self.trainer.learning_rate,
                "seed": self.seed,
                "penalty": self.penalty,
                "penalty_weight": self.penalty_weight,
                "parameter_names": list(self._names(nf)),
            }
        )
        start = perf_counter()

        def callback(point, trainer_history):
            history.parameters.append(point.tolist())
            history.evaluations.append(self._evaluations)
            history.seconds.append(perf_counter() - start)

        vector, h = self.trainer.fit(
            lambda v: float(np.real(objective(v))),
            init,
            gradient=grad,
            validation=validation if X_val is not None else None,
            callback=callback,
        )
        history.losses, history.validation_losses = h.losses, h.validation_losses
        history.gradient_norms = h.gradient_norms
        self.vector = np.asarray(vector, float)
        self.params, self.head = self._split(self.vector, nf)
        self.history = history
        return self

    def _require_fit(self):
        if self.params is None:
            raise RuntimeError("Call fit first")

    def get_params(self):
        self._require_fit()
        assert self.params is not None
        return dict(self.params), self.head.copy()

    def set_params(self, params, head):
        missing = set(self.quantum_parameters) - set(params)
        if missing:
            raise ValueError(f"Missing parameters {sorted(missing)}")
        self.params, self.head = dict(params), np.asarray(head, float)
        return self

    def resource_report(self, X=None):
        from .resources import resource_report

        self._require_fit()
        return resource_report(self, X)


class CVRegressor(_CVModel):
    """Affine classical head on CV readout features; MSE training (normalised when penalised)."""

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("Regression targets must be a finite vector")
        return y

    def _scale(self, y):
        return float(np.var(y)) if self.penalty is not None and np.var(y) > 0 else 1.0

    def _head_names(self, nf):
        return tuple(f"head.w{j}" for j in range(nf)) + ("head.b",)

    def _head_init(self, nf):
        return np.r_[np.ones(nf) / nf, 0.0]

    def _predict(self, f, head):
        return f @ head[:-1] + head[-1]

    def _loss(self, f, head, y):
        return np.mean((self._predict(f, head) - y) ** 2)

    def predict(self, X, *, mode=None, shots=None, seed=None):
        self._require_fit()
        noise = None
        if shots is not None:
            settings = 2 if self.readout == "qp" else 1
            noise = np.random.default_rng(seed).standard_normal(
                (len(X), settings, len(self.readout_modes))
            )
        return np.real(
            self._predict(
                self.features(self.params, X, mode=mode, shots=shots, noise=noise), self.head
            )
        )

    def score(self, X, y, **kw):
        prediction = self.predict(X, **kw)
        y = self._targets(y)
        return {
            "mse": float(np.mean((prediction - y) ** 2)),
            "mae": float(np.mean(np.abs(prediction - y))),
        }


class CVClassifier(_CVModel):
    """Binary (logistic or physical-homodyne head) or multiclass (softmax head) classifier."""

    def __init__(self, program, feature_map=None, *, head="logistic", classes=2, **kw):
        super().__init__(program, feature_map, **kw)
        if head not in ("logistic", "homodyne", "softmax"):
            raise ValueError("head must be logistic, homodyne or softmax")
        if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
            raise ValueError("classes must be an integer >= 2")
        if (classes > 2) != (head == "softmax"):
            raise ValueError("Use the softmax head exactly when classes > 2")
        if head == "homodyne" and (self.readout != "q" or self.shots is not None):
            raise ValueError(
                "The homodyne head reads the q quadrature of the first readout mode "
                "exactly; use readout='q' and no training shots"
            )
        self.head_kind, self.classes = head, classes

    def _targets(self, y):
        y = np.asarray(y)
        if y.ndim != 1 or not np.isin(y, np.arange(self.classes)).all():
            raise ValueError(f"Labels must be integers in range({self.classes})")
        return y.astype(int)

    def _scale(self, y):
        return 1.0

    def _head_names(self, nf):
        if self.head_kind == "homodyne":
            return ("head.threshold",)
        if self.head_kind == "logistic":
            return tuple(f"head.w{j}" for j in range(nf)) + ("head.b",)
        return tuple(f"head.W{k}.{j}" for k in range(self.classes) for j in range(nf)) + tuple(
            f"head.b{k}" for k in range(self.classes)
        )

    def _head_init(self, nf):
        return np.zeros(len(self._head_names(nf)))

    def _logits(self, f, head):
        if self.head_kind == "logistic":
            return f @ head[:-1] + head[-1]
        nf, k = f.shape[1], self.classes
        return f @ head[: k * nf].reshape(k, nf).T + head[k * nf :]

    def _loss(self, f, head, y):
        if self.head_kind == "homodyne":  # f = (mean q, var q) of the first readout mode
            z = (f[:, 0] - head[0]) / np.sqrt(f[:, 1])
            return -np.mean(np.where(y == 1, log_ndtr(z), log_ndtr(-z)))
        z = self._logits(f, head)
        if self.head_kind == "logistic":
            return np.mean(np.where(y == 1, _softplus(-z), _softplus(z)))
        shift = np.max(np.real(z), axis=1, keepdims=True)
        lse = shift[:, 0] + np.log(np.sum(np.exp(z - shift), axis=1))
        return np.mean(lse - z[np.arange(len(y)), y])

    def features(self, params, X, *, mode=None, shots=None, noise=None):
        if self.head_kind != "homodyne":
            return super().features(params, X, mode=mode, shots=shots, noise=noise)
        means, covs = self.moments(params, X, mode=mode)
        return np.stack([means[:, 0], covs[:, 0, 0]], axis=1)

    def predict_proba(self, X, *, mode=None, shots=None, seed=None):
        """Class probabilities (N, classes). With the homodyne head and ``shots`` the
        probability is the observed fraction of S binned homodyne outcomes."""
        self._require_fit()
        if self.head_kind == "homodyne":
            means, covs = self.moments(self.params, X, mode=mode)
            p1 = np.real(ndtr((means[:, 0] - self.head[0]) / np.sqrt(covs[:, 0, 0])))
            if shots is not None:
                p1 = np.random.default_rng(seed).binomial(shots, p1) / shots
            return np.stack([1 - p1, p1], axis=1)
        noise = None
        if shots is not None:
            settings = 2 if self.readout == "qp" else 1
            noise = np.random.default_rng(seed).standard_normal(
                (len(X), settings, len(self.readout_modes))
            )
        z = np.real(
            self._logits(
                self.features(self.params, X, mode=mode, shots=shots, noise=noise), self.head
            )
        )
        if self.head_kind == "logistic":
            p1 = 1 / (1 + np.exp(-z))
            return np.stack([1 - p1, p1], axis=1)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X, **kw):
        return np.argmax(self.predict_proba(X, **kw), axis=1)

    def score(self, X, y, **kw):
        y = self._targets(y)
        proba = self.predict_proba(X, **kw)
        return {
            "accuracy": float(np.mean(np.argmax(proba, axis=1) == y)),
            "log_loss": float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1)))),
        }
