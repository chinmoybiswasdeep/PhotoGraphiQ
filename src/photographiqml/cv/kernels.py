"""CV quantum kernels from closed-form Gaussian-state overlaps (hbar = 2).

``overlap``: K(x, x') = Tr[rho(x) rho(x')] (Hilbert-Schmidt). A Gram matrix of vectors in
operator space, hence symmetric positive semidefinite for any states; K(x, x) = purity <= 1.
``normalized``: Tr[rho rho'] / sqrt(Tr rho^2 Tr rho'^2), unit diagonal, still PSD.
``fidelity``: squared Uhlmann fidelity (Banchi-Braunstein-Pirandola for mixed states). Equal to
the overlap when either state is pure; for mixed states it differs and PSD is not guaranteed.

For displacement encoding of coherent states, K = exp(-scale^2 |x - x'|^2 / 4): a Gaussian RBF
kernel. A Gaussian unitary applied to every state leaves all three kernels unchanged; only a
noisy channel (finite-squeezing compilation) changes them, which is what ``mode="physical"``
exposes. The Gram matrix is meant for classical kernel methods (SVM, kernel ridge).
"""

from __future__ import annotations

import numpy as np

import photographiq as pg

from .gaussian import CVFeatureMap

KINDS = ("overlap", "normalized", "fidelity")


class CVKernel:
    def __init__(
        self,
        feature_map=None,
        program=None,
        params=None,
        *,
        kind="overlap",
        mode="ideal",
        squeezing=None,
        modes=None,
    ):
        if mode not in ("ideal", "physical"):
            raise ValueError("mode must be 'ideal' or 'physical'")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.feature_map = feature_map or CVFeatureMap(modes or (program.modes if program else 1))
        if program is not None and program.modes != self.feature_map.modes:
            raise ValueError("Feature map and program must act on the same modes")
        self.program, self.params = program, dict(params or {})
        self.kind, self.mode, self.squeezing = kind, mode, squeezing

    def states(self, X):
        X = self.feature_map.check(X)
        nodes = tuple(range(self.feature_map.modes))
        channel = None
        if self.program is not None and not getattr(self.program, "data_dependent", False):
            channel = self.program.channel(
                self.params, (), mode=self.mode, squeezing=self.squeezing
            )
        out = []
        for x in X:
            d, V = self.feature_map.moments(x)
            if self.program is not None:
                A, t, Y, _ = channel or self.program.channel(
                    self.params, x, mode=self.mode, squeezing=self.squeezing
                )
                d, V = A @ d + t, A @ V @ A.T + Y
            out.append(pg.GaussianState(np.real(d), np.real(V + V.T) / 2, nodes))
        return out

    def _value(self, a, b):
        if self.kind == "fidelity":
            return a.fidelity(b)
        value = a.overlap(b)
        if self.kind == "normalized":
            value /= np.sqrt(a.purity() * b.purity())
        return value

    def __call__(self, X1, X2):
        s1, s2 = self.states(X1), self.states(X2)
        return np.array([[self._value(a, b) for b in s2] for a in s1])

    def gram_matrix(self, X):
        s = self.states(X)
        gram = np.empty((len(s), len(s)))
        for i, a in enumerate(s):
            for j in range(i, len(s)):
                gram[i, j] = gram[j, i] = self._value(a, s[j])
        return gram

    def diagnostics(self, X):
        gram = self.gram_matrix(X)
        if not len(gram):
            raise ValueError("Diagnostics require at least one sample")
        return {
            "symmetry_error": float(np.max(abs(gram - gram.T))),
            "max_diagonal": float(np.max(np.diag(gram))),
            "min_diagonal": float(np.min(np.diag(gram))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
        }
