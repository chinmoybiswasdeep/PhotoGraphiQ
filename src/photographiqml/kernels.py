"""Paper Eq. 5 feature map realized through a logical MuTA layer."""

import numpy as np

from .ansatz.muta import MuTA


class MuTAKernel:
    def __init__(self):
        self.model = MuTA(2, 1, one_column=True)

    def features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
            raise ValueError(
                "Kernel inputs must have shape (N,2) with finite entries; no scaling is implicit"
            )
        states = []
        for x0, x1 in X:
            angles = {
                "alpha.w0.c0": x0,
                "alpha.w1.c0": x1,
                "alpha.w1.c1": np.cos(x0) * np.cos(x1),
                "alpha.w0.c2": x0,
                "alpha.w1.c2": x1,
            }
            states.append(self.model.run([1, 0, 0, 0], angles).state)
        return np.asarray(states).reshape(-1, 4)

    def __call__(self, X1, X2):
        return abs(self.features(X1).conj() @ self.features(X2).T) ** 2

    def gram_matrix(self, X):
        states = self.features(X)
        return abs(states.conj() @ states.T) ** 2

    def diagnostics(self, X):
        gram = self.gram_matrix(X)
        if not len(gram):
            raise ValueError("PSD diagnostics require at least one sample")
        return {
            "symmetry_error": float(np.max(abs(gram - gram.T))),
            "diagonal_error": float(np.max(abs(np.diag(gram) - 1))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
        }
