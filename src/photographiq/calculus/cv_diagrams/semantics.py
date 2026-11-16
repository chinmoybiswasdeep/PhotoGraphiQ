"""Gaussian semantics of typed CV diagrams: unitary, channel, conditional instrument.

Hybrid classical-quantum Gaussian semantics. Every live coordinate -- the (q, p) of each live
mode and every classical record -- is an affine function of the input quadratures plus an
input-independent Gaussian part:

    z = A x_in + b + xi,     xi ~ N(0, Y)     (Wigner variables for modes, values for records)

A homodyne measurement at angle theta replaces the two coordinates of its mode by one record
coordinate h . (q, p) (h = (cos theta, sin theta)) plus detector noise (1 - eta)/eta + noise:
the joint law of the record and the remaining Wigner variables is the Wigner marginal over the
conjugate quadrature, which is exactly Gaussian, so

* the record density and the conditional output state are Gaussian conditioning (Schur
  complement) of the joint law on the record coordinates (:meth:`HybridGaussian.conditional`);
* the outcome-averaged process (records discarded) is the Gaussian channel obtained by
  dropping the record coordinates (:meth:`HybridGaussian.channel`).

Feed-forward displacements are affine in records, so they are linear maps on the joint
coordinates. Non-Gaussian boxes raise ``NotImplementedError`` (use ``fock.py``).
This is an independent implementation: tests compare it with
``photographiq.analysis.gaussian_channel`` and with trajectory execution on the Gaussian
backend (sequential Schur-complement conditioning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

from ... import gaussian as gm
from .ir import (
    CZ,
    BeamSplitter,
    ClassicalSource,
    CVDiagram,
    Discard,
    Displacement,
    Feedforward,
    GaussianChannelBox,
    GaussianLoss,
    Homodyne,
    PhasePolynomial,
    Prepare,
    PrepareState,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)


def box_symplectic(box):
    """(S, d) of a Gaussian unitary box on ``box.modes`` (interleaved)."""
    if isinstance(box, Rotation):
        return gm.rotation(box.theta), np.zeros(2)
    if isinstance(box, Squeezer):
        return gm.squeezing(box.r), np.zeros(2)
    if isinstance(box, Shear):
        return np.array([[1.0, 0.0], [box.s, 1.0]]), np.zeros(2)
    if isinstance(box, Displacement):
        return np.eye(2), np.array([box.dq, box.dp], float)
    if isinstance(box, BeamSplitter):
        return gm.beamsplitter(box.theta), np.zeros(4)
    if isinstance(box, CZ):
        return gm.cz(box.g), np.zeros(4)
    if isinstance(box, Symplectic):
        return box.S, box.d
    if isinstance(box, PhasePolynomial) and box.gaussian:
        c = list(box.coefficients) + [0.0] * 3
        # exp(i f(q)): p -> p + 2 f'(q)  (hbar = 2), f = c0 + c1 q + c2 q^2
        return np.array([[1.0, 0.0], [4 * c[2], 1.0]]), np.array([0.0, 2 * c[1]])
    raise TypeError(f"{type(box).__name__} is not a Gaussian unitary")


@dataclass(frozen=True)
class HybridGaussian:
    """z = A x_in + b + xi, cov(xi) = Y, over labelled coordinates ('q'|'p'|'r', label)."""

    coords: tuple
    A: np.ndarray
    b: np.ndarray
    Y: np.ndarray
    inputs: tuple
    outputs: tuple
    records: tuple

    def _index(self, kind, label):
        return self.coords.index((kind, label))

    def _rows(self, labels):
        return [self._index(k, m) for m in labels for k in ("q", "p")]

    def channel(self):
        """Outcome-averaged channel on the outputs: (X, Y, d) with x_out = X x_in + d + noise."""
        idx = self._rows(self.outputs)
        return self.A[idx], self.Y[np.ix_(idx, idx)], self.b[idx]

    def unitary(self, atol=1e-10):
        X, Y, d = self.channel()
        if self.records or set(self.inputs) != set(self.outputs) or np.abs(Y).max(initial=0) > atol:
            raise ValueError("Diagram is not a Gaussian unitary (records, noise or wire change)")
        return X, d

    def joint(self, mean_in=None, cov_in=None):
        """(mean, covariance) of (outputs, records) for a Gaussian input state."""
        n = 2 * len(self.inputs)
        mean_in = np.zeros(n) if mean_in is None else np.asarray(mean_in, float)
        cov_in = np.eye(n) if cov_in is None else np.asarray(cov_in, float)
        idx = self._rows(self.outputs) + [self._index("r", k) for k in self.records]
        A, b = self.A[idx], self.b[idx]
        return A @ mean_in + b, A @ cov_in @ A.T + self.Y[np.ix_(idx, idx)]

    def conditional(self, outcomes, mean_in=None, cov_in=None):
        """Conditional output (mean, covariance) and the joint record density for ``outcomes``.

        ``outcomes`` must give every record. The density is a probability *density* in the
        record units (product of homodyne calibrations), not a probability.
        """
        missing = set(self.records) - set(outcomes)
        if missing:
            raise ValueError(f"Missing outcomes for records {sorted(map(repr, missing))}")
        mean, cov = self.joint(mean_in, cov_in)
        k = 2 * len(self.outputs)
        m = np.array([float(outcomes[r]) for r in self.records])
        ca, cb, cc = cov[:k, :k], cov[:k, k:], cov[k:, k:]
        if not len(m):
            return mean, cov, 1.0
        gain = np.linalg.solve(cc, cb.T).T
        cond_mean = mean[:k] + gain @ (m - mean[k:])
        cond_cov = ca - gain @ cb.T
        density = float(multivariate_normal.pdf(m, mean[k:], cc, allow_singular=False))
        return cond_mean, (cond_cov + cond_cov.T) / 2, density


def evaluate(diagram: CVDiagram) -> HybridGaussian:
    """Hybrid Gaussian semantics of a Gaussian diagram (see module docstring)."""
    if not diagram.is_gaussian:
        bad = sorted({type(b).__name__ for b in diagram.boxes if not b.gaussian})
        raise NotImplementedError(
            f"Non-Gaussian boxes {bad} have no Gaussian semantics; use calculus.cv_diagrams.fock"
        )
    n_in = 2 * len(diagram.inputs)
    coords = [(k, m) for m in diagram.inputs for k in ("q", "p")]
    A = np.eye(n_in)
    b = np.zeros(n_in)
    Y = np.zeros((n_in, n_in))

    def grow(new, mean, cov):
        nonlocal A, b, Y
        coords.extend(new)
        A = np.vstack([A, np.zeros((len(new), n_in))])
        b = np.r_[b, mean]
        old = len(Y)
        Y2 = np.zeros((old + len(new),) * 2)
        Y2[:old, :old], Y2[old:, old:] = Y, cov
        Y = Y2

    def drop(rows):
        nonlocal A, b, Y
        keep = [i for i in range(len(coords)) if i not in rows]
        for i in sorted(rows, reverse=True):
            coords.pop(i)
        A, b, Y = A[keep], b[keep], Y[np.ix_(keep, keep)]

    def transform(rows, M, shift=0.0):
        nonlocal Y
        A[rows] = M @ A[rows]
        b[rows] = M @ b[rows] + shift
        Y[rows] = M @ Y[rows]
        Y[:, rows] = Y[:, rows] @ M.T

    def rows_of(modes):
        return [coords.index((k, m)) for m in modes for k in ("q", "p")]

    records = []
    for box in diagram.boxes:
        if isinstance(box, Prepare):
            grow(
                [("q", box.mode), ("p", box.mode)],
                np.asarray(box.mean, float),
                np.asarray(box.covariance, float),
            )
        elif isinstance(box, PrepareState):
            grow([(k, m) for m in box.labels for k in ("q", "p")], box.mean, box.covariance)
        elif box.unitary and not isinstance(box, Feedforward):
            S, d = box_symplectic(box)
            transform(rows_of(box.modes), S, d)
        elif isinstance(box, GaussianLoss):
            rows = rows_of(box.modes)
            transform(rows, np.sqrt(box.eta) * np.eye(2))
            Y[np.ix_(rows, rows)] += (1 - box.eta) * (2 * box.thermal + 1) * np.eye(2)
        elif isinstance(box, GaussianChannelBox):
            rows = rows_of(box.modes)
            transform(rows, box.X, box.d)
            Y[np.ix_(rows, rows)] += box.Y
        elif isinstance(box, Homodyne):
            rows = rows_of((box.mode,))
            h = np.array([np.cos(box.theta), np.sin(box.theta)])
            row_A, row_b = h @ A[rows], h @ b[rows]
            cross = h @ Y[rows]
            var = h @ Y[np.ix_(rows, rows)] @ h + (1 - box.efficiency) / box.efficiency + box.noise
            grow([("r", box.key)], np.array([row_b]), np.array([[var]]))
            A[-1] = row_A
            Y[-1, :-1] = Y[:-1, -1] = cross
            drop(rows)
            records.append(box.key)
        elif isinstance(box, ClassicalSource):
            grow([("r", box.key)], np.array([box.mean]), np.array([[box.variance]]))
            records.append(box.key)
        elif isinstance(box, Feedforward):
            rows = rows_of((box.mode,))
            for key, (gq, gp) in box.gains:
                r = coords.index(("r", key))
                # T = I + gq e_q e_r^T + gp e_p e_r^T as rank-one row/column updates (O(n))
                for row, gain in ((rows[0], gq), (rows[1], gp)):
                    if gain:
                        A[row] += gain * A[r]
                        b[row] += gain * b[r]
                        Y[row] += gain * Y[r]
                        Y[:, row] += gain * Y[:, r]
            b[rows] += np.asarray(box.constant, float)
        elif isinstance(box, Discard):
            drop(rows_of((box.mode,)))
        else:
            raise NotImplementedError(f"No Gaussian semantics for {type(box).__name__}")
    return HybridGaussian(
        tuple(coords), A, b, (Y + Y.T) / 2, diagram.inputs, diagram.outputs, tuple(records)
    )


def channel_semantics(diagram):
    """(X, Y, d) of the outcome-averaged process inputs -> outputs."""
    return evaluate(diagram).channel()


def unitary_semantics(diagram):
    """(S, d) of a Gaussian unitary diagram; raises if it has records, noise or wire changes."""
    return evaluate(diagram).unitary()


def conditional_semantics(diagram, outcomes, mean_in=None, cov_in=None):
    """(conditional output mean, covariance, record density) for given outcomes and input."""
    return evaluate(diagram).conditional(outcomes, mean_in, cov_in)


def is_completely_positive(X, Y, tol=1e-9):
    """Y + i Omega - i X Omega X^T >= 0 (hbar = 2, vacuum I)."""
    n = len(X) // 2
    test = Y + 1j * gm.omega(n) - 1j * X @ gm.omega(n) @ X.T
    return bool(
        np.linalg.eigvalsh((test + test.conj().T) / 2).min()
        >= -tol * max(1.0, abs(Y).max(initial=0))
    )


def compose_channels(first, second):
    """(X2 X1, X2 Y1 X2^T + Y2, X2 d1 + d2): ``first`` then ``second`` on the same modes."""
    X1, Y1, d1 = first
    X2, Y2, d2 = second
    return X2 @ X1, X2 @ Y1 @ X2.T + Y2, X2 @ d1 + d2
