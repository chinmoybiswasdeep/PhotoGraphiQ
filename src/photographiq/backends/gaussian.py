"""NumPy Gaussian reference backend with exact destructive conditioning."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, norm

from .. import gaussian as gm
from ..measurements import Generaldyne, Heterodyne, Homodyne
from ..states import GaussianInput, GaussianState
from .base import BaseBackend


class GaussianBackend(BaseBackend):
    capabilities = frozenset(
        {
            "gaussian",
            "homodyne",
            "noisy_homodyne",
            "heterodyne",
            "generaldyne",
            "loss",
            "quadratic_phase",
            "postselection",
        }
    )

    def quadratic_phase(self, node, s):
        self.transform((node,), np.array([[1.0, 0.0], [s, 1.0]]))

    def __init__(self):
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = GaussianState(np.zeros(0), np.zeros((0, 0)), ())
        self.last_measurement: dict = {}

    def _store(self, mean, cov, nodes=None):
        # Re-symmetrise: S V S^T leaves O(eps |V|) asymmetry, which exceeds the absolute
        # tolerance of GaussianState.validate once anti-squeezed variances are ~1e5.
        nodes = self.state.nodes if nodes is None else nodes
        self.state = GaussianState(mean, (cov + cov.T) / 2, nodes)

    def prepare(self, node, squeezing=1.0, state=None):
        if node in self.state.nodes:
            raise ValueError("Node already prepared")
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Resource squeezing must be finite and nonnegative")
        if state is None:
            state = GaussianInput.squeezed(-squeezing)
        if not isinstance(state, GaussianInput):
            raise NotImplementedError("Gaussian backend requires GaussianInput")
        local = state.state(node)
        cov = (
            local.covariance
            if not self.state.nodes
            else block_diag(self.state.covariance, local.covariance)
        )
        self.state = GaussianState(
            np.r_[self.state.mean, local.mean], cov, self.state.nodes + (node,)
        )

    def set_state(self, state):
        self.state = state.copy().validate()

    def transform(self, nodes, matrix):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate modes")
        indices = [
            j
            for n in nodes
            for j in (2 * self.state.nodes.index(n), 2 * self.state.nodes.index(n) + 1)
        ]
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (len(indices), len(indices)) or not np.allclose(
            matrix @ gm.omega(len(nodes)) @ matrix.T, gm.omega(len(nodes))
        ):
            raise ValueError("Transformation must be symplectic")
        full = gm.embed(matrix, indices, len(self.state.mean))
        self._store(full @ self.state.mean, full @ self.state.covariance @ full.T)

    def entangle(self, u, v, weight=1.0):
        self.transform((u, v), gm.cz(weight))

    def rotate(self, node, angle):
        self.transform((node,), gm.rotation(angle))

    def squeeze(self, node, r):
        self.transform((node,), gm.squeezing(r))

    def beamsplitter(self, u, v, theta):
        self.transform((u, v), gm.beamsplitter(theta))

    def displace(self, node, q=0.0, p=0.0):
        if not np.isfinite([q, p]).all():
            raise ValueError("Nonfinite displacement")
        i = self.state.nodes.index(node)
        self.state.mean[2 * i : 2 * i + 2] += [q, p]

    def measure(self, node, measurement, angle=0.0, *, outcome=None):
        """Destructively measure one mode and condition the rest exactly (Schur complement).

        Homodyne records q cos(angle) + p sin(angle) calibrated to the incident quadrature,
        with variance Var + (1-eta)/eta + noise. Heterodyne/Generaldyne record (q, p) with
        covariance V + I or V + seed covariance. ``outcome`` fixes the record (postselection)
        instead of sampling it; the update is the same in both cases. ``last_measurement``
        holds the exact Gaussian density of the record given the pre-measurement state.

        Args:
            node (object): Hashable mode label.
            measurement (object): Homodyne, Heterodyne or Generaldyne description.
            angle (float): Homodyne angle in radians.
            outcome (object): Fixed real scalar (Homodyne) or real pair, or None to sample.

        Returns:
            result (float or tuple): Sampled or postselected calibrated outcome.

        Raises:
            NotImplementedError: Photon counting has non-Gaussian conditional states.
            ValueError: Outcome must be finite and real, with the measurement's shape.
        """
        postselected = outcome is not None
        i = self.state.nodes.index(node)
        measured = [2 * i, 2 * i + 1]
        remaining = [j for j in range(len(self.state.mean)) if j not in measured]
        a = self.state.covariance[np.ix_(remaining, remaining)]
        b = self.state.covariance[np.ix_(measured, measured)]
        c = self.state.covariance[np.ix_(remaining, measured)]
        mu = self.state.mean[measured]
        if isinstance(measurement, Homodyne):
            vector = np.array([np.cos(angle), np.sin(angle)])
            # Outcome is calibrated to the incident quadrature. Inefficiency adds vacuum noise.
            variance = float(
                vector @ b @ vector
                + (1 - measurement.efficiency) / measurement.efficiency
                + measurement.noise
            )
            if variance <= 0 or not np.isfinite(variance):
                raise ValueError("Singular homodyne variance")
            expected = float(vector @ mu)
            if outcome is None:
                outcome = self.rng.normal(expected, np.sqrt(variance))
            elif not np.isscalar(outcome) or not np.isreal(outcome) or not np.isfinite(outcome):
                raise ValueError("Homodyne outcome must be a finite real scalar")
            outcome = float(outcome)
            density = norm.pdf(outcome, expected, np.sqrt(variance))
            cross = c @ vector
            mean = self.state.mean[remaining] + cross * ((outcome - expected) / variance)
            cov = a - np.outer(cross, cross) / variance
        elif isinstance(measurement, (Heterodyne, Generaldyne)):
            detector = (
                np.eye(2)
                if isinstance(measurement, Heterodyne)
                else np.asarray(measurement.covariance)
            )
            total = b + detector
            if outcome is None:
                vector_outcome = self.rng.multivariate_normal(mu, total)
            else:
                vector_outcome = np.asarray(outcome)
                if (
                    vector_outcome.shape != (2,)
                    or not np.isrealobj(vector_outcome)
                    or not np.isfinite(vector_outcome).all()
                ):
                    raise ValueError("Phase-space outcome must be a finite real pair (q, p)")
                vector_outcome = vector_outcome.astype(float)
            density = multivariate_normal.pdf(vector_outcome, mu, total)
            gain = np.linalg.solve(total, c.T).T
            mean = self.state.mean[remaining] + gain @ (vector_outcome - mu)
            cov = a - gain @ c.T
            outcome = tuple(map(float, vector_outcome))
        else:
            raise NotImplementedError(
                "Photon counting has non-Gaussian conditional states; use a Fock backend "
                "(native-fock, native-mixed-fock, piquasso-fock or piquasso-mixed-fock)"
            )
        if postselected and not density > 0:  # same contract as the Fock backends
            raise ValueError("Postselected outcome has zero numerical density")
        self._store(mean, cov, tuple(n for n in self.state.nodes if n != node))
        self.last_measurement = {"kind": "density", "value": float(density)}
        return outcome

    def loss(self, node, transmissivity, thermal_photons=0.0):
        if not 0 <= transmissivity <= 1 or not np.isfinite(thermal_photons) or thermal_photons < 0:
            raise ValueError("Invalid loss parameters")
        i = self.state.nodes.index(node)
        x = np.eye(len(self.state.mean))
        x[2 * i, 2 * i] = x[2 * i + 1, 2 * i + 1] = np.sqrt(transmissivity)
        noise = np.zeros_like(x)
        noise[2 * i, 2 * i] = noise[2 * i + 1, 2 * i + 1] = (1 - transmissivity) * (
            2 * thermal_photons + 1
        )
        self._store(x @ self.state.mean, x @ self.state.covariance @ x.T + noise)

    def get_state(self, nodes=None):
        return self.state.copy() if nodes is None else self.state.reduced(nodes)
