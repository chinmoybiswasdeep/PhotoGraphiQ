"""Piquasso physical gates, with an explicit exact Gaussian measurement adapter.

No private Piquasso APIs are used. Statistical covariance V = sigma_Piquasso/2.
"""

from __future__ import annotations

import numpy as np
import piquasso as pq

from ..states import GaussianInput, GaussianState
from .gaussian import GaussianBackend


class PiquassoBackend(GaussianBackend):
    def _native(self):
        state = pq.GaussianState(
            d=len(self.state.nodes), connector=pq.NumpyConnector(), config=pq.Config(hbar=2.0)
        )
        state.xpxp_mean_vector = self.state.mean.copy()
        state.xpxp_covariance_matrix = 2 * self.state.covariance
        return state

    def _gate(self, nodes, instruction):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate modes")
        modes = tuple(self.state.nodes.index(n) for n in nodes)
        with pq.Program() as program:
            pq.Q(*modes) | instruction
        native = (
            pq.GaussianSimulator(d=len(self.state.nodes), config=pq.Config(hbar=2.0))
            .execute(program, initial_state=self._native())
            .state
        )
        self.state = GaussianState(
            native.xpxp_mean_vector, native.xpxp_covariance_matrix / 2, self.state.nodes
        )

    def prepare(self, node, squeezing=1.0, state=None):
        """Prepare a fresh labelled mode; resource squeezing is momentum squeezing.

        Args:
            node (object): Hashable mode label.
            squeezing (float): Finite momentum resource squeezing; nonnegative.
            state (object): Supported state preparation or independent state snapshot.

        Raises:
            ValueError: Invalid resource squeezing.
        """
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Invalid resource squeezing")
        super().prepare(node, squeezing=0.0, state=GaussianInput() if state is None else state)
        if state is None:
            self._gate((node,), pq.Squeezing(r=-squeezing))

    def entangle(self, u, v, weight=1.0):
        """Apply weighted controlled-Z to two existing modes.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            weight (float): Real controlled-Z edge weight.
        """
        self._gate((u, v), pq.ControlledZ(s=weight))

    def displace(self, node, q=0.0, p=0.0):
        """Apply or append quadrature translations q and p in hbar=2 coordinates.

        Args:
            node (object): Hashable mode label.
            q (float): Position translation; hbar=2 quadrature units.
            p (float): Momentum translation; hbar=2 quadrature units.
        """
        alpha = complex(q, p) / 2
        self._gate((node,), pq.Displacement(r=abs(alpha), phi=np.angle(alpha)))

    def rotate(self, node, angle):
        """Append a rotation by angle radians to this optical circuit.

        Args:
            node (object): Hashable mode label.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
        """
        self._gate((node,), pq.Phaseshifter(phi=angle))

    def squeeze(self, node, r):
        """Append or apply q squeezing by parameter r.

        Args:
            node (object): Hashable mode label.
            r (float): Dimensionless squeezing parameter.
        """
        self._gate((node,), pq.Squeezing(r=r))

    def quadratic_phase(self, node, s):
        self._gate((node,), pq.QuadraticPhase(s=s))

    def beamsplitter(self, u, v, theta):
        """Mix two optical modes using the package real beamsplitter convention.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            theta (float): Beamsplitter mixing angle in radians.
        """
        self._gate((u, v), pq.Beamsplitter(theta=theta, phi=0.0))

    def loss(self, node, transmissivity, thermal_photons=0.0):
        """Apply attenuation with thermal environment noise where supported.

        Args:
            node (object): Hashable mode label.
            transmissivity (float): Intensity transmission in [0,1].
            thermal_photons (float): Nonnegative mean environment occupation.

        Raises:
            ValueError: Invalid loss parameters.
        """
        if not 0 <= transmissivity <= 1 or not np.isfinite(thermal_photons) or thermal_photons < 0:
            raise ValueError("Invalid loss parameters")
        self._gate(
            (node,),
            pq.Attenuator(
                theta=np.arccos(np.sqrt(transmissivity)), mean_thermal_excitation=thermal_photons
            ),
        )

    def import_state(self, state, nodes, *, source_hbar):
        """Inject an externally prepared, possibly correlated Piquasso Gaussian state."""
        if not isinstance(state, pq.GaussianState):
            raise TypeError("Expected Piquasso GaussianState")
        if not np.isfinite(source_hbar) or source_hbar <= 0:
            raise ValueError("Declare the external state's positive hbar convention")
        self.set_state(
            GaussianState(
                state.xpxp_mean_vector * np.sqrt(2 / source_hbar),
                state.xpxp_covariance_matrix / source_hbar,
                tuple(nodes),
            )
        )

    def export_state(self):
        """Return an independent native Piquasso Gaussian representation."""
        return self._native()
