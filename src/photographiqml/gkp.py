"""Controlled GKP resource bridge; no implicit qubit-to-homodyne substitution."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

import photographiq as pg

from .logical import execute, statevector


@dataclass(frozen=True)
class GKPBridge:
    """One-mode finite-resource preparation and explicitly ideal logical targets.

    ``encode`` prepares a normalized finite superposition; because codewords
    overlap, this is not asserted to be an isometric quantum channel.
    """

    cutoff: int = 32
    peak_width: float = 0.4
    envelope: float = 0.4
    peaks: int = 8
    grid_points: int = 4097

    @property
    def resource_options(self):
        return {
            "peak_width": self.peak_width,
            "envelope": self.envelope,
            "peaks": self.peaks,
            "grid_points": self.grid_points,
        }

    def encode(self, logical_state):
        vector = statevector(logical_state, 1)
        return self.code.encode(*vector)

    @cached_property
    def code(self):
        return pg.GKPCode(cutoff=self.cutoff, **self.resource_options)

    def diagnostics(self):
        projections = [self.code.resource(bit).project(self.cutoff) for bit in (0, 1)]
        basis = np.column_stack([p[0].amplitudes for p in projections])
        gram = basis.conj().T @ basis
        return {
            "cutoff": self.cutoff,
            "captured_weights": [float(p[1]) for p in projections],
            "codeword_overlap": float(abs(gram[0, 1])),
            "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
            "physical_muta_validated": False,
        }

    def logical_target(self, model, input_state, parameters=None):
        """Execute ideal MuTA; result remains labeled logical, never finite GKP."""
        return execute(model, input_state, model._parameters.bind(parameters))

    def run(self, model, input_state=None, parameters=None, *, config=None, **options):
        from .lowering import GKPPhysicalConfig
        from .physical import run_physical

        if model.representation in ("gkp", "gkp-resource"):
            raise NotImplementedError(
                "Legacy resource models retain their safety boundary; explicitly choose PhysicalMuTA or validate a fixed logical model through the supported signed-X protocol"
            )
        config = (
            config
            or getattr(model, "physical_config", None)
            or GKPPhysicalConfig(cutoff=self.cutoff, **self.resource_options)
        )
        return run_physical(model, input_state, parameters, config=config, **options)

    def resource_readout(self, basis="Z", *, decoder="nearest"):
        """Calibrated one-mode ensemble readout; soft posteriors are not universal."""
        if basis not in ("X", "Z") or decoder not in ("nearest", "soft"):
            raise ValueError("Resource readout requires X/Z and nearest/soft")
        selected = (
            pg.NearestCellDecoder()
            if decoder == "nearest"
            else pg.SoftDecisionDecoder(self.code, basis=basis)
        )
        return self.code.logical_measurement(basis, decoder=selected)

    def measurement_convergence(self, values, **options):
        """Upstream one-resource study, distinct from whole-model convergence."""
        return pg.measurement_convergence(self.code, values, **options)
