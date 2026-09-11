"""Density-matrix trajectories with explicit total-photon truncation, hbar=2."""

import numpy as np
import piquasso as pq
from scipy.integrate import quad
from scipy.optimize import brentq

from ..fock_measurements import wavefunctions
from ..measurements import Homodyne, PhotonNumber
from ..states import FockDensityMatrix, GaussianInput
from .fock import PiquassoFockBackend


class MixedFockBackend(PiquassoFockBackend):
    """Mixed Fock evolution using native Piquasso gates and local conditioning.

    Args:
        cutoff (int): Exclusive total-photon cutoff.
        max_matrix_bytes (object): Limit for one complex128 density matrix; workspaces
            need additional memory.
        **kwargs (dict): Pure backend norm and boundary diagnostic settings.

    Raises:
        MemoryError: A density matrix exceeds the configured allocation budget.
    """

    capabilities = PiquassoFockBackend.capabilities | frozenset(
        {
            "mixed_fock",
            "loss",
            "noisy_homodyne",
            "custom_kraus_instrument",
            "finite_outcome_povm",
            "mixed_conditional_update",
        }
    )

    def __init__(self, cutoff=None, *, max_matrix_bytes=256_000_000, **kwargs):
        if (
            isinstance(max_matrix_bytes, bool)
            or not isinstance(max_matrix_bytes, int)
            or max_matrix_bytes < 16
        ):
            raise ValueError("max_matrix_bytes must be an integer >= 16")
        self.max_matrix_bytes = max_matrix_bytes
        super().__init__(cutoff=cutoff, **kwargs)

    def _guard(self, modes):
        super()._guard(modes)
        size = 16 * self.dimension(modes) ** 2
        if size > self.max_matrix_bytes:
            raise MemoryError(
                f"Density matrix needs {size} bytes; reduce modes/cutoff or set max_matrix_bytes"
            )

    def _density(self, matrix, basis):
        # Native occupations can use int32 while local resources use Python
        # ints. Numba's dynamically indexed tuples require homogeneous types.
        basis = tuple(tuple(int(n) for n in b) for b in basis)
        modes = len(basis[0])
        self._guard(modes)
        with pq.Program() as program:
            for i, j in zip(*np.nonzero(matrix), strict=True):
                pq.Q() | pq.DensityMatrix(basis[i], basis[j], complex(matrix[i, j]))
        return pq.FockSimulator(d=modes, config=self._config()).execute(program).state

    def validate_preparation(self, state, modes=1):
        if isinstance(state, FockDensityMatrix):
            if state.modes != modes or any(sum(b) >= self.cutoff for b in state.basis):
                raise ValueError("Density-matrix basis must match resource modes and total cutoff")
            self._guard(modes)
        elif isinstance(state, GaussianInput):
            state.state(0).validate()
            if modes != 1:
                raise ValueError("GaussianInput requires one mode")
        else:
            super().validate_preparation(state, modes)

    def prepare(self, node, squeezing=1.0, state=None):
        """Prepare a fresh labelled mode; resource squeezing is momentum squeezing.

        Args:
            node (object): Hashable mode label.
            squeezing (float): Finite momentum resource squeezing; nonnegative.
            state (object): Supported state preparation or independent state snapshot.
        """
        self.prepare_resource((node,), state, squeezing=squeezing)

    def prepare_resource(self, nodes, state, *, squeezing=1.0):
        nodes = tuple(nodes)
        self.validate_preparation(state, len(nodes))
        if len(set(nodes)) != len(nodes) or set(nodes) & set(self.nodes):
            raise ValueError("Resource nodes must be new and unique")
        self._guard(len(self.nodes) + len(nodes))
        if isinstance(state, FockDensityMatrix):
            local_basis, local = state.basis, np.asarray(state.matrix)
        elif isinstance(state, GaussianInput):
            with pq.Program() as program:
                pq.Q() | pq.Vacuum()
            native = pq.GaussianSimulator(d=1, config=self._config()).execute(program).state
            native.xpxp_mean_vector = np.array(state.mean)
            native.xpxp_covariance_matrix = 2 * np.array(state.covariance)
            local = native.density_matrix
            local_basis = tuple((n,) for n in range(self.cutoff))
        else:
            helper = PiquassoFockBackend(self.cutoff, norm_tolerance=self.norm_tolerance)
            if len(nodes) == 1:
                helper.prepare(nodes[0], squeezing, state)
            else:
                helper.prepare_resource(nodes, state)
            local_state = helper.get_state()
            self.retained_norms.extend(helper.retained_norms)
            self.diagnostics.extend(dict(item) for item in helper.diagnostics)
            local, local_basis = local_state.density_matrix, local_state.basis
        old_basis = ((),) if self.native is None else tuple(self.native.fock_probabilities_map)
        old = np.ones((1, 1)) if self.native is None else self.native.density_matrix
        pairs = [
            (i, j)
            for i, b in enumerate(old_basis)
            for j, c in enumerate(local_basis)
            if sum(b + c) < self.cutoff
        ]
        basis = tuple(old_basis[i] + local_basis[j] for i, j in pairs)
        ii, jj = np.array(pairs).T
        matrix = old[np.ix_(ii, ii)] * local[np.ix_(jj, jj)]
        self.native = self._check(self._density(matrix, basis))
        self.nodes += nodes

    def _gate(self, nodes, instruction):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate mode")
        self._guard(len(self.nodes))
        with pq.Program() as program:
            pq.Q(*(self.nodes.index(n) for n in nodes)) | instruction
        self.native = self._check(
            pq.FockSimulator(d=len(self.nodes), config=self._config())
            .execute(program, initial_state=self.native)
            .state,
            type(instruction).__name__,
        )

    def loss(self, node, transmissivity, thermal_photons=0.0):
        """Apply attenuation with thermal environment noise where supported.

        Args:
            node (object): Hashable mode label.
            transmissivity (float): Intensity transmission in [0,1].
            thermal_photons (float): Nonnegative mean environment occupation.

        Raises:
            ValueError: Loss requires 0 <= transmissivity <= 1 and thermal_photons >= 0.
        """
        if (
            not np.isfinite([transmissivity, thermal_photons]).all()
            or not 0 <= transmissivity <= 1
            or thermal_photons < 0
        ):
            raise ValueError("Loss requires 0 <= transmissivity <= 1 and thermal_photons >= 0")
        # Thermal attenuation = vacuum attenuation eta/G followed by a
        # quantum-limited amplifier G=1+(1-eta)*n_env. The upstream Fock
        # attenuator rejects nonzero thermal occupation, so use explicit Kraus
        # operators for that path. This decomposition retains correlations.
        from math import comb

        gain = 1 + (1 - transmissivity) * thermal_photons
        eta = transmissivity / gain
        basis = tuple(self.native.fock_probabilities_map)
        lookup = {b: i for i, b in enumerate(basis)}
        mode = self.nodes.index(node)
        rho = self.native.density_matrix
        lost_state = np.zeros_like(rho)
        for lost in range(self.cutoff):
            sources, targets, factors = [], [], []
            for index, b in enumerate(basis):
                n = b[mode]
                if n >= lost:
                    target = list(b)
                    target[mode] -= lost
                    sources.append(index)
                    targets.append(lookup[tuple(target)])
                    factors.append(np.sqrt(comb(n, lost) * (1 - eta) ** lost * eta ** (n - lost)))
            lost_state[np.ix_(targets, targets)] += rho[np.ix_(sources, sources)] * np.outer(
                factors, factors
            )
        if gain == 1:
            output = lost_state
        else:
            output = np.zeros_like(rho)
            for added in range(self.cutoff):
                sources, targets, factors = [], [], []
                for index, b in enumerate(basis):
                    if sum(b) + added < self.cutoff:
                        target = list(b)
                        target[mode] += added
                        sources.append(index)
                        targets.append(lookup[tuple(target)])
                        factors.append(
                            np.sqrt(
                                comb(b[mode] + added, added)
                                * (gain - 1) ** added
                                / gain ** (b[mode] + added + 1)
                            )
                        )
                output[np.ix_(targets, targets)] += lost_state[np.ix_(sources, sources)] * np.outer(
                    factors, factors
                )
        self.native = self._check(self._density(output, basis), "ThermalLoss")

    def ladder(self, node, addition):
        basis = tuple(self.native.fock_probabilities_map)
        lookup = {b: j for j, b in enumerate(basis)}
        op = np.zeros((len(basis), len(basis)))
        expected = 0.0
        mode = self.nodes.index(node)
        rho = self.native.density_matrix
        for j, b in enumerate(basis):
            factor = b[mode] + int(addition)
            expected += factor * rho[j, j].real
            target = list(b)
            target[mode] += 1 if addition else -1
            if factor and tuple(target) in lookup:
                op[lookup[tuple(target)], j] = np.sqrt(factor)
        projected = op @ rho @ op.T
        mass = float(np.trace(projected).real)
        if mass <= 1e-28:
            raise ValueError("Ideal ladder operation has zero norm")
        if expected - mass > 1e-12:
            raise ValueError("Photon addition exceeds cutoff; increase cutoff")
        self.native = self._density(projected / mass, basis)
        self.diagnostics.append(
            {
                "operation": "PhotonAdd" if addition else "PhotonSubtract",
                "ladder_norm_squared": mass,
            }
        )

    def measure(self, node, measurement, angle=0.0, *, outcome=None):
        """Execute a destructive measurement on the density matrix.

        Args:
            node (object): Hashable mode label.
            measurement (object): Homodyne, Heterodyne, Generaldyne or PhotonNumber description.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
            outcome (object): Fixed postselection value, or None to sample.

        Returns:
            result (float or int): Sampled or postselected measurement outcome.

        Raises:
            NotImplementedError: Mixed Fock supports PhotonNumber and Homodyne.
            ValueError: Measurement outcome must be finite and real.
            ValueError: Postselection has zero numerical probability/density.
            ArithmeticError: Noisy homodyne convolution did not converge.
            ValueError: Photon count must be an integer within cutoff.
        """
        from ..encoded import PhysicalGKPReadout
        from ..instruments import MeasurementInstrument, destructive_instrument

        if isinstance(measurement, PhysicalGKPReadout):
            return measurement.execute(self, node, outcome)
        if isinstance(measurement, MeasurementInstrument):
            return destructive_instrument(self, node, measurement, outcome, mixed=True)
        if not isinstance(measurement, (Homodyne, PhotonNumber)):
            raise NotImplementedError("Mixed Fock supports PhotonNumber and Homodyne")
        if isinstance(measurement, Homodyne) and measurement.efficiency != 1:
            self.loss(node, measurement.efficiency)
        basis = tuple(self.native.fock_probabilities_map)
        rho = self.native.density_matrix
        mode = self.nodes.index(node)
        survivors = tuple(dict.fromkeys(b[:mode] + b[mode + 1 :] for b in basis))
        lookup = {b: i for i, b in enumerate(survivors)}
        counts = np.array([b[mode] for b in basis])
        rows = np.array([lookup[b[:mode] + b[mode + 1 :]] for b in basis])

        def project(value):
            if isinstance(measurement, PhotonNumber):
                weights = (counts == value).astype(complex)
            else:
                # Outcomes use incident-quadrature calibration, as GaussianBackend.
                scale = np.sqrt(measurement.efficiency)
                weights = (
                    np.sqrt(scale)
                    * wavefunctions(value * scale, self.cutoff)[counts]
                    * np.exp(-1j * angle * counts)
                )
            op = np.zeros((len(survivors), len(basis)), complex)
            op[rows, np.arange(len(basis))] = weights
            return op @ rho @ op.conj().T

        def conditioned(value):
            if not isinstance(measurement, Homodyne) or measurement.noise == 0:
                return project(value)
            # Electronics noise is a classical convolution in detected units.
            from scipy.integrate import quad_vec

            sigma = np.sqrt(measurement.noise)
            matrix, error = quad_vec(
                lambda z: project(value + sigma * z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
                -10,
                10,
                epsabs=1e-10,
                epsrel=1e-9,
            )
            if error > 1e-7:
                raise ArithmeticError("Noisy homodyne convolution did not converge")
            return matrix

        def density(value):
            return float(np.trace(conditioned(value)).real)

        if isinstance(measurement, PhotonNumber):
            probabilities = np.bincount(counts, weights=rho.diagonal().real, minlength=self.cutoff)
            if outcome is None:
                outcome = int(self.rng.choice(self.cutoff, p=probabilities / probabilities.sum()))
            if (
                isinstance(outcome, bool)
                or not isinstance(outcome, (int, np.integer))
                or not 0 <= outcome < self.cutoff
            ):
                raise ValueError("Photon count must be an integer within cutoff")
        elif outcome is None:
            bound = (2 * np.sqrt(self.cutoff) + 10) / np.sqrt(
                measurement.efficiency
            ) + 10 * np.sqrt(measurement.noise)

            def integrate(end):
                mass, error = quad(density, -bound, end, epsabs=1e-9, limit=300)
                if not np.isfinite(mass) or error > 1e-7:
                    raise ArithmeticError("Mixed homodyne CDF integration failed")
                return mass

            mass = integrate(bound)
            if abs(mass - 1) > 1e-7:
                raise ArithmeticError("Mixed homodyne density is not normalized")
            target = self.rng.random() * mass
            outcome = brentq(lambda x: integrate(x) - target, -bound, bound)
        if not np.isscalar(outcome) or not np.isreal(outcome) or not np.isfinite(outcome):
            raise ValueError("Measurement outcome must be finite and real")
        projected = conditioned(outcome)
        mass = float(np.trace(projected).real)
        if mass <= 1e-300:
            raise ValueError("Postselection has zero numerical probability/density")
        self.nodes = self.nodes[:mode] + self.nodes[mode + 1 :]
        self.native = self._density(projected / mass, survivors) if self.nodes else None
        self.last_measurement = {
            "kind": "probability" if isinstance(measurement, PhotonNumber) else "density",
            "value": mass,
        }
        return outcome
