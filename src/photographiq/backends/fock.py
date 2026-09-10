"""Experimental pure-Fock execution with explicit total-photon truncation.

Photon counting and ideal homodyne are conditional. Mixed channels are rejected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import piquasso as pq

from ..measurements import Homodyne, PhotonNumber
from ..resources import CatResource, CubicPhaseResource
from ..states import FockInput, FockSuperposition, GaussianInput
from .base import BaseBackend


@dataclass
class FockResultState:
    native: Any
    nodes: tuple
    retained_norms: tuple
    diagnostics: tuple = ()

    def __repr__(self):
        return f"FockResultState(modes={len(self.nodes)}, diagnostics={len(self.diagnostics)}, norm={self.norm:.6g})"

    @property
    def basis(self):
        """Ordered occupation tuples corresponding to vector/matrix indices."""
        return tuple(self.probabilities)

    @property
    def state_vector(self):
        """Copy pure-state amplitudes; mixed reduced states have no state vector.

        Raises:
            ValueError: A mixed reduced state has no state vector.
        """
        if self.native is None:
            return np.ones(1, dtype=complex)
        if not hasattr(self.native, "state_vector"):
            raise ValueError("A mixed reduced state has no state vector")
        return np.array(self.native.state_vector, copy=True)

    @property
    def norm(self):
        """Return the state norm or density-matrix trace."""
        return 1.0 if self.native is None else float(self.native.norm)

    def quadrature_moment(self, node, order, angle=0.0):
        """Raw quadrature moment through order four, with complete ladder paths."""
        from ..fock_analysis import quadrature_moment

        return quadrature_moment(self, node, order, angle)

    def photon_moment(self, node, order=2):
        """Raw photon-number moment through order four."""
        if not isinstance(order, int) or isinstance(order, bool) or not 0 <= order <= 4:
            raise ValueError("Moment order must be an integer from zero through four")
        i = self.nodes.index(node)
        return float(sum(b[i] ** order * p for b, p in self.probabilities.items()))

    def reduced(self, nodes):
        """Return the partial trace on requested nodes in exactly that label order.

        Args:
            nodes (tuple): Ordered mode labels.

        Returns:
            result (object): Reduced state snapshot.

        Raises:
            ValueError: Invalid reduced-state labels.
        """
        nodes = tuple(nodes)
        if len(set(nodes)) != len(nodes) or not set(nodes) <= set(self.nodes):
            raise ValueError("Invalid reduced-state labels")
        if nodes == self.nodes:
            return FockResultState(
                self.native.copy() if self.native is not None else None,
                nodes,
                self.retained_norms,
                self.diagnostics,
            )
        native = self.native.reduced(tuple(self.nodes.index(n) for n in nodes)) if nodes else None
        return FockResultState(native, nodes, self.retained_norms, self.diagnostics)

    def quadrature(self, node, angle=0.0):
        """Mean/variance with exact ladder moments, including vacuum boundary term."""
        if not np.isfinite(angle):
            raise ValueError("Angle must be finite")
        state = self.reduced((node,))
        rho = state.density_matrix
        n = np.arange(len(rho))
        a = sum(np.sqrt(k) * rho[k, k - 1] for k in range(1, len(rho)))
        a2 = sum(np.sqrt(k * (k - 1)) * rho[k, k - 2] for k in range(2, len(rho)))
        mean = 2 * np.real(np.exp(-1j * angle) * a)
        second = 2 * np.dot(n, rho.diagonal()).real + 1 + 2 * np.real(np.exp(-2j * angle) * a2)
        return float(mean), float(second - mean**2)

    def fidelity(self, other):
        """Return squared Uhlmann fidelity after occupation-basis alignment.

        Args:
            other (object): State with matching node labels and order.

        Returns:
            result (float): Squared Uhlmann fidelity.
        """
        from ..fock_analysis import fidelity

        return fidelity(self, other)

    def trace_distance(self, other):
        """Return half the trace norm of the occupation-aligned state difference.

        Args:
            other (object): State with matching node labels and order.

        Returns:
            result (float): Trace distance.
        """
        from ..fock_analysis import trace_distance

        return trace_distance(self, other)

    def wigner(self, q, p, node=None):
        """Compute a single-mode reduced Wigner quasiprobability grid.

        Args:
            q (float): Position translation; hbar=2 quadrature units.
            p (float): Momentum translation; hbar=2 quadrature units.
            node (object): Hashable mode label.
        """
        from ..fock_analysis import wigner

        return wigner(self, q, p, node)

    @property
    def density_matrix(self):
        """Copy the density matrix; allocation scales quadratically with Fock dimension."""
        return (
            np.ones((1, 1))
            if self.native is None
            else np.array(self.native.density_matrix, copy=True)
        )

    @property
    def probabilities(self):
        """Return occupation-tuple probabilities in basis order."""
        return {(): 1.0} if self.native is None else dict(self.native.fock_probabilities_map)

    def photon_number(self, node):
        """Return mean photon occupation on a labelled mode.

        Args:
            node (object): Hashable mode label.

        Returns:
            result (float): Mean occupation.
        """
        i = self.nodes.index(node)
        return float(sum(basis[i] * p for basis, p in self.probabilities.items()))

    def parity(self, nodes=None):
        """Return the joint photon-number parity expectation on selected modes.

        Args:
            nodes (tuple): Ordered mode labels.

        Returns:
            result (float): Parity expectation.
        """
        indices = range(len(self.nodes)) if nodes is None else [self.nodes.index(n) for n in nodes]
        return float(
            sum(
                (-1) ** sum(basis[i] for i in indices) * p
                for basis, p in self.probabilities.items()
            )
        )


class PiquassoFockBackend(BaseBackend):
    capabilities = frozenset(
        {
            "gaussian",
            "fock_input",
            "multimode_fock",
            "cat_state",
            "cubic_phase",
            "kerr",
            "photon_counting",
            "homodyne",
            "quadratic_phase",
            "photon_addition",
            "photon_subtraction",
            "postselection",
            "high_order_moments",
        }
    )

    def __init__(
        self, cutoff=None, *, norm_tolerance=1e-3, max_dimension=1_000_000, boundary_warning=0.02
    ):
        if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 2:
            raise ValueError("piquasso-fock requires an explicit total-photon cutoff >= 2")
        if not 0 < norm_tolerance < 1:
            raise ValueError("Invalid norm tolerance")
        self.cutoff, self.norm_tolerance = cutoff, norm_tolerance
        if (
            not isinstance(max_dimension, int)
            or isinstance(max_dimension, bool)
            or max_dimension < 1
            or not np.isfinite(boundary_warning)
            or not 0 < boundary_warning <= 1
        ):
            raise ValueError("Invalid dimension/boundary diagnostic settings")
        self.max_dimension, self.boundary_warning = max_dimension, boundary_warning
        self.reset()

    def reset(self, seed=None):
        """Clear backend state and initialize the trajectory random-number generator.

        Args:
            seed (object): Random seed or SeedSequence; zero is valid.
        """
        self.rng = np.random.default_rng(seed)
        self.nodes: tuple = ()
        self.native: Any = None
        self.retained_norms: list[float] = []
        self.diagnostics: list[dict] = []
        self.last_measurement: dict = {}

    def dimension(self, modes):
        """Number of occupations with total photon number strictly below cutoff."""
        if not isinstance(modes, int) or isinstance(modes, bool) or modes < 0:
            raise ValueError("Mode count must be a nonnegative integer")
        return comb(modes + self.cutoff - 1, modes)

    def _guard(self, modes):
        dimension = self.dimension(modes)
        if dimension > self.max_dimension:
            raise MemoryError(
                f"Fock dimension {dimension} exceeds max_dimension={self.max_dimension}"
            )
        if dimension >= 100_000:
            warnings.warn(
                f"Fock dimension {dimension}: vector alone needs {16 * dimension} bytes; "
                "gate workspaces can be much larger",
                RuntimeWarning,
                stacklevel=3,
            )

    def _config(self):
        return pq.Config(cutoff=self.cutoff, hbar=2.0)

    def _check(self, state, operation="preparation"):
        norm = float(state.norm)
        self.retained_norms.append(norm)
        if not np.isfinite(norm) or abs(norm - 1) > self.norm_tolerance:
            raise ValueError(f"Fock truncation norm {norm:.8g}; increase cutoff={self.cutoff}")
        if abs(norm - 1) > 1e-8:
            warnings.warn(
                f"{operation}: retained Fock norm {norm:.8g}; test a larger cutoff",
                RuntimeWarning,
                stacklevel=3,
            )
        state.normalize()
        boundary = float(
            sum(
                p
                for b, p in state.fock_probabilities_map.items()
                if sum(b) >= max(1, self.cutoff - 2)
            )
        )
        self.diagnostics.append(
            {
                "operation": operation,
                "retained_norm": norm,
                "boundary_population": boundary,
                "cutoff": self.cutoff,
                "dimension": len(state.fock_probabilities_map),
            }
        )
        if boundary > self.boundary_warning:
            warnings.warn(
                f"{operation}: boundary population {boundary:.3g}; cutoff convergence "
                "is required even when norm is one",
                RuntimeWarning,
                stacklevel=3,
            )
        return state

    def _vector(self, amplitudes, modes):
        self._guard(modes)
        with pq.Program() as program:
            for basis, coefficient in amplitudes.items():
                if abs(coefficient) > 0:
                    pq.Q() | pq.NumberState(tuple(basis)) * coefficient
        return pq.PureFockSimulator(d=modes, config=self._config()).execute(program).state

    def validate_preparation(self, state, modes=1):
        """Reject unsupported/mismatched resources before native state allocation."""
        from ..gkp import GKPResource

        if isinstance(state, GKPResource):
            if modes != 1:
                raise ValueError("GKPResource requires one mode")
            return
        if isinstance(state, FockInput):
            if modes != 1 or len(state.amplitudes) > self.cutoff:
                raise ValueError("Input mode count or support exceeds cutoff")
        elif isinstance(state, FockSuperposition):
            if state.modes != modes:
                raise ValueError("Resource mode count does not match preparation labels")
            if any(sum(b) >= self.cutoff and abs(a) > 0 for b, a in state.amplitude_map.items()):
                raise ValueError("Resource support exceeds total-photon cutoff")
        elif isinstance(state, GaussianInput):
            state.state(0).validate()
            if modes != 1:
                raise ValueError("GaussianInput is single mode")
            # Physicality and purity are distinct. Default np.isclose would
            # silently purify thermal inputs with small but nonzero mixedness.
            if not np.isclose(np.linalg.det(state.covariance), 1, atol=1e-10, rtol=0):
                raise NotImplementedError("Mixed Gaussian inputs require a mixed Fock backend")
        elif state is None or isinstance(state, (CatResource, CubicPhaseResource)):
            if modes != 1:
                raise ValueError("This resource requires one mode")
        else:
            raise NotImplementedError("Only pure supported input descriptions can be prepared")

    def prepare(self, node, squeezing=1.0, state=None):
        """Prepare a fresh labelled mode; resource squeezing is momentum squeezing.

        Args:
            node (object): Hashable mode label.
            squeezing (float): Finite momentum resource squeezing; nonnegative.
            state (object): Supported state preparation or independent state snapshot.

        Raises:
            ValueError: Duplicate Fock node.
            ValueError: Invalid squeezing.
            ValueError: Input exceeds cutoff.
            NotImplementedError: Unsupported Fock input.
        """
        self.validate_preparation(state)
        if node in self.nodes:
            raise ValueError("Duplicate Fock node")
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Invalid squeezing")
        self._guard(len(self.nodes) + 1)
        resource_mass = 1.0
        from ..gkp import GKPResource

        if isinstance(state, GKPResource):
            state, resource_mass = state.project(self.cutoff)
        if isinstance(state, CatResource):
            from scipy.special import gammaln

            intensity = abs(state.alpha) ** 2
            if intensity > 0:
                n = np.arange(self.cutoff)
                weights = (
                    np.exp(-intensity + n * np.log(intensity) - gammaln(n + 1))
                    * (1 + state.parity * (-1) ** n) ** 2
                    / (
                        2
                        * (
                            1 + np.exp(-2 * intensity)
                            if state.parity == 1
                            else -np.expm1(-2 * intensity)
                        )
                    )
                )
                resource_mass = float(weights.sum())
            state = FockInput.cat(state.alpha, self.cutoff, state.parity)
        if isinstance(state, CubicPhaseResource):
            self.prepare(node, squeezing=state.squeezing)
            self.cubic_phase(node, state.gamma)
            return
        if isinstance(state, FockSuperposition):
            self.prepare_resource((node,), state)
            return
        if isinstance(state, FockInput):
            if len(state.amplitudes) > self.cutoff:
                raise ValueError("Input exceeds cutoff")
            local = {(n,): a * np.sqrt(resource_mass) for n, a in enumerate(state.amplitudes)}
        else:
            if state is not None and not isinstance(state, GaussianInput):
                raise NotImplementedError("Unsupported Fock input")
            with pq.Program() as program:
                pq.Q() | pq.Vacuum()
                if state is None:
                    pq.Q(0) | pq.Squeezing(r=-squeezing)
                else:
                    cov = np.array(state.covariance)
                    values, vectors = np.linalg.eigh(cov)
                    r = -0.5 * np.log(values[0])
                    phi = np.arctan2(vectors[1, 0], vectors[0, 0])
                    pq.Q(0) | pq.Squeezing(r=r)
                    pq.Q(0) | pq.Phaseshifter(phi=phi)
                    alpha = complex(*state.mean) / 2
                    pq.Q(0) | pq.Displacement(r=abs(alpha), phi=np.angle(alpha))
            local_state = self._check(
                pq.PureFockSimulator(d=1, config=self._config()).execute(program).state
            )
            local = local_state.fock_amplitudes_map
        old = {(): 1.0} if self.native is None else self.native.fock_amplitudes_map
        tensor = {
            tuple(b) + tuple(n): a * c
            for b, a in old.items()
            for n, c in local.items()
            if sum(b) + sum(n) < self.cutoff
        }
        self.native = self._check(self._vector(tensor, len(self.nodes) + 1))
        self.nodes += (node,)

    def prepare_resource(self, nodes, state):
        nodes = tuple(nodes)
        self.validate_preparation(state, len(nodes))
        if not isinstance(state, FockSuperposition) or state.modes != len(nodes):
            raise ValueError("Provide a FockSuperposition matching the ordered resource nodes")
        if len(set(nodes)) != len(nodes) or set(nodes) & set(self.nodes):
            raise ValueError("Duplicate resource nodes")
        if any(sum(b) >= self.cutoff and abs(a) > 0 for b, a in state.amplitude_map.items()):
            raise ValueError("Resource support exceeds total-photon cutoff")
        self._guard(len(self.nodes) + len(nodes))
        old = {(): 1.0} if self.native is None else self.native.fock_amplitudes_map
        tensor = {
            b + n: a * c
            for b, a in old.items()
            for n, c in state.amplitude_map.items()
            if sum(b) + sum(n) < self.cutoff
        }
        self.native = self._check(self._vector(tensor, len(self.nodes) + len(nodes)))
        self.nodes += nodes

    def _gate(self, nodes, instruction):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate mode")
        with pq.Program() as program:
            pq.Q(*(self.nodes.index(n) for n in nodes)) | instruction
        result = pq.PureFockSimulator(d=len(self.nodes), config=self._config()).execute(
            program, initial_state=self.native
        )
        self.native = self._check(result.state, type(instruction).__name__)

    def entangle(self, u, v, weight=1.0):
        """Apply weighted controlled-Z to two existing modes.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            weight (float): Real controlled-Z edge weight.
        """
        passive = np.array([[1, 1j * weight / 2], [1j * weight / 2, 1]])
        active = np.array([[0, 1j * weight / 2], [1j * weight / 2, 0]])
        self._gate((u, v), pq.GaussianTransform(passive=passive, active=active))

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

    def beamsplitter(self, u, v, theta):
        """Mix two optical modes using the package real beamsplitter convention.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            theta (float): Beamsplitter mixing angle in radians.
        """
        self._gate((u, v), pq.Beamsplitter(theta=theta, phi=0))

    def cubic_phase(self, node, gamma):
        """Apply exp(i gamma q^3/6) on a Fock-capable backend.

        Args:
            node (object): Hashable mode label.
            gamma (float): Cubic coefficient in exp(i gamma q³/6).
        """
        self._gate((node,), pq.CubicPhase(gamma=gamma))

    def kerr(self, node, kappa):
        self._gate((node,), pq.Kerr(xi=kappa))

    def quadratic_phase(self, node, s):
        self._gate((node,), pq.QuadraticPhase(s=s))

    def ladder(self, node, addition):
        i = self.nodes.index(node)
        projected = {}
        expected = retained = 0.0
        for basis, amplitude in self.native.fock_amplitudes_map.items():
            factor = basis[i] + int(addition)
            if factor == 0:
                continue
            value = amplitude * np.sqrt(factor)
            expected += abs(value) ** 2
            target = list(basis)
            target[i] += 1 if addition else -1
            if sum(target) < self.cutoff:
                projected[tuple(target)] = value
                retained += abs(value) ** 2
        if expected <= 1e-28:
            raise ValueError("Ideal ladder operation has zero norm")
        if expected - retained > 1e-12:
            raise ValueError("Photon addition exceeds cutoff; increase cutoff")
        self.native = self._vector(
            {b: a / np.sqrt(retained) for b, a in projected.items()}, len(self.nodes)
        )
        self.diagnostics.append(
            {
                "operation": "PhotonAdd" if addition else "PhotonSubtract",
                "ladder_norm_squared": float(expected),
                "retained_norm": retained / expected,
                "cutoff": self.cutoff,
                "dimension": self.dimension(len(self.nodes)),
                "boundary_population": float(
                    sum(
                        p
                        for b, p in self.native.fock_probabilities_map.items()
                        if sum(b) >= max(1, self.cutoff - 2)
                    )
                ),
            }
        )

    def measure(self, node, measurement, angle=0.0, *, outcome=None):
        """Append a destructive measurement; its key becomes a later classical dependency.

        Args:
            node (object): Hashable mode label.
            measurement (object): Homodyne, Heterodyne, Generaldyne or PhotonNumber description.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
            outcome (object): Outcome as described by this object’s contract.

        Returns:
            result (object): Pattern when constructing; sampled outcome when executing.

        Raises:
            NotImplementedError: Fock backend supports ideal Homodyne and PhotonNumber measurements only.
            ValueError: Photon-count outcome must be an integer within cutoff.
            ValueError: Photon-count postselection has zero probability.
            NotImplementedError: Noisy Fock homodyne requires mixed-state conditioning.
        """
        if isinstance(measurement, Homodyne):
            if measurement.efficiency != 1 or measurement.noise != 0:
                raise NotImplementedError("Noisy Fock homodyne requires mixed-state conditioning")
            from ..fock_measurements import homodyne_projection

            value, projected, density = homodyne_projection(
                self.native.fock_amplitudes_map,
                self.nodes.index(node),
                self.cutoff,
                angle,
                self.rng,
                outcome,
            )
            self.nodes = tuple(n for n in self.nodes if n != node)
            self.native = self._vector(projected, len(self.nodes)) if self.nodes else None
            self.last_measurement = {"kind": "density", "value": density}
            return value
        if not isinstance(measurement, PhotonNumber):
            raise NotImplementedError(
                "Fock backend supports ideal Homodyne and PhotonNumber measurements only"
            )
        i = self.nodes.index(node)
        amplitudes = self.native.fock_amplitudes_map
        probabilities = np.zeros(self.cutoff)
        for basis, amplitude in amplitudes.items():
            probabilities[basis[i]] += abs(amplitude) ** 2
        if outcome is None:
            outcome = int(self.rng.choice(self.cutoff, p=probabilities / probabilities.sum()))
        if (
            not isinstance(outcome, (int, np.integer))
            or isinstance(outcome, bool)
            or not 0 <= outcome < self.cutoff
        ):
            raise ValueError("Photon-count outcome must be an integer within cutoff")
        if probabilities[outcome] <= 1e-300:
            raise ValueError("Photon-count postselection has zero probability")
        self.last_measurement = {"kind": "probability", "value": float(probabilities[outcome])}
        projected = {
            b[:i] + b[i + 1 :]: a / np.sqrt(probabilities[outcome])
            for b, a in amplitudes.items()
            if b[i] == outcome
        }
        self.nodes = self.nodes[:i] + self.nodes[i + 1 :]
        self.native = self._vector(projected, len(self.nodes)) if self.nodes else None
        return outcome

    def get_state(self, nodes=None):
        """Return an independent state snapshot, optionally reduced and reordered.

        Args:
            nodes (tuple): Ordered mode labels.

        Returns:
            result (object): Independent backend state snapshot.
        """
        nodes = self.nodes if nodes is None else tuple(nodes)
        if not nodes:
            return FockResultState(None, (), tuple(self.retained_norms), tuple(self.diagnostics))
        native = (
            self.native
            if nodes == self.nodes
            else self.native.reduced(tuple(self.nodes.index(n) for n in nodes))
        )
        return FockResultState(
            native.copy(),
            nodes,
            tuple(self.retained_norms),
            tuple(dict(d) for d in self.diagnostics),
        )
