"""Experimental pure-Fock execution with explicit total-photon truncation.

Photon counting is conditional; homodyne/heterodyne conditioning is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import piquasso as pq

from ..measurements import PhotonNumber
from ..states import FockInput, GaussianInput
from .base import BaseBackend


@dataclass
class FockResultState:
    native: Any
    nodes: tuple
    retained_norms: tuple

    @property
    def density_matrix(self):
        return (
            np.ones((1, 1))
            if self.native is None
            else np.array(self.native.density_matrix, copy=True)
        )

    @property
    def probabilities(self):
        return {(): 1.0} if self.native is None else dict(self.native.fock_probabilities_map)

    def photon_number(self, node):
        i = self.nodes.index(node)
        return float(sum(basis[i] * p for basis, p in self.probabilities.items()))

    def parity(self, nodes=None):
        indices = range(len(self.nodes)) if nodes is None else [self.nodes.index(n) for n in nodes]
        return float(
            sum(
                (-1) ** sum(basis[i] for i in indices) * p
                for basis, p in self.probabilities.items()
            )
        )


class PiquassoFockBackend(BaseBackend):
    def __init__(self, cutoff=None, *, norm_tolerance=1e-3):
        if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 2:
            raise ValueError("piquasso-fock requires an explicit total-photon cutoff >= 2")
        if not 0 < norm_tolerance < 1:
            raise ValueError("Invalid norm tolerance")
        self.cutoff, self.norm_tolerance = cutoff, norm_tolerance
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.nodes: tuple = ()
        self.native: Any = None
        self.retained_norms: list[float] = []

    def _config(self):
        return pq.Config(cutoff=self.cutoff, hbar=2.0)

    def _check(self, state):
        norm = float(state.norm)
        self.retained_norms.append(norm)
        if not np.isfinite(norm) or abs(norm - 1) > self.norm_tolerance:
            raise ValueError(f"Fock truncation norm {norm:.8g}; increase cutoff={self.cutoff}")
        state.normalize()
        return state

    def _vector(self, amplitudes, modes):
        with pq.Program() as program:
            for basis, coefficient in amplitudes.items():
                if abs(coefficient) > 0:
                    pq.Q() | pq.NumberState(tuple(basis)) * coefficient
        return pq.PureFockSimulator(d=modes, config=self._config()).execute(program).state

    def prepare(self, node, squeezing=1.0, state=None):
        if node in self.nodes:
            raise ValueError("Duplicate Fock node")
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Invalid squeezing")
        if isinstance(state, FockInput):
            if len(state.amplitudes) > self.cutoff:
                raise ValueError("Input exceeds cutoff")
            local = {(n,): a for n, a in enumerate(state.amplitudes)}
        else:
            if state is not None and not isinstance(state, GaussianInput):
                raise NotImplementedError("Unsupported Fock input")
            with pq.Program() as program:
                pq.Q() | pq.Vacuum()
                if state is None:
                    pq.Q(0) | pq.Squeezing(r=-squeezing)
                else:
                    cov = np.array(state.covariance)
                    if not np.isclose(np.linalg.det(cov), 1):
                        raise NotImplementedError(
                            "Mixed Gaussian inputs require a mixed Fock backend"
                        )
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

    def _gate(self, nodes, instruction):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate mode")
        with pq.Program() as program:
            pq.Q(*(self.nodes.index(n) for n in nodes)) | instruction
        result = pq.PureFockSimulator(d=len(self.nodes), config=self._config()).execute(
            program, initial_state=self.native
        )
        self.native = self._check(result.state)

    def entangle(self, u, v, weight=1.0):
        passive = np.array([[1, 1j * weight / 2], [1j * weight / 2, 1]])
        active = np.array([[0, 1j * weight / 2], [1j * weight / 2, 0]])
        self._gate((u, v), pq.GaussianTransform(passive=passive, active=active))

    def displace(self, node, q=0.0, p=0.0):
        alpha = complex(q, p) / 2
        self._gate((node,), pq.Displacement(r=abs(alpha), phi=np.angle(alpha)))

    def rotate(self, node, angle):
        self._gate((node,), pq.Phaseshifter(phi=angle))

    def squeeze(self, node, r):
        self._gate((node,), pq.Squeezing(r=r))

    def beamsplitter(self, u, v, theta):
        self._gate((u, v), pq.Beamsplitter(theta=theta, phi=0))

    def cubic_phase(self, node, gamma):
        self._gate((node,), pq.CubicPhase(gamma=gamma))

    def measure(self, node, measurement, angle=0.0):
        if not isinstance(measurement, PhotonNumber):
            raise NotImplementedError(
                "Adaptive Fock homodyne/heterodyne has no conditional-state adapter"
            )
        i = self.nodes.index(node)
        amplitudes = self.native.fock_amplitudes_map
        probabilities = np.zeros(self.cutoff)
        for basis, amplitude in amplitudes.items():
            probabilities[basis[i]] += abs(amplitude) ** 2
        outcome = int(self.rng.choice(self.cutoff, p=probabilities / probabilities.sum()))
        projected = {
            b[:i] + b[i + 1 :]: a / np.sqrt(probabilities[outcome])
            for b, a in amplitudes.items()
            if b[i] == outcome
        }
        self.nodes = self.nodes[:i] + self.nodes[i + 1 :]
        self.native = self._vector(projected, len(self.nodes)) if self.nodes else None
        return outcome

    def get_state(self, nodes=None):
        nodes = self.nodes if nodes is None else tuple(nodes)
        if not nodes:
            return FockResultState(None, (), tuple(self.retained_norms))
        native = (
            self.native
            if nodes == self.nodes
            else self.native.reduced(tuple(self.nodes.index(n) for n in nodes))
        )
        return FockResultState(native.copy(), nodes, tuple(self.retained_norms))
