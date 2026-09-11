"""Finite dimensional completely positive instruments, independent of a backend."""

from typing import Any

import numpy as np


def density_matrix(state, dimension):
    """Validate a normalized vector or density matrix without repairing it."""
    rho = np.asarray(state, complex)
    if rho.shape == (dimension,):
        rho = np.outer(rho, rho.conj())
    if (
        rho.shape != (dimension, dimension)
        or not np.isfinite(rho).all()
        or not np.allclose(rho, rho.conj().T, atol=1e-10, rtol=0)
        or not np.isclose(np.trace(rho), 1, atol=1e-10, rtol=0)
        or np.linalg.eigvalsh(rho).min() < -1e-10
    ):
        raise ValueError("State must be finite, normalized, Hermitian and positive")
    return rho


class MeasurementInstrument:
    """Complete finite-outcome instrument given label -> sequence of Kraus matrices.

    Matrices may be rectangular but must share input and output dimensions.
    ``conditional`` retains the output system. Pattern ``Measure`` instead
    discards that system, retaining the conditional state of all other modes.
    Effects alone do not specify a retained-system update.
    """

    def __init__(self, kraus):
        if not kraus:
            raise ValueError("An instrument needs outcomes")
        self.labels = tuple(kraus)
        self._kraus = tuple(tuple(np.array(m, complex, copy=True) for m in kraus[k]) for k in kraus)
        matrices = [m for branch in self._kraus for m in branch]
        if any(not branch for branch in self._kraus) or not matrices:
            raise ValueError("Every outcome needs Kraus operators")
        shape = matrices[0].shape
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Kraus matrices must be nonempty matrices")
        if any(m.shape != shape or not np.isfinite(m).all() for m in matrices):
            raise ValueError("Kraus matrices must have equal shapes and finite entries")
        self.output_dimension, self.dimension = shape
        self._effects = tuple(sum(m.conj().T @ m for m in branch) for branch in self._kraus)
        if not np.allclose(sum(self._effects), np.eye(self.dimension), atol=1e-10, rtol=0):
            raise ValueError("Instrument effects must sum to identity; include failure outcomes")
        for m in [*matrices, *self._effects]:
            m.setflags(write=False)

    @property
    def effects(self):
        return dict(zip(self.labels, (m.copy() for m in self._effects), strict=True))

    @property
    def required_capabilities(self):
        # Rank-one effects guarantee pure surviving states after destructive readout.
        rank_one = all(np.linalg.eigvalsh(e)[-2:-1].sum() < 1e-12 for e in self._effects)
        return frozenset({"rank_one_instrument" if rank_one else "custom_kraus_instrument"})

    def validate(self):
        return self

    def validate_backend(self, engine):
        engine.require(*self.required_capabilities)
        if engine.cutoff != self.dimension:
            raise ValueError("Instrument input dimension must equal backend cutoff")

    def probabilities(self, state):
        rho = density_matrix(state, self.dimension)
        return {k: float(np.trace(e @ rho).real) for k, e in self.effects.items()}

    def conditional(self, state, outcome):
        rho = density_matrix(state, self.dimension)
        branch = self._kraus[self.labels.index(outcome)]
        updated = sum(m @ rho @ m.conj().T for m in branch)
        probability = float(np.trace(updated).real)
        if probability <= 1e-300:
            raise ValueError("Outcome has zero probability")
        return probability, updated / probability


def destructive_instrument(engine, node, instrument, outcome, *, mixed):
    """Apply a local instrument then trace its output without truncating that output."""
    instrument.validate_backend(engine)
    basis = tuple(engine.native.fock_probabilities_map)
    mode = engine.nodes.index(node)
    survivors = tuple(dict.fromkeys(b[:mode] + b[mode + 1 :] for b in basis))
    lookup = {b: i for i, b in enumerate(survivors)}
    rows = np.array([lookup[b[:mode] + b[mode + 1 :]] for b in basis])
    counts = np.array([b[mode] for b in basis])
    if mixed:
        rho = engine.native.density_matrix
    else:
        vector = engine.native.state_vector
    branches = []
    probabilities = []
    updated: Any
    for effect in instrument._effects:
        if mixed:
            weighted = rho * effect[counts[None, :], counts[:, None]]
            updated = np.zeros((len(survivors), len(survivors)), complex)
            np.add.at(updated, (rows[:, None], rows[None, :]), weighted)
            mass = float(np.trace(updated).real)
        else:
            values, vectors = np.linalg.eigh(effect)
            weights = np.sqrt(max(0, values[-1])) * vectors[:, -1].conj()
            updated = np.zeros(len(survivors), complex)
            np.add.at(updated, rows, vector * weights[counts])
            mass = float(np.vdot(updated, updated).real)
        branches.append(updated)
        probabilities.append(max(0, mass))
    if not np.isclose(sum(probabilities), 1, atol=1e-9, rtol=0):
        raise ArithmeticError("Instrument probabilities are not complete")
    if outcome is None:
        index = engine.rng.choice(
            len(probabilities), p=np.array(probabilities) / sum(probabilities)
        )
        outcome = instrument.labels[index]
    else:
        index = instrument.labels.index(outcome)
    mass = probabilities[index]
    if mass <= 1e-300:
        raise ValueError("Outcome has zero probability")
    engine.nodes = engine.nodes[:mode] + engine.nodes[mode + 1 :]
    if not engine.nodes:
        engine.native = None
    elif mixed:
        engine.native = engine._density(branches[index] / mass, survivors)
    else:
        engine.native = engine._vector(
            dict(zip(survivors, branches[index] / np.sqrt(mass), strict=True)), len(engine.nodes)
        )
    engine.last_measurement = {"kind": "probability", "value": mass}
    return outcome
