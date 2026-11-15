"""Ideal qubit target operations from Appendix B, not a CV simulator."""

from dataclasses import dataclass

import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def statevector(state, n_wires):
    state = np.asarray(state, dtype=complex)
    if state.shape != (2**n_wires,) or not np.isfinite(state).all():
        raise ValueError(f"Logical input must have shape ({2**n_wires},) and finite entries")
    if not np.isclose(np.vdot(state, state).real, 1, atol=1e-10, rtol=1e-10):
        raise ValueError("Logical input must be normalized; no implicit normalization is applied")
    return state.copy()


def local_gate(state, gate, wire, n_wires):
    tensor = np.moveaxis(state.reshape([2] * n_wires), wire, 0)
    tensor = (gate @ tensor.reshape(2, -1)).reshape(tensor.shape)
    return np.moveaxis(tensor, 0, wire).reshape(-1)


def cz(state, u, v, n_wires):
    bits = np.arange(2**n_wires)
    sign = 1 - 2 * (((bits >> (n_wires - 1 - u)) & 1) * ((bits >> (n_wires - 1 - v)) & 1))
    return state * sign


@dataclass(frozen=True)
class LogicalResult:
    """Ideal logical output in input-wire order; no physical measurement record."""

    state: np.ndarray
    representation: str = "logical"

    @property
    def probabilities(self):
        return abs(self.state) ** 2

    @property
    def density_matrix(self):
        return np.outer(self.state, self.state.conj())

    def expectation(self, observable):
        observable = np.asarray(observable, dtype=complex)
        d = len(self.state)
        if (
            observable.shape != (d, d)
            or not np.isfinite(observable).all()
            or not np.allclose(observable, observable.conj().T)
        ):
            raise ValueError("Observable must be a finite Hermitian matrix of output dimension")
        return float(np.vdot(self.state, observable @ self.state).real)


def execute(model, state, angles):
    """Appendix B frontier translation with remaining junction edges."""
    state = statevector(state, model.n_wires)
    frontier = list(model.input_nodes)
    remaining = set(model.graph.nodes)
    for node in model.measurement_order:
        w, c = node
        for neighbor in model.graph.neighbors(node):
            if neighbor in remaining and neighbor != (w, c + 1):
                if neighbor not in frontier:
                    raise RuntimeError("Invalid flow: junction neighbor is not on frontier")
                state = cz(state, w, neighbor[0], model.n_wires)
        a = angles[model.parameter_name(node)]
        gate = H @ np.diag(np.exp(0.5j * a * np.array([1, -1])))
        state = local_gate(state, gate, w, model.n_wires)
        frontier[w] = (w, c + 1)
        remaining.remove(node)
    return LogicalResult(state)
