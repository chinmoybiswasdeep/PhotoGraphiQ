import numpy as np
import pytest

from photographiq.backends import GaussianBackend, PiquassoBackend
from photographiq.graph import CVGraph
from photographiq.pattern import Pattern
from photographiq.simulator import simulate
from photographiq.states import GaussianInput


@pytest.mark.parametrize("backend", [GaussianBackend, PiquassoBackend])
def test_vacuum_squeeze_displacement_rotation(backend):
    b = backend()
    b.prepare("a", 0)
    np.testing.assert_allclose(b.get_state().covariance, np.eye(2))
    b.squeeze("a", 0.4)
    b.displace("a", 0.7, -0.2)
    b.rotate("a", np.pi / 2)
    np.testing.assert_allclose(b.get_state().mean, [0.2, 0.7], atol=1e-14)
    np.testing.assert_allclose(b.get_state().covariance, np.diag(np.exp([0.8, -0.8])), atol=1e-14)


@pytest.mark.parametrize("backend", ["gaussian", "piquasso"])
def test_weighted_graph_nullifiers(backend):
    a = np.array([[0, 0.7, -0.2], [0.7, 0, 1.3], [-0.2, 1.3, 0]])
    graph = CVGraph.from_adjacency(a, squeezing={0: 0.3, 1: 0.5, 2: 0.8})
    state = simulate(Pattern(graph), backend=backend).state
    q = np.diag(np.exp([0.6, 1.0, 1.6]))
    p = np.diag(np.exp([-0.6, -1.0, -1.6]))
    np.testing.assert_allclose(state.covariance[::2, ::2], q, atol=1e-13)
    np.testing.assert_allclose(state.covariance[1::2, 1::2], p + a @ q @ a.T, atol=1e-13)
    np.testing.assert_allclose(
        graph.nullifiers() @ state.covariance @ graph.nullifiers().T, p, atol=1e-13
    )


def test_beamsplitter_and_loss_agree():
    states = []
    for cls in (GaussianBackend, PiquassoBackend):
        b = cls()
        b.prepare(0, state=GaussianInput.coherent(0.3 + 0.2j))
        b.prepare(1, 0.4)
        b.beamsplitter(0, 1, 0.3)
        b.loss(0, 0.6, 0.2)
        states.append(b.get_state())
    np.testing.assert_allclose(states[0].mean, states[1].mean, atol=1e-13)
    np.testing.assert_allclose(states[0].covariance, states[1].covariance, atol=1e-13)
