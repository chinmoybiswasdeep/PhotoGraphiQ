import math

import networkx as nx
import numpy as np
import pytest
from scipy.linalg import block_diag

import photographiq as pg
from photographiq.gaussian import beamsplitter, squeezing


def paper_U(A, s):
    # Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
    # branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License,
    # Copyright (c) 2026 Chinmoy Biswas; notebook 0b_cv_mbqrc_photographiqml_colab, cell 20.
    # Independent analytic oracle U = C_Z^(G) (+)_i F D(s_i).
    N = len(A)
    U = np.zeros((2 * N, 2 * N))
    for i in range(N):
        U[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array([[0, -1], [1, 0.0]]) @ np.array(
            [[1, 0], [s[i], 1.0]]
        )
    CZ = np.eye(2 * N)
    CZ[1::2, 0::2] = A
    return CZ @ U


def cluster(n, topology):
    graph = pg.CVGraph.line(n) if topology == "chain" else pg.CVGraph.ring(n)
    if topology == "2nn":
        for i in range(n):
            if not graph.network.has_edge(i, (i + 2) % n):
                graph.add_edge(i, (i + 2) % n, weight=0.1)
    return graph


@pytest.mark.parametrize("topology,n", [("chain", 4), ("ring", 5), ("2nn", 6)])
@pytest.mark.parametrize("r", [0.5, 1.5])
def test_multimode_flow_pattern_matches_graph_teleportation_oracle(topology, n, r):
    # Each input R_i couples to C_i of the cluster; measuring all R_i teleports through it.
    base = cluster(n, topology)
    shears = np.random.default_rng(n).uniform(-1, 1, n)
    inputs = tuple(("R", i) for i in range(n))
    outputs = tuple(("C", i) for i in range(n))
    network = nx.Graph()
    # flow_pattern requires graph node order to list outputs in declared order.
    network.add_nodes_from(inputs + outputs)
    for u, v, data in base.network.edges(data=True):
        network.add_edge(("C", u), ("C", v), weight=data["weight"])
    for i in range(n):
        network.add_edge(("R", i), ("C", i), weight=1.0)
    graph = pg.CVGraph(
        network,
        squeezing={**{x: 0.0 for x in inputs}, **{x: r for x in outputs}},
        inputs=inputs,
        outputs=outputs,
    )
    pattern = pg.cvflow.flow_pattern(
        graph, inputs, shears={x: float(k) for x, k in zip(inputs, shears)}
    )
    channel = pg.gaussian_channel(pattern)
    assert channel.outputs == outputs
    np.testing.assert_allclose(channel.matrix, paper_U(base.adjacency(), shears), atol=1e-12)
    # Finite squeezing leaves only the resource p-variance e^{-2r} on each output.
    np.testing.assert_allclose(
        channel.noise, np.exp(-2 * r) * np.diag(np.tile([0.0, 1.0], n)), atol=1e-12
    )
    np.testing.assert_allclose(channel.displacement, 0, atol=1e-12)


def test_quadratic_phase_channel_matches_analytic_map_and_simulate():
    s = 0.45
    shear = np.array([[1, 0], [s, 1.0]])
    pattern = pg.Pattern(inputs=(0,))
    pattern.append(pg.QuadraticPhase(0, pg.Parameter("s")))
    pattern.append(pg.Output((0,)))
    channel = pg.gaussian_channel(pattern, parameters={"s": s})
    np.testing.assert_allclose(channel.matrix, shear, atol=1e-15)
    np.testing.assert_allclose(channel.noise, 0, atol=1e-15)
    state = pg.GaussianState(np.array([0.5, 0.2]), np.diag([2.0, 0.5]), (0,))
    out = channel.apply(state)
    np.testing.assert_allclose(out.mean, shear @ state.mean, atol=1e-14)
    np.testing.assert_allclose(out.covariance, shear @ state.covariance @ shear.T, atol=1e-14)
    for backend in ("gaussian", "piquasso"):
        sim = pg.simulate(
            pattern, backend=backend, initial_state=state, parameters={"s": s}, seed=0
        ).state
        np.testing.assert_allclose(sim.mean, out.mean, atol=1e-12)
        np.testing.assert_allclose(sim.covariance, out.covariance, atol=1e-12)


def test_outcome_dependent_shear_is_rejected():
    pattern = pg.Pattern(inputs=(0, 1))
    pattern.measure(0, pg.Homodyne.q())
    pattern.append(pg.QuadraticPhase(1, pg.Outcome(0)))
    pattern.append(pg.Output((1,)))
    with pytest.raises(NotImplementedError, match="Outcome-dependent"):
        pg.gaussian_channel(pattern)


def correlated_input():
    s = beamsplitter(0.4) @ block_diag(squeezing(0.3), squeezing(-0.2))
    cov = s @ np.diag([1.3, 1.3, 1.1, 1.1]) @ s.T
    return pg.GaussianState(np.array([0.4, -0.3, 0.1, 0.2]), cov, (0, 1))


def test_rotated_homodyne_equals_quadratic_phase_then_p_homodyne():
    # Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
    # branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License,
    # Copyright (c) 2026 Chinmoy Biswas; notebook 0b_cv_mbqrc_photographiqml_colab, cell 20
    # (validation test 06). Homodyne(atan2(1, s)) / sin(theta) = p + s q.
    s = 0.6
    theta = math.atan2(1, s)
    a = pg.Pattern(inputs=(0, 1))
    a.measure(0, pg.Homodyne(theta))
    a.displace(1, p=0.7 * pg.Outcome(0) / math.sin(theta))
    b = pg.Pattern(inputs=(0, 1))
    b.append(pg.QuadraticPhase(0, s))
    b.measure(0, pg.Homodyne.p())
    b.displace(1, p=0.7 * pg.Outcome(0))
    ca, cb = pg.gaussian_channel(a), pg.gaussian_channel(b)
    np.testing.assert_allclose(ca.matrix, cb.matrix, atol=1e-14)
    np.testing.assert_allclose(ca.noise, cb.noise, atol=1e-14)
    np.testing.assert_allclose(ca.displacement, cb.displacement, atol=1e-14)
    # Conditional trajectories with the same seed: same state, outcome scaled by sqrt(1+s^2).
    state = correlated_input()
    ra = pg.simulate(a, backend="gaussian", initial_state=state, seed=11)
    rb = pg.simulate(b, backend="gaussian", initial_state=state, seed=11)
    assert np.isclose(ra.outcomes[0] * math.sqrt(1 + s * s), rb.outcomes[0])
    np.testing.assert_allclose(ra.state.mean, rb.state.mean, atol=1e-12)
    np.testing.assert_allclose(ra.state.covariance, rb.state.covariance, atol=1e-12)
