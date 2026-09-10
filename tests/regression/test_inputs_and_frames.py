import numpy as np
import pytest

import photographiq as pg
from photographiq.backends import GaussianBackend


def test_correlated_input_and_output_reordering():
    b = GaussianBackend()
    b.prepare("a", 0.3)
    b.prepare("b", 0.4)
    b.entangle("a", "b", 0.6)
    initial = b.get_state()
    pattern = pg.Pattern(inputs=("a", "b")).append(pg.Output(("b", "a")))
    result = pg.simulate(pattern, initial_state=initial)
    np.testing.assert_allclose(result.state.covariance, initial.reduced(("b", "a")).covariance)
    with pytest.raises(ValueError):
        pg.simulate(pattern, initial_state=initial, inputs={"a": pg.GaussianInput()})


@pytest.mark.parametrize(
    "measurement",
    [pg.Homodyne(0.3, efficiency=0.8), pg.Heterodyne(), pg.Generaldyne(np.diag([0.5, 2]))],
)
def test_virtual_measurement_and_loss(measurement):
    p = pg.Pattern(pg.CVGraph.line(2, squeezing=0.5))
    p.displace(0, q=0.4, p=0.3).displace(1, q=0.1, p=-0.2)
    p.append(pg.Loss(0, 0.8, 0.1)).measure(0, measurement)
    value = pg.Outcome(0) if isinstance(measurement, pg.Homodyne) else pg.Outcome(0, component=0)
    p.displace(1, q=value)
    a, b = [pg.simulate(p, seed=37, frame=frame) for frame in (False, True)]
    np.testing.assert_allclose(a.outcomes[0], b.outcomes[0], atol=1e-12)
    np.testing.assert_allclose(a.state.mean, b.state.mean, atol=1e-12)
    np.testing.assert_allclose(a.state.covariance, b.state.covariance, atol=1e-12)


def test_symbolic_flow_coefficients_are_frozen():
    from photographiq.cvflow import flow_pattern

    graph = pg.CVGraph.line(2, inputs=(0,), outputs=(1,))
    graph.network[0][1]["weight"] = pg.Parameter("g")
    p = flow_pattern(graph, [0], parameters={"g": 0.7})
    assert "g" not in p.parameters
    channel = pg.gaussian_channel(p)
    np.testing.assert_allclose(channel.matrix, [[0, -1 / 0.7], [0.7, 0]], atol=1e-12)
