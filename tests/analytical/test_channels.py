import numpy as np
import pytest

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.gaussian import beamsplitter, cz, rotation, squeezing, wire_channel


@pytest.mark.parametrize(
    "circuit,expected",
    [
        (pg.Circuit(1).rotate(0, 0.43), rotation(0.43)),
        (pg.Circuit(1).squeeze(0, 0.6), squeezing(0.6)),
        (pg.Circuit(1).identity(0), np.eye(2)),
        (pg.Circuit(2).cz(0, 1, 0.7), cz(0.7)),
        (pg.Circuit(2).beamsplitter(0, 1, 0.37), beamsplitter(0.37)),
        (pg.Circuit(2).beamsplitter(0, 1, np.pi), beamsplitter(np.pi)),
    ],
)
def test_compiled_gaussian_gate_channel(circuit, expected):
    channel = gaussian_channel(circuit.compile(squeezing=0.7))
    np.testing.assert_allclose(channel.matrix, expected, atol=1e-10)
    assert np.linalg.eigvalsh(channel.noise).min() > -1e-9


def test_wire_noise_and_bk_noise_exact():
    ks, r = [0.2, -0.4, 0.7], 0.6
    channel = gaussian_channel(pg.protocols.wire(ks, squeezing=r))
    s, noise = wire_channel(ks, r)
    np.testing.assert_allclose(channel.matrix, s, atol=1e-13)
    np.testing.assert_allclose(channel.noise, noise, atol=1e-13)
    bk = gaussian_channel(pg.protocols.teleportation(squeezing=r))
    np.testing.assert_allclose(bk.matrix, np.eye(2), atol=1e-13)
    np.testing.assert_allclose(bk.noise, 2 * np.exp(-2 * r) * np.eye(2), atol=1e-13)


def test_displacement_and_adaptive_rejection():
    channel = gaussian_channel(pg.protocols.displacement(0.4, -0.2))
    np.testing.assert_allclose(channel.displacement, [0.4, -0.2])
    with pytest.raises(NotImplementedError):
        gaussian_channel(pg.protocols.adaptive(), parameters={"k": 0.2})
