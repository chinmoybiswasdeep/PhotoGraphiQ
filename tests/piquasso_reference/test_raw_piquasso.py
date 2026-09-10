"""Raw Piquasso references with explicit covariance and detector conventions."""

import numpy as np
import piquasso as pq

import photographiq as pg
from photographiq.analysis import gaussian_channel


def test_raw_graph_state_and_displacement():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=-0.3)
        pq.Q(1) | pq.Squeezing(r=-0.6)
        pq.Q(0, 1) | pq.ControlledZ(s=-0.7)
        pq.Q(1) | pq.Displacement(r=abs(0.2 + 0.1j), phi=np.angle(0.2 + 0.1j))
    raw = pq.GaussianSimulator(d=2, config=pq.Config(hbar=2)).execute(program).state
    graph = pg.CVGraph.from_adjacency([[0, -0.7], [-0.7, 0]], squeezing={0: 0.3, 1: 0.6})
    actual = pg.simulate(pg.Pattern(graph).displace(1, q=0.4, p=0.2)).state
    np.testing.assert_allclose(actual.mean, raw.xpxp_mean_vector, atol=1e-13)
    np.testing.assert_allclose(actual.covariance, raw.xpxp_covariance_matrix / 2, atol=1e-13)


def test_raw_generaldyne_conditional_state():
    # Verify native conditional moments at its own sample. This does NOT validate
    # the physical distribution of that sample (see the separate characterization).
    theta, z = 0.4, 0.25
    with pq.Program() as prep:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Squeezing(r=-0.5)
        pq.Q(1) | pq.Squeezing(r=-0.7)
        pq.Q(0, 1) | pq.ControlledZ(s=0.6)
        pq.Q(0) | pq.Phaseshifter(phi=-theta)
    simulator = pq.GaussianSimulator(d=2, config=pq.Config(hbar=2, seed_sequence=93))
    before = simulator.execute(prep).state
    with pq.Program() as measurement:
        pq.Q(0) | pq.GeneraldyneMeasurement(detection_covariance=np.diag([z * z, z**-2]))
    result = simulator.execute(measurement, initial_state=before, shots=1)
    sample = np.asarray(result.samples[0])
    v, mean = before.xpxp_covariance_matrix / 2, before.xpxp_mean_vector
    gain = v[2:, :2] @ np.linalg.inv(v[:2, :2] + np.diag([z * z, z**-2]))
    np.testing.assert_allclose(
        result.state.xpxp_mean_vector, mean[2:] + gain @ (sample - mean[:2]), atol=1e-12
    )
    np.testing.assert_allclose(
        result.state.xpxp_covariance_matrix / 2, v[2:, 2:] - gain @ v[:2, 2:], atol=1e-12
    )


def test_pinned_raw_sampler_covariance_characterization():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.HeterodyneMeasurement()
    result = pq.GaussianSimulator(d=1, config=pq.Config(hbar=2, seed_sequence=734)).execute(
        program, shots=2500
    )
    # Pinned 8.0.1 currently samples vacuum heterodyne variance 4, whereas its
    # stored vacuum statistical variance is 1 and physical heterodyne variance 2.
    assert np.all(np.abs(np.var(result.samples, axis=0) - 4) < 0.4)


def test_ideal_compiler_target_against_raw_gaussian_circuit():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.3, phi=0.2)
        pq.Q(0) | pq.Phaseshifter(phi=0.7)
        pq.Q(0) | pq.Squeezing(r=0.4)
    raw = pq.GaussianSimulator(d=1).execute(program).state
    circuit = pg.Circuit(1).rotate(0, 0.7).squeeze(0, 0.4)
    channel = gaussian_channel(circuit.compile(squeezing=0.8))
    source = pg.GaussianInput.coherent(0.3 * np.exp(0.2j)).state(0)
    actual = channel.apply(source)
    np.testing.assert_allclose(actual.mean, raw.xpxp_mean_vector, atol=1e-12)
    np.testing.assert_allclose(
        actual.covariance - channel.noise, raw.xpxp_covariance_matrix / 2, atol=1e-12
    )


def test_external_state_hbar_conversion():
    from photographiq.backends import PiquassoBackend

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=0.2, phi=0)
    raw = pq.GaussianSimulator(d=1, config=pq.Config(hbar=1)).execute(program).state
    b = PiquassoBackend()
    b.import_state(raw, ("input",), source_hbar=1)
    np.testing.assert_allclose(b.get_state().mean, [0.4, 0])
    np.testing.assert_allclose(b.get_state().covariance, np.eye(2))
