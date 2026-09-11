"""Two-mode total-cutoff validation using an independent q-space integral."""

import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend
from tests.v03_reference import hermites


def run_case(cutoff, coefficients=((1, 1), (1, 1))):
    code = pg.GKPCode(0.6, 0.55, cutoff, 5, 2049)
    a, b = [np.array(code.encode(*c).amplitudes) for c in coefficients]
    mapping = {(i, j): a[i] * b[j] for i in range(cutoff) for j in range(cutoff - i)}
    mass = sum(abs(x) ** 2 for x in mapping.values())
    source = pg.FockSuperposition.from_mapping({k: v / np.sqrt(mass) for k, v in mapping.items()})
    pattern = pg.Pattern(inputs=(0, 1)).append(code.logical_cz(0, 1))
    result = pg.simulate(
        pattern, initial_state=source, backend=PiquassoFockBackend(cutoff, norm_tolerance=0.1)
    )
    # Reconstruct the same total-cutoff input in quadrature, then multiply by CZ.
    q = np.linspace(-20, 20, 801)
    h = hermites(q, cutoff)
    coefficients_matrix = np.zeros((cutoff, cutoff), complex)
    for (i, j), v in source.amplitude_map.items():
        coefficients_matrix[i, j] = v
    psi = h @ coefficients_matrix @ h.T * np.exp(0.5j * q[:, None] * q[None, :])
    weights = simpson(np.eye(len(q)), x=q, axis=0)
    projected = h.T @ (weights[:, None] * psi * weights[None, :]) @ h
    reference = np.array([projected[i, j] for i, j in result.state.basis])
    reference /= np.linalg.norm(reference)
    return code, result, reference, mass


@pytest.mark.parametrize(
    "coefficients",
    [((1, 0), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, 1), (0, 1)), ((1, 1), (1, 1))],
)
def test_two_mode_cz_independent_reference(coefficients):
    errors = []
    for cutoff in (24, 40, 64):
        code, result, reference, mass = run_case(cutoff, coefficients)
        assert mass > 0.98
        errors.append(1 - abs(np.vdot(reference, result.state.state_vector)) ** 2)
        joint = pg.multimode_readout(result.state, {0: "Z", 1: "X"})
        assert sum(joint["joint_probabilities"].values()) == pytest.approx(1, abs=1e-10)
        for probabilities in joint["marginal_probabilities"].values():
            assert sum(probabilities) == pytest.approx(1, abs=1e-10)

    assert errors[-1] < 1e-7
    assert errors[-1] < errors[0]


def test_cz_measurement_step_has_correct_conditional_state_and_density():
    code, result, reference, _ = run_case(40)
    raw = 0.27
    bra = hermites(np.array([raw]), 40)[0] * np.exp(-0.5j * np.pi * np.arange(40))
    expected = np.zeros(40, complex)
    for (i, j), v in zip(result.state.basis, result.state.state_vector, strict=True):
        expected[j] += bra[i] * v
    probability = float(np.vdot(expected, expected).real)
    source = pg.FockSuperposition(result.state.basis, tuple(result.state.state_vector))
    pattern = pg.Pattern(inputs=(0, 1)).measure(0, code.logical_measurement("X"), key="x")
    measured = pg.simulate(
        pattern,
        initial_state=source,
        backend="piquasso-fock",
        cutoff=40,
        measurement_outcomes={"x": raw},
    )
    assert measured.measurement_statistics["x"]["value"] == pytest.approx(probability, abs=1e-10)
    assert (
        abs(np.vdot(expected / np.sqrt(probability), measured.state.state_vector)) ** 2 > 1 - 1e-10
    )
    assert measured.outcomes["x"].raw_outcome == raw
