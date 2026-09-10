import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from tests.v03_reference import basis, gate, hermites, vector


def cubic_observation(gamma, r, cutoff, source, outcome=0.35):
    initial = vector(source, cutoff)
    pattern = pg.Circuit(1).cubic_phase(0, gamma).compile(squeezing=r)
    result = pg.simulate(
        pattern,
        inputs={0: pg.FockInput(tuple(initial))},
        backend="piquasso-fock",
        cutoff=cutoff,
        measurement_outcomes={("cubic", 1): outcome},
    )
    q = np.linspace(-14, 14, 6001)
    functions = hermites(q, cutoff)
    psi = functions @ initial
    envelope = (
        np.exp(-((q + outcome) ** 2) / (4 * np.exp(2 * r))) / (2 * np.pi * np.exp(2 * r)) ** 0.25
    )
    filtered = psi * envelope * np.exp(1j * gamma * q**3 / 6)
    probability = simpson(abs(filtered) ** 2, x=q)
    expected = simpson(functions * filtered[:, None], x=q, axis=0)
    expected /= np.linalg.norm(expected)
    ideal = gate(basis(1, cutoff), "CubicPhase", gamma) @ initial
    actual = result.state.state_vector
    mean = simpson(q * abs(filtered) ** 2, x=q) / probability
    return {
        "gamma": gamma,
        "squeezing": r,
        "cutoff": cutoff,
        "source": source,
        "outcome": outcome,
        "filter_infidelity": max(0.0, float(1 - abs(np.vdot(expected, actual)) ** 2)),
        "ideal_infidelity": max(0.0, float(1 - abs(np.vdot(ideal, actual)) ** 2)),
        "density": float(np.exp(result.log_likelihood)),
        "density_error": float(abs(np.exp(result.log_likelihood) - probability)),
        "mean_error": float(abs(result.state.quadrature(0)[0] - mean)),
        "retained_norm": float(min(result.state.retained_norms)),
    }


@pytest.mark.parametrize("source", ["vacuum", "coherent", "cat"])
@pytest.mark.parametrize("gamma,r", [(0.03, 0.2), (-0.06, 0.4)])
def test_compiled_cubic_matches_finite_resource_kernel(gamma, r, source):
    rows = [cubic_observation(gamma, r, c, source) for c in (48, 96)]
    assert rows[-1]["filter_infidelity"] < 1e-7, rows
    assert rows[-1]["density_error"] < 1e-6, rows
    assert rows[-1]["mean_error"] < 2e-6, rows
    assert rows[-1]["filter_infidelity"] <= rows[0]["filter_infidelity"] + 1e-12


def test_finite_resource_error_separate_from_synthesis_error():
    rows = [cubic_observation(0.03, r, 64, "vacuum", 0.0) for r in (0.1, 0.4, 0.7)]
    assert rows[-1]["ideal_infidelity"] < rows[0]["ideal_infidelity"]
    assert all(row["ideal_infidelity"] > row["filter_infidelity"] for row in rows)
