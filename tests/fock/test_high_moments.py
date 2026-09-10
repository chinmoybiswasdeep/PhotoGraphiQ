import numpy as np
import pytest

import photographiq as pg
from tests.fock.reference import operators


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_number_moments_include_boundary_ladder_paths(n):
    state = pg.simulate(
        pg.Pattern().append(pg.Prepare(0, state=pg.FockInput.number(n))),
        backend="piquasso-fock",
        cutoff=max(2, n + 1),
    ).state
    for angle in (0, np.pi / 2, 0.37):
        expected = [1, 0, 2 * n + 1, 0, 6 * n * n + 6 * n + 3]
        for k, value in enumerate(expected):
            # Exact number-state polynomial identities, even at the top basis.
            assert np.isclose(state.quadrature_moment(0, k, angle), value, atol=1e-11, rtol=1e-13)
    assert np.isclose(state.photon_moment(0), n * n)


def test_moments_against_padded_independent_matrices_and_convergence_api():
    p = pg.Pattern().extend(
        [
            pg.Prepare(0, state=pg.FockInput((1 / np.sqrt(2), 1j / np.sqrt(2)))),
            pg.CubicPhase(0, 0.5),
            pg.Kerr(0, 0.3),
        ]
    )
    study = pg.cutoff_convergence(p, [16, 32, 64], high_order_moments=True)
    for result, row in zip(study.results, study.rows, strict=True):
        vector = np.pad(result.state.state_vector, (0, 4))
        *_, q, p_op = operators(len(vector))
        for axis, op in (("q", q), ("p", p_op)):
            for order in (2, 3, 4):
                expected = np.vdot(vector, np.linalg.matrix_power(op, order) @ vector).real
                # Padded dense oracle includes all intermediate paths through
                # order four; relative tolerance scales with large tail moments.
                assert np.isclose(
                    row["high_order_moments"][0][f"{axis}{order}"], expected, atol=1e-10, rtol=1e-12
                )
    with pytest.raises(ValueError):
        study.results[0].state.quadrature_moment(0, 5)


def test_mixed_reduction_moments():
    source = pg.FockSuperposition.from_mapping({(0, 0): 1 / np.sqrt(2), (1, 1): 1 / np.sqrt(2)})
    state = pg.simulate(
        pg.Pattern(inputs=(0, 1)), initial_state=source, backend="piquasso-fock", cutoff=5
    ).state.reduced((0,))
    assert np.isclose(state.quadrature_moment(0, 4), 9)
    assert np.isclose(state.photon_moment(0), 0.5)


@pytest.mark.parametrize("gamma", [0.2, 0.5, 0.8])
def test_cubic_high_moments_converge_to_noncommuting_analytic_identity(gamma):
    expected = np.array(
        [1 + 3 * gamma**2, gamma + 15 * gamma**3, 3 + 10 * gamma**2 + 105 * gamma**4]
    )
    errors = []
    for cutoff in (80, 160):
        state = pg.simulate(
            pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, gamma)]),
            backend="piquasso-fock",
            cutoff=cutoff,
        ).state
        actual = np.array([state.quadrature_moment(0, k, np.pi / 2) for k in (2, 3, 4)])
        errors.append(np.max(abs(actual - expected) / expected))
    # Derived by differentiating exp(-q²/4+i gamma q³/6), including commutators.
    # High moments amplify the Fock tail: require 0.02% relative accuracy at
    # c=160 and improvement from c=80, rather than a gate-matrix tolerance.
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=2e-4)
    assert errors[-1] <= errors[0] + 1e-10
