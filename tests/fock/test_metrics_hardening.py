import numpy as np
import pytest

import photographiq as pg
from tests.fock.test_states_and_metrics import state


def mixed_qubit(p, angle, cutoff=6):
    c, s = np.cos(angle), np.sin(angle)
    source = pg.FockSuperposition.from_mapping(
        {
            (0, 0): np.sqrt(p) * c,
            (1, 0): np.sqrt(p) * s,
            (0, 1): -np.sqrt(1 - p) * s,
            (1, 1): np.sqrt(1 - p) * c,
        }
    )
    output = pg.simulate(
        pg.Pattern(inputs=("a", "b")), initial_state=source, backend="piquasso-fock", cutoff=cutoff
    ).state
    u = np.array([[c, -s], [s, c]])
    return output.reduced(("a",)), u @ np.diag([p, 1 - p]) @ u.T


@pytest.mark.parametrize("p,q,angle", [(0.5, 0.5, 0), (0.2, 0.8, 0.3), (0.05, 0.7, 0.8)])
def test_mixed_mixed_squared_fidelity_against_qubit_identity(p, q, angle):
    a, rho = mixed_qubit(p, 0, 5)
    b, sigma = mixed_qubit(q, angle, 8)
    expected = np.trace(rho @ sigma) + 2 * np.sqrt(np.linalg.det(rho) * np.linalg.det(sigma))
    # Rank-two embedded matrices: square roots near zero eigenvalues have less
    # than full float64 relative accuracy. No Fock or quadrature error here.
    assert np.isclose(a.fidelity(b), expected, atol=3e-8, rtol=0)
    distance = np.sqrt(-np.linalg.det(rho - sigma))
    assert np.isclose(a.trace_distance(b), distance, atol=1e-12, rtol=0)
    assert np.allclose(np.diag(a.density_matrix).real, list(a.probabilities.values()))


def test_labels_reduction_empty_and_global_phase():
    a = state(pg.FockInput((np.sqrt(0.3), 1j * np.sqrt(0.7))), 8)
    b = state(pg.FockInput(tuple(np.exp(0.8j) * np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)]))), 12)
    assert a.fidelity(b) > 1 - 1e-14
    assert a.trace_distance(b) < 1e-14
    empty_a, empty_b = a.reduced(()), b.reduced(())
    assert empty_a.fidelity(empty_b) == 1
    assert empty_a.trace_distance(empty_b) == 0
    source = pg.FockSuperposition.from_mapping({(0, 1): np.sqrt(0.3), (2, 0): 1j * np.sqrt(0.7)})
    full = pg.simulate(
        pg.Pattern(inputs=("a", "b")), initial_state=source, backend="piquasso-fock", cutoff=6
    ).state
    reordered = full.reduced(("b", "a"))
    assert reordered.nodes == ("b", "a")
    assert np.isclose(reordered.probabilities[(1, 0)], 0.3)
    assert np.isclose(reordered.probabilities[(0, 2)], 0.7)
    with pytest.raises(ValueError, match="order"):
        full.fidelity(reordered)
    assert full.fidelity(reordered.reduced(("a", "b"))) > 1 - 3e-8
    reduced = full.reduced(("a",))
    with pytest.raises(ValueError, match="mixed"):
        _ = reduced.state_vector
    assert np.linalg.eigvalsh(reduced.density_matrix).min() > -1e-13
    assert np.isclose(reduced.norm, 1)


def test_pure_metrics_and_norm_do_not_allocate_density(monkeypatch):
    a, b = state(pg.FockInput.number(2), 8), state(pg.FockInput.number(2), 12)

    def forbidden(self):
        pytest.fail("Pure metrics allocated a quadratic-sized density matrix")

    monkeypatch.setattr(type(a.native), "density_matrix", property(forbidden))
    assert a.norm == 1
    assert a.fidelity(b) == 1
    assert a.trace_distance(b) == 0
