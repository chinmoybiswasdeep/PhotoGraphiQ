import numpy as np
import pytest

import photographiq as pg
from tests.v03_reference import physical, vector


@pytest.mark.parametrize(
    "kind", ["vacuum", "one", "coherent", "cat", "superposition", "thermal", "mixture"]
)
def test_density_preparations_and_permuted_basis(kind):
    cutoff = 12
    if kind == "thermal":
        rho = np.diag((0.2 / 1.2) ** np.arange(cutoff))
        rho /= np.trace(rho)
    elif kind == "mixture":
        rho = np.diag([0.2, 0.3, 0.5] + [0.0] * (cutoff - 3))
    else:
        v = vector(kind, cutoff)
        rho = np.outer(v, v.conj())
    order = np.random.default_rng(4).permutation(cutoff)
    source = pg.FockDensityMatrix(rho[np.ix_(order, order)], tuple((int(n),) for n in order))
    state = pg.simulate(
        pg.Pattern(inputs=("x",)),
        initial_state=source,
        backend="piquasso-mixed-fock",
        cutoff=cutoff,
    ).state
    physical(state.density_matrix)
    np.testing.assert_allclose(state.density_matrix, rho, atol=1e-13)


def test_entangled_reduction_and_output_order():
    source = pg.FockDensityMatrix([[0.4, 0.2j], [-0.2j, 0.6]], ((0, 1), (2, 0)))
    full = pg.simulate(
        pg.Pattern(inputs=("a", "b")), initial_state=source, backend="piquasso-mixed-fock", cutoff=5
    ).state
    for nodes in [("a", "b"), ("b", "a"), ("a",), ("b",)]:
        state = full.reduced(nodes)
        physical(state.density_matrix)
        assert state.nodes == nodes
    assert full.reduced(("b", "a")).probabilities[(1, 0)] == pytest.approx(0.4)
    np.testing.assert_allclose(
        np.diag(full.reduced(("a",)).density_matrix).real, [0.4, 0, 0.6, 0, 0]
    )
