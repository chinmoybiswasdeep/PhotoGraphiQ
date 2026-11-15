import numpy as np
import pytest

from photographiqml import MuTA
from photographiqml.expressivity import pauli_lie_dimension, state_fisher


def test_lie_algebra_regimes():
    assert pauli_lie_dimension(["X", "Z"]) == 3
    assert pauli_lie_dimension(["XI", "ZI", "IX", "IZ"]) == 6
    assert pauli_lie_dimension(["XI", "ZI", "IX", "IZ", "XX"]) == 15
    assert pauli_lie_dimension([]) == 0
    with pytest.raises(ValueError):
        pauli_lie_dimension(["A"])


def test_fisher_psd_and_refinement():
    model = MuTA(1)
    values = np.array(list(model.initialize(1).values()))
    fisher = state_fisher(model, [1, 0], values)
    assert np.allclose(fisher, fisher.T)
    assert np.linalg.eigvalsh(fisher).min() > -1e-10
    assert np.linalg.matrix_rank(fisher, tol=1e-8) <= 2
    assert np.allclose(fisher, state_fisher(model, [1, 0], values, step=1e-6), atol=1e-8)
