"""Finite-sector identities: these errors cannot be excused as cutoff tails."""

import numpy as np
import pytest

import photographiq as pg
from photographiq import autodiff as ad

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_double_precision_requirement_is_enforced():
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            ad.fock_state(pg.Pattern(inputs=(0,)), cutoff=4)
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("cutoff", [2, 3, 5])
def test_beamsplitter_top_sector_preserves_trace_and_binomial_weights(cutoff):
    n, theta = cutoff - 1, 0.3
    pattern = pg.Pattern(inputs=("a", "b")).append(pg.BeamSplitter("a", "b", theta))
    result = ad.fock_state(pattern, cutoff=cutoff, inputs={"b": pg.FockInput.number(n)})
    rho = np.asarray(result.density_matrix)
    assert np.trace(rho).real == pytest.approx(1, abs=2e-13)
    from math import comb

    for i, (a, b) in enumerate(result.basis):
        expected = (
            comb(n, a) * np.sin(theta) ** (2 * a) * np.cos(theta) ** (2 * b) if a + b == n else 0
        )
        assert rho[i, i].real == pytest.approx(expected, abs=2e-13)


def test_beamsplitter_boundary_gradient():
    pattern = pg.Pattern(inputs=(0, 1)).append(pg.BeamSplitter(0, 1, pg.Parameter("t")))

    def mean(t):
        return ad.expectation(
            pattern, {"t": t}, cutoff=2, inputs={1: pg.FockInput.number(1)}, node=0
        )

    assert float(jax.grad(mean)(0.3)) == pytest.approx(np.sin(0.6), abs=2e-12)


def test_addition_does_not_silently_discard_top_occupation():
    p = pg.Pattern(inputs=(0,)).append(pg.PhotonAdd(0))
    with pytest.raises(ValueError, match="cutoff"):
        ad.fock_state(p, inputs={0: pg.FockInput((0, 2**-0.5, 2**-0.5))}, cutoff=3)


def test_tensor_preparation_reports_and_rejects_unresolved_projection():
    source = pg.FockInput((2**-0.5, 2**-0.5))
    with pytest.raises(ValueError, match="truncation"):
        ad.fock_state(pg.Pattern(inputs=(0, 1)), inputs={0: source, 1: source}, cutoff=2)


def test_beamsplitter_coherence_sign_matches_independent_convention():
    from tests.v03_reference import basis, gate

    occupations = basis(2, 4)
    v = np.zeros(len(occupations), complex)
    v[occupations.index((1, 0))] = 1
    expected = gate(occupations, "BeamSplitter", 0.3) @ v
    p = pg.Pattern(inputs=(0, 1)).append(pg.BeamSplitter(0, 1, 0.3))
    state = ad.fock_state(p, cutoff=4, inputs={0: pg.FockInput.number(1)})
    np.testing.assert_allclose(
        state.density_matrix, np.outer(expected, expected.conj()), atol=2e-13
    )
