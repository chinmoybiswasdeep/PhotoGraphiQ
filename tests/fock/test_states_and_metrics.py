import numpy as np
import piquasso as pq
import pytest

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend


def state(resource, cutoff=20):
    return pg.simulate(
        pg.Pattern().append(pg.Prepare("a", state=resource)), backend="piquasso-fock", cutoff=cutoff
    ).state


def test_sparse_multimode_import_reduction_and_counting():
    source = pg.FockSuperposition.from_mapping({(0, 0): np.sqrt(0.3), (1, 1): np.sqrt(0.7)})
    p = pg.Pattern(inputs=("a", "b"))
    output = pg.simulate(p, initial_state=source, backend="piquasso-fock", cutoff=5).state
    assert output.nodes == ("a", "b")
    reduced = output.reduced(("b",))
    np.testing.assert_allclose(reduced.density_matrix[:2, :2], np.diag([0.3, 0.7]))
    with pytest.raises(ValueError, match="mixed"):
        _ = reduced.state_vector
    imported = pg.FockSuperposition.from_piquasso(output.native)
    assert imported.modes == 2
    p.measure("a", pg.PhotonNumber(), key="n").measure("b", pg.PhotonNumber(), key="k")
    result = pg.simulate(
        p,
        initial_state=source,
        backend="piquasso-fock",
        cutoff=5,
        measurement_outcomes={"n": 1, "k": 1},
    )
    assert np.isclose(np.exp(result.log_likelihood), 0.7)
    assert result.state.norm == 1
    p2 = pg.Pattern().append(pg.PrepareResource(("a", "b"), source))
    assert pg.Pattern.from_json(p2.to_json()).outputs == ("a", "b")
    assert pg.simulate(p2, backend="piquasso-fock", cutoff=5).state.fidelity(output) > 1 - 1e-7


@pytest.mark.parametrize("parity", [-1, 1])
def test_cat_parity_number_and_stable_large_cutoff(parity):
    alpha = 1.2 + 0.3j
    output = state(pg.CatResource(alpha, parity), 30)
    assert np.isclose(output.parity(), parity)
    mean = abs(alpha) ** 2 * (
        (1 - parity * np.exp(-2 * abs(alpha) ** 2)) / (1 + parity * np.exp(-2 * abs(alpha) ** 2))
    )
    assert np.isclose(output.photon_number("a"), mean)
    assert len(pg.FockInput.cat(alpha, 400, parity).amplitudes) == 400


def test_wigner_normalization_negativity_and_complex_displacement():
    axis = np.linspace(-7, 7, 181)
    one = state(pg.FockInput.number(1))
    grid = one.wigner(axis, axis)
    q, p = np.meshgrid(axis, axis)
    r2 = q * q + p * p
    np.testing.assert_allclose(grid.values, (r2 - 1) * np.exp(-r2 / 2) / (2 * np.pi), atol=1e-14)
    assert abs(grid.captured_mass - 1) < 1e-8
    assert abs(grid.negative_volume - (2 * np.exp(-0.5) - 1)) < 4e-4
    coherent = state(pg.GaussianInput.coherent(0.4 + 0.6j))
    grid = coherent.wigner(axis, axis)
    np.testing.assert_allclose(
        grid.values, np.exp(-((q - 0.8) ** 2 + (p - 1.2) ** 2) / 2) / (2 * np.pi), atol=1e-9
    )
    np.testing.assert_allclose(coherent.quadrature("a"), (0.8, 1), atol=1e-10)
    np.testing.assert_allclose(coherent.quadrature("a", np.pi / 2), (1.2, 1), atol=1e-10)
    # Top occupied basis: q^2 includes transitions outside the stored support.
    np.testing.assert_allclose(state(pg.FockInput.number(3), 6).quadrature("a"), (0, 7))


def test_metrics_align_cutoffs_and_mixed_states():
    a, b = state(pg.FockInput.number(0), 5), state(pg.FockInput.number(0), 10)
    assert np.isclose(a.fidelity(b), 1)
    assert a.trace_distance(b) < 1e-14
    one = state(pg.FockInput.number(1), 8)
    assert a.fidelity(one) == 0
    assert np.isclose(a.trace_distance(one), 1)
    p = pg.Pattern().append(
        pg.PrepareResource(
            ("a", "b"),
            pg.FockSuperposition.from_mapping({(0, 0): np.sqrt(0.4), (1, 1): np.sqrt(0.6)}),
        )
    )
    mixed = pg.simulate(p, backend="piquasso-fock", cutoff=5).state.reduced(("a",))
    assert np.isclose(a.fidelity(mixed), 0.4)
    assert np.isclose(a.trace_distance(mixed), 0.6)


def test_cat_projection_weight_and_tiny_odd_cat():
    with pytest.warns(RuntimeWarning, match="retained Fock norm"):
        result = state(pg.CatResource(1.5), 12)
    assert 0.999 < min(result.retained_norms) < 1
    with pytest.raises(ValueError, match="truncation"):
        state(pg.CatResource(2.5), 6)
    tiny = pg.FockInput.cat(1e-200, 8, parity=-1)
    assert np.isclose(abs(tiny.amplitudes[1]), 1)


def test_inputs_capability_preflight_and_allocation_errors():
    with pytest.raises(ValueError):
        pg.FockSuperposition(((0, 1), (1,)), (1, 0))
    with pytest.raises(ValueError):
        pg.FockSuperposition(((0,), (0,)), (1, 0))
    with pytest.raises(ValueError):
        pg.FockInput((float("nan"),))
    with pytest.raises(ValueError):
        pg.FockInput.number(True)
    with pytest.raises(ValueError):
        pg.CatResource(0, -1)
    backend = PiquassoFockBackend(10, max_dimension=30)
    assert backend.dimension(2) == 55
    assert backend.supports("cat_state") and not backend.supports("loss")
    with pytest.raises(MemoryError):
        backend.prepare_resource((0, 1), pg.FockSuperposition.number((0, 0)))
    p = pg.Pattern().append(pg.Prepare(0, 0)).append(pg.Kerr(0, 0.5))
    with pytest.raises(NotImplementedError, match="kerr"):
        pg.simulate(p)
    p = pg.Pattern().append(pg.Prepare(0, 0)).append(pg.Loss(0, 0.9))
    with pytest.raises(NotImplementedError, match="loss"):
        pg.simulate(p, backend=backend)
    assert backend.nodes == ()  # failure before any preparation
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
    mixed = pq.FockSimulator(d=1, config=pq.Config(cutoff=5)).execute(program).state
    with pytest.raises(TypeError):
        pg.FockSuperposition.from_piquasso(mixed)
