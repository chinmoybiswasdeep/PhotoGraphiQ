import numpy as np
import pytest

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend


def test_kraus_is_not_effect_update():
    gamma = 0.3
    m0 = np.diag([1, np.sqrt(1 - gamma)])
    m1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    instrument = pg.MeasurementInstrument({"all": (m0, m1)})
    rho = np.array([[0.4, 0.2j], [-0.2j, 0.6]])
    p, result = instrument.conditional(rho, "all")
    assert p == pytest.approx(1)
    np.testing.assert_allclose(result, m0 @ rho @ m0.T + m1 @ rho @ m1.T)
    assert np.linalg.eigvalsh(result).min() > 0
    np.testing.assert_allclose(instrument.effects["all"], np.eye(2))


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("outcome", ["+", "-"])
def test_complex_rank_one_entangled_conditioning(mixed, outcome):
    vectors = [np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)]
    instrument = pg.MeasurementInstrument(
        dict(zip(("+", "-"), [(v.conj()[None, :],) for v in vectors], strict=True))
    )
    source = pg.FockSuperposition.from_mapping({(0, 0): 1 / np.sqrt(2), (1, 0): 1j / np.sqrt(2)})
    # Use |01>+i|10> with total cutoff 2, hence each local instrument is complete.
    source = pg.FockSuperposition.from_mapping({(0, 1): 1 / np.sqrt(2), (1, 0): 1j / np.sqrt(2)})
    pattern = pg.Pattern(inputs=("a", "b")).measure("a", instrument, key="r")
    result = pg.simulate(
        pattern,
        initial_state=source,
        cutoff=2,
        backend="piquasso-mixed-fock" if mixed else "piquasso-fock",
        measurement_outcomes={"r": outcome},
    )
    bra = vectors[("+", "-").index(outcome)].conj()
    expected = np.array([1j * bra[1], bra[0]])
    np.testing.assert_allclose(
        result.state.density_matrix, np.outer(expected, expected.conj()), atol=1e-12
    )
    assert result.measurement_statistics["r"]["value"] == pytest.approx(0.5)


def test_multikraus_partial_measurement_preserves_mixedness_and_rejects_pure_preflight(monkeypatch):
    instrument = pg.MeasurementInstrument({"all": (np.eye(2),)})
    source = pg.FockSuperposition.from_mapping({(0, 1): 1 / np.sqrt(2), (1, 0): 1 / np.sqrt(2)})
    p = pg.Pattern(inputs=(0, 1)).measure(0, instrument)
    result = pg.simulate(p, initial_state=source, backend="piquasso-mixed-fock", cutoff=2)
    np.testing.assert_allclose(result.state.density_matrix, np.eye(2) / 2)
    engine = PiquassoFockBackend(2)
    monkeypatch.setattr(engine, "reset", lambda *args: pytest.fail("preflight must precede reset"))
    with pytest.raises(NotImplementedError, match="custom_kraus"):
        pg.simulate(p, initial_state=source, backend=engine)


@pytest.mark.parametrize(
    "kraus",
    [
        {},
        {0: ()},
        {0: (np.eye(2) * 0.5,)},
        {0: (np.ones(2),)},
        {0: (np.array([[np.nan]]),)},
        {0: (np.eye(2), np.eye(3))},
    ],
)
def test_invalid_instruments(kraus):
    with pytest.raises(ValueError):
        pg.MeasurementInstrument(kraus)


def test_validation_probabilities_and_impossible_branch():
    instrument = pg.MeasurementInstrument({0: (np.diag([1, 0]),), 1: (np.diag([0, 1]),)})
    assert instrument.probabilities([1, 0]) == {0: 1, 1: 0}
    with pytest.raises(ValueError, match="zero"):
        instrument.conditional([1, 0], 1)
    for state in ([1, 1], [[1, 1], [0, 0]], [[2, 0], [0, -1]], [np.nan, 0]):
        with pytest.raises(ValueError):
            instrument.probabilities(state)
    p = pg.Pattern(inputs=(0,)).measure(0, instrument)
    with pytest.raises(ValueError, match="cutoff"):
        pg.simulate(p, backend="piquasso-fock", cutoff=3)
    for backend in ("piquasso-fock", "piquasso-mixed-fock"):
        with pytest.raises(ValueError, match="zero"):
            pg.simulate(
                p,
                inputs={0: pg.FockInput((1, 0))},
                backend=backend,
                cutoff=2,
                measurement_outcomes={0: 1},
            )


def test_finite_code_dual_povm_distinguishes_ambiguity_from_leakage():
    code = pg.GKPCode(cutoff=32, peak_width=0.8, envelope=0.4)
    instrument = code.discrimination_instrument()
    for bit, state in enumerate((code.zero(), code.one())):
        probabilities = instrument.probabilities(state.amplitudes)
        assert probabilities[1 - bit] == pytest.approx(0, abs=1e-12)
        assert probabilities["outside-code"] == pytest.approx(0, abs=1e-12)
        assert probabilities["inconclusive"] > 0.01
        assert sum(probabilities.values()) == pytest.approx(1)
    values, vectors = np.linalg.eigh(code.code_projector)
    assert values[0] < 1e-12
    assert instrument.probabilities(vectors[:, 0])["outside-code"] == pytest.approx(1)
