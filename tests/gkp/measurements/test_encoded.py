import numpy as np
import pytest

import photographiq as pg


@pytest.mark.parametrize("state", [[1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j]])
@pytest.mark.parametrize("alpha", [0, 0.31, np.pi / 4, np.pi / 2, 3.2])
def test_ideal_reference(state, alpha):
    state = np.array(state) / np.linalg.norm(state)
    observable = np.array([[0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 0]])
    expected = (1 + np.vdot(state, observable @ state).real) / 2
    assert pg.IdealLogicalXYMeasurement(alpha).probabilities(state) == pytest.approx(
        (expected, 1 - expected)
    )


@pytest.mark.parametrize("basis,raw", [("X", -5.1), ("Z", 2.6)])
@pytest.mark.parametrize("backend", ["piquasso-fock", "piquasso-mixed-fock"])
def test_physical_analog_result_and_feedforward(basis, raw, backend):
    measurement = pg.PhysicalGKPReadout(basis, frame=pg.LogicalPauliFrame(1, 1))
    p = pg.Pattern(inputs=(0,)).measure(0, measurement, key="analog")
    p.append(
        pg.Signal("bit", pg.CallableExpression(lambda r: r["analog"].bit, frozenset({"analog"})))
    )
    result = pg.simulate(p, backend=backend, cutoff=8, measurement_outcomes={"analog": raw})
    decoded = result.outcomes["analog"]
    residual, bit = pg.gkp.decode_shift(raw)
    assert decoded.bit == bit ^ 1
    assert decoded.residual == residual
    assert decoded.raw_outcome == raw
    assert decoded.confidence is None and decoded.code_subspace_leakage is None
    assert result.records["bit"] == bit ^ 1
    assert result.measurement_statistics["analog"]["kind"] == "density"


def test_unsupported_measurements_never_lower_to_rotated_homodyne():
    code = pg.GKPCode()
    for angle in (np.pi / 2, np.pi / 4, 0.123, 1e-9, np.pi + 1e-9):
        with pytest.raises(NotImplementedError):
            code.logical_measurement("XY", alpha=angle)
    assert code.logical_measurement("XY", alpha=0).basis == "X"
    assert code.logical_measurement("XY", alpha=np.pi).flip == 1
    assert code.logical_measurement("XY", alpha=2 * np.pi).flip == 0
    with pytest.raises(NotImplementedError):
        code.logical_measurement("Y")
    with pytest.raises(NotImplementedError):
        pg.LogicalMeasurementSynthesis().lower(code, np.pi / 4)
    with pytest.raises(NotImplementedError):
        pg.simulate(pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement()))


def test_resources_gram_projector_and_gate_interface():
    code = pg.GKPCode(cutoff=32)
    assert abs(code.gram[0, 1]) > 1e-5
    for coefficients in ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j)):
        state = code.encode(*coefficients)
        assert code.leakage(state.amplitudes) == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(code.plus().amplitudes, code.encode(1, 1).amplitudes)
    np.testing.assert_allclose(code.minus().amplitudes, code.encode(1, -1).amplitudes)
    assert code.logical_gate(0, "S") == pg.QuadraticPhase(0, 1)
    assert code.logical_gate(0, "H") == pg.Rotate(0, np.pi / 2)
    assert code.logical_gate(0, "X") == pg.gkp.logical_displacement(0)
    assert code.logical_gate(0, "Z") == pg.gkp.logical_displacement(0, "Z")
    assert code.logical_cz(0, 1) == pg.Entangle(0, 1, 1)
    with pytest.raises(NotImplementedError):
        code.logical_gate(0, "T")
    with pytest.raises(ValueError):
        code.encode(0, 0)


@pytest.mark.parametrize("cell", [-10000, -3, -1, 0, 1, 4, 10000])
def test_decoder_cells_boundaries(cell):
    length = pg.gkp.SPACING
    for fraction in (0, 0.1, -0.1, 0.5):
        raw = (cell + fraction) * length
        result = pg.NearestCellDecoder().decode(raw)
        expected = cell + (fraction == 0.5)
        assert result.cell == expected
        assert result.bit == expected % 2
        assert result.residual == pytest.approx(raw - expected * length)


@pytest.mark.parametrize("x,z", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_frame_against_qubit_conjugation(x, z):
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])
    frame = pg.LogicalPauliFrame(x, z)
    alpha = 0.37

    def observable(a):
        return np.array([[0, np.exp(-1j * a)], [np.exp(1j * a), 0]])

    F = np.linalg.matrix_power(X, x) @ np.linalg.matrix_power(Z, z)
    np.testing.assert_allclose(
        F.conj().T @ observable(alpha) @ F, observable(frame.xy_angle(alpha))
    )
    assert frame.hadamard() == pg.LogicalPauliFrame(z, x)
    assert frame.phase() == pg.LogicalPauliFrame(x, x ^ z)
    assert frame.compose(frame) == pg.LogicalPauliFrame()
    other = pg.LogicalPauliFrame(1, 0)
    assert frame.cz(other) == (pg.LogicalPauliFrame(x, z ^ 1), pg.LogicalPauliFrame(1, x))
