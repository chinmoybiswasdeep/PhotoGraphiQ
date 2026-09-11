import numpy as np
import pytest

import photographiq as pg


def test_invalid_encoded_parameters_fail_explicitly():
    for cutoff in (1, True, 2.5):
        with pytest.raises(ValueError):
            pg.GKPCode(cutoff=cutoff)
    for alpha in (None, np.nan, np.inf, 1j, [0]):
        with pytest.raises(ValueError):
            pg.GKPCode().logical_measurement("XY", alpha=alpha)
    with pytest.raises(ValueError):
        pg.GKPCode().logical_measurement("Z", alpha=0)
    with pytest.raises(ValueError):
        pg.GKPCode().logical_cz(0, 0)
    with pytest.raises(ValueError, match="singular"):
        pg.GKPCode(cutoff=2).code_projector
    for bits in ((1.0, 0), (2, 0), (0, -1)):
        with pytest.raises(ValueError):
            pg.LogicalPauliFrame(*bits)
    with pytest.raises(ValueError):
        pg.LogicalPauliFrame().correction("unknown")
    for kwargs in ({"flip": 2}, {"flip": 1.0}, {"decoder": None}, {"frame": None}):
        with pytest.raises(ValueError):
            pg.PhysicalGKPReadout(**kwargs)
    with pytest.raises(ValueError):
        pg.IdealLogicalXYMeasurement(np.nan).probabilities([1, 0])
    with pytest.raises(NotImplementedError):
        pg.modular_effects(4, "Y")
    with pytest.raises(ValueError):
        pg.modular_effects(1)


def test_soft_frame_probabilities_reverse_with_labels():
    code = pg.GKPCode(cutoff=24)
    decoder = pg.SoftDecisionDecoder(code)
    p = pg.Pattern(inputs=(0,)).measure(
        0, pg.PhysicalGKPReadout("Z", decoder, pg.LogicalPauliFrame(1, 0))
    )
    r = pg.simulate(
        p,
        inputs={0: code.zero()},
        backend="piquasso-fock",
        cutoff=24,
        measurement_outcomes={0: 0.2},
    )
    assert r.outcomes[0].probabilities == tuple(reversed(decoder.decode(0.2).probabilities))


def test_mixed_input_instrument_and_multimode_contract():
    instrument = pg.MeasurementInstrument({"all": (np.eye(3),)})
    source = pg.FockDensityMatrix([[0.4, 0.1j], [-0.1j, 0.6]], ((0, 1), (1, 0)))
    p = pg.Pattern(inputs=(0, 1)).measure(0, instrument)
    result = pg.simulate(p, initial_state=source, backend="piquasso-mixed-fock", cutoff=3)
    np.testing.assert_allclose(result.state.density_matrix, np.diag([0.6, 0.4, 0]))
    with pytest.raises(ValueError):
        pg.multimode_readout(result.state, {})
    with pytest.raises(ValueError):
        pg.multimode_readout(result.state, {1: "Z"}, frames={0: pg.LogicalPauliFrame()})
    normal = pg.multimode_readout(result.state, {1: "Z"})
    flipped = pg.multimode_readout(result.state, {1: "Z"}, frames={1: pg.LogicalPauliFrame(1, 0)})
    assert normal["marginal_probabilities"][1] == tuple(
        reversed(flipped["marginal_probabilities"][1])
    )
