"""Physical logical-XY GKP instrument: conventions, independent references and boundaries.

The untruncated position-grid model in tests/validation/gkp_xy_reference.py is the
independent finite-energy reference (no Fock truncation, its own gates and cell integrals);
the ideal qubit is the logical reference. Tolerances separate Fock truncation (engine vs
grid) from finite-energy physics (grid vs ideal).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import photographiq as pg
from photographiq.encoded import quarter_turns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from gkp_xy_reference import GridModel, ideal_probabilities  # noqa: E402

pytest.importorskip("photographiq._native")

WIDTH, ENVELOPE, PEAKS = 0.6, 0.6, 6
CARDINAL = {"0": (1, 0), "1": (0, 1), "+": (1, 1), "-": (1, -1), "+i": (1, 1j), "-i": (1, -1j)}


@pytest.fixture(scope="module")
def grid():
    return GridModel(WIDTH, ENVELOPE, PEAKS)


def code(cutoff):
    return pg.GKPCode(
        cutoff=cutoff, peak_width=WIDTH, envelope=ENVELOPE, peaks=PEAKS, grid_points=2049
    )


def measure(cutoff, alpha, logical, protocol="auto", outcomes=None, seed=0, **options):
    gkp = code(cutoff)
    pattern = pg.Pattern(inputs=("d",))
    pattern.append(
        pg.Measure(
            "d", gkp.logical_measurement("XY", alpha=alpha, protocol=protocol, **options), "m"
        )
    )
    pattern.append(pg.Output(()))
    extra = {"measurement_outcomes": {"m": outcomes}} if outcomes is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pg.simulate(
            pattern,
            seed=seed,
            backend="native-fock",
            cutoff=cutoff,
            inputs={"d": gkp.encode(*logical)},
            **extra,
        )
    return result


def test_quarter_turns_uses_an_absolute_tolerance():
    assert [quarter_turns(k * np.pi / 2) for k in range(5)] == [0, 1, 2, 3, 0]
    assert quarter_turns(-np.pi / 2) == 3
    assert quarter_turns(np.pi / 2 + 1e-9) is None
    assert quarter_turns(0.7) is None


def test_explicit_protocol_is_required_and_inputs_are_validated():
    gkp = code(48)
    assert isinstance(gkp.logical_measurement("Y", protocol="clifford"), pg.GKPLogicalXYMeasurement)
    assert gkp.logical_measurement("Y", protocol="clifford").angle == pytest.approx(np.pi / 2)
    with pytest.raises(NotImplementedError):  # legacy factory: no implicit synthesis
        gkp.logical_measurement("XY", alpha=0.7)
    with pytest.raises(NotImplementedError):
        gkp.logical_measurement("XY", alpha=0.7, protocol="clifford")
    with pytest.raises(ValueError):
        pg.GKPLogicalXYMeasurement(angle=0.7, protocol="injection")  # no ancilla code
    for bad in (
        {"protocol": "rotated-homodyne"},
        {"max_rounds": -1},
        {"ancilla_transmissivity": 0.0},
        {"detector_efficiency": 1.5},
        {"homodyne_noise": -1.0},
    ):
        with pytest.raises(ValueError):
            pg.GKPLogicalXYMeasurement(angle=0.7, code=gkp, **bad)
    with pytest.raises(ValueError):
        gkp.logical_measurement("Z", protocol="auto")
    noisy = pg.GKPLogicalXYMeasurement(angle=0.7, code=gkp, ancilla_transmissivity=0.9)
    assert "mixed_fock" in noisy.required_capabilities
    with pytest.raises(NotImplementedError):  # pure engine cannot represent the loss channel
        noisy.validate_backend(pg.registry.create_backend("native-fock", 48))
    with pytest.raises(NotImplementedError):
        pg.LogicalMeasurementSynthesis().lower(gkp, 0.7)
    assert isinstance(
        pg.LogicalMeasurementSynthesis("injection").lower(gkp, 0.7), pg.GKPLogicalXYMeasurement
    )


def test_logical_frame_rules_for_the_xy_family():
    rng = np.random.default_rng(4)
    x, z = np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    for _ in range(20):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi /= np.linalg.norm(psi)
        alpha = rng.uniform(0, 2 * np.pi)
        p = ideal_probabilities(psi, alpha)
        # X M(a) X = M(-a): measuring X psi at -a reproduces psi at a.
        assert ideal_probabilities(x @ psi, -alpha) == pytest.approx(p, abs=1e-12)
        # Z M(a) Z = -M(a): measuring Z psi at a flips the outcome.
        assert ideal_probabilities(z @ psi, alpha) == pytest.approx(p[::-1], abs=1e-12)
        frame = pg.LogicalPauliFrame(1, 1)
        assert np.cos(frame.xy_angle(alpha)) == pytest.approx(-np.cos(-alpha))


def test_physical_y_on_cardinal_states_matches_the_untruncated_reference(grid):
    """Clifford Y readout (Gaussian S^dagger shear + X readout): engine vs grid vs ideal."""
    for label, logical in CARDINAL.items():
        record = measure(48, np.pi / 2, logical, "clifford").records["m"]
        engine = record.readout_probabilities[0]
        reference = grid.outcome_probabilities(
            grid.encode(*logical), np.pi / 2, protocol="clifford"
        )[0]
        ideal = ideal_probabilities(logical, np.pi / 2)[0]
        assert record.applied_operations == ("S_dagger",) and record.rounds == ()
        assert abs(engine - reference) < 5e-4, label  # Fock truncation and grid discretization
        assert abs(reference - ideal) < 0.15, label  # finite-energy error at width 0.6
        if label in ("+i", "-i"):  # the Y eigenstates fix the sign convention
            assert (engine > 0.5) == (label == "+i")


def test_injection_record_likelihood_matches_the_reference(grid):
    """Postselected analog records: engine joint density and decoded path vs the grid model."""
    cases = (
        (np.pi / 4, [0.31], 0.4),  # success in round 1
        (0.7, [0.2], 1.1),
        (0.7, [2.4, 0.1], 0.3),  # one heralded failure, then success
        (np.pi / 4, [2.7, 0.2], -0.9),  # failure leaves a Clifford remainder: Y-state injection
    )
    for alpha, ancilla, data in cases:
        record = measure(64, alpha, (1, 1), outcomes={"ancilla": ancilla, "data": data}).records[
            "m"
        ]
        density, bits, bit = grid.record_density(grid.encode(1, 1), alpha, ancilla, data)
        assert [r["bit"] for r in record.rounds] == bits and record.bit == bit
        assert record.joint_density == pytest.approx(density, rel=1e-4)
        assert record.heralded_success


def test_postselected_records_must_match_the_protocol():
    with pytest.raises(ValueError, match="fewer ancilla"):
        measure(48, np.pi / 4, (1, 1), outcomes={"ancilla": [2.7], "data": 0.0})
    with pytest.raises(ValueError, match="more ancilla"):
        measure(48, np.pi / 4, (1, 1), outcomes={"ancilla": [0.2, 0.1], "data": 0.0})
    with pytest.raises(ValueError):
        measure(48, np.pi / 4, (1, 1), outcomes=0.3)


def test_exhausted_round_budget_is_a_reported_heralded_failure():
    record = measure(48, 0.7, (1, 1), max_rounds=0).records["m"]
    assert record.heralded_success is False and record.rounds == ()
    assert "round budget exhausted" in record.applied_operations[-1]


def test_reference_is_trace_preserving_and_reaches_the_ideal_limit():
    sharp = GridModel(0.25, 0.25, 8)
    state = sharp.encode(np.cos(0.4), np.exp(0.9j) * np.sin(0.4))
    ideal = ideal_probabilities([np.cos(0.4), np.exp(0.9j) * np.sin(0.4)], np.pi / 3)[0]
    rows = sharp.protocol_map(sharp.operator(state), np.pi / 3, max_rounds=4)
    total = sum(r["value"] for r in rows)
    success = sum(r["value"].real for r in rows if r["success"])
    heralded = sum(r["value"].real for r in rows if r["success"] and r["bit"] == 0) / success
    # Integration over every continuous outcome (not point sums). The residual ~1e-7 is grid
    # quadrature of the codeword norm at dx = 0.07 for sharp (width 0.25) peaks.
    assert abs(total - 1) < 1e-6
    assert success == pytest.approx(1 - 2.0**-4, abs=1e-3)
    assert heralded == pytest.approx(ideal, abs=0.01)
