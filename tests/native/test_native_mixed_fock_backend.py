"""NativeMixedFockBackend: pure/mixed agreement, Piquasso mixed reference, Gaussian moments.

Loss never raises photon number, so pure-loss Kraus maps are exact on the truncated
space; thermal noise (amplifier) and Gaussian gates are truncation-limited, and the
cutoffs below keep those effects under the stated tolerances (measured margins in the
comments).
"""

import warnings

import numpy as np
import pytest

import photographiq as pg
from photographiq import native

try:
    native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)

from photographiq.backends.native_fock import NativeFockBackend, NativeMixedFockBackend


def _quiet(pattern, backend, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pg.simulate(pattern, backend=backend, **kwargs)


def _piquasso_mixed(cutoff):
    pytest.importorskip("piquasso")
    from photographiq.backends.mixed_fock import MixedFockBackend

    return MixedFockBackend(cutoff=cutoff)


def _non_gaussian_pattern():
    return pg.Pattern(inputs=(0,)).extend(
        [
            pg.Prepare(1, 0.3),
            pg.Entangle(0, 1, 0.6),
            pg.Kerr(0, 0.2),
            pg.Measure(0, pg.Homodyne(0.3), "m"),
            pg.CubicPhase(1, 0.05),
            pg.PhotonSubtract(1),
            pg.Prepare(2, 0.0, pg.FockInput.number(1)),
            pg.BeamSplitter(1, 2, 0.4),
            pg.Measure(2, pg.PhotonNumber(), "n"),
            pg.Output((1,)),
        ]
    )


def test_pure_inputs_give_the_pure_backend_state():
    cutoff = 24
    inputs = {0: pg.FockInput.cat(1.0, cutoff, 1)}
    outcomes = {"m": 0.2, "n": 1}
    pure = _quiet(
        _non_gaussian_pattern(),
        NativeFockBackend(cutoff),
        inputs=inputs,
        measurement_outcomes=outcomes,
    )
    mixed = _quiet(
        _non_gaussian_pattern(),
        NativeMixedFockBackend(cutoff),
        inputs=inputs,
        measurement_outcomes=outcomes,
    )
    np.testing.assert_allclose(mixed.state.density_matrix, pure.state.density_matrix, atol=1e-13)
    for key in outcomes:
        assert mixed.measurement_statistics[key]["value"] == pytest.approx(
            pure.measurement_statistics[key]["value"], rel=1e-12
        )


def test_noisy_channels_and_mixed_inputs_match_piquasso_mixed():
    # Measured at cutoff 18: max |d rho| = 5.6e-17 (native 0.11 s, Piquasso 182 s).
    cutoff = 10
    thermal = pg.GaussianInput(mean=(0.3, -0.2), covariance=((1.6, 0.2), (0.2, 1.1)))
    rho = np.array([[0.5, 0.2j, 0], [-0.2j, 0.3, 0.1], [0, 0.1, 0.2]])
    pattern = pg.Pattern(inputs=(0,)).extend(
        [
            pg.Prepare(1, 0.0, pg.FockDensityMatrix(rho, ((0,), (1,), (2,)))),
            pg.BeamSplitter(0, 1, 0.5),
            pg.Loss(0, 0.8, 0.2),
            pg.Rotate(1, 0.3),
            pg.Measure(1, pg.Homodyne(0.4, efficiency=0.85, noise=0.05), "m"),
            pg.Output((0,)),
        ]
    )
    kwargs = dict(inputs={0: thermal}, measurement_outcomes={"m": 0.25})
    reference = _quiet(pattern, _piquasso_mixed(cutoff), **kwargs)
    result = _quiet(pattern, NativeMixedFockBackend(cutoff), **kwargs)
    np.testing.assert_allclose(
        result.state.density_matrix, reference.state.density_matrix, atol=1e-10
    )
    assert result.measurement_statistics["m"]["value"] == pytest.approx(
        reference.measurement_statistics["m"]["value"], rel=1e-10
    )
    assert [d["operation"] for d in result.state.diagnostics] == [
        d["operation"] for d in reference.state.diagnostics
    ]


def test_ladder_and_counting_match_piquasso_mixed():
    cutoff = 12
    pattern = pg.Pattern(inputs=(0, 1)).extend(
        [
            pg.Squeeze(0, 0.1),
            pg.Displace(1, 0.2, 0.1),
            pg.BeamSplitter(0, 1, 0.7),
            pg.Loss(1, 0.7),
            pg.PhotonAdd(0),
            pg.PhotonSubtract(1),
            pg.Measure(1, pg.PhotonNumber(), "n"),
            pg.Output((0,)),
        ]
    )
    reference = _quiet(pattern, _piquasso_mixed(cutoff), measurement_outcomes={"n": 2})
    result = _quiet(pattern, NativeMixedFockBackend(cutoff), measurement_outcomes={"n": 2})
    np.testing.assert_allclose(
        result.state.density_matrix, reference.state.density_matrix, atol=1e-12
    )
    assert result.measurement_statistics["n"]["value"] == pytest.approx(
        reference.measurement_statistics["n"]["value"], rel=1e-12
    )


@pytest.mark.parametrize("thermal_photons", [0.0, 0.3])
def test_loss_matches_exact_gaussian_moments(thermal_photons):
    pattern = pg.Pattern(inputs=("a", "b")).extend(
        [
            pg.Displace("a", 0.6, -0.3),
            pg.Squeeze("a", 0.15),
            pg.Entangle("a", "b", 0.2),
            pg.Loss("a", 0.7, thermal_photons),
            pg.Loss("b", 0.9, thermal_photons),
        ]
    )
    exact = pg.simulate(pattern, backend="gaussian").state
    state = _quiet(pattern, NativeMixedFockBackend(cutoff=24)).state
    for node in ("a", "b"):
        for angle in (0.0, 0.4, np.pi / 2):
            np.testing.assert_allclose(
                state.quadrature(node, angle), exact.quadrature(node, angle), atol=1e-9
            )
        assert state.photon_number(node) == pytest.approx(exact.photon_number(node), abs=1e-9)


def test_thermal_gaussian_input_is_geometric_and_physical():
    nbar, cutoff = 0.7, 30
    thermal = pg.GaussianInput(covariance=((2 * nbar + 1, 0), (0, 2 * nbar + 1)))
    state = _quiet(
        pg.Pattern(inputs=(0,)), NativeMixedFockBackend(cutoff), inputs={0: thermal}
    ).state
    p = np.array([state.probabilities[(n,)] for n in range(cutoff)])
    tail = (nbar / (1 + nbar)) ** cutoff  # removed by renormalisation
    exact = nbar ** np.arange(cutoff) / (1 + nbar) ** np.arange(1, cutoff + 1) / (1 - tail)
    np.testing.assert_allclose(p, exact, atol=1e-15)
    rho = state.density_matrix
    assert np.abs(rho - np.diag(np.diag(rho))).max() < 1e-15


def test_states_stay_physical():
    cutoff = 20
    pattern = pg.Pattern(inputs=(0,)).extend(
        [
            pg.Prepare(1, 0.3),
            pg.Entangle(0, 1, 0.5),
            pg.Loss(1, 0.6, 0.4),
            pg.CubicPhase(1, 0.1),
            pg.Measure(0, pg.Homodyne(0.2, efficiency=0.9), "m"),
            pg.Output((1,)),
        ]
    )
    # Postselected: sampling would spend its time in the inherited Python CDF inversion.
    result = _quiet(
        pattern,
        NativeMixedFockBackend(cutoff),
        inputs={0: pg.CatResource(0.8)},
        measurement_outcomes={"m": 0.3},
    )
    report = result.state.native.physicality()
    assert report["hermiticity"] < 1e-15
    assert abs(report["trace"] - 1) < 1e-13
    assert report["min_eigenvalue"] > -1e-13
    assert report["max_eigenvalue"] <= 1 + 1e-13
    assert not hasattr(result.state.native, "state_vector")


def test_multi_kraus_instrument_matches_explicit_partial_trace():
    """Random 2-outcome, 4-Kraus instrument on mode A of a 2-mode mixed state.

    Reference: the unmeasured density matrix contracted by index sums with each effect
    E = sum K^dag K (probability) and with the Kraus maps (conditional state of B).
    """
    from photographiq.instruments import MeasurementInstrument

    d, rng = 8, np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(4 * d, d)) + 1j * rng.normal(size=(4 * d, d)))
    kraus = [q[i * d : (i + 1) * d] for i in range(4)]
    branches = {"a": kraus[0:2], "b": kraus[2:4]}
    instrument = MeasurementInstrument(branches)

    def pattern(measure):
        commands = [
            pg.Displace("A", 0.3, 0.1),
            pg.Squeeze("B", 0.1),
            pg.BeamSplitter("A", "B", 0.6),
            pg.Loss("B", 0.8, 0.1),
        ]
        tail = [pg.Measure("A", instrument, "m"), pg.Output(("B",))] if measure else []
        return pg.Pattern(inputs=("A", "B")).extend(commands + (tail or [pg.Output(("A", "B"))]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = pg.simulate(pattern(False), backend="native-mixed-fock", cutoff=d).state
        basis, rho = list(full.probabilities), full.density_matrix
        for label, ops in branches.items():
            result = pg.simulate(
                pattern(True),
                backend="native-mixed-fock",
                cutoff=d,
                measurement_outcomes={"m": label},
            )
            ref = np.zeros((d, d), complex)
            for k in ops:
                for i, (na, nb) in enumerate(basis):
                    for j, (ma, mb) in enumerate(basis):
                        # sum_a K[a, na] rho K^dag[ma, a] restricted to mode B indices
                        ref[nb, mb] += (k[:, na] @ k[:, ma].conj()) * rho[i, j]
            prob = np.trace(ref).real
            assert result.measurement_statistics["m"]["value"] == pytest.approx(prob, abs=1e-12)
            out = result.state.density_matrix
            np.testing.assert_allclose(out, ref[: out.shape[0], : out.shape[1]] / prob, atol=1e-12)
