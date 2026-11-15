"""NativeFockBackend as a drop-in for PiquassoFockBackend, plus independent references.

Reference hierarchy: exact Gaussian moments (GaussianBackend), closed forms, a direct
position-space integral for the cubic-phase state, then Piquasso. Differences to
Piquasso are truncation effects: Piquasso applies CZ through an Euler decomposition
with a truncation after each factor, the native engine applies exact P U P. Observed
1 - F for the CZ/cat pattern below: 1.3e-5, 3.0e-7, 6.6e-9, 1.1e-10 at cutoffs
20, 28, 36, 44, so the tolerances are set one order above the cutoff-36 value.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.special import eval_hermite, factorial

import photographiq as pg
from photographiq import native
from photographiq.fock_analysis import fidelity

try:
    native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)

from photographiq.backends.native_fock import NativeFockBackend, NativeMixedFockBackend


def _piquasso_fock(cutoff):
    pytest.importorskip("piquasso")
    from photographiq.backends.fock import PiquassoFockBackend

    return PiquassoFockBackend(cutoff=cutoff)


def _cat_cz_pattern():
    return pg.Pattern(inputs=(0,)).extend(
        [
            pg.Prepare(1, 0.4),
            pg.Entangle(0, 1, 0.7),
            pg.Measure(0, pg.Homodyne(0.3), "m"),
            pg.Kerr(1, 0.2),
            pg.CubicPhase(1, 0.05),
            pg.PhotonSubtract(1),
            pg.Output((1,)),
        ]
    )


def _counting_pattern():
    return pg.Pattern(inputs=(0, 1)).extend(
        [
            pg.Squeeze(0, 0.3),
            pg.BeamSplitter(0, 1, 0.6),
            pg.QuadraticPhase(1, 0.3),
            pg.Rotate(1, 0.4),
            pg.Measure(0, pg.PhotonNumber(), "n"),
            pg.Output((1,)),
        ]
    )


def _run(pattern, backend, cutoff, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pg.simulate(
            pattern, backend=backend, inputs={0: pg.FockInput.cat(1.0, cutoff, 1)}, **kwargs
        )


def test_gaussian_pattern_matches_exact_gaussian_moments():
    pattern = pg.Pattern(inputs=("a", "b")).extend(
        [
            pg.Squeeze("a", 0.08),
            pg.Rotate("a", 0.2),
            pg.Displace("b", 0.12, -0.08),
            pg.BeamSplitter("a", "b", 0.2),
            pg.Entangle("a", "b", 0.08),
            pg.QuadraticPhase("b", 0.1),
        ]
    )
    exact = pg.simulate(pattern, backend="gaussian").state
    for backend in (NativeFockBackend(cutoff=16), NativeMixedFockBackend(cutoff=16)):
        state = pg.simulate(pattern, backend=backend).state
        for node in ("a", "b"):
            for angle in (0.0, 0.3, np.pi / 2):
                np.testing.assert_allclose(
                    state.quadrature(node, angle), exact.quadrature(node, angle), atol=2e-7, rtol=0
                )
            assert state.photon_number(node) == pytest.approx(exact.photon_number(node), abs=2e-7)


@pytest.mark.parametrize(
    "pattern,outcomes,tolerance",
    [(_cat_cz_pattern, {"m": 0.4}, 1e-7), (_counting_pattern, {"n": 1}, 1e-12)],
)
def test_non_gaussian_patterns_match_piquasso(pattern, outcomes, tolerance):
    cutoff = 36
    reference = _run(
        pattern(), _piquasso_fock(cutoff), cutoff, measurement_outcomes=outcomes, seed=1
    )
    result = _run(
        pattern(), NativeFockBackend(cutoff=cutoff), cutoff, measurement_outcomes=outcomes, seed=1
    )
    assert 1 - fidelity(reference.state, result.state) < tolerance
    key = next(iter(outcomes))
    ours, theirs = (r.measurement_statistics[key] for r in (result, reference))
    assert ours["kind"] == theirs["kind"]
    assert ours["value"] == pytest.approx(theirs["value"], rel=tolerance)
    assert [d["operation"] for d in result.state.diagnostics] == [
        d["operation"] for d in reference.state.diagnostics
    ]
    assert result.state.nodes == reference.state.nodes == (1,)
    assert result.state.basis == reference.state.basis


def test_piquasso_difference_vanishes_with_cutoff():
    differences = []
    for cutoff in (20, 28, 36):
        a = _run(_cat_cz_pattern(), _piquasso_fock(cutoff), cutoff, measurement_outcomes={"m": 0.4})
        b = _run(
            _cat_cz_pattern(),
            NativeFockBackend(cutoff=cutoff),
            cutoff,
            measurement_outcomes={"m": 0.4},
        )
        differences.append(1 - fidelity(a.state, b.state))
    assert differences[0] > 10 * differences[1] > 100 * differences[2]


def _cubic_phase_state_exact(r, gamma, cutoff):
    """<n| exp(i gamma q^3/6) |p-squeezed vacuum>, by direct position-space integration."""
    x = np.linspace(-40, 40, 400001)
    variance = np.exp(2 * r)  # Prepare(node, r): Var(q) = e^{2r}
    psi = (2 * np.pi * variance) ** -0.25 * np.exp(-(x**2) / (4 * variance) + 1j * gamma * x**3 / 6)
    n = np.arange(cutoff)[:, None]
    y = x / np.sqrt(2)
    basis = (
        eval_hermite(n, y[None])
        * np.exp(-(y**2) / 2)
        / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
    )
    basis /= 2**0.25  # hbar = 2 position wavefunctions
    integrand = basis * psi[None]
    return np.sum((integrand[:, 1:] + integrand[:, :-1]) / 2 * np.diff(x), axis=1)


def test_cubic_phase_converges_to_exact_state_and_matches_piquasso_definition():
    r, gamma = 0.3, 0.1
    pattern = pg.Pattern(inputs=()).extend(
        [pg.Prepare(0, r), pg.CubicPhase(0, gamma), pg.Output((0,))]
    )
    errors = {}
    for cutoff in (16, 24, 32):
        exact = _cubic_phase_state_exact(r, gamma, cutoff)
        for method in ("projected", "truncated_generator"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                state = pg.simulate(
                    pattern, backend=NativeFockBackend(cutoff, cubic_phase=method)
                ).state
            overlap = abs(np.vdot(exact, state.state_vector)) ** 2 / np.vdot(exact, exact).real
            errors[method, cutoff] = 1 - overlap
    for method in ("projected", "truncated_generator"):
        assert errors[method, 16] > errors[method, 24] > errors[method, 32]
        assert errors[method, 32] < 1e-8
    # The default (truncated generator) reproduces Piquasso's CubicPhase exactly.
    reference = pg.simulate(pattern, backend=_piquasso_fock(24)).state
    ours = pg.simulate(pattern, backend=NativeFockBackend(24)).state
    assert 1 - fidelity(reference, ours) < 1e-13
    # Projected matrix elements expose leakage that the truncated generator hides.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        leaky = pg.simulate(pattern, backend=NativeFockBackend(12, cubic_phase="projected")).state
    assert leaky.diagnostics[-1]["operation"] == "CubicPhase"
    assert leaky.diagnostics[-1]["retained_norm"] < 1 - 1e-6
    assert ours.diagnostics[-1]["retained_norm"] == pytest.approx(1, abs=1e-12)


def test_cubic_phase_moments_converge_for_both_definitions():
    """<p^k> of exp(i gamma q^3/6)|0>: analytic 1 + 3g^2, g + 15g^3, 3 + 10g^2 + 105g^4.

    Observed max relative errors at cutoffs 80/160/320, gamma = 0.8: projected
    8.9e-3/2.8e-4/1.4e-6, truncated generator 5.0e-3/1.2e-4/4.2e-7 (bounds: 2x observed)."""
    gamma = 0.8
    exact = np.array([1 + 3 * gamma**2, gamma + 15 * gamma**3, 3 + 10 * gamma**2 + 105 * gamma**4])
    bounds = {"projected": (1.8e-2, 5.6e-4), "truncated_generator": (1.0e-2, 2.4e-4)}
    pattern = pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, gamma)])
    for method, limits in bounds.items():
        errors = []
        for cutoff in (80, 160):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                state = pg.simulate(
                    pattern, backend=NativeFockBackend(cutoff, cubic_phase=method)
                ).state
            moments = np.array([state.quadrature_moment(0, k, np.pi / 2) for k in (2, 3, 4)])
            errors.append(np.max(abs(moments - exact) / exact))
        assert errors[0] < limits[0] and errors[1] < limits[1] and errors[1] < errors[0] / 10


def test_kerr_cross_kerr_and_hong_ou_mandel():
    backend = NativeFockBackend(cutoff=6)
    amplitudes = {(0, 0): 0.5, (1, 1): 0.5, (2, 1): 0.5, (0, 3): 0.5}
    backend.prepare_resource(("u", "v"), pg.FockSuperposition.from_mapping(amplitudes))
    backend.kerr("u", 0.3)
    backend.cross_kerr("u", "v", 0.7)
    state = backend.get_state().native.fock_amplitudes_map
    for (nu, nv), amplitude in amplitudes.items():
        assert state[nu, nv] == pytest.approx(
            amplitude * np.exp(1j * (0.3 * nu**2 + 0.7 * nu * nv)), abs=1e-15
        )
    hom = pg.Pattern(inputs=()).extend(
        [
            pg.PrepareResource(("a", "b"), pg.FockSuperposition.from_mapping({(1, 1): 1.0})),
            pg.BeamSplitter("a", "b", np.pi / 4),
            pg.Output(("a", "b")),
        ]
    )
    probabilities = pg.simulate(hom, backend=NativeFockBackend(cutoff=3)).state.probabilities
    assert probabilities[(1, 1)] < 1e-30
    assert probabilities[(2, 0)] == pytest.approx(0.5, abs=1e-15)
    assert probabilities[(0, 2)] == pytest.approx(0.5, abs=1e-15)


def test_cross_kerr_matches_piquasso_cross_kerr_gate():
    """Independent reference: Piquasso's CrossKerr(xi) = exp(i xi n_u n_v) (no Pattern command)."""
    pq = pytest.importorskip("piquasso")
    from photographiq.backends.mixed_fock import MixedFockBackend

    cutoff = 12
    pairs = (
        (NativeFockBackend(cutoff), _piquasso_fock(cutoff)),
        (NativeMixedFockBackend(cutoff), MixedFockBackend(cutoff)),
    )
    for ours, theirs in pairs:
        for backend in (ours, theirs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                backend.prepare("u", 0.0, pg.GaussianInput.coherent(0.6 + 0.3j))
                backend.prepare("v", 0.0, pg.GaussianInput.coherent(-0.5j))
        ours.cross_kerr("u", "v", 0.7)
        theirs._gate(("u", "v"), pq.CrossKerr(xi=0.7))
        np.testing.assert_allclose(
            ours.get_state().density_matrix, theirs.get_state().density_matrix, atol=1e-14
        )


def test_sampled_trajectories_match_piquasso():
    cutoff = 32
    shots = {}
    for label, factory in (("native", NativeFockBackend), ("piquasso", None)):
        values = []
        for seed in range(4):
            backend = factory(cutoff=cutoff) if factory else _piquasso_fock(cutoff)
            values.append(_run(_counting_pattern(), backend, cutoff, seed=seed).outcomes["n"])
            backend = factory(cutoff=cutoff) if factory else _piquasso_fock(cutoff)
            values.append(_run(_cat_cz_pattern(), backend, cutoff, seed=seed).outcomes["m"])
        shots[label] = values
    np.testing.assert_allclose(shots["native"], shots["piquasso"], atol=1e-6, rtol=0)


def test_instrument_and_gkp_readout_match_piquasso():
    cutoff = 4
    vectors = [np.array([1, 1j, 0, 0]) / np.sqrt(2), np.array([1, -1j, 0, 0]) / np.sqrt(2)]
    kraus = {"+": vectors[0].conj(), "-": vectors[1].conj(), "2": np.eye(4)[2], "3": np.eye(4)[3]}
    instrument = pg.MeasurementInstrument({k: (v[None, :],) for k, v in kraus.items()})
    source = pg.FockSuperposition.from_mapping({(0, 1): 0.6, (1, 0): 0.8j})
    pattern = pg.Pattern(inputs=()).extend(
        [
            pg.PrepareResource(("a", "b"), source),
            pg.Measure("a", instrument, "k"),
            pg.Output(("b",)),
        ]
    )
    for outcome in ("+", "-"):
        a = pg.simulate(
            pattern, backend=_piquasso_fock(cutoff), measurement_outcomes={"k": outcome}
        )
        b = pg.simulate(
            pattern, backend=NativeFockBackend(cutoff), measurement_outcomes={"k": outcome}
        )
        assert b.measurement_statistics["k"]["value"] == pytest.approx(
            a.measurement_statistics["k"]["value"], abs=1e-15
        )
        np.testing.assert_allclose(b.state.density_matrix, a.state.density_matrix, atol=1e-15)
    readout = pg.Pattern(inputs=(0,)).measure(0, pg.PhysicalGKPReadout("Z"), key="g")
    a = pg.simulate(readout, backend=_piquasso_fock(8), measurement_outcomes={"g": 2.6})
    b = pg.simulate(readout, backend=NativeFockBackend(8), measurement_outcomes={"g": 2.6})
    assert b.outcomes["g"] == a.outcomes["g"]
    assert b.measurement_statistics["g"]["value"] == pytest.approx(
        a.measurement_statistics["g"]["value"], rel=1e-12
    )


def test_truncation_errors_warnings_and_snapshots():
    with pytest.raises(ValueError, match="Fock truncation norm"):
        NativeFockBackend(cutoff=4).prepare(0, 0.0, pg.GaussianInput.coherent(2.0))
    backend = NativeFockBackend(cutoff=12)
    backend.prepare(0, 0.0, pg.GaussianInput.coherent(1.2))
    with pytest.warns(RuntimeWarning, match="retained Fock norm"):
        backend.displace(0, 0.5, 0.0)
    backend.prepare(1, 0.2)
    before = backend.get_state()
    with pytest.raises(ValueError, match="Fock truncation norm"):
        backend.displace(0, 6.0, 0.0)  # a failed norm check keeps the pre-gate state
    np.testing.assert_array_equal(backend.get_state().density_matrix, before.density_matrix)
    with pytest.raises(ValueError, match="Repeated gate mode"):
        backend.entangle(0, 0)
    with pytest.raises(MemoryError):
        NativeFockBackend(cutoff=40, max_dimension=100).prepare_resource(
            (0, 1), pg.FockSuperposition.from_mapping({(0, 0): 1.0})
        )
    snapshot = backend.get_state((1, 0))
    assert snapshot.nodes == (1, 0) and not hasattr(snapshot.native, "state_vector")
    with pytest.raises(ValueError):
        snapshot.state_vector
    full = backend.get_state()
    np.testing.assert_allclose(
        snapshot.reduced((0,)).density_matrix, full.reduced((0,)).density_matrix, atol=1e-15
    )
    backend.rotate(0, 0.3)  # snapshots are independent of later evolution
    assert full.native is not backend.native
    assert full.norm == pytest.approx(1, abs=1e-14)
    with pytest.raises(ValueError, match="cubic_phase"):
        NativeFockBackend(cutoff=4, cubic_phase="expm")


def test_native_backends_never_import_piquasso():
    script = (
        "import sys, photographiq as pg\n"
        "from photographiq.backends.native_fock import NativeFockBackend, NativeMixedFockBackend\n"
        "p = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.3), pg.Entangle(0, 1, 0.5),"
        " pg.Loss(1, 0.9), pg.Measure(0, pg.Homodyne(0.2), 'm'), pg.CubicPhase(1, 0.1),"
        " pg.Output((1,))])\n"
        "for b in (NativeFockBackend, NativeMixedFockBackend):\n"
        "    q = p if b is NativeMixedFockBackend else pg.Pattern(inputs=(0,)).extend("
        "[c for c in p.commands if type(c).__name__ != 'Loss'])\n"
        "    pg.simulate(q, backend=b(cutoff=20), inputs={0: pg.CatResource(1.0)}, seed=3)\n"
        "assert not any(m == 'piquasso' or m.startswith('piquasso.') for m in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
