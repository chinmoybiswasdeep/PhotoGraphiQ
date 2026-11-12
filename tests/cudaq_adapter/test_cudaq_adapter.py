"""CUDA-Q Dynamics adapter: generator conventions vs native-fock, truncation mapping,
Lindblad loss vs the native mixed-Fock channel, and translation wiring.

Nothing here runs CUDA-Q unless ``cudaq`` is importable (it is Linux-only).
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

import photographiq.cudaq_dynamics as cq
from photographiq import native

HAS_CUDAQ = importlib.util.find_spec("cudaq") is not None
needs_native = pytest.mark.skipif(
    not native.AVAILABLE or "fock" not in native.components(), reason="native fock missing"
)
CUTOFF = 6


def _random(n, rng):
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    return psi / np.linalg.norm(psi)


def _phase_aligned_error(a, b):
    """Max |a - e^{i phi} b|: native Gaussian gates are defined up to a global phase."""
    overlap = np.vdot(b, a)
    return float(np.abs(a - overlap / abs(overlap) * b).max())


def _native_pure(method, modes, args, psi, **kwargs):
    from photographiq.backends.native_fock import NativeFockBackend, NativePureFockState

    backend = NativeFockBackend(CUTOFF, norm_tolerance=0.9, **kwargs)
    backend.nodes = tuple(range(modes))
    backend.native = NativePureFockState(psi.copy(), modes, CUTOFF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # retained-norm warnings: P U psi
        getattr(backend, method)(*args)
    return backend.native.state_vector


def _reference(h, modes, local_dim, psi):
    """P expm(-i H_D) psi, normalised (native normalises after each gate)."""
    dims = {m: local_dim for m in range(modes)}
    wide = cq.total_photon_to_per_mode(psi, dims, CUTOFF)
    out = expm_multiply(-1j * sparse.csr_array(cq.to_matrix(h, dims)), wide)
    out = cq.per_mode_to_total_photon(out, dims, CUTOFF)
    return out / np.linalg.norm(out)


# (native method, modes, native args, generator, native kwargs, enlarged local dimension)
GATES = [
    ("rotate", 1, (0, 0.9), cq.rotation(0, 0.9), {}, 60),
    ("displace", 1, (0, 0.6, -0.4), cq.displacement(0, 0.6, -0.4), {}, 60),
    ("squeeze", 1, (0, 0.3), cq.squeezing(0, 0.3), {}, 60),
    ("quadratic_phase", 1, (0, 0.4), cq.quadratic_phase(0, 0.4), {}, 60),
    ("kerr", 1, (0, 0.3), cq.kerr(0, 0.3), {}, 60),
    ("cubic_phase", 1, (0, 0.1), cq.cubic_phase(0, 0.1), {"cubic_phase": "projected"}, 60),
    ("beamsplitter", 2, (0, 1, 0.7), cq.beamsplitter(0, 1, 0.7), {}, 30),
    ("entangle", 2, (0, 1, 0.5), cq.controlled_z(0, 1, 0.5), {}, 30),
    ("cross_kerr", 2, (0, 1, 0.2), cq.cross_kerr(0, 1, 0.2), {}, 30),
]


@needs_native
@pytest.mark.parametrize("method, modes, args, h, kwargs, dim", GATES, ids=[g[0] for g in GATES])
def test_generator_matches_native_fock(method, modes, args, h, kwargs, dim):
    rng = np.random.default_rng(7)
    size = len(cq._total_photon_basis(modes, CUTOFF))
    for _ in range(3):
        psi = _random(size, rng)
        expected = _native_pure(method, modes, args, psi, **kwargs)
        # Truncation convergence: two enlarged local dimensions agree with native.
        for local_dim in (dim, dim + 10):
            assert _phase_aligned_error(expected, _reference(h, modes, local_dim, psi)) < 1e-10
        # Negative control: the opposite sign is a different gate.
        assert _phase_aligned_error(expected, _reference(-1 * h, modes, dim, psi)) > 1e-3


@needs_native
def test_cubic_truncated_generator_differs_from_projected_generator():
    """Native default expm(i g Q^3/6) cubes the truncated Q; to_matrix projects q^3."""
    psi = _random(CUTOFF, np.random.default_rng(3))
    expected = _native_pure("cubic_phase", 1, (0, 0.4), psi)
    q = cq.to_matrix(cq.position(0), {0: CUTOFF})
    assert np.abs(expected - expm(1j * 0.4 / 6 * q @ q @ q) @ psi).max() < 1e-12
    projected = cq.to_matrix(cq.cubic_phase(0, 0.4), {0: CUTOFF})
    assert np.abs(expected - expm(-1j * projected) @ psi).max() > 1e-3


@pytest.mark.parametrize("h", [g[3] for g in GATES], ids=[g[0] for g in GATES])
def test_generators_hermitian(h):
    assert h.is_hermitian()
    matrix = cq.to_matrix(h, {m: 8 for m in h.modes})
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)


def test_hamiltonian_algebra():
    a, ad = cq.annihilate(0), cq.create(0)
    commutator = a * ad - ad * a
    assert commutator.terms == {(): 1}
    assert not (1j * cq.number(0)).is_hermitian()
    q3 = (cq.position(0) ** 3).terms
    assert q3[((0, 3, 0),)] == 1 and q3[((0, 2, 1),)] == 3 and q3[((0, 1, 0),)] == 3
    assert cq.cross_kerr(0, 1, 1.0).terms == {((0, 1, 1), (1, 1, 1)): -1}
    # Normal-ordered monomials are exact projected elements: (a^dag a)^2 at d = 4.
    n = np.diag(np.arange(4.0))
    assert np.allclose(cq.to_matrix(cq.number(0) ** 2, {0: 4}), n @ n)


def test_basis_mapping_round_trips():
    from photographiq.autodiff import _basis

    rng = np.random.default_rng(1)
    dims = {0: 7, 1: 6, 2: 6}
    size = len(_basis(3, 5))
    psi = _random(size, rng)
    wide = cq.total_photon_to_per_mode(psi, dims, 5)
    assert np.allclose(cq.per_mode_to_total_photon(wide, dims, 5), psi)
    rho = np.outer(psi, psi.conj())
    assert np.allclose(
        cq.per_mode_to_total_photon(cq.total_photon_to_per_mode(rho, dims, 5), dims, 5), rho
    )
    # Explicit index check: occupation (n0, n1, n2) sits at ravel_multi_index in C order.
    for k, occupation in enumerate(_basis(3, 5)):
        assert wide[np.ravel_multi_index(occupation, (7, 6, 6))] == psi[k]
    with pytest.raises(ValueError, match="cutoff"):
        cq.per_mode_to_total_photon(wide, {0: 4, 1: 6, 2: 6}, 5)


@needs_native
def test_basis_matches_native_order():
    from photographiq.backends.native_fock import _occupations

    for modes, cutoff in [(1, 5), (2, 6), (3, 4)]:
        assert cq._total_photon_basis(modes, cutoff) == tuple(
            map(tuple, _occupations(modes, cutoff).tolist())
        )


@needs_native
@pytest.mark.parametrize("gamma, t, angle", [(0.4, 1.0, 0.0), (1.3, 0.7, 0.8)])
def test_lindblad_loss_matches_native_mixed_fock(gamma, t, angle):
    from photographiq.backends.native_fock import NativeMixedFockBackend, NativeMixedFockState

    rng = np.random.default_rng(11)
    dims = {0: CUTOFF, 1: CUTOFF}
    size = len(cq._total_photon_basis(2, CUTOFF))
    vectors = [_random(size, rng) for _ in range(3)]
    rho = sum(w * np.outer(v, v.conj()) for w, v in zip((0.5, 0.3, 0.2), vectors, strict=True))
    backend = NativeMixedFockBackend(CUTOFF)
    backend.nodes = (0, 1)
    backend.native = NativeMixedFockState(rho.copy(), 2, CUTOFF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # random states fill the cutoff
        backend.rotate(1, angle * t)  # rotation commutes with pure loss
        backend.loss(1, np.exp(-gamma * t))
    result = cq.reference_evolve(
        cq.rotation(1, angle),
        dims,
        [0.0, t / 2, t],
        cq.total_photon_to_per_mode(rho, dims, CUTOFF),
        collapse_operators=[np.sqrt(gamma) * cq.annihilate(1)],
        observables=[cq.number(1)],
        store_intermediate_results=True,
    )
    # Loss and rotation never raise the photon number: the total-photon space is exact.
    final = cq.per_mode_to_total_photon(result.state, dims, CUTOFF)
    assert np.abs(final - backend.native.density_matrix).max() < 1e-10
    assert len(result.intermediate_states) == 3
    n1 = np.trace(cq.to_matrix(cq.number(1), dims) @ result.state).real
    assert result.expectation_values[-1, 0].real == pytest.approx(n1, abs=1e-12)
    assert result.record["engine"] == "numpy-reference"


def test_reference_unitary_evolution_and_validation():
    dims = {0: 10}
    psi = np.zeros(10, complex)
    psi[1] = 1
    result = cq.reference_evolve(cq.number(0), dims, [0, 0.5, 2.0], psi)
    assert np.allclose(result.state[1], np.exp(-2j))
    assert result.intermediate_states is None and result.expectation_values is None
    with pytest.raises(ValueError, match="Hermitian"):
        cq.reference_evolve(1j * cq.number(0), dims, [0, 1], psi)
    with pytest.raises(ValueError, match="increasing"):
        cq.reference_evolve(cq.number(0), dims, [1, 0], psi)
    with pytest.raises(ValueError, match="mode"):
        cq.reference_evolve(cq.number(1), dims, [0, 1], psi)


@pytest.mark.skipif(HAS_CUDAQ, reason="cudaq is installed")
def test_cudaq_missing_raises_import_error():
    with pytest.raises(ImportError, match="Linux"):
        cq.to_cudaq(cq.number(0))
    with pytest.raises(ImportError, match="Linux"):
        cq.evolve(cq.number(0), {0: 4}, [0, 1], np.eye(4)[0])
    assert cq.cudaq_probe()["available"] is False
    assert cq.cudaq_capabilities()["verified"] is False


def _fake_cudaq(version):
    """Minimal stand-in: boson ops build BosonicHamiltonians and record each call."""
    calls = []

    def recorder(name, build):
        def op(*args):
            calls.append((name, *args))
            return build(*args)

        return op

    boson = types.SimpleNamespace(
        create=recorder("create", cq.create),
        annihilate=recorder("annihilate", cq.annihilate),
        identity=recorder("identity", lambda: cq.BosonicHamiltonian.identity()),
        empty=recorder("empty", lambda: cq.BosonicHamiltonian()),
    )
    module = types.ModuleType("cudaq")
    module.__version__ = version
    module.operators = types.SimpleNamespace(boson=boson)
    return module, calls


def test_to_cudaq_translation_wiring_with_fake_module(monkeypatch):
    """Translation WIRING only (a fake cudaq, not CUDA-Q): the calls and their algebra."""
    fake, calls = _fake_cudaq("CUDA-Q Version 0.16.0 (https://github.com/NVIDIA/cuda-quantum)")
    monkeypatch.setitem(sys.modules, "cudaq", fake)
    h = cq.squeezing(0, 0.3) + cq.cross_kerr(0, 1, 0.2) + 0.5
    out = cq.to_cudaq(h)
    assert out.terms == h.terms
    assert ("create", 0) in calls and ("annihilate", 1) in calls and ("identity",) in calls
    assert sum(1 for c in calls if c[0] == "identity") == len(h.terms)
    assert cq.to_cudaq(cq.BosonicHamiltonian()).terms == {}
    assert calls[-1] == ("empty",)


def test_to_cudaq_refuses_old_version(monkeypatch):
    fake, _ = _fake_cudaq("CUDA-Q Version 0.9.1")
    monkeypatch.setitem(sys.modules, "cudaq", fake)
    with pytest.raises(ImportError, match="0.9"):
        cq.to_cudaq(cq.number(0))


@pytest.mark.skipif(not HAS_CUDAQ, reason="requires CUDA-Q (Linux only)")
def test_real_cudaq_evolve_matches_reference():
    dims = {0: 12, 1: 12}
    h = cq.beamsplitter(0, 1, 0.4) + cq.kerr(0, 0.1)
    psi = np.zeros(144, complex)
    psi[np.ravel_multi_index((2, 1), (12, 12))] = 1
    times = np.linspace(0, 1, 21)
    kwargs = {"observables": [cq.number(0)], "store_intermediate_results": False}
    got = cq.evolve(h, dims, times, psi, **kwargs)
    want = cq.reference_evolve(h, dims, times, psi, **kwargs)
    assert got.record["engine"] == "cudaq" and got.record["verified"] is False
    assert _phase_aligned_error(want.state, got.state) < 1e-6
    lossy = cq.evolve(h, dims, times, psi, collapse_operators=[np.sqrt(0.3) * cq.annihilate(1)])
    ref = cq.reference_evolve(
        h, dims, times, psi, collapse_operators=[np.sqrt(0.3) * cq.annihilate(1)]
    )
    assert np.abs(lossy.state - ref.state).max() < 1e-6
