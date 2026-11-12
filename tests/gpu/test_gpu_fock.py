"""photographiq.gpu.fock local gates against the native Fock engine, CPU path and GPU path.

Every numerical test runs for device="cpu" (always) and device="gpu" (skipped without CuPy or
a CUDA device); the module is skipped if the native ``fock`` component is not compiled (it
supplies the basis). References are the native kernels ``apply_pure`` / ``apply_mixed`` /
``phase_pure`` / ``phase_mixed`` state by state, and ``NativeFockBackend`` /
``NativeMixedFockBackend`` for whole gate sequences. Both sides evaluate the same products
in a different summation order (cuSPARSE / scipy CSR vs Eigen per fibre). Largest deviations
observed over seeds 0-9, shapes (modes, cutoff) up to (2, 20), (3, 12), (5, 8), all gates
below, relative to the largest reference entry: complex128 1.3e-15 (norm / trace vs native
3.3e-16), tolerance RTOL = 1e-12; complex64 4.4e-7, bound C64_RTOL = 1e-5.
"""

import warnings

import numpy as np
import pytest

from photographiq import gpu, native
from photographiq.gpu import device as gpu_device
from photographiq.gpu import fock as gf

pytestmark = pytest.mark.skipif(
    "fock" not in native.components(), reason="native fock component (the basis) not compiled"
)
RTOL = 1e-12
C64_RTOL = 1e-5
needs_gpu = pytest.mark.skipif(not gpu.available(), reason="CuPy/CUDA device missing")
DEVICES = ["cpu", pytest.param("gpu", marks=needs_gpu)]
SHAPES = [(2, 8), (3, 7), (4, 6)]  # (modes, cutoff)


def engine():
    return native.require("fock")


def close(actual, reference, rtol=RTOL):
    reference = np.asarray(reference)
    scale = float(np.abs(reference).max())
    np.testing.assert_allclose(actual, reference, rtol=0, atol=rtol * scale)


def gates(cutoff, rng):
    """name -> (local modes, op) or (local modes, phases): the native engine's own matrices."""
    f = engine()
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = rng.uniform(0.2, 1.0)
    theta = rng.uniform(0, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    bs = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
    r = rng.uniform(-0.4, 0.4)
    n = np.arange(cutoff)
    pair = f.basis(2, cutoff)
    return {
        "displace": ((1,), f.gaussian_unitary(np.eye(2), rng.normal(scale=0.4, size=2), cutoff)),
        "squeeze": (
            (0,),
            f.gaussian_unitary(np.diag([np.exp(-r), np.exp(r)]), np.zeros(2), cutoff),
        ),
        "beamsplitter": ((1, 0), f.gaussian_unitary(bs, np.zeros(4), cutoff)),
        "cz": ((0, 1), f.gaussian_unitary(cz, np.zeros(4), cutoff)),
        "cubic_phase": ((1,), f.cubic_phase_truncated_generator(rng.uniform(0.05, 0.3), cutoff)),
        "cubic_projected": ((0,), f.cubic_phase_projected(0.1, cutoff, 1e-12)[0]),
        "kerr": ((1,), ("phase", rng.uniform(0, 1) * n**2)),
        "cross_kerr": ((0, 1), ("phase", rng.uniform(0, 1) * pair[:, 0] * pair[:, 1])),
    }


def random_pure(rng, batch, dim):
    x = rng.normal(size=(batch, dim)) + 1j * rng.normal(size=(batch, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def random_mixed(rng, batch, dim, rank=3):
    a = rng.normal(size=(batch, dim, rank)) + 1j * rng.normal(size=(batch, dim, rank))
    rho = a @ np.conj(np.swapaxes(a, 1, 2))
    return rho / np.trace(rho, axis1=1, axis2=2)[:, None, None]


def ours(states, modes, cutoff, local, gate, **kw):
    if isinstance(gate, tuple):
        return gf.apply_phase(states, modes, cutoff, local, gate[1], **kw)
    return gf.apply_operator(states, modes, cutoff, local, gate, **kw)


def reference(states, modes, cutoff, local, gate):
    f, out = engine(), np.array(states, copy=True)
    kind = "mixed" if out.ndim == 3 else "pure"
    for state in out:
        if isinstance(gate, tuple):
            getattr(f, f"phase_{kind}")(state, modes, cutoff, list(local), gate[1])
        else:
            getattr(f, f"apply_{kind}")(state, modes, cutoff, list(local), gate)
    return out


def mode_sets(modes, local):
    """The gate's local modes, plus a reversed / shifted placement on larger states."""
    yield local
    if modes > 2:
        yield tuple(modes - 1 - m for m in local)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("modes,cutoff", SHAPES)
@pytest.mark.parametrize("kind", ["pure", "mixed"])
def test_gates_match_native_engine(device, modes, cutoff, kind):
    rng = np.random.default_rng(modes * 100 + cutoff)
    dim = engine().dimension(modes, cutoff)
    states = random_pure(rng, 5, dim) if kind == "pure" else random_mixed(rng, 3, dim)
    for name, (local, gate) in gates(cutoff, rng).items():
        for where in mode_sets(modes, local):
            result = ours(states, modes, cutoff, where, gate, device=device)
            assert result.execution.device.startswith("cpu" if device == "cpu" else "cuda:")
            assert result.execution.dtype == "complex128"
            close(result.states, reference(states, modes, cutoff, where, gate))


@pytest.mark.parametrize("modes,cutoff", SHAPES)
def test_arbitrary_operators_and_non_hermitian_matrices_match_native(modes, cutoff):
    """Index maps, not just unitarity: random complex U and random (non-Hermitian) matrices."""
    rng = np.random.default_rng(7)
    dim = engine().dimension(modes, cutoff)
    for local in [(0,), (modes - 1, 0), tuple(range(modes))]:
        size = engine().dimension(len(local), cutoff)
        u = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        x = rng.normal(size=(4, dim)) + 1j * rng.normal(size=(4, dim))
        a = rng.normal(size=(2, dim, dim)) + 1j * rng.normal(size=(2, dim, dim))
        for states in (x, a):
            expected = reference(states, modes, cutoff, local, u)
            close(gf.apply_operator(states, modes, cutoff, local, u, device="cpu").states, expected)
            if gpu.available():
                close(gf.apply_operator(states, modes, cutoff, local, u).states, expected)


@needs_gpu
@pytest.mark.parametrize("modes,cutoff", SHAPES)
def test_gpu_matches_cpu_path(modes, cutoff):
    rng = np.random.default_rng(3)
    dim = engine().dimension(modes, cutoff)
    for states in (random_pure(rng, 4, dim), random_mixed(rng, 2, dim)):
        for local, gate in gates(cutoff, rng).values():
            close(
                ours(states, modes, cutoff, local, gate).states,
                ours(states, modes, cutoff, local, gate, device="cpu").states,
            )


@pytest.mark.parametrize("device", DEVICES)
def test_norm_and_trace(device):
    """Photon-number-conserving gates keep norm / trace / hermiticity to rounding; others lose
    exactly the native engine's leakage."""
    modes, cutoff = 3, 7
    rng = np.random.default_rng(11)
    dim = engine().dimension(modes, cutoff)
    pure, mixed = random_pure(rng, 6, dim), random_mixed(rng, 3, dim)
    for name, (local, gate) in gates(cutoff, rng).items():
        psi = ours(pure, modes, cutoff, local, gate, device=device).states
        rho = ours(mixed, modes, cutoff, local, gate, device=device).states
        norms = np.linalg.norm(psi, axis=1) ** 2
        traces = np.trace(rho, axis1=1, axis2=2)
        np.testing.assert_allclose(rho, np.conj(np.swapaxes(rho, 1, 2)), rtol=0, atol=1e-14)
        if name in ("beamsplitter", "kerr", "cross_kerr"):
            np.testing.assert_allclose(norms, 1, rtol=0, atol=1e-13)
            np.testing.assert_allclose(traces, 1, rtol=0, atol=1e-13)
        ref_psi = reference(pure, modes, cutoff, local, gate)
        ref_rho = reference(mixed, modes, cutoff, local, gate)
        np.testing.assert_allclose(norms, np.linalg.norm(ref_psi, axis=1) ** 2, rtol=1e-12)
        np.testing.assert_allclose(traces, np.trace(ref_rho, axis1=1, axis2=2), rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["pure", "mixed"])
def test_gate_sequence_matches_native_backend(device, kind):
    """B copies through the backend's gate sequence (renormalised after each gate as the
    backend does) equal the backend's state."""
    from photographiq.backends.native_fock import NativeFockBackend, NativeMixedFockBackend

    modes, cutoff = 3, 8
    cls = NativeFockBackend if kind == "pure" else NativeMixedFockBackend
    backend = cls(cutoff=cutoff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # retained norm 1 - 1e-8 at cutoff 8
        for m in range(modes):
            backend.prepare(m, 0.1)
            backend.displace(m, 0.15, -0.05)
    states = np.repeat(backend.native._data[None], 4, axis=0)
    f = engine()
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = 0.3
    c, s = np.cos(0.4), np.sin(0.4)
    sequence = [
        (
            lambda b: b.squeeze(0, 0.2),
            (0,),
            f.gaussian_unitary(np.diag([np.exp(-0.2), np.exp(0.2)]), np.zeros(2), cutoff),
        ),
        (
            lambda b: b.beamsplitter(0, 2, 0.4),
            (0, 2),
            f.gaussian_unitary(np.kron([[c, -s], [s, c]], np.eye(2)), np.zeros(4), cutoff),
        ),
        (lambda b: b.entangle(1, 2, 0.3), (1, 2), f.gaussian_unitary(cz, np.zeros(4), cutoff)),
        (lambda b: b.kerr(1, 0.2), (1,), ("phase", 0.2 * np.arange(cutoff) ** 2)),
        (lambda b: b.cubic_phase(2, 0.1), (2,), f.cubic_phase_truncated_generator(0.1, cutoff)),
        (
            lambda b: b.displace(1, 0.1, 0.2),
            (1,),
            f.gaussian_unitary(np.eye(2), np.array([0.1, 0.2]), cutoff),
        ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for step, local, gate in sequence:
            step(backend)
            states = ours(states, modes, cutoff, local, gate, device=device).states
            if kind == "pure":
                states = states / np.linalg.norm(states, axis=1, keepdims=True)
            else:
                states = states / np.trace(states, axis1=1, axis2=2)[:, None, None]
    for state in states:
        close(state, backend.native._data)


@pytest.mark.parametrize("device", DEVICES)
def test_complex64_is_explicit_and_bounded(device):
    modes, cutoff = 3, 10
    rng = np.random.default_rng(5)
    dim = engine().dimension(modes, cutoff)
    for states in (random_pure(rng, 4, dim), random_mixed(rng, 2, dim)):
        for local, gate in gates(cutoff, rng).values():
            low = ours(states, modes, cutoff, local, gate, device=device, dtype="complex64")
            assert low.states.dtype == np.complex64 and low.execution.dtype == "complex64"
            close(low.states, reference(states, modes, cutoff, local, gate), rtol=C64_RTOL)


def test_basis_and_argument_errors():
    rng = np.random.default_rng(0)
    modes, cutoff = 2, 6
    dim, size = engine().dimension(modes, cutoff), engine().dimension(1, cutoff)
    psi, op = random_pure(rng, 2, dim), np.eye(size)
    with pytest.raises(ValueError, match="truncated basis of 2 modes at cutoff 7"):
        gf.apply_operator(psi, modes, cutoff + 1, [0], np.eye(size + 1), device="cpu")
    with pytest.raises(ValueError, match="truncated basis of 3 modes"):
        gf.apply_operator(psi, 3, cutoff, [0], op, device="cpu")
    with pytest.raises(ValueError, match="truncated basis"):
        gf.apply_operator(psi[0], modes, cutoff, [0], op, device="cpu")  # no batch axis
    with pytest.raises(ValueError, match="truncated basis"):
        gf.apply_operator(np.zeros((1, dim, dim + 1)), modes, cutoff, [0], op, device="cpu")
    with pytest.raises(ValueError, match=r"op must be \(6, 6\)"):
        gf.apply_operator(psi, modes, cutoff, [0], np.eye(size + 1), device="cpu")
    with pytest.raises(ValueError, match=r"op must be \(21, 21\)"):
        gf.apply_operator(psi, modes, cutoff, [0, 1], op, device="cpu")
    for local in ([0, 0], [2], [-1], []):
        with pytest.raises(ValueError, match="distinct indices"):
            gf.apply_operator(psi, modes, cutoff, local, op, device="cpu")
    with pytest.raises(TypeError):
        gf.apply_operator(psi, modes, cutoff, [0.0], op, device="cpu")
    with pytest.raises(ValueError, match="phases must be 6"):
        gf.apply_phase(psi, modes, cutoff, [0], np.zeros(5), device="cpu")
    with pytest.raises(ValueError, match="complex128 or complex64"):
        gf.apply_operator(psi, modes, cutoff, [0], op, device="cpu", dtype="float64")
    with pytest.raises(ValueError, match="'gpu' or 'cpu'"):
        gf.apply_operator(psi, modes, cutoff, [0], op, device="auto")
    with pytest.raises(ValueError, match="requires device='gpu'"):
        gf.apply_phase(psi, modes, cutoff, [0], np.zeros(6), device="cpu", on_device=True)


def test_missing_cupy_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(gpu_device, "cupy", gpu_device._CuPy("photographiq_no_such_cupy", "gpu"))
    psi = random_pure(np.random.default_rng(0), 2, engine().dimension(2, 5))
    with pytest.raises(ImportError, match="requires CuPy"):
        gf.apply_operator(psi, 2, 5, [0], np.eye(5))
    with pytest.raises(ImportError, match="requires CuPy"):
        gf.apply_phase(psi, 2, 5, [0], np.zeros(5))
    execution = gf.apply_operator(psi, 2, 5, [0], np.eye(5), device="cpu").execution
    assert (execution.kernel, execution.device, execution.backend) == (
        "apply_operator",
        "cpu",
        "numpy",
    )


@needs_gpu
def test_gpu_results_report_and_live_on_the_gpu():
    import cupy as cp

    current = cp.cuda.Device().id
    rng = np.random.default_rng(1)
    dim = engine().dimension(3, 6)
    for states in (random_pure(rng, 3, dim), random_mixed(rng, 2, dim)):
        for call in (
            lambda **kw: gf.apply_operator(states, 3, 6, [2], np.eye(6), **kw),
            lambda **kw: gf.apply_phase(cp.asarray(states), 3, 6, [1], np.ones(6), **kw),
        ):
            off, on = call(), call(on_device=True)
            assert isinstance(off.states, np.ndarray) and not off.execution.returned_on_device
            assert isinstance(on.states, cp.ndarray) and on.execution.returned_on_device
            assert on.execution.device == f"cuda:{current}" and on.execution.backend == "cupy"
            attributes = cp.cuda.runtime.pointerGetAttributes(on.states.data.ptr)
            assert attributes.type == cp.cuda.runtime.memoryTypeDevice
            assert attributes.device == current
            np.testing.assert_array_equal(on.states.get(), off.states)
    with pytest.raises(TypeError):  # the CPU path refuses device arrays instead of copying
        gf.apply_phase(cp.asarray(states), 3, 6, [1], np.ones(6), device="cpu")
