"""photographiq.gpu Gaussian kernels against the NumPy reference backend, CPU path and native.

Every test runs for device="cpu" (always) and device="gpu" (skipped without CuPy or a CUDA
device). The reference is GaussianBackend / GaussianState evaluated state by state. Both
sides evaluate the same float64 formulas in a different summation order (local versus dense
updates, cuBLAS/cuSOLVER versus LAPACK). Largest deviations observed over seeds 0-19 (RTX 4050,
CuPy 14.2), relative to the largest reference entry: symplectic update 7e-16, homodyne
means/covariances 3e-15, densities 1e-14, purity 7e-14, symplectic eigenvalues 4e-14, and
fidelity 8e-11 on the GPU (the BBP formula goes through a general eigensolver and is
ill-conditioned near purity, see GaussianState.fidelity). Tolerances: RTOL = 1e-11 with
ATOL = 1e-11 * max(|reference|, 1), and 1e-8 for fidelity, i.e. a margin of >= 100 while any
formula error is O(1).
"""

import numpy as np
import pytest
from scipy.stats import norm

import photographiq as pg
from photographiq import gaussian as gm
from photographiq import gpu
from photographiq.backends.gaussian import GaussianBackend

RTOL = 1e-11
FIDELITY_RTOL = 1e-8
DEVICES = [
    "cpu",
    pytest.param(
        "gpu", marks=pytest.mark.skipif(not gpu.available(), reason="CuPy/CUDA device missing")
    ),
]
BATCHES = (1, 7, 33)


def close(actual, reference, rtol=RTOL):
    reference = np.asarray(reference)
    scale = max(float(np.abs(reference).max(initial=0.0)), 1.0)
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=rtol * scale)


def rclose(actual, reference, rtol=RTOL):
    """Purely relative comparison for positive per-state scalars of any magnitude."""
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=0)


def random_symplectic(rng, n, depth=None):
    """Product of random squeezers, rotations, beam splitters and CZ gates on n modes."""
    s = np.eye(2 * n)
    for _ in range(depth or 3 * n):
        kind = rng.integers(4) if n > 1 else rng.integers(2)
        if kind == 0:
            j, g = [rng.integers(n)], gm.squeezing(rng.uniform(-0.8, 0.8))
        elif kind == 1:
            j, g = [rng.integers(n)], gm.rotation(rng.uniform(0, 2 * np.pi))
        else:
            j = list(rng.choice(n, 2, replace=False))
            g = gm.beamsplitter(rng.uniform(0, np.pi)) if kind == 2 else gm.cz(rng.normal())
        idx = [q for m in j for q in (2 * m, 2 * m + 1)]
        s = gm.embed(g, idx, 2 * n) @ s
    return s


def random_states(rng, batch, n, pure=False):
    """Physical states V = S diag(nu) S^T with nu >= 1 (nu = 1 if pure) and random means."""
    means = rng.normal(scale=2.0, size=(batch, 2 * n))
    covs = np.empty((batch, 2 * n, 2 * n))
    for b in range(batch):
        s = random_symplectic(rng, n)
        nu = np.ones(n) if pure else 1 + rng.exponential(0.5, n)
        v = s @ np.diag(np.repeat(nu, 2)) @ s.T
        covs[b] = (v + v.T) / 2
    return means, covs


def state(mean, cov):
    return pg.GaussianState(mean, cov, tuple(range(len(mean) // 2)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("modes", [[1], [0, 3], [4, 2, 0], [0, 1, 2, 3, 4], [3, 1, 4, 0, 2]])
def test_apply_symplectic_matches_reference_backend(device, batch, modes):
    rng = np.random.default_rng(batch * 31 + len(modes))
    n = 5
    means, covs = random_states(rng, batch, n)
    mats = np.stack([random_symplectic(rng, len(modes)) for _ in range(batch)])
    result = gpu.apply_symplectic(means, covs, modes, mats, device=device)
    for b in range(batch):
        ref = GaussianBackend()
        ref.set_state(state(means[b], covs[b]))
        ref.transform(tuple(modes), mats[b])
        close(result.means[b], ref.state.mean)
        close(result.covariances[b], ref.state.covariance)
    np.testing.assert_array_equal(result.covariances, np.swapaxes(result.covariances, 1, 2))
    # one shared matrix broadcast over the batch equals the stacked form
    shared = gpu.apply_symplectic(means, covs, modes, mats[0], device=device)
    stacked = gpu.apply_symplectic(means, covs, modes, np.repeat(mats[:1], batch, 0), device=device)
    close(shared.covariances, stacked.covariances)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("efficiency,noise", [(1.0, 0.0), (0.9, 0.0), (0.5, 0.2)])
def test_condition_homodyne_matches_reference_backend(device, batch, efficiency, noise):
    rng = np.random.default_rng(batch * 7 + int(10 * efficiency))
    n, mode = 4, 2
    means, covs = random_states(rng, batch, n)
    angles = rng.uniform(-np.pi, np.pi, batch)
    outcomes = rng.normal(scale=2.0, size=batch)
    result = gpu.condition_homodyne(
        means, covs, mode, angles, outcomes, efficiency, noise, device=device
    )
    moments = gpu.homodyne_moments(means, covs, mode, angles, efficiency, noise, device=device)
    logpdf = gpu.homodyne_log_density(
        means, covs, mode, angles, outcomes, efficiency, noise, device=device
    )
    for b in range(batch):
        ref = GaussianBackend()
        ref.set_state(state(means[b], covs[b]))
        st = ref.get_state()
        mu, var = st.quadrature(mode, angles[b])
        var += (1 - efficiency) / efficiency + noise
        ref.measure(mode, pg.Homodyne(angles[b], efficiency, noise), angles[b], outcome=outcomes[b])
        close(result.means[b], ref.state.mean)
        close(result.covariances[b], ref.state.covariance)
        rclose(result.densities[b], ref.last_measurement["value"])
        close(moments.expected[b], mu)
        close(moments.variance[b], var)
        close(logpdf.values[b], norm.logpdf(outcomes[b], mu, np.sqrt(var)))
    close(result.log_densities, logpdf.values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("efficiency", [1.0, 0.7])
def test_conditioned_states_are_physical(device, efficiency):
    rng = np.random.default_rng(5)
    n = 5
    means, covs = random_states(rng, 20, n)
    result = gpu.condition_homodyne(
        means, covs, 0, rng.uniform(0, np.pi, 20), rng.normal(size=20), efficiency, device=device
    )
    om = gm.omega(n - 1)
    for cov in result.covariances:
        np.testing.assert_array_equal(cov, cov.T)
        assert np.linalg.eigvalsh(cov + 1j * om).min() > -1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_diagnostics_match_gaussian_state(device):
    rng = np.random.default_rng(11)
    means, covs = random_states(rng, 12, 3)
    means2, covs2 = random_states(rng, 12, 3)
    pm, pc = random_states(rng, 12, 3, pure=True)
    states1 = [state(m, c) for m, c in zip(means, covs)]
    states2 = [state(m, c) for m, c in zip(means2, covs2)]
    pures = [state(m, c) for m, c in zip(pm, pc)]
    rclose(gpu.purity(covs, device=device).values, [s.purity() for s in states1])
    close(
        gpu.symplectic_eigenvalues(covs, device=device).values,
        [s.symplectic_eigenvalues() for s in states1],
    )
    mixed = gpu.fidelity(means, covs, means2, covs2, device=device).values
    rclose(mixed, [a.fidelity(b) for a, b in zip(states1, states2)], rtol=FIDELITY_RTOL)
    # pure/mixed pairs take the overlap branch, mixed/mixed the BBP branch, in one batch
    mm, mc = np.r_[pm[:6], means[6:]], np.r_[pc[:6], covs[6:]]
    expected = [a.fidelity(b) for a, b in zip(pures[:6] + states1[6:], states2)]
    rclose(gpu.fidelity(mm, mc, means2, covs2, device=device).values, expected, rtol=FIDELITY_RTOL)
    same = gpu.fidelity(means, covs, means, covs, device=device).values
    rclose(same, np.ones(12), rtol=FIDELITY_RTOL)


@pytest.mark.skipif(not gpu.available(), reason="CuPy/CUDA device missing")
def test_gpu_and_cpu_paths_agree():
    rng = np.random.default_rng(2)
    means, covs = random_states(rng, 64, 6)
    mats = np.stack([random_symplectic(rng, 2) for _ in range(64)])
    angles, outcomes = rng.uniform(0, np.pi, 64), rng.normal(size=64)
    matrices = [
        lambda d: gpu.apply_symplectic(means, covs, [1, 4], mats, device=d).covariances,
        lambda d: (
            gpu.condition_homodyne(
                means, covs, 3, angles, outcomes, 0.8, 0.05, device=d
            ).covariances
        ),
    ]
    for call in matrices:
        close(call("gpu"), call("cpu"), rtol=1e-12)
    # per-state scalars span many orders of magnitude: compare purely relatively (atol=0)
    scalars = [
        (
            lambda d: gpu.condition_homodyne(means, covs, 3, angles, outcomes, device=d).densities,
            RTOL,
        ),
        (lambda d: gpu.purity(covs, device=d).values, RTOL),
        (
            lambda d: gpu.fidelity(means, covs, means[::-1], covs[::-1], device=d).values,
            FIDELITY_RTOL,
        ),
    ]
    for call, rtol in scalars:
        rclose(call("gpu"), call("cpu"), rtol)


@pytest.mark.parametrize("device", DEVICES)
def test_matches_native_batch_kernels(device):
    ng = pytest.importorskip(
        "photographiq.backends.native_gaussian", reason="native 'gaussian' component not built"
    )
    rng = np.random.default_rng(9)
    means, covs = random_states(rng, 16, 5)
    s = gm.beamsplitter(0.37)
    native_means, native_covs = ng.batch_apply_symplectic(means, covs, [1, 3], s)
    ours = gpu.apply_symplectic(means, covs, [1, 3], s, device=device)
    close(ours.means, native_means)
    close(ours.covariances, native_covs)
    nm, nc, nd = ng.batch_condition_homodyne(means, covs, 2, 0.4, 0.1, 0.9, 0.0)
    ours = gpu.condition_homodyne(means, covs, 2, 0.4, 0.1, 0.9, 0.0, device=device)
    close(ours.means, nm)
    close(ours.covariances, nc)
    close(ours.densities, nd)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_is_explicit_and_close(device):
    rng = np.random.default_rng(4)
    means, covs = random_states(rng, 8, 3)
    r64 = gpu.condition_homodyne(means, covs, 1, 0.3, 0.2, 0.9, device=device)
    r32 = gpu.condition_homodyne(means, covs, 1, 0.3, 0.2, 0.9, device=device, dtype="float32")
    assert r64.covariances.dtype == np.float64 and r64.execution.dtype == "float64"
    assert r32.covariances.dtype == np.float32 and r32.execution.dtype == "float32"
    close(r32.covariances, r64.covariances, rtol=1e-4)
    s = gm.beamsplitter(0.3)
    s64 = gpu.apply_symplectic(means, covs, [0, 2], s, device=device)
    s32 = gpu.apply_symplectic(means, covs, [0, 2], s, device=device, dtype="float32")
    assert s32.covariances.dtype == np.float32
    close(s32.covariances, s64.covariances, rtol=1e-5)
    with pytest.raises(ValueError, match="float64 or float32"):
        gpu.purity(covs, device=device, dtype="float16")


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_input_raises(device):
    rng = np.random.default_rng(0)
    means, covs = random_states(rng, 3, 2)
    with pytest.raises(ValueError, match="symplectic"):
        gpu.apply_symplectic(means, covs, [0], np.diag([2.0, 1.0]), device=device)
    with pytest.raises(ValueError, match="distinct"):
        gpu.apply_symplectic(means, covs, [0, 0], np.eye(4), device=device)
    with pytest.raises(ValueError, match="efficiency"):
        gpu.condition_homodyne(means, covs, 0, 0.0, 0.0, 0.0, device=device)
    with pytest.raises(ValueError, match="Singular"):
        bad = covs.copy()
        bad[:, 0, 0] = -5.0
        gpu.homodyne_moments(means, bad, 0, 0.0, device=device)
    with pytest.raises(ValueError, match="positive definite"):
        gpu.symplectic_eigenvalues(-covs, device=device)
    with pytest.raises(ValueError, match="Expected means"):
        gpu.condition_homodyne(means[:, :3], covs, 0, 0.0, 0.0, device=device)
    # non-finite outcomes raise like GaussianBackend.measure instead of returning NaN states
    for bad in (np.nan, np.inf, [0.1, np.nan, 0.2]):
        with pytest.raises(ValueError, match="outcomes must be finite"):
            gpu.condition_homodyne(means, covs, 0, 0.0, bad, device=device)
        with pytest.raises(ValueError, match="outcomes must be finite"):
            gpu.homodyne_log_density(means, covs, 0, 0.0, bad, device=device)
    # non-integer mode indices are rejected, not truncated (1.7 -> mode 1)
    with pytest.raises(TypeError):
        gpu.homodyne_moments(means, covs, 1.7, 0.0, device=device)
    with pytest.raises(TypeError):
        gpu.apply_symplectic(means, covs, [0.9], gm.rotation(0.3), device=device)
