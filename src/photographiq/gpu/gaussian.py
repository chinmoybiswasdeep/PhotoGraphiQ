"""Batched Gaussian kernels on a CUDA device (CuPy) with an explicit NumPy CPU path.

Conventions are those of :class:`photographiq.backends.gaussian.GaussianBackend`: hbar = 2,
interleaved quadratures (q0, p0, q1, p1, ...), statistical covariance with vacuum V = I,
Omega = (+)[[0, 1], [-1, 0]], homodyne record q cos t + p sin t calibrated to the incident
quadrature with variance Var + (1 - eta)/eta + noise. A batch is ``means`` (B, 2n) and
``covariances`` (B, 2n, 2n); modes are integer indices, not labels.

Every kernel takes ``device`` ("gpu" or "cpu", no default guessing beyond "gpu"), ``dtype``
("float64" default, "float32" only when asked) and ``on_device`` (return CuPy arrays instead
of copying back to NumPy). Inputs may be NumPy or CuPy arrays. The same array code runs on
both devices (``xp`` is NumPy or CuPy), so the CPU path is the GPU algorithm, and the result's
``execution`` records which device produced it. With ``device="gpu"`` and no usable CuPy the
call raises; it never falls back to the CPU. Results returned as NumPy are synchronised by
the device-to-host copy; with ``on_device=True`` the work may still be queued on the stream.

Covariance inputs are assumed symmetric and physical; no O(n^3) physicality check is made.
Validity checks that are cheap (symplecticity of the gate matrices, positive homodyne
variance, finite diagnostics) are made and raise ``ValueError``; each costs one device sync.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..gaussian import omega
from .device import Execution, cpu_execution, gpu_execution, require

__all__ = [
    "BatchValues",
    "GaussianBatch",
    "HomodyneConditioning",
    "HomodyneMoments",
    "apply_symplectic",
    "condition_homodyne",
    "fidelity",
    "homodyne_log_density",
    "homodyne_moments",
    "purity",
    "symplectic_eigenvalues",
]

_LOG_2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class GaussianBatch:
    means: Any
    covariances: Any
    execution: Execution


@dataclass(frozen=True)
class HomodyneMoments:
    expected: Any
    variance: Any
    execution: Execution


@dataclass(frozen=True)
class HomodyneConditioning:
    """Conditioned moments (mode removed) and the outcome (log-)densities of the input states."""

    means: Any
    covariances: Any
    log_densities: Any
    densities: Any
    execution: Execution


@dataclass(frozen=True)
class BatchValues:
    values: Any
    execution: Execution


# ----- plumbing --------------------------------------------------------------------------
def _resolve(device: str, dtype, on_device: bool):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
        raise ValueError("dtype must be float64 or float32")
    if device == "cpu":
        if on_device:
            raise ValueError("on_device=True requires device='gpu'")
        return np, dtype
    if device == "gpu":
        return require(), dtype
    raise ValueError("device must be 'gpu' or 'cpu'")


def _batch(xp, means, covariances, dtype, copy: bool = False):
    convert = (
        (lambda a: xp.array(a, dtype=dtype, copy=True))
        if copy
        else (lambda a: xp.asarray(a, dtype=dtype))
    )
    means, covs = convert(means), convert(covariances)
    if means.ndim != 2 or means.shape[1] % 2 or covs.shape != means.shape + means.shape[1:]:
        raise ValueError("Expected means (B, 2n) and covariances (B, 2n, 2n)")
    return means, covs


def _per_state(xp, values, batch: int, dtype, name: str):
    values = xp.asarray(values, dtype=dtype)
    if values.ndim > 1 or values.size not in (1, batch):
        raise ValueError(f"{name} must be a scalar or have shape (B,)")
    return xp.broadcast_to(values.reshape(-1), (batch,))


def _check(xp, condition, message: str) -> None:
    if not bool(xp.all(condition)):  # one device -> host sync on the GPU path
        raise ValueError(message)


def _finish(xp, kernel: str, dtype, on_device: bool, *arrays):
    if xp is np:
        return arrays, cpu_execution(kernel, dtype.name)
    execution = gpu_execution(kernel, arrays[0], dtype.name, on_device)
    if not on_device:
        arrays = tuple(xp.asnumpy(a) for a in arrays)
    return arrays, execution


def _complex(dtype):
    return np.complex128 if dtype == np.float64 else np.complex64


# ----- symplectic updates ----------------------------------------------------------------
def apply_symplectic(
    means,
    covariances,
    modes: Sequence[int],
    matrices,
    *,
    device: str = "gpu",
    dtype="float64",
    on_device: bool = False,
) -> GaussianBatch:
    """d -> S d and V -> S V S^T for the 2k x 2k matrix S on ``modes`` of every state.

    Local O(B n k) update: the touched rows R = S V[idx, :] (V's symmetric part on those
    rows) and block R[:, idx] S^T, written back to rows and columns so the output is exactly
    symmetric. ``modes`` lists distinct mode indices (any order, all modes allowed: then this
    is the dense update). ``matrices`` is one (2k, 2k) matrix or a (B, 2k, 2k) stack.

    Raises:
        ValueError: Shapes, repeated/out-of-range modes, or a matrix failing S Omega S^T =
            Omega with ``numpy.allclose`` default tolerances (the reference backend's check),
            evaluated in float64 before the cast to ``dtype``.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    means, covs = _batch(xp, means, covariances, dtype, copy=True)
    batch, dim = means.shape
    modes = [operator.index(m) for m in modes]  # 1.7 must not become mode 1
    if not modes or len(set(modes)) != len(modes) or not all(0 <= m < dim // 2 for m in modes):
        raise ValueError("modes must be distinct indices of the state's modes")
    k = 2 * len(modes)
    given = xp.asarray(matrices)
    if given.shape not in ((k, k), (batch, k, k)):
        raise ValueError(f"matrices must have shape ({k}, {k}) or ({batch}, {k}, {k})")
    # Checked in float64 on the matrices as given (before any cast to the compute dtype);
    # float32-given matrices get atol 1e-5 since 1e-8 is below their rounding error.
    check = given.astype(np.float64)
    om = xp.asarray(omega(len(modes)))
    _check(
        xp,
        xp.allclose(
            check @ om @ xp.swapaxes(check, -1, -2),
            om,
            atol=1e-5 if given.dtype == np.float32 else 1e-8,
        ),
        "Transformation must be symplectic",
    )
    mats = given.astype(dtype)
    idx = xp.asarray([j for m in modes for j in (2 * m, 2 * m + 1)])
    means[:, idx] = (mats @ means[:, idx, None])[..., 0]
    rows = mats @ (0.5 * (covs[:, idx, :] + xp.swapaxes(covs[:, :, idx], 1, 2)))
    block = rows[:, :, idx] @ xp.swapaxes(mats, -1, -2)
    rows[:, :, idx] = 0.5 * (block + xp.swapaxes(block, 1, 2))
    covs[:, idx, :] = rows
    covs[:, :, idx] = xp.swapaxes(rows, 1, 2)
    (means, covs), execution = _finish(xp, "apply_symplectic", dtype, on_device, means, covs)
    return GaussianBatch(means, covs, execution)


# ----- homodyne --------------------------------------------------------------------------
def _homodyne(xp, means, covs, mode, angles, efficiency, noise, dtype, outcomes=None):
    batch, dim = means.shape
    mode = operator.index(mode)
    if not 0 <= mode < dim // 2:
        raise ValueError("mode index out of range")
    if not 0 < efficiency <= 1 or not np.isfinite(noise) or noise < 0:
        raise ValueError("Require 0 < efficiency <= 1 and finite noise >= 0")
    i = 2 * mode
    angles = _per_state(xp, angles, batch, dtype, "angles")
    c, s = xp.cos(angles), xp.sin(angles)
    expected = c * means[:, i] + s * means[:, i + 1]
    variance = (
        c * c * covs[:, i, i]
        + c * s * (covs[:, i, i + 1] + covs[:, i + 1, i])
        + s * s * covs[:, i + 1, i + 1]
        + ((1 - efficiency) / efficiency + noise)
    )
    flags = [xp.all((variance > 0) & xp.isfinite(variance))]
    if outcomes is not None:
        outcomes = _per_state(xp, outcomes, batch, dtype, "outcomes")
        # GaussianBackend.measure rejects non-finite outcomes; NaN must not propagate silently.
        flags.append(xp.all(xp.isfinite(outcomes)))
    valid = xp.stack(flags).tolist()  # both checks in one device -> host sync
    if not valid[0]:
        raise ValueError("Singular homodyne variance")
    if outcomes is None:
        return i, c, s, expected, variance
    if not valid[1]:
        raise ValueError("Homodyne outcomes must be finite")
    return i, c, s, expected, variance, outcomes


def _log_density(xp, outcomes, expected, variance):
    return -0.5 * (_LOG_2PI + xp.log(variance) + (outcomes - expected) ** 2 / variance)


def homodyne_moments(
    means,
    covariances,
    mode: int,
    angles,
    efficiency: float = 1.0,
    noise: float = 0.0,
    *,
    device: str = "gpu",
    dtype="float64",
    on_device: bool = False,
) -> HomodyneMoments:
    """Per-state mean and variance of the calibrated homodyne record on mode index ``mode``."""
    xp, dtype = _resolve(device, dtype, on_device)
    means, covs = _batch(xp, means, covariances, dtype)
    _, _, _, expected, variance = _homodyne(xp, means, covs, mode, angles, efficiency, noise, dtype)
    (expected, variance), execution = _finish(
        xp, "homodyne_moments", dtype, on_device, expected, variance
    )
    return HomodyneMoments(expected, variance, execution)


def homodyne_log_density(
    means,
    covariances,
    mode: int,
    angles,
    outcomes,
    efficiency: float = 1.0,
    noise: float = 0.0,
    *,
    device: str = "gpu",
    dtype="float64",
    on_device: bool = False,
) -> BatchValues:
    """Per-state log density of the given calibrated homodyne outcomes (no conditioning)."""
    xp, dtype = _resolve(device, dtype, on_device)
    means, covs = _batch(xp, means, covariances, dtype)
    *_, expected, variance, outcomes = _homodyne(
        xp, means, covs, mode, angles, efficiency, noise, dtype, outcomes
    )
    values = _log_density(xp, outcomes, expected, variance)
    (values,), execution = _finish(xp, "homodyne_log_density", dtype, on_device, values)
    return BatchValues(values, execution)


def condition_homodyne(
    means,
    covariances,
    mode: int,
    angles,
    outcomes,
    efficiency: float = 1.0,
    noise: float = 0.0,
    *,
    device: str = "gpu",
    dtype="float64",
    on_device: bool = False,
) -> HomodyneConditioning:
    """Destructive homodyne on mode index ``mode`` of every state, conditioned on GIVEN outcomes.

    Schur complement of :meth:`GaussianBackend.measure`: with c = V[rest, mode] (cos, sin)
    and sigma^2 the record variance, d' = d_rest + c (x - <x>)/sigma^2 and
    V' = V_rest - c c^T / sigma^2 (then symmetrised). Densities are those of each outcome
    given its pre-measurement state; a batch of chosen outcomes is a set of postselected
    branches, not a sampled trajectory. Unlike ``GaussianBackend.measure``, an outcome whose
    density underflows to 0 does not raise: ``densities`` is 0 for it, ``log_densities`` stays
    finite and exact, and its conditioned moments are still returned.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    means, covs = _batch(xp, means, covariances, dtype)
    batch, dim = means.shape
    i, c, s, expected, variance, outcomes = _homodyne(
        xp, means, covs, mode, angles, efficiency, noise, dtype, outcomes
    )
    rest = xp.asarray(np.r_[0:i, i + 2 : dim])
    cross = c[:, None] * covs[:, rest, i] + s[:, None] * covs[:, rest, i + 1]
    new_means = means[:, rest] + cross * ((outcomes - expected) / variance)[:, None]
    new_covs = covs[:, rest[:, None], rest[None, :]] - (
        cross[:, :, None] * cross[:, None, :] / variance[:, None, None]
    )
    new_covs = 0.5 * (new_covs + xp.swapaxes(new_covs, 1, 2))
    log_densities = _log_density(xp, outcomes, expected, variance)
    (new_means, new_covs, log_densities, densities), execution = _finish(
        xp,
        "condition_homodyne",
        dtype,
        on_device,
        new_means,
        new_covs,
        log_densities,
        xp.exp(log_densities),
    )
    return HomodyneConditioning(new_means, new_covs, log_densities, densities, execution)


# ----- diagnostics -----------------------------------------------------------------------
def purity(
    covariances, *, device: str = "gpu", dtype="float64", on_device: bool = False
) -> BatchValues:
    """Tr(rho^2) = det(V)^(-1/2) per state (``GaussianState.purity``), via batched slogdet."""
    xp, dtype = _resolve(device, dtype, on_device)
    covs = xp.asarray(covariances, dtype=dtype)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2] or covs.shape[1] % 2:
        raise ValueError("Expected covariances (B, 2n, 2n)")
    values = xp.exp(-0.5 * xp.linalg.slogdet(covs)[1])
    (values,), execution = _finish(xp, "purity", dtype, on_device, values)
    return BatchValues(values, execution)


def _symplectic_eigenvalues(xp, covs, dtype):
    # Same algorithm as GaussianState: V = L L^T, eigenvalues of Hermitian i L^T Omega L.
    n = covs.shape[1] // 2
    lower = xp.linalg.cholesky(covs)
    om = xp.asarray(omega(n), dtype=dtype)
    herm = 1j * (xp.swapaxes(lower, 1, 2) @ om @ lower).astype(_complex(dtype))
    values = xp.sort(xp.abs(xp.linalg.eigvalsh(herm)), axis=1)[:, ::2]
    # CuPy reports a failed Cholesky as NaN instead of raising.
    _check(xp, xp.isfinite(values), "Covariance is not positive definite")
    return values


def symplectic_eigenvalues(
    covariances, *, device: str = "gpu", dtype="float64", on_device: bool = False
) -> BatchValues:
    """Sorted symplectic eigenvalues (B, n); raises for non positive-definite covariances.

    Unlike ``GaussianState.symplectic_eigenvalues`` there is no general-eigensolver fallback
    for unphysical (indefinite) input: such input raises ``ValueError``.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    covs = xp.asarray(covariances, dtype=dtype)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2] or covs.shape[1] % 2:
        raise ValueError("Expected covariances (B, 2n, 2n)")
    values = _symplectic_eigenvalues(xp, covs, dtype)
    (values,), execution = _finish(xp, "symplectic_eigenvalues", dtype, on_device, values)
    return BatchValues(values, execution)


def fidelity(
    means1,
    covariances1,
    means2,
    covariances2,
    pure_tolerance: float = 1e-9,
    *,
    device: str = "gpu",
    dtype="float64",
    on_device: bool = False,
) -> BatchValues:
    """Squared Uhlmann fidelity of state pairs, with the rule of ``GaussianState.fidelity``.

    Pairs in which either state is pure (all symplectic eigenvalues within ``pure_tolerance``
    of 1) use the exact F = Tr(rho sigma); other pairs use the Banchi-Braunstein-Pirandola
    formula exactly as ``photographiq.states._bbp_fidelity`` (same accuracy caveats for
    nearly pure, strongly squeezed states). Both batches must use the same mode order.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    m1, v1 = _batch(xp, means1, covariances1, dtype)
    m2, v2 = _batch(xp, means2, covariances2, dtype)
    if m1.shape != m2.shape:
        raise ValueError("Both batches must have the same shape")
    n = m1.shape[1] // 2
    pure = xp.zeros(len(m1), dtype=bool)
    for cov in (v1, v2):
        pure |= xp.all(xp.abs(_symplectic_eigenvalues(xp, cov, dtype) - 1) < pure_tolerance, 1)
    total = v1 + v2
    delta = m1 - m2
    exponent = (delta * xp.linalg.solve(total, delta[..., None])[..., 0]).sum(axis=1)
    gaussian = xp.exp(-0.5 * exponent)
    overlap = 2.0**n * gaussian / xp.sqrt(xp.linalg.det(total))
    # BBP in units with vacuum covariance I/2 (see states._bbp_fidelity).
    om = xp.asarray(omega(n), dtype=dtype)
    va, vb = v1 / 2, v2 / 2
    aux = om.T @ xp.linalg.solve(va + vb, om / 4 + vb @ om @ va)
    mu = xp.linalg.eigvals(aux @ om).astype(_complex(dtype))
    product = xp.prod(1 + xp.sqrt(1 + 1 / (4 * mu**2)), axis=1).real
    sign_aux, log_aux = xp.linalg.slogdet(2 * aux)
    sign_total, log_total = xp.linalg.slogdet(va + vb)
    ratio = product * sign_aux * sign_total * xp.exp(log_aux - log_total)
    _check(xp, pure | (ratio >= 0), "Fidelity determinant ratio is not positive")
    values = xp.where(pure, overlap, xp.sqrt(xp.where(pure, 1.0, ratio)) * gaussian)
    _check(xp, xp.isfinite(values), "Non-finite fidelity")
    (values,), execution = _finish(xp, "fidelity", dtype, on_device, values)
    return BatchValues(values, execution)
