"""Local gates on batches of Fock states in the total-photon truncated basis (CuPy / NumPy).

Basis and conventions are those of the native Fock engine
(:mod:`photographiq.backends.native_fock`): occupations with TOTAL photon number < cutoff,
in its order (sectors of total photon number, first mode descending inside a sector). The
basis and ranks come from the native engine itself (``basis``, ``rank``), never from a
re-implementation. A batch is ``(B, D)`` for pure states or ``(B, D, D)`` for density
matrices, D = C(modes + cutoff - 1, modes); a single state is a batch of one.

A k-mode local operator U (``(S, S)``, S = C(k + cutoff - 1, k), e.g. from the native
``gaussian_unitary`` / ``cubic_phase_*``) acts on every fibre of fixed photon numbers of the
other modes as its top-left block, exactly as the native ``apply_pure`` / ``apply_mixed``
(P U P on the truncated space, no renormalisation). That block structure is one sparse
D x D matrix G (cached index maps; only its values are gathered from U per call), applied
as one sparse-dense product: psi -> G psi for all B states at once, rho -> G rho G^dag as
two products. The CPU path runs the same algorithm with scipy.sparse, the GPU path with
cuSPARSE (cupyx.scipy.sparse).

Every kernel takes ``device`` ("gpu" default or "cpu"), ``dtype`` ("complex128" default or
"complex64") and ``on_device`` (return CuPy arrays). With ``device="gpu"`` and no usable
CuPy the call raises; it never falls back to the CPU. The result's ``execution`` records
the device that produced it.
"""

from __future__ import annotations

import copy
import functools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import native
from .device import Execution, require
from .gaussian import _finish

__all__ = ["FockBatch", "apply_operator", "apply_phase"]


@dataclass(frozen=True)
class FockBatch:
    states: Any
    execution: Execution


def _resolve(device: str, dtype, on_device: bool):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.complex128), np.dtype(np.complex64)):
        raise ValueError("dtype must be complex128 or complex64")
    if device == "cpu":
        if on_device:
            raise ValueError("on_device=True requires device='gpu'")
        return np, dtype
    if device == "gpu":
        return require(), dtype
    raise ValueError("device must be 'gpu' or 'cpu'")


@functools.lru_cache(maxsize=32)
def _rank(modes: int, cutoff: int, local: tuple[int, ...]) -> np.ndarray:
    """Rank of each basis state's occupations on ``local`` in the local truncated basis."""
    engine = native.require("fock")
    occupations = engine.basis(modes, cutoff)[:, list(local)]
    rank = engine.rank(np.ascontiguousarray(occupations), cutoff)
    rank.setflags(write=False)
    return rank


@functools.lru_cache(maxsize=32)
def _maps(modes: int, cutoff: int, local: tuple[int, ...]):
    """CSR structure (indptr, indices) of the gate on ``local`` and each entry's index into U.

    Row g couples to every g' with the same occupations on the other modes (its fibre),
    with value U[r(g), r(g')], r = ``_rank``: the native engine's ``make_fibers`` with
    ``op.topLeftCorner(size, size)`` per fibre.
    """
    engine = native.require("fock")
    rank = _rank(modes, cutoff, local)
    rest = [m for m in range(modes) if m not in local]
    if rest:
        occupations = engine.basis(modes, cutoff)[:, rest]
        fibre = engine.rank(np.ascontiguousarray(occupations), cutoff)
    else:
        fibre = np.zeros(len(rank), dtype=np.int64)
    size = np.bincount(fibre)
    members = np.argsort(fibre, kind="stable")  # fibre by fibre, ascending global index
    start = np.cumsum(size) - size
    counts = size[fibre]
    indptr = np.concatenate([[0], np.cumsum(counts)])
    if indptr[-1] >= 2**31:
        raise ValueError("Gate has more than 2^31 nonzeros; reduce modes or cutoff")
    within = np.arange(indptr[-1]) - np.repeat(indptr[:-1], counts)
    indices = members[np.repeat(start[fibre], counts) + within]
    flat = np.repeat(rank, counts) * engine.dimension(len(local), cutoff) + rank[indices]
    arrays = (indptr.astype(np.int32), indices.astype(np.int32), flat)
    for a in arrays:
        a.setflags(write=False)
    return arrays


@functools.lru_cache(maxsize=32)
def _template(key: tuple, dtype: str, device_id: int | None):
    """Zero-valued CSR gate on the cached structure, and the entry -> U index map.

    The canonical-format check (on the GPU a reduction plus a device sync) runs once here and
    is cached on the template; calls take a shallow copy with their own values.
    """
    indptr, indices, flat = _maps(*key)
    dim = len(indptr) - 1
    if device_id is None:
        import scipy.sparse

        values = np.zeros(len(indices), dtype)
        gate = scipy.sparse.csr_matrix((values, indices, indptr), shape=(dim, dim))
    else:
        import cupyx.scipy.sparse

        cp = require()
        with cp.cuda.Device(device_id):
            values, flat = cp.zeros(len(indices), dtype), cp.asarray(flat)
            structure = (values, cp.asarray(indices), cp.asarray(indptr))
            gate = cupyx.scipy.sparse.csr_matrix(structure, shape=(dim, dim))
    if not gate.has_canonical_format:  # sorted, duplicate-free rows by construction
        raise RuntimeError("internal error: Fock gate structure is not canonical")
    return gate, flat


@functools.lru_cache(maxsize=32)
def _device_rank(key: tuple, device_id: int):
    cp = require()
    with cp.cuda.Device(device_id):
        return cp.asarray(_rank(*key))


def _prepare(xp, states, modes, cutoff, local, dtype):
    """Cast and check the batch against the basis; return it, the key and the dimensions."""
    engine = native.require("fock")
    modes, cutoff = operator.index(modes), operator.index(cutoff)
    local = tuple(operator.index(m) for m in local)  # 1.7 must not become mode 1
    if not local or len(set(local)) != len(local) or not all(0 <= m < modes for m in local):
        raise ValueError("local modes must be distinct indices of the state's modes")
    dim = engine.dimension(modes, cutoff)  # ValueError for modes < 0 or cutoff < 1
    states = xp.asarray(states, dtype=dtype)
    if states.ndim not in (2, 3) or states.shape[1:] not in ((dim,), (dim, dim)):
        raise ValueError(
            f"states of shape {states.shape} do not match the truncated basis of {modes} modes "
            f"at cutoff {cutoff}: expected (B, {dim}) pure states or (B, {dim}, {dim}) density "
            "matrices"
        )
    return states, (modes, cutoff, local), dim, engine.dimension(len(local), cutoff)


def apply_operator(
    states,
    modes: int,
    cutoff: int,
    local: Sequence[int],
    op,
    *,
    device: str = "gpu",
    dtype="complex128",
    on_device: bool = False,
) -> FockBatch:
    """psi -> P U P psi (``(B, D)``) or rho -> (P U P) rho (P U P)^dag (``(B, D, D)``).

    ``local`` lists the ordered gate modes; ``op`` is U in the local truncated basis,
    ``(S, S)`` with S the dimension of ``len(local)`` modes at ``cutoff``. Matches the native
    ``apply_pure`` / ``apply_mixed`` (no renormalisation; the input is not modified).

    Raises:
        ValueError: states not of the basis dimension of (modes, cutoff), ``op`` not
            ``(S, S)``, repeated or out-of-range local modes, or a bad device/dtype.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    states, key, dim, size = _prepare(xp, states, modes, cutoff, local, dtype)
    op = xp.asarray(op, dtype=dtype)
    if op.shape != (size, size):
        raise ValueError(
            f"op must be ({size}, {size}) for {len(key[2])} local modes at cutoff {cutoff}, "
            f"got {op.shape}"
        )
    template, flat = _template(key, dtype.name, None if xp is np else xp.cuda.Device().id)
    gate = copy.copy(template)  # shares the structure and its cached canonical flag
    gate.data = op.ravel()[flat]
    if states.ndim == 2:
        out = (gate @ states.T).T
    else:
        # Columns (b, c): G rho^dag, then G (G rho^dag)^dag = G rho G^dag for any rho.
        batch = len(states)
        half = gate @ xp.conj(states).transpose(2, 0, 1).reshape(dim, batch * dim)
        half = xp.conj(half.reshape(dim, batch, dim)).transpose(2, 1, 0).reshape(dim, -1)
        out = (gate @ half).reshape(dim, batch, dim).transpose(1, 0, 2)
    (out,), execution = _finish(xp, "apply_operator", dtype, on_device, xp.ascontiguousarray(out))
    return FockBatch(out, execution)


def apply_phase(
    states,
    modes: int,
    cutoff: int,
    local: Sequence[int],
    phases,
    *,
    device: str = "gpu",
    dtype="complex128",
    on_device: bool = False,
) -> FockBatch:
    """Diagonal gate exp(i phases[r]) with r the rank of the local occupations.

    ``phases`` (real, length S) is indexed like the native ``phase_pure`` / ``phase_mixed``:
    Kerr is ``kappa * n**2`` and rotation ``angle * n`` for ``n = arange(cutoff)`` on one
    mode; cross-Kerr is ``kappa * n_u * n_v`` over the two-mode basis. Density matrices
    become D rho D^dag.
    """
    xp, dtype = _resolve(device, dtype, on_device)
    states, key, _, size = _prepare(xp, states, modes, cutoff, local, dtype)
    phases = np.asarray(phases, dtype=float)
    if phases.shape != (size,) or not np.isfinite(phases).all():
        raise ValueError(f"phases must be {size} finite values for {len(key[2])} local modes")
    rank = _rank(*key) if xp is np else _device_rank(key, xp.cuda.Device().id)
    weights = xp.asarray(np.exp(1j * phases), dtype=dtype)[rank]
    if states.ndim == 2:
        out = states * weights
    else:
        out = states * weights[:, None] * xp.conj(weights)
    (out,), execution = _finish(xp, "apply_phase", dtype, on_device, out)
    return FockBatch(out, execution)
