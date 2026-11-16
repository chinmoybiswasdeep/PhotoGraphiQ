"""Hybrid Gaussian-Fock trajectory executor for CV-MBQC patterns (experimental API).

Executes a ``photographiq.Pattern`` in command order on a per-mode truncated Fock state:

* preparations: Gaussian inputs via the Gaussian-to-Fock bridge, Fock inputs truncated;
* Gaussian and non-Gaussian gates: U = exp(-i H) of the verified generators of
  :mod:`photographiq.cudaq_dynamics` on the gate's modes (truncated-generator semantics, the
  semantics of CUDA-Q's per-mode truncation), ``engine="numpy-reference"`` (local dense expm)
  or ``engine="cudaq"`` (``cudaq_dynamics.evolve`` on the full state: never replaced by NumPy);
* loss: the Lindblad channel L = sqrt(1 + n_th) a, sqrt(n_th) a^dagger for t = -ln(eta);
* homodyne: the physical Fock instrument (:mod:`.homodyne`), postselected or sampled;
* photon counting: projective number measurement;
* feed-forward: ``Displace`` with outcome expressions evaluated on the actual records.

Not a registered pattern backend: use :func:`run` / :func:`run_shots`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Any

import numpy as np
from scipy.linalg import expm

from .. import commands as c
from .. import cudaq_dynamics as cq
from ..expressions import resolve
from ..measurements import Homodyne, PhotonNumber
from ..resources import CubicPhaseResource
from ..states import FockInput, GaussianInput
from . import homodyne as hd
from .planner import plan
from .states import FockRep, fock_input, gaussian_to_fock

ENGINES = ("numpy-reference", "cudaq")
BOUNDARY_WARNING = 0.02  # as the native Fock backends


@dataclass
class HybridResult:
    state: FockRep | None
    records: dict
    densities: dict  # homodyne record -> density (or bin probability)
    probabilities: dict  # discrete record -> probability
    retained_weight: float
    tail_mass: dict
    execution_record: list = field(default_factory=list)
    plan: object = None
    runtime: float = 0.0
    engine: str = "numpy-reference"
    truncation: dict = field(default_factory=dict)  # boundary-population diagnostics

    @property
    def log_likelihood(self):
        return float(
            sum(np.log(v) for v in self.densities.values())
            + sum(np.log(v) for v in self.probabilities.values())
        )

    def expectation(self, observable):
        """Tr(rho O) for a :class:`cudaq_dynamics.BosonicHamiltonian` on output positions 0..n-1."""
        if self.state is None:
            raise ValueError("No surviving output state")
        rho = self.state.density_matrix()
        return complex(
            np.trace(cq.to_matrix(observable, {k: d for k, d in enumerate(self.state.dims)}) @ rho)
        )


def _gate_hamiltonian(cmd, value):
    """(local Hamiltonian on local indices 0.., mode labels) of a unitary command."""
    if isinstance(cmd, c.Rotate):
        return cq.rotation(0, value(cmd.angle)), (cmd.node,)
    if isinstance(cmd, c.Squeeze):
        return cq.squeezing(0, value(cmd.r)), (cmd.node,)
    if isinstance(cmd, c.QuadraticPhase):
        return cq.quadratic_phase(0, value(cmd.s)), (cmd.node,)
    if isinstance(cmd, c.Displace):
        return cq.displacement(0, value(cmd.q), value(cmd.p)), (cmd.node,)
    if isinstance(cmd, c.Kerr):
        return cq.kerr(0, value(cmd.kappa)), (cmd.node,)
    if isinstance(cmd, c.CubicPhase):
        return cq.cubic_phase(0, value(cmd.gamma)), (cmd.node,)
    if isinstance(cmd, c.BeamSplitter):
        return cq.beamsplitter(0, 1, value(cmd.theta)), (cmd.u, cmd.v)
    if isinstance(cmd, c.Entangle):
        return cq.controlled_z(0, 1, value(cmd.weight)), (cmd.u, cmd.v)
    raise NotImplementedError(type(cmd).__name__)


def _apply_local(rep, U, modes):
    """Apply a local operator U (on ``modes``, kron order) to the state."""
    axes = [rep.modes.index(m) for m in modes]
    dims = rep.dims
    local = tuple(dims[a] for a in axes)
    Ut = U.reshape(local + local)
    k = len(axes)
    if rep.data.ndim == 1:
        t = rep.data.reshape(dims)
        t = np.tensordot(Ut, t, axes=(list(range(k, 2 * k)), axes))
        t = np.moveaxis(t, list(range(k)), axes)
        return rep.__class__(
            t.reshape(-1), rep.modes, dims, rep.retained_weight, rep.physical_parameters
        )
    n = len(dims)
    t = rep.data.reshape(dims + dims)
    t = np.tensordot(Ut, t, axes=(list(range(k, 2 * k)), axes))
    t = np.moveaxis(t, list(range(k)), axes)
    t = np.tensordot(t, Ut.conj(), axes=([n + a for a in axes], list(range(k, 2 * k))))
    t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), [n + a for a in axes])
    D = prod(dims)
    return rep.__class__(
        t.reshape(D, D), rep.modes, dims, rep.retained_weight, rep.physical_parameters
    )


@lru_cache(maxsize=256)
def _loss_superoperator(eta, thermal, dim):
    """Single-mode channel Lambda[i', j', i, j] = expm(t Liouvillian) on the truncated mode (cached).

    Jump operators sqrt(1 + n_th) a and sqrt(n_th) a^dagger (the adapter's truncated
    matrices), t = -ln(eta); row-major vec(A rho B) = (A kron B^T) vec(rho).
    """
    if not 0 < eta <= 1 or thermal < 0:
        raise ValueError("Hybrid loss needs 0 < eta <= 1 and n_th >= 0")
    if eta == 1:
        return np.einsum("ac,bd->abcd", np.eye(dim), np.eye(dim))
    a = cq.to_matrix(cq.annihilate(0), {0: dim})
    eye = np.eye(dim)
    gen = np.zeros((dim * dim, dim * dim), complex)
    for J in [np.sqrt(1 + thermal) * a] + ([np.sqrt(thermal) * a.conj().T] if thermal else []):
        JdJ = J.conj().T @ J
        gen += np.kron(J, J.conj()) - 0.5 * np.kron(JdJ, eye) - 0.5 * np.kron(eye, JdJ.T)
    return expm(-np.log(eta) * gen).reshape(dim, dim, dim, dim)


def _apply_channel(rep, lam, mode):
    a = rep.modes.index(mode)
    dims = rep.dims
    n = len(dims)
    t = rep.data.reshape(dims + dims)
    t = np.tensordot(lam, t, axes=([2, 3], [a, n + a]))  # -> (i', j', rest...)
    order = list(range(2, 2 + n + n - 2))
    order.insert(a, 0)
    order.insert(n + a, 1)
    t = np.transpose(t, order)  # final axis k <- current axis order[k]
    D = prod(dims)
    return rep.__class__(
        t.reshape(D, D), rep.modes, dims, rep.retained_weight, rep.physical_parameters
    )


def _initial(state, node, dim):
    if isinstance(state, FockRep):
        return state
    if state is None:
        state = GaussianInput()
    if isinstance(state, GaussianInput):
        return gaussian_to_fock(
            np.asarray(state.mean, float),
            np.asarray(state.covariance, float),
            (dim,),
            modes=(node,),
        )
    if isinstance(state, FockInput):
        return fock_input(state.amplitudes, node, dim)
    raise NotImplementedError(f"Unsupported input {type(state).__name__}")


def run(
    pattern,
    *,
    dims=12,
    inputs=None,
    measurement_outcomes=None,
    seed=None,
    engine="numpy-reference",
    mixed=None,
    resolution=None,
    parameters=None,
    grid=4001,
) -> HybridResult:
    """Execute one trajectory of ``pattern`` on the hybrid Fock executor.

    ``dims``: per-mode local dimension (int, or {mode: int, "default": int}).
    ``measurement_outcomes``: postselected records (homodyne values / photon counts).
    ``engine``: ``"numpy-reference"`` or ``"cudaq"`` (raises ImportError without CUDA-Q).
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    start = time.perf_counter()
    the_plan = plan(pattern, dims, mixed=mixed)
    if not the_plan.executable:
        raise NotImplementedError(f"Unsupported commands: {the_plan.unsupported}")
    for cmd in pattern.commands:
        if (
            isinstance(cmd, c.Measure)
            and isinstance(cmd.measurement, Homodyne)
            and (cmd.measurement.efficiency != 1 or cmd.measurement.noise)
        ):
            raise NotImplementedError(
                "Inefficient / noisy homodyne is not implemented in the hybrid executor"
            )
    if engine == "cudaq":
        cq._require_cudaq()  # explicit: never substitute the NumPy reference
    rng = np.random.default_rng(seed)
    params = dict(parameters or {})
    inputs = dict(inputs or {})
    outcomes = dict(measurement_outcomes or {})
    rep: Any = None  # FockRep once the first mode exists
    for node in pattern.inputs:
        part = _initial(inputs.get(node), node, the_plan.dims[node])
        rep = part if rep is None else rep.tensor(part)
    if rep is not None and the_plan.mixed:
        rep = rep.to_mixed()
    records: dict[Any, Any] = {}
    densities: dict[Any, float] = {}
    probabilities: dict[Any, float] = {}
    tails: dict[Any, float] = {}
    log: list[dict[str, Any]] = []

    def value(x):
        return float(resolve(x, records, params))

    def evolve(H, modes):
        nonlocal rep
        t0 = time.perf_counter()
        if engine == "numpy-reference":
            local = {k: rep.dims[rep.modes.index(m)] for k, m in enumerate(modes)}
            U = expm(-1j * cq.to_matrix(H, local))
            rep = _apply_local(rep, U, modes)
            record = {"engine": "numpy-reference", "device": "cpu"}
        else:  # full-state CUDA-Q evolution, H relabelled to global positions
            mapping = {k: rep.modes.index(m) for k, m in enumerate(modes)}
            Hg = cq.BosonicHamiltonian(
                {
                    tuple(sorted((mapping[mm], cr, an) for mm, cr, an in key)): v
                    for key, v in H.terms.items()
                }
            )
            res = cq.evolve(Hg, {k: d for k, d in enumerate(rep.dims)}, [0.0, 1.0], rep.data)
            rep = FockRep(
                res.state, rep.modes, rep.dims, rep.retained_weight, rep.physical_parameters
            )
            record = dict(res.record)
        log.append(
            {
                "op": "unitary",
                "modes": list(map(repr, modes)),
                "seconds": time.perf_counter() - t0,
                "boundary_population": rep.boundary_population(),
                **record,
            }
        )

    for cmd in pattern.commands:
        if isinstance(cmd, c.Prepare):
            resource = cmd.state if isinstance(cmd.state, CubicPhaseResource) else None
            if (
                cmd.state is None or resource is not None
            ):  # p-squeezed S(-r)|0>, as the Fock backends
                state = GaussianInput.squeezed(
                    -(resource.squeezing if resource else value(cmd.squeezing))
                )
            else:
                state = cmd.state
            part = _initial(state, cmd.node, the_plan.dims[cmd.node])
            if rep is None:
                rep = part
            elif rep.data.ndim == 2 or part.data.ndim == 2:
                rep = rep.to_mixed().tensor(part.to_mixed())
            else:
                rep = rep.tensor(part)
            if the_plan.mixed:
                rep = rep.to_mixed()
            log.append(
                {
                    "op": "prepare",
                    "node": repr(cmd.node),
                    "retained_weight": part.retained_weight,
                    "boundary_population": rep.boundary_population(),
                }
            )
            if resource is not None:  # CubicPhaseResource = CP(gamma) S(-r)|0>
                evolve(cq.cubic_phase(0, resource.gamma), (cmd.node,))
        elif isinstance(
            cmd,
            (
                c.Rotate,
                c.Squeeze,
                c.QuadraticPhase,
                c.Displace,
                c.Kerr,
                c.CubicPhase,
                c.BeamSplitter,
                c.Entangle,
            ),
        ):
            H, modes = _gate_hamiltonian(cmd, value)
            evolve(H, modes)
        elif isinstance(cmd, c.Loss):
            eta, nth = value(cmd.transmissivity), value(cmd.thermal_photons)
            rep = rep.to_mixed()
            t0 = time.perf_counter()
            if engine == "cudaq":
                k = rep.modes.index(cmd.node)
                jumps = [np.sqrt(1 + nth) * cq.annihilate(k)] + (
                    [np.sqrt(nth) * cq.create(k)] if nth else []
                )
                res = cq.evolve(
                    cq.BosonicHamiltonian(),
                    {i: d for i, d in enumerate(rep.dims)},
                    [0.0, -np.log(eta)],
                    rep.data,
                    collapse_operators=jumps,
                )
                rep = FockRep(
                    res.state, rep.modes, rep.dims, rep.retained_weight, rep.physical_parameters
                )
                record = dict(res.record)
            else:
                rep = _apply_channel(
                    rep,
                    _loss_superoperator(
                        float(eta), float(nth), rep.dims[rep.modes.index(cmd.node)]
                    ),
                    cmd.node,
                )
                record = {"engine": "numpy-reference", "device": "cpu"}
            log.append(
                {
                    "op": "lindblad-loss",
                    "node": repr(cmd.node),
                    "eta": eta,
                    "seconds": time.perf_counter() - t0,
                    "boundary_population": rep.boundary_population(),
                    **record,
                }
            )
        elif isinstance(cmd, c.Measure) and isinstance(cmd.measurement, Homodyne):
            theta = value(cmd.measurement.angle)
            given = outcomes.get(cmd.result_key)
            out = hd.measure(
                rep, cmd.node, theta, outcome=given, rng=rng, grid=grid, resolution=resolution
            )
            records[cmd.result_key] = out.outcome
            densities[cmd.result_key] = out.density
            tails[cmd.result_key] = out.tail_mass
            remaining = tuple(m for m in rep.modes if m != cmd.node)
            rep = out.state if remaining else None
            log.append(
                {
                    "op": "homodyne",
                    "node": repr(cmd.node),
                    "theta": theta,
                    "outcome": out.outcome,
                    "density": out.density,
                    "tail_mass": out.tail_mass,
                    "convention": out.convention,
                    "boundary_population": out.state.boundary_population()
                    if out.state is not None
                    else 0.0,
                    "conditional_boundary": out.conditional_boundary,
                }
            )
        elif isinstance(cmd, c.Measure) and isinstance(cmd.measurement, PhotonNumber):
            k = rep.modes.index(cmd.node)
            if rep.data.ndim == 1:
                t = np.moveaxis(rep.data.reshape(rep.dims), k, 0)
                probs = np.array([np.vdot(t[n], t[n]).real for n in range(rep.dims[k])])
            else:
                n_ = len(rep.dims)
                t = np.moveaxis(rep.data.reshape(rep.dims + rep.dims), [k, n_ + k], [0, 1])
                R = prod(rep.dims) // rep.dims[k]
                probs = np.array([np.trace(t[n, n].reshape(R, R)).real for n in range(rep.dims[k])])
            given = outcomes.get(cmd.result_key)
            n = (
                int(given)
                if given is not None
                else int(rng.choice(len(probs), p=probs / probs.sum()))
            )
            p = float(probs[n])
            if p <= 0:
                raise ValueError("Photon-count outcome has zero probability")
            remaining = tuple(m for m in rep.modes if m != cmd.node)
            if remaining:
                rd = rep.dims[:k] + rep.dims[k + 1 :]
                if rep.data.ndim == 1:
                    data = t[n].reshape(-1) / np.sqrt(p)
                else:
                    data = t[n, n].reshape(prod(rd), prod(rd)) / p
                rep = FockRep(data, remaining, rd, rep.retained_weight, rep.physical_parameters)
            else:
                rep = None
            records[cmd.result_key] = n
            probabilities[cmd.result_key] = p
            log.append(
                {"op": "photon-count", "node": repr(cmd.node), "outcome": n, "probability": p}
            )
        elif isinstance(cmd, c.Output):
            continue
        else:
            raise NotImplementedError(type(cmd).__name__)
    boundary = max((e.get("boundary_population", 0.0) for e in log), default=0.0)
    if boundary > BOUNDARY_WARNING:
        warnings.warn(
            f"Hybrid boundary population {boundary:.3g}: increase the local dimensions",
            RuntimeWarning,
            stacklevel=2,
        )
    truncation = {
        "max_boundary_population": boundary,
        "levels": 2,
        "warning_threshold": BOUNDARY_WARNING,
        "max_conditional_boundary": max(
            (e.get("conditional_boundary", 0.0) for e in log), default=0.0
        ),
        "note": "per-mode population with n >= d - 2 after every operation; gates are norm preserving "
        "(truncated-generator semantics), so a small lost weight alone does not certify convergence; "
        "conditional_boundary = share of each homodyne density carried by the top 2 levels (rare "
        "outcomes amplify truncation error by 1/p(x))",
    }
    if rep is not None and pattern.outputs:
        if tuple(rep.modes) != tuple(pattern.outputs):
            rep = (
                rep.reduced(pattern.outputs)
                if rep.data.ndim == 2
                else _reorder(rep, pattern.outputs)
            )
    return HybridResult(
        rep,
        records,
        densities,
        probabilities,
        rep.retained_weight if rep is not None else 1.0,
        tails,
        log,
        the_plan,
        time.perf_counter() - start,
        engine,
        truncation,
    )


def _reorder(rep, modes):
    axes = [rep.modes.index(m) for m in modes]
    t = np.transpose(rep.data.reshape(rep.dims), axes).reshape(-1)
    return FockRep(
        t,
        tuple(modes),
        tuple(rep.dims[a] for a in axes),
        rep.retained_weight,
        rep.physical_parameters,
    )


def run_shots(pattern, shots, *, seed=None, **kwargs):
    """Independent sampled trajectories with reproducible child seeds."""
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    children = np.random.SeedSequence(seed).spawn(shots)
    return [run(pattern, seed=np.random.default_rng(child), **kwargs) for child in children]


def ensemble_statistics(results, observable):
    """Mean and standard error of Tr(rho O) over sampled trajectories (finite-shot estimate)."""
    values = np.array([np.real(r.expectation(observable)) for r in results])
    return {
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else float("nan"),
        "shots": len(values),
        "note": "finite-shot estimate, not an exact expectation",
    }
