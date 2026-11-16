"""Physical homodyne instrument on truncated Fock states (hbar = 2).

Quadrature wavefunctions. With q = a + a^dagger, the q eigenfunctions of the number states are
psi_0(x) = (2 pi)^{-1/4} exp(-x^2 / 4) and psi_{n+1}(x) = (x psi_n - sqrt(n) psi_{n-1}) /
sqrt(n + 1) (stable three-term recurrence of normalised Hermite functions; never a high-order
polynomial). Measuring x_theta = q cos(theta) + p sin(theta) equals measuring q after
R(-theta) = exp(-i theta n) (PhotoGraphiQ's R(phi) = exp(i phi n) maps q -> q cos phi - p sin phi),
so <x_theta = x|n> = e^{-i n theta} psi_n(x).

Instrument. For a pure state with coefficients c[..., n_m, ...] the unnormalised conditional
amplitude of the other modes is psi~(x) = sum_n c[..., n, ...] e^{-i n theta} psi_n(x), the outcome
density p(x) = ||psi~(x)||^2 (a density in the calibrated outcome x, the same units as the
Gaussian backend's records) and the conditional state psi~/sqrt(p). For a density matrix,
rho~(x) = <x_theta| rho |x_theta> over the measured mode. Finite-resolution detection with bins B
is implemented as fine homodyne followed by coarse graining: P(B) = int_B p(x) dx and the
conditional state is the mixture int_B p(x) rho(x) dx / P(B) (the sub-bin record is discarded);
this convention is stated in every result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .states import FockRep


def quadrature_wavefunctions(x, dim):
    """psi_n(x), n < dim, shape (dim, len(x)) (hbar = 2, q-representation)."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.zeros((dim, x.size))
    out[0] = (2 * np.pi) ** -0.25 * np.exp(-(x**2) / 4)
    if dim > 1:
        out[1] = x * out[0]
    for n in range(1, dim - 1):
        out[n + 1] = (x * out[n] - np.sqrt(n) * out[n - 1]) / np.sqrt(n + 1)
    return out


def _effects(x, theta, dim):
    return np.exp(-1j * np.arange(dim) * theta)[:, None] * quadrature_wavefunctions(x, dim)


@dataclass
class HomodyneOutcome:
    outcome: float
    density: float  # p(x) at the outcome (a density), or P(bin) for binned detection
    state: FockRep | None  # normalised conditional state of the remaining modes
    tail_mass: float  # outcome probability outside the integration window (sampling)
    window: tuple
    resolution: float | None
    convention: str
    conditional_boundary: float = 0.0  # share of p(x) carried by the measured mode's top 2 levels


def _amplitudes(rep, k, x, theta, top=None):
    """psi~(x) for each x: array (len(x), D_rest) pure, or rho~ (len(x), D_rest, D_rest).
    ``top`` keeps only the measured mode's levels n >= dim - top (truncation diagnostic)."""
    dims = rep.dims
    e = _effects(x, theta, dims[k])  # (dim_k, nx)
    if top is not None:
        e[: max(1, dims[k] - top)] = 0
    if rep.data.ndim == 1:
        t = np.moveaxis(rep.data.reshape(dims), k, 0).reshape(dims[k], -1)
        return np.einsum("nx,nr->xr", e, t)
    t = rep.data.reshape(dims + dims)
    n = len(dims)
    t = np.moveaxis(t, [k, n + k], [0, 1])
    rest = int(np.prod([d for i, d in enumerate(dims) if i != k]))
    t = t.reshape(dims[k], dims[k], rest, rest)
    return np.einsum("nx,mx,nmrs->xrs", e, e.conj(), t)


def density(rep: FockRep, mode, theta, x):
    """Outcome density p(x) of a homodyne at angle theta on ``mode``."""
    # p(x) = e(x)^T rho_k e(x)^*, rho_k the measured mode's reduced state (cheap for any size).
    rho = rep.reduced((mode,)).data
    e = _effects(np.atleast_1d(x), theta, rho.shape[0])
    return np.real(np.einsum("nx,nm,mx->x", e, rho, e.conj()))


def _remaining(rep, k, data, weight):
    modes = rep.modes[:k] + rep.modes[k + 1 :]
    dims = rep.dims[:k] + rep.dims[k + 1 :]
    return FockRep(
        data,
        modes,
        dims,
        rep.retained_weight * weight,
        dict(rep.physical_parameters),
        rep.precision,
    )


def window(rep: FockRep, mode, theta, width=10.0):
    """Integration window: mean +- width std of the measured quadrature, enclosing the Fock
    support radius ~ 2 sqrt(dim) of the truncated mode (so no occupied tail is cut)."""
    k = rep.modes.index(mode)
    mean, cov = rep.reduced((mode,)).moments()
    h = np.array([np.cos(theta), np.sin(theta)])
    m, s = h @ mean, np.sqrt(max(h @ cov @ h, 1e-12))
    reach = 2 * np.sqrt(rep.dims[k]) + 4
    return min(m - width * s, -reach), max(m + width * s, reach)


def measure(rep: FockRep, mode, theta, *, outcome=None, rng=None, grid=4001, resolution=None):
    """Apply the homodyne instrument; returns a :class:`HomodyneOutcome`.

    ``outcome`` postselects (density evaluated exactly at it); otherwise the outcome is sampled
    by inverse-CDF on a ``grid``-point window with linear interpolation inside a cell; the
    probability outside the window is reported as ``tail_mass``. With ``resolution`` the
    detector bins outcomes into cells of that width centred on multiples of the resolution.
    """
    k = rep.modes.index(mode)
    lo, hi = window(rep, mode, theta)
    xs = np.linspace(lo, hi, grid)
    p = density(rep, mode, theta, xs)
    dx = xs[1] - xs[0]
    mass = float(np.sum((p[1:] + p[:-1]) / 2) * dx)
    tail = max(0.0, 1.0 - mass)
    if resolution is None:
        if outcome is None:
            rng = np.random.default_rng() if rng is None else rng
            cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * dx)])
            u = rng.uniform(0, cdf[-1])
            outcome = float(np.interp(u, cdf, xs))
        amp = _amplitudes(rep, k, np.array([float(outcome)]), theta)[0]
        if amp.ndim == 1:
            pdf = float(np.vdot(amp, amp).real)
            state = (
                _remaining(rep, k, amp / np.sqrt(pdf), 1.0)
                if pdf > 0 and len(rep.modes) > 1
                else None
            )
        else:
            pdf = float(np.trace(amp).real)
            state = _remaining(rep, k, amp / pdf, 1.0) if pdf > 0 and len(rep.modes) > 1 else None
        if pdf <= 0:
            raise ValueError("Homodyne outcome has zero probability density")
        return HomodyneOutcome(
            float(outcome),
            pdf,
            state,
            tail,
            (lo, hi),
            None,
            "ideal homodyne; conditional state psi~(x)/sqrt(p(x))",
            _top_share(rep, k, float(outcome), theta, pdf),
        )
    # finite resolution: bins [c - r/2, c + r/2)
    if outcome is None:
        rng = np.random.default_rng() if rng is None else rng
        cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * dx)])
        x0 = float(np.interp(rng.uniform(0, cdf[-1]), cdf, xs))
        centre = resolution * np.round(x0 / resolution)
    else:
        centre = float(outcome)
    sub = 64
    fine = centre + resolution * ((np.arange(sub) + 0.5) / sub - 0.5)
    amps = _amplitudes(rep, k, fine, theta)
    if amps.ndim == 2:
        weights = np.sum(np.abs(amps) ** 2, axis=1)
        rho = np.einsum("xr,xs->rs", amps, amps.conj()) * (resolution / sub)
    else:
        weights = np.real(np.einsum("xrr->x", amps))
        rho = np.sum(amps, axis=0) * (resolution / sub)
    prob = float(np.sum(weights) * resolution / sub)
    if prob <= 0:
        raise ValueError("Bin has zero probability")
    state = _remaining(rep, k, rho / prob, 1.0) if len(rep.modes) > 1 else None
    return HomodyneOutcome(
        float(centre),
        prob,
        state,
        tail,
        (lo, hi),
        float(resolution),
        "binned homodyne: fine measurement then coarse graining; conditional "
        "state = int_bin p(x) rho(x) dx / P(bin) (midpoint rule, 64 points)",
        _top_share(rep, k, float(centre), theta, float(density(rep, mode, theta, centre)[0])),
    )


def _top_share(rep, k, x, theta, pdf, levels=2):
    """||psi~_top(x)||^2 / p(x): conditioning on a rare outcome amplifies truncation error in the
    high levels by 1 / p(x), which the state's own boundary population does not show."""
    amp = _amplitudes(rep, k, np.array([x]), theta, top=levels)[0]
    top = np.vdot(amp, amp).real if amp.ndim == 1 else np.trace(amp).real
    return float(top / pdf) if pdf > 0 else float("inf")
