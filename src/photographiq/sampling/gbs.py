r"""Gaussian boson sampling from PhotoGraphiQ Gaussian states.

Conversion (Hamilton et al. PRL 119, 170501 (2017); Kruse et al. PRA 100, 032326 (2019);
Quesada et al. PRA 98, 062322 (2018)). A :class:`~photographiq.states.GaussianState`
holds the interleaved mean ``d`` and statistical covariance ``V`` at hbar = 2 (vacuum
``V = I``). With ``a_j = (q_j + i p_j)/2`` and ``xi = (a_1..a_m, a_1^dag..a_m^dag)``,
``xi = W x`` where ``W`` has rows ``(e_q + i e_p)/2`` and ``(e_q - i e_p)/2``, so

* ``beta = <xi> = W d = (alpha, alpha^*)`` with ``alpha_j = (d_{q_j} + i d_{p_j})/2``;
* ``sigma = <{dxi, dxi^dag}>/2 = W V W^dag`` and ``sigma_Q = sigma + I/2`` (vacuum
  ``sigma_Q = I``). As ``W = U/sqrt(2)`` with ``U`` unitary, ``sigma_Q = U (V + I)/2 U^dag``;
* ``A = X (I - sigma_Q^{-1})^*`` with ``X = [[0, I], [I, 0]]``, and the loop vector
  ``gamma = sigma_Q^{-1} beta``;
* vacuum probability ``p_0 = exp(-beta^dag sigma_Q^{-1} beta / 2) / sqrt(det sigma_Q)
  = 2^m exp(-d^T (V+I)^{-1} d / 2) / sqrt(det(V+I))``.

Derivation of ``(A, gamma)``: with ``z = (alpha, alpha^*)`` and coherent states
``|alpha>``, ``e^{|alpha|^2} <alpha|rho|alpha> = p_0 exp(z^T X (I - sigma_Q^{-1}) z / 2
+ beta^dag sigma_Q^{-1} z) = sum_{n,k} rho_{nk} alpha^{*n} alpha^k / sqrt(n! k!)``; the
Taylor coefficients of the exponential are loop hafnians of
``(X (I - sigma_Q^{-1}), (sigma_Q^{-1})^* beta^*)``. Diagonal (probability) elements
are real, so the complex-conjugate pair ``(A, gamma)`` used here gives identical
probabilities. This ``A`` is the complex conjugate of thewalrus' ``A`` (and ``gamma`` of
its ``gamma``), so both conventions give the same probabilities; mixing ``gamma`` from one
convention with ``A`` from the other is wrong for displaced states. The test-suite
compares against Piquasso and Fock-space simulations.

Photon-number-resolving (PNR) probability of the pattern ``n``:
``P(n) = p_0 lhaf(A_{n+n}; gamma_{n+n}) / prod_i n_i!``, where ``A_{n+n}`` repeats index
``i`` and ``i+m`` ``n_i`` times (a hafnian when ``d = 0``). For pure states
``A = B (+) B^*`` and ``P(n) = p_0 |lhaf(B_n; gamma_n)|^2 / prod n_i!`` (dimension ``N``
instead of ``2N`` for ``N`` photons). Repeated photons are handled by duplicating
indices (cost ``2^{N/2}`` (pure) or ``2^N`` (mixed) power-trace terms).

Threshold detectors (click = at least one photon). With ``O = I - sigma_Q^{-1}`` and
``S`` the clicked modes, ``P(S) = p_0 ltor(O_S, gamma'_S)`` with
``gamma' = gamma^* = (sigma_Q^{-1} beta)^*`` (``Tor(O_S)`` when ``d = 0``), which is the
inclusion-exclusion sum over vacuum projections of the marginals. ``threshold_probability``
evaluates the same sum in an equivalent real form that conditions on the dark modes ``Z``
first. With ``M = V + I``, ``E = ((V - I)_SS - M_SZ M_ZZ^{-1} M_ZS) / 2`` (so the
conditioned block is ``C = 2 (I + E)``, ``E = 0`` for vacuum), ``e = d_S - M_SZ M_ZZ^{-1} d_Z``
and the dark-mode vacuum probability
``P_Z = 2^{|Z|} exp(-d_Z^T M_ZZ^{-1} d_Z / 2) / sqrt(det M_ZZ)``,

``P(S) = P_Z ([S empty] + sum_{T subset S} (-1)^{|T|} w_T)``,
``w_T = expm1(-sum log1p(eig E_T) / 2 - e_T^T C_T^{-1} e_T / 2)``,

i.e. every vacuum-projection term ``1 + w_T`` enters through its deviation from 1, computed
to relative accuracy (``V - I`` is exact in floating point). The alternating sum still
cancels as the number of clicks grows: against a 50-digit reference the relative error at
32 modes / 16 clicks (r = 0.5, eta = 0.9, ``P ~ 9e-13``) is 2.9e-3 (Python) and 7.5e-3
(native), where Piquasso gives 3.0e-4 (benchmarks/README.md). It varies by about an order
of magnitude with eps-level changes of the conditioning, which is why the rounding bound
below, not a quoted figure, is the guide; ``precision=`` evaluates the same sum in mpmath.

Rounding bound. ``threshold_probability`` returns the relative error ``B / |P|`` of the
first-order absolute bound (unit roundoff ``u = eps/2``)

``B = P_Z (eps sum_T |w_T| + S) + P_Z C + |P| (eps_Z + u)``,
``C = (1 + sqrt3)/8 ||V_c|| Tr|dV_c| + ||V_c||^{1/2} ||de|| / 2``.

``P_Z (eps sum_T |w_T| + S)`` is the alternating sum's rounding. ``eps sum_T |w_T|`` alone
(the earlier estimate) misses exponents whose eigenvalue contributions cancel (squeezing
0.05, 6 clicks: error 4.2e-16, estimate 5.2e-17), so ``S`` bounds
``u sum_T (1 + w_T) (sum_i |log1p mu_i(E_T)| / 2 + q_T)`` in closed form (``_sum_bound``),
times ``max(1, 16 / sqrt(2^k - 1))`` for ``k`` clicks. This per-term model is first order
and empirical, like the earlier estimate: it is checked against mpmath, not proven. Without
the factor it failed for few clicks at small squeezing (exact for one term, so the
eigensolver's error ~ u ||E_T|| shows up to 3.9x the model at k <= 2); the excess falls
roughly as ``1/sqrt(2^k - 1)`` as per-term errors partly cancel (k = 8: 0.36x). With it,
the sum error stayed below 0.52 of the sum bound in 4140 random states on both backends
(k = 1..9, squeezing 5e-5..2, transmissivity 0.5..1, displaced and not), and the total
error below 0.43 of ``B`` in 2000 further ones. For ``k >= 9`` the factor is 1.

``C`` and ``eps_Z`` cover the dark-mode conditioning. It uses one Cholesky factor
``M_ZZ = L L^T`` with backward error ``|dM| <= gamma_{3n+1} |L| |L^T|`` (Higham, *Accuracy
and Stability of Numerical Algorithms*, Thms 10.3-10.4), propagated to first order into
componentwise bounds on ``dE``, ``de`` and the relative error ``eps_Z`` of ``P_Z``.
``P / P_Z = tr(Pi rho_c)`` is the click probability of the conditional Gaussian state
``rho_c`` (covariance ``V_c = I + 2E``, mean ``e``) with ``0 <= Pi <= I``. A covariance
change ``N = sum_k lambda_k u_k u_k^T`` changes ``rho_c`` by
``-(1/8) sum_k lambda_k [A_k, [A_k, rho_c]]`` with ``A_k = (Omega u_k)^T R`` (additive
Gaussian noise to second order), and ``|tr(Pi [A, [A, rho]])| <= (1 + sqrt3) <dA^2>`` by
Hoelder and ``<dA^4> = 3 <dA^2>^2``. A mean change ``de`` changes it by ``i [G, rho_c]`` with
``|tr(Pi i [G, rho_c])| <= <dG^2>^{1/2}`` and ``<dG^2> = de^T Omega^T V_c Omega de / 4``.

A result with ``|P| <= B`` is consistent with zero. Negative results always raise a
``ThresholdPrecisionWarning``, and so does ``B / |P| > 1e-6``. ``precision=<digits>``
evaluates the sum with mpmath instead.

Loss: ``V -> eta V + (1 - eta) I`` and ``d -> sqrt(eta) d`` per mode (pure-loss channel).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from scipy.linalg import cho_solve

from .. import gaussian as gm
from ..states import GaussianState
from .boson_sampling import output_patterns
from .linalg import Backend, _kernels, hafnian, loop_hafnian
from .unitaries import passive_symplectic

# The pure-state formula is used only when A = B (+) B^* holds to rounding; the mixed
# formula is exact for every state, so ambiguous (nearly pure) states take the exact route.
_PURE_RTOL = 100 * np.finfo(float).eps


@dataclass(frozen=True)
class GBSMatrices:
    """Complex-amplitude representation of a Gaussian state (block ``(a, a^dag)`` order)."""

    sigma_q: np.ndarray
    a: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    vacuum_probability: float
    mean: np.ndarray
    covariance: np.ndarray

    @property
    def modes(self) -> int:
        return len(self.beta) // 2

    @property
    def displaced(self) -> bool:
        return bool(np.any(self.beta != 0))

    @property
    def pure(self) -> bool:
        """``A = B (+) B^*`` to rounding (off-diagonal block <= 100 eps max(1, max|A|)).

        Nearly pure states (e.g. loss 1 - 1e-11) are *not* pure here: the pure formula
        would neglect the off-diagonal block, an error proportional to its size.
        """
        m = self.modes
        scale = max(1.0, float(np.abs(self.a).max(initial=0.0)))
        return bool(np.abs(self.a[:m, m:]).max(initial=0.0) <= _PURE_RTOL * scale)


def _quadrature_to_ladder(modes: int) -> np.ndarray:
    w = np.zeros((2 * modes, 2 * modes), dtype=complex)
    for j in range(modes):
        w[j, 2 * j], w[j, 2 * j + 1] = 0.5, 0.5j
        w[modes + j, 2 * j], w[modes + j, 2 * j + 1] = 0.5, -0.5j
    return w


def gbs_matrices(state: GaussianState) -> GBSMatrices:
    """Compute ``sigma_Q``, ``A``, ``beta``, ``gamma`` and ``p_0`` (see module doc)."""
    m = len(state.nodes)
    w = _quadrature_to_ladder(m)
    sigma_q = w @ state.covariance @ w.conj().T + np.eye(2 * m) / 2
    sigma_q = (sigma_q + sigma_q.conj().T) / 2
    q_inv = np.linalg.inv(sigma_q)
    x = np.block([[np.zeros((m, m)), np.eye(m)], [np.eye(m), np.zeros((m, m))]])
    a = x @ (np.eye(2 * m) - q_inv).conj()
    a = (a + a.T) / 2
    beta = w @ state.mean
    vac = state.covariance + np.eye(2 * m)
    _, logdet = np.linalg.slogdet(vac)
    p0 = math.exp(
        m * math.log(2.0) - 0.5 * state.mean @ np.linalg.solve(vac, state.mean) - 0.5 * logdet
    )
    return GBSMatrices(sigma_q, a, beta, q_inv @ beta, p0, state.mean, state.covariance)


def _pattern(pattern: Sequence[int], modes: int) -> np.ndarray:
    n = np.asarray(pattern)
    if n.shape != (modes,) or not np.issubdtype(n.dtype, np.integer) or (n < 0).any():
        raise ValueError(f"Pattern must be {modes} nonnegative integers")
    return n.astype(np.int64)


def pnr_probability(
    state: GaussianState | GBSMatrices,
    pattern: Sequence[int],
    *,
    method: Literal["auto", "pure", "mixed"] = "auto",
    backend: Backend = "auto",
) -> float:
    """Probability of the photon-number pattern ``pattern`` (ordered as ``state.nodes``).

    ``method="pure"`` uses ``|lhaf(B_n)|^2`` and raises ``ValueError`` for mixed states;
    ``"mixed"`` always uses ``lhaf(A_{n+n})``; ``"auto"`` picks ``"pure"`` only for states pure to rounding (see ``GBSMatrices.pure``).
    """
    g = state if isinstance(state, GBSMatrices) else gbs_matrices(state)
    m = g.modes
    n = _pattern(pattern, m)
    if method == "pure" and not g.pure:
        raise ValueError("State is not pure; use method='mixed'")
    if method not in ("auto", "pure", "mixed"):
        raise ValueError(f"Unknown method {method!r}")
    pure = g.pure if method == "auto" else method == "pure"
    norm = math.prod(math.factorial(int(k)) for k in n)
    if pure:
        index = np.repeat(np.arange(m), n)
        b = g.a[np.ix_(index, index)]
        if g.displaced:
            value = abs(loop_hafnian(b, g.gamma[index], backend=backend)) ** 2
        else:
            value = abs(hafnian(b, backend=backend)) ** 2
    else:
        index = np.r_[np.repeat(np.arange(m), n), np.repeat(np.arange(m, 2 * m), n)]
        a = g.a[np.ix_(index, index)]
        if g.displaced:
            value = loop_hafnian(a, g.gamma[index], backend=backend).real
        else:
            value = hafnian(a, backend=backend).real
    return g.vacuum_probability * value / norm


@dataclass(frozen=True)
class TruncatedDistribution:
    """PNR probabilities of every pattern with total photon number ``< cutoff``.

    ``truncated_mass = 1 - captured_mass`` is the probability of all omitted patterns
    (total photon number ``>= cutoff``); it is reported, never redistributed. Rounding
    can make it negative at the ``1e-15`` level.
    """

    probabilities: dict[tuple[int, ...], float]
    cutoff: int
    captured_mass: float
    truncated_mass: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "truncated_mass", 1.0 - self.captured_mass)


def pnr_distribution(
    state: GaussianState, cutoff: int, *, backend: Backend = "auto"
) -> TruncatedDistribution:
    """All PNR probabilities with total photon number below ``cutoff``."""
    if int(cutoff) != cutoff or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    g = gbs_matrices(state)
    probs = {
        t: pnr_probability(g, t, backend=backend)
        for total in range(int(cutoff))
        for t in output_patterns(g.modes, total)
    }
    return TruncatedDistribution(probs, int(cutoff), math.fsum(probs.values()))


class ThresholdPrecisionWarning(RuntimeWarning):
    """A threshold probability is negative, consistent with zero within its rounding bound,
    or has an estimated relative rounding error above ``1e-6`` (module doc)."""


_THRESHOLD_WARN = 1e-6
_U = float(np.finfo(float).eps) / 2  # unit roundoff
_SUM_SAFETY = 16.0  # factor max(1, 16 / sqrt(2^k - 1)) on the per-term model (module doc)


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)``."""
    return k * _U / (1 - k * _U)


def _dark_conditioned(g: GBSMatrices, clicks: np.ndarray):
    """``(P_Z, E, e)`` of the module doc and first-order bounds on their rounding.

    Returns ``(P_Z, E, e, eps_Z, dvc, de)``: ``eps_Z`` bounds the relative error of ``P_Z``,
    ``dvc`` bounds ``Tr|dV_c|`` (``V_c = I + 2E``) and ``de`` bounds ``||de||_2``.
    """
    m = g.modes
    s, z = np.flatnonzero(clicks), np.flatnonzero(clicks == 0)
    qs = np.ravel(np.column_stack([2 * s, 2 * s + 1])).astype(int)
    qz = np.ravel(np.column_stack([2 * z, 2 * z + 1])).astype(int)
    big = g.covariance + np.eye(2 * m)
    excess = g.covariance - np.eye(2 * m)
    d = g.mean
    # V_ii - 1 is exact for V_ii in [1/2, 2] (Sterbenz) and off the diagonal; bound it anyway.
    excess_ss = excess[np.ix_(qs, qs)]
    formed = _U * np.diag(np.abs(np.diag(excess_ss)))
    if not len(qz):
        # Tr|2 dE| <= sum of the column 2-norms of the componentwise bound on 2 dE.
        return 1.0, excess / 2, d.copy(), 0.0, float(np.linalg.norm(formed, axis=0).sum()), 0.0
    n = len(qz)
    mzz = big[np.ix_(qz, qz)]
    b = big[np.ix_(qz, qs)]
    chol = np.linalg.cholesky(mzz)
    sol = cho_solve((chol, True), np.column_stack([b, d[qz], np.eye(n)]))
    x_mat, x_vec, mzz_inv = sol[:, : len(qs)], sol[:, len(qs)], sol[:, len(qs) + 1 :]
    g_mat = b.T @ x_mat
    e_mat = (excess_ss - g_mat) / 2
    bx = b.T @ x_vec
    shift = d[qs] - bx
    quad = float(d[qz] @ x_vec)
    log_diag = np.log(np.diag(chol))
    log_pz = len(z) * math.log(2.0) - 0.5 * quad - log_diag.sum()
    # First-order rounding bounds (module doc). With M_ZZ + dM = L L^T and the solves exact
    # for M_ZZ + dM', |dM|, |dM'| <= gamma_{3n+1} K, K = |L| |L^T|; B^T M_ZZ^{-1} dM X is
    # X^T dM X, so the Schur complement's solve error is <= gamma X^T K X componentwise.
    k_mat = np.abs(chol) @ np.abs(chol).T
    ax, av, ab = np.abs(x_mat), np.abs(x_vec), np.abs(b)
    g_solve, g_dot = _gamma(3 * n + 1), _gamma(n)
    d_e = (
        0.5 * (g_solve * ax.T @ k_mat @ ax + g_dot * ab.T @ ax + formed)
        + 2 * _U * np.abs(e_mat)  # subtraction (u |2E| / 2) and symmetrization (u |E|)
    )
    d_shift = g_solve * ax.T @ k_mat @ av + g_dot * ab.T @ av + _U * (np.abs(d[qs]) + np.abs(bx))
    eps_z = (
        0.5 * _gamma(n + 1) * float((np.abs(mzz_inv) * k_mat).sum())  # log det: tr(M^-1 dM)/2
        + 0.5 * (g_solve * av @ k_mat @ av + g_dot * np.abs(d[qz]) @ av)  # d^T M^-1 d
        + _gamma(n + 3) * (len(z) * math.log(2.0) + 0.5 * abs(quad) + np.abs(log_diag).sum())
        + _U  # exp
    )
    dvc = 2 * float(np.linalg.norm(d_e, axis=0).sum())
    return (
        math.exp(log_pz),
        (e_mat + e_mat.T) / 2,
        shift,
        float(eps_z),
        dvc,
        float(np.linalg.norm(d_shift)),
    )


def _sum_bound(excess: np.ndarray, shift: np.ndarray, magnitude: float) -> tuple[float, float]:
    """Rounding bound of the alternating sum (module doc) and ``||V_c||_2 = ||I + 2E||_2``.

    Per term, the exponent's rounding is modelled as ``u (sum_i |log1p mu_i| / 2 + q_T)``
    (``mu_i`` the eigenvalues of ``E_T``, ``q_T`` the displacement quadratic form),
    amplified by ``1 + w_T``, plus ``eps |w_T|`` for ``expm1`` and the summation. Interlacing
    gives ``mu_i >= lambda_min(E)``, so ``|log1p mu| <= rho |mu|`` with
    ``rho = log1p(lambda_min) / lambda_min`` (1 if ``lambda_min >= 0``),
    ``sum_i |mu_i| <= tr(|E|_T)`` and ``q_T <= ||e||^2 / (4 (1 + lambda_min))``, and
    ``sum_T tr(|E|_T) = 2^{k-1} tr|E|``, ``sum_T (1 + w_T) y_T <= sum_T y_T + max y sum |w_T|``
    make it a closed form in the kernel's ``sum_T |w_T|``. The calibrated factor
    ``max(1, 16 / sqrt(2^k - 1))`` covers few-click states (module doc).
    """
    eps = float(np.finfo(float).eps)
    if not len(shift):
        return eps * magnitude, 0.0
    k = len(shift) // 2
    lam = np.linalg.eigvalsh(excess)
    low = float(lam[0])
    rho = math.log1p(low) / low if low < 0 else 1.0
    det = 0.5 * rho * float(np.abs(lam).sum())  # >= sum_i |log1p mu_i(E_T)| / 2 for all T
    quad = float(shift @ shift) / (4 * (1 + low))
    model = 2 ** (k - 1) * det + (2**k - 1) * quad + magnitude * (det + quad)
    # Per-term constant up to ~4 (eigensolver error is ~u ||E_T||, not u |mu_i|), partly
    # cancelling over the 2^k - 1 terms: calibrated factor, module doc.
    model *= max(1.0, _SUM_SAFETY / math.sqrt(2.0**k - 1))
    return eps * magnitude + _U * model, max(1 + 2 * float(lam[-1]), 0.0)


def _vacuum_sum(excess: np.ndarray, shift: np.ndarray) -> tuple[float, float]:
    """``[k = 0] + sum_T (-1)^{|T|} w_T`` and ``sum_T |w_T|`` in double precision.

    Raises:
        ValueError: A block ``I + E_T`` is not positive definite (as the native kernel).
    """
    k = len(shift) // 2
    displaced = bool(np.any(shift))
    terms = []
    for mask in range(1, 1 << k):
        t = [i for i in range(k) if mask >> i & 1]
        idx = [q for i in t for q in (2 * i, 2 * i + 1)]
        sub = excess[np.ix_(idx, idx)]
        eig = np.linalg.eigvalsh(sub)
        if eig[0] <= -1.0:
            raise ValueError("Conditioned covariance block is not positive definite")
        exponent = -0.5 * np.log1p(eig).sum()
        if displaced:
            c = 2 * (np.eye(len(idx)) + sub)
            exponent -= 0.5 * shift[idx] @ np.linalg.solve(c, shift[idx])
        w = math.expm1(exponent)
        terms.append(-w if len(t) % 2 else w)
    return float(k == 0) + math.fsum(terms), float(k == 0) + math.fsum(map(abs, terms))


def _vacuum_sum_mp(g: GBSMatrices, clicks: np.ndarray, digits: int) -> tuple[float, float]:
    """``P(S)`` and its absolute rounding estimate in ``digits``-digit mpmath arithmetic."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ImportError(
            "threshold_probability(precision=...) needs mpmath: "
            "pip install 'photographiq[precision]'"
        ) from exc

    with mp.workdps(digits):
        m = g.modes
        s, z = np.flatnonzero(clicks), np.flatnonzero(clicks == 0)
        qs = [int(q) for i in s for q in (2 * i, 2 * i + 1)]
        qz = [int(q) for i in z for q in (2 * i, 2 * i + 1)]
        big = mp.matrix(g.covariance.tolist()) + mp.eye(2 * m)
        d = mp.matrix(g.mean.tolist())

        def sub(rows, cols):
            return mp.matrix([[big[a, b] for b in cols] for a in rows])

        def vec(rows):
            return mp.matrix([d[a] for a in rows])

        if not qs:
            quad = (d.T * mp.inverse(big) * d)[0] if m else mp.mpf(0)
            pz = mp.mpf(2) ** m * mp.exp(-quad / 2) / mp.sqrt(mp.det(big))
            return float(pz), float(pz * mp.eps)
        if qz:
            mzz_inv = mp.inverse(sub(qz, qz))
            schur = sub(qs, qs) - sub(qs, qz) * mzz_inv * sub(qz, qs)
            shift = vec(qs) - sub(qs, qz) * mzz_inv * vec(qz)
            quad = (vec(qz).T * mzz_inv * vec(qz))[0]
            pz = mp.mpf(2) ** len(z) * mp.exp(-quad / 2) / mp.sqrt(mp.det(sub(qz, qz)))
        else:
            schur, shift, pz = sub(qs, qs), vec(qs), mp.mpf(1)
        terms = []
        for mask in range(1 << len(s)):
            t = [i for i in range(len(s)) if mask >> i & 1]
            idx = [q for i in t for q in (2 * i, 2 * i + 1)]
            if not idx:
                terms.append(mp.mpf(1))
                continue
            c = mp.matrix([[schur[a, b] for b in idx] for a in idx])
            e = mp.matrix([shift[a] for a in idx])
            quad = (e.T * mp.inverse(c) * e)[0]
            value = mp.mpf(2) ** len(t) * mp.exp(-quad / 2) / mp.sqrt(mp.det(c))
            terms.append(-value if len(t) % 2 else value)
        total = pz * mp.fsum(terms)
        spread = pz * mp.fsum(abs(t) for t in terms) * mp.eps
        return float(total), float(spread)


def _threshold_double(g: GBSMatrices, clicks: np.ndarray, kernels) -> tuple[float, float]:
    """Double-precision ``P(S)`` and its absolute rounding bound ``B`` (module doc)."""
    pz, excess, shift, eps_z, dvc, de = _dark_conditioned(g, clicks)
    if kernels is not None:
        total, magnitude = kernels.threshold_vacuum_sum(excess, shift)
    else:
        total, magnitude = _vacuum_sum(excess, shift)
    value = pz * total
    summed, vc_norm = _sum_bound(excess, shift, magnitude)
    conditioning = (1 + math.sqrt(3)) / 8 * vc_norm * dvc + 0.5 * math.sqrt(vc_norm) * de
    return value, pz * (summed + conditioning) + abs(value) * (eps_z + _U)


def threshold_probability(
    state: GaussianState | GBSMatrices,
    clicks: Sequence[int],
    *,
    backend: Backend = "auto",
    precision: int | None = None,
    return_error: bool = False,
) -> float | tuple[float, float]:
    """Probability of the threshold-detector pattern ``clicks`` (0 = no click, 1 = click).

    Args:
        state: Gaussian state or its precomputed :class:`GBSMatrices`.
        clicks: One 0/1 entry per mode.
        backend: ``"native"``, ``"python"`` or ``"auto"`` for the double-precision sum.
        precision: Decimal digits (>= 16) of an mpmath evaluation instead of double
            precision (pure Python and slow: ``2^clicks`` multiprecision determinants).
            The covariance and mean are taken exactly as stored in double precision.
        return_error: Also return the estimated relative rounding error: the absolute
            bound ``B`` of the module doc divided by ``|value|`` (``inf`` for ``value = 0``).
            In double precision ``B`` covers the alternating sum and the dark-mode
            conditioning. With ``precision`` it is ``P_Z mp.eps sum_T |1 + w_T|``
            (``mp.eps`` about ``10^-precision``).

    Warns:
        ThresholdPrecisionWarning: The value is negative, is consistent with zero
            (``|value| <= B``), or has estimated relative rounding error above ``1e-6``.
            A value that is negative beyond ``B`` means that the stored covariance and mean
            are not a physical state to working accuracy (e.g. a covariance that rounding
            pushed below the vacuum).
    """
    g = state if isinstance(state, GBSMatrices) else gbs_matrices(state)
    c = _pattern(clicks, g.modes)
    if (c > 1).any():
        raise ValueError("Threshold patterns contain only 0 and 1")
    kernels = _kernels(backend)  # validates backend on every path
    if precision is not None:
        if isinstance(precision, bool) or not isinstance(precision, int) or precision < 16:
            raise ValueError("precision is a number of decimal digits >= 16")
        value, spread = _vacuum_sum_mp(g, c, precision)
    else:
        value, spread = _threshold_double(g, c, kernels)
    error = spread / abs(value) if value else math.inf
    where = f"alternating inclusion-exclusion sum over {int(c.sum())} clicked modes"
    hint = "pass precision=<digits> for a multiprecision evaluation"
    message = None
    if value < 0 and -value > spread:
        message = (
            f"Threshold probability {value:.3e} is negative beyond its absolute rounding bound "
            f"{spread:.1e} ({where}): the stored covariance and mean are not a physical state "
            "to this accuracy (e.g. a covariance below the vacuum by rounding of its entries)"
        )
    elif abs(value) <= spread:
        message = (
            f"Threshold probability {value:.3e} is consistent with zero within its absolute "
            f"rounding bound {spread:.1e} ({where}); {hint}"
        )
    elif error > _THRESHOLD_WARN:
        message = (
            f"Threshold probability {value:.3e} has estimated relative rounding error "
            f"{error:.1e} ({where}); {hint}"
        )
    if message is not None:
        warnings.warn(message, ThresholdPrecisionWarning, stacklevel=2)
    return (value, error) if return_error else value


def threshold_distribution(
    state: GaussianState, *, backend: Backend = "auto"
) -> dict[tuple[int, ...], float]:
    """Exact probabilities of all ``2^m`` click patterns."""
    g = gbs_matrices(state)
    return {
        c: cast(float, threshold_probability(g, c, backend=backend))
        for c in itertools.product((0, 1), repeat=g.modes)
    }


def apply_loss(state: GaussianState, transmissivity: float | Sequence[float]) -> GaussianState:
    """Pure-loss channel per mode: ``V -> eta V + (1-eta) I`` (on the 2x2 blocks), ``d -> sqrt(eta) d``."""
    m = len(state.nodes)
    eta = np.broadcast_to(np.asarray(transmissivity, dtype=float), (m,))
    if not np.all((eta >= 0) & (eta <= 1)):
        raise ValueError("Transmissivities must lie in [0, 1]")
    scale = np.repeat(np.sqrt(eta), 2)
    cov = scale[:, None] * state.covariance * scale + np.diag(np.repeat(1 - eta, 2))
    return GaussianState(scale * state.mean, (cov + cov.T) / 2, state.nodes)


def squeezed_interferometer_state(
    squeezing: Sequence[float],
    unitary,
    transmissivity: float | Sequence[float] = 1.0,
    displacement: Sequence[complex] | None = None,
) -> GaussianState:
    """GBS input state: ``Squeeze(r_i)`` on vacuum, interferometer ``U``, loss, displacement.

    ``Squeeze(r)`` is the PhotoGraphiQ gate ``S = diag(e^{-r}, e^{r})`` (``r > 0`` squeezes
    q); ``U`` acts as ``a -> U a``; then per-mode loss ``transmissivity``; finally the
    coherent displacement ``alpha_i`` adds ``(2 Re alpha_i, 2 Im alpha_i)`` to the mean.
    Mode labels are ``0..m-1``.
    """
    r = np.asarray(squeezing, dtype=float)
    s_u = passive_symplectic(unitary)
    m = len(r)
    if r.ndim != 1 or s_u.shape != (2 * m, 2 * m):
        raise ValueError("Supply one squeezing value per interferometer mode")
    s = s_u @ np.diag(np.ravel([np.diag(gm.squeezing(x)) for x in r]))
    state = apply_loss(GaussianState(np.zeros(2 * m), s @ s.T, tuple(range(m))), transmissivity)
    if displacement is None:
        return state
    alpha = np.asarray(displacement, dtype=complex)
    if alpha.shape != (m,):
        raise ValueError("Supply one displacement per mode")
    shift = np.ravel(np.column_stack([2 * alpha.real, 2 * alpha.imag]))
    return GaussianState(state.mean + shift, state.covariance, state.nodes)


@dataclass(frozen=True)
class PNRSamples:
    """Exact PNR samples; rows of ``-1`` mark shots that fell in the truncated mass."""

    samples: np.ndarray
    cutoff: int
    truncated_mass: float

    @property
    def overflow(self) -> int:
        return int((self.samples[:, 0] < 0).sum()) if len(self.samples) else 0


def sample_pnr(
    state: GaussianState,
    shots: int,
    cutoff: int,
    *,
    rng: np.random.Generator | int | None = None,
    backend: Backend = "auto",
) -> PNRSamples:
    """Exact sampling from the enumerated PNR distribution below ``cutoff``.

    The truncated mass is an explicit outcome: a shot lands there with probability
    ``truncated_mass`` and is returned as a row of ``-1`` (it is not resampled).
    Probabilities (and the truncated mass) that rounding makes negative by at most
    ``1e-12`` times the total mass are clipped to zero before sampling; a larger negative
    raises ``ValueError`` (see ``_sampling_weights``).
    """
    dist = pnr_distribution(state, cutoff, backend=backend)
    patterns = np.array(list(dist.probabilities), dtype=np.int64)
    p = np.append(np.fromiter(dist.probabilities.values(), float), dist.truncated_mass)
    rows = np.vstack([patterns, -np.ones((1, patterns.shape[1]), dtype=np.int64)])
    choice = np.random.default_rng(rng).choice(len(p), size=shots, p=_sampling_weights(p))
    return PNRSamples(rows[choice], dist.cutoff, dist.truncated_mass)


def sample_threshold(
    state: GaussianState,
    shots: int,
    *,
    rng: np.random.Generator | int | None = None,
    backend: Backend = "auto",
) -> np.ndarray:
    """Exact threshold-detector samples from all ``2^m`` click probabilities.

    Probabilities that rounding makes negative by at most ``1e-12`` times the total mass
    are clipped to zero before sampling (``threshold_probability`` still warns about each);
    a larger negative raises ``ValueError`` (see ``_sampling_weights``).
    """
    dist = threshold_distribution(state, backend=backend)
    patterns = np.array(list(dist), dtype=np.int64)
    p = _sampling_weights(np.fromiter(dist.values(), float))
    choice = np.random.default_rng(rng).choice(len(p), size=shots, p=p)
    return patterns[choice]


# Largest negative probability, relative to the total mass, that the samplers treat as
# rounding. The largest absolute rounding error of a single probability seen in the
# test-suite is 2.9e-15 (16-photon PNR), and the largest threshold rounding bound in the
# reference cases is 4.5e-14, so 1e-12 leaves a > 20x margin. Clipping changes the sampled
# distribution by at most 1e-12 per clipped outcome in total variation.
_CLIP_RTOL = 1e-12


def _sampling_weights(probabilities: np.ndarray) -> np.ndarray:
    """Normalized sampling weights; clip rounding-level negatives, raise for larger ones."""
    scale = float(np.abs(probabilities).sum())
    worst = float(probabilities.min(initial=0.0))
    if worst < -_CLIP_RTOL * scale:
        raise ValueError(
            f"Probability {worst:.3e} is negative beyond rounding "
            f"({_CLIP_RTOL:g} x total mass {scale:.3e}): the state is not physical or the "
            "evaluation lost accuracy"
        )
    p = np.clip(probabilities, 0.0, None)
    return p / p.sum()
