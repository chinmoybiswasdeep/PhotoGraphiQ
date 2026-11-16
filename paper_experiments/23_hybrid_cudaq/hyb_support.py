"""Shared helpers for HYB-01..HYB-12 (hybrid Gaussian-Fock CV-MBQC, CUDA-Q Dynamics adapter).

Reuses the R61-R68 conventions (``cvzx_support.save``): CSV + JSON rows, environment metadata,
figure (PDF/PNG/SVG) and caption; the JSON carries ``configuration``, ``validation_report``,
``source_revision`` and ``status`` (pass / fail / blocked).
"""

import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "22_cv_diagrams"))

from cvzx_support import Timer, common, plt, revision, save  # noqa: E402,F401

import photographiq as pg  # noqa: E402

COLORS = {
    "hybrid": "#c66d27",
    "native": "#24677b",
    "gaussian": "#5b4b8a",
    "piquasso": "#627a36",
    "reference": "#8c8c8c",
    "accent": "#b23a48",
}


def infidelity(a, b):
    """1 - |<a|b>|^2 for (zero-padded) pure states."""
    n = max(len(a), len(b))
    A, B = np.zeros(n, complex), np.zeros(n, complex)
    A[: len(a)], B[: len(b)] = a, b
    return float(max(0.0, 1 - abs(np.vdot(A, B)) ** 2 / np.vdot(A, A).real / np.vdot(B, B).real))


def trace_distance(a, b):
    """Trace distance of (zero-padded) density matrices; vectors are promoted."""
    a = np.outer(a, a.conj()) if np.ndim(a) == 1 else np.asarray(a)
    b = np.outer(b, b.conj()) if np.ndim(b) == 1 else np.asarray(b)
    n = max(len(a), len(b))
    A, B = np.zeros((n, n), complex), np.zeros((n, n), complex)
    A[: len(a), : len(a)], B[: len(b), : len(b)] = a, b
    return float(0.5 * np.abs(np.linalg.eigvalsh(A - B)).sum())


def simulate(pattern, **kwargs):
    """pg.simulate with retained-norm warnings silenced (cutoff sweeps record them explicitly)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pg.simulate(pattern, **kwargs)


def wire(theta=1.1, r=0.6, losses=()):
    """One-step CZ teleportation with correction q -> -m / sin(theta); optional (node, eta) losses."""
    p = pg.Pattern(inputs=(0,))
    p.extend([pg.Prepare(1, r), pg.Entangle(0, 1)] + [pg.Loss(n, eta) for n, eta in losses])
    p.measure(0, pg.Homodyne(theta))
    p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
    return p.append(pg.Output((1,))).validate()


def three_mode_cubic(gamma=0.15, r=0.4, theta=1.1):
    """Primary acceptance pattern: wire step + cubic injection (three live modes, two homodynes,
    quadratic and nonlinear feed-forward QP(-2 gamma m), Z(-gamma m^2))."""
    p = pg.Pattern(inputs=(0,))
    p.extend(
        [pg.Prepare(1, r), pg.Prepare(2, state=pg.CubicPhaseResource(gamma, r)), pg.Entangle(0, 1)]
    )
    p.measure(0, pg.Homodyne(theta))
    p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
    p.extend([pg.Rotate(2, np.pi / 2), pg.Entangle(1, 2, -1.0), pg.Rotate(2, -np.pi / 2)])
    p.append(pg.Measure(2, pg.Homodyne(0.0), "m"))
    p.extend(
        [
            pg.QuadraticPhase(1, -2 * gamma * pg.Outcome("m")),
            pg.Displace(1, p=-gamma * pg.Outcome("m") ** 2),
        ]
    )
    return p.append(pg.Output((1,))).validate()


def kerr_wire(kappa=0.1, theta=0.5, losses=()):
    """Kerr-enhanced MBQC step: K(kappa) on the input before the homodyne (non-Gaussian basis)."""
    p = pg.Pattern(inputs=(0,))
    p.extend(
        [pg.Prepare(1, 0.4), pg.Entangle(0, 1), pg.Kerr(0, kappa)]
        + [pg.Loss(n, eta) for n, eta in losses]
    )
    p.measure(0, pg.Homodyne(theta))
    p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
    return p.append(pg.Output((1,))).validate()
