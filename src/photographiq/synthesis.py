"""Experimental product-formula synthesis over Gaussian and cubic primitives.

Finite-energy injection, finite-Fock projection and product-formula error are
separate limits. No returned step count is an a priori operator-norm guarantee.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SynthesisReport:
    """Algorithm and resource count for a synthesized finite circuit."""

    method: str
    steps: int
    primitive_count: int
    warning: str = "Refine synthesis steps, resource squeezing and cutoff independently."
    approximate: bool = True
    order: float | None = None
    target_gate: str = "quadrature_polynomial"
    target_parameter: object = None


def _power(circuit, mode, angle, degree, strength):
    # R(-theta) followed by a q gate followed by R(theta) implements x_theta.
    circuit.rotate(mode, -angle)
    if degree == 1:
        circuit.displace(mode, p=2 * strength)
    elif degree == 2:
        circuit.gaussian(mode, np.array([[1.0, 0.0], [4 * strength, 1.0]]))
    elif degree == 3:
        circuit.cubic_phase(mode, 6 * strength)
    circuit.rotate(mode, angle)


def _q4(circuit, mode, strength):
    # [q^3, Weyl(q^2 p)] = 6 i q^4. The ordered group commutator
    # exp(-i t B)exp(-i s A)exp(i t B)exp(i s A) = exp(6 i s t q^4)+O(t^3).
    s = np.sqrt(abs(strength) / 6)
    t = np.copysign(s, strength)

    def a(value):
        _power(circuit, mode, 0.0, 3, value)

    def b(value):
        # B = ((q+p)^3 - (q-p)^3 - 2 p^3)/6, Weyl ordered.
        terms = [(np.pi / 4, 2**1.5 / 6), (-np.pi / 4, -(2**1.5) / 6), (np.pi / 2, -1 / 3)]
        # Symmetric splitting retains inverse consistency and O(t^3) error.
        for angle, coefficient in terms + terms[::-1]:
            _power(circuit, mode, angle, 3, value * coefficient / 2)

    a(s)
    b(t)
    a(-s)
    b(-t)


def quadrature_polynomial(terms, *, steps=8, modes=1, mode=0):
    """Synthesize ``exp(i sum(c*x_theta**degree))`` for degrees 1 through 4.

    Args:
        terms (object): Iterable of ``(coefficient, angle, degree)`` in hbar=2 units.
        steps (object): Positive product-formula refinement count.
        modes (int): Number of optical modes.
        mode (int): Target mode.

    Returns:
        result (tuple): Circuit and SynthesisReport; quartics use cubic commutators.

    Raises:
        ValueError: Nonfinite terms, unsupported degree, or invalid step count.

    Example:
        ``circuit, report = quadrature_polynomial([(0.01, 0., 4)], steps=16)``.
    """
    from .compiler import Circuit

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    terms = tuple(terms)
    if any(
        not np.isfinite([c, angle]).all() or isinstance(d, bool) or d not in (1, 2, 3, 4)
        for c, angle, d in terms
    ):
        raise ValueError("Terms require finite coefficient/angle and degree 1, 2, 3, or 4")
    circuit = Circuit(modes)
    for _ in range(steps):
        for coefficient, angle, degree in terms:
            if coefficient == 0:
                continue
            strength = coefficient / steps
            if degree == 4:
                circuit.rotate(mode, -angle)
                _q4(circuit, mode, strength)
                circuit.rotate(mode, angle)
            else:
                _power(circuit, mode, angle, degree, strength)
    active = [(c, angle, degree) for c, angle, degree in terms if c != 0]
    quartic = any(degree == 4 for _, _, degree in active)
    approximate = quartic or len(active) > 1
    return circuit, SynthesisReport(
        "Lie product formula with cubic commutators"
        if quartic
        else "Lie product formula"
        if approximate
        else "Exact quadrature power",
        steps,
        len(circuit.gates),
        approximate=approximate,
        order=0.5 if quartic else 1.0 if approximate else None,
        target_parameter=terms,
    )


def synthesize_kerr(kappa, *, steps=8, modes=1, mode=0):
    """Approximate exp(i*kappa*n^2) using quadrature quartics and quadratics.

    In Weyl ordering at hbar=2, n² = sum(x_theta^4)/24 - (q²+p²)/4,
    with theta=0, pi/2, pi/4, -pi/4. Refinement is required.
    """
    terms = [(kappa / 24, theta, 4) for theta in (0, np.pi / 2, np.pi / 4, -np.pi / 4)]
    terms += [(-kappa / 4, theta, 2) for theta in (0, np.pi / 2)]
    circuit, report = quadrature_polynomial(terms, steps=steps, modes=modes, mode=mode)
    return circuit, replace(report, target_gate="Kerr", target_parameter=kappa)
