"""Independent finite-space references; no production numerical helpers."""

from itertools import product
from math import factorial

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermitenorm, gammaln


def basis(modes, cutoff):
    return tuple(
        sorted(
            (b for b in product(range(cutoff), repeat=modes) if sum(b) < cutoff),
            key=lambda b: (sum(b), tuple(-x for x in b)),
        )
    )


def ladder(occupations, mode):
    lookup = {b: i for i, b in enumerate(occupations)}
    a = np.zeros((len(occupations), len(occupations)), complex)
    for j, b in enumerate(occupations):
        if b[mode]:
            target = list(b)
            target[mode] -= 1
            a[lookup[tuple(target)], j] = np.sqrt(b[mode])
    return a


def gate(occupations, name, value):
    a = ladder(occupations, 0)
    q, p = a + a.conj().T, -1j * (a - a.conj().T)
    n = np.diag([b[0] for b in occupations])
    if name == "Rotate":
        g = 1j * value * n
    elif name == "Displace":
        g = 0.5j * (value[1] * q - value[0] * p)
    elif name == "Squeeze":
        g = value * (a @ a - a.conj().T @ a.conj().T) / 2
    elif name == "QuadraticPhase":
        g = 0.25j * value * (q @ q)
    elif name == "CubicPhase":
        g = 1j * value * (q @ q @ q) / 6
    elif name == "Kerr":
        g = 1j * value * (n @ n)
    else:
        b = ladder(occupations, 1)
        if name == "BeamSplitter":
            g = value * (b.conj().T @ a - a.conj().T @ b)
        elif name == "Entangle":
            x = b + b.conj().T
            g = 0.25j * value * (q @ x + x @ q)
        else:
            raise ValueError(name)
    return expm(g)


def vector(kind, cutoff):
    v = np.zeros(cutoff, complex)
    if kind in ("vacuum", "one"):
        v[int(kind == "one")] = 1
    elif kind == "superposition":
        v[0], v[2] = 2**-0.5, 2**-0.5
    else:
        alpha = 0.45 + 0.2j
        v = np.array(
            [
                np.exp(-(abs(alpha) ** 2) / 2) * alpha**n / np.sqrt(float(factorial(n)))
                for n in range(cutoff)
            ]
        )
        if kind == "cat":
            v *= 1 + (-1.0) ** np.arange(cutoff)
    return v / np.linalg.norm(v)


def hermites(q, cutoff):
    n = np.arange(cutoff)
    return (
        np.exp(-(np.asarray(q)[..., None] ** 2) / 4 - gammaln(n + 1) / 2)
        * eval_hermitenorm(n, np.asarray(q)[..., None])
        / (2 * np.pi) ** 0.25
    )


def physical(rho, tol=2e-12):
    np.testing.assert_allclose(rho, rho.conj().T, atol=tol, rtol=0)
    np.testing.assert_allclose(np.trace(rho), 1, atol=tol, rtol=0)
    assert np.linalg.eigvalsh(rho).min() >= -tol


def circuit_unitary(circuit, cutoff):
    """Evaluate emitted ideal single-mode primitives, caching repeated pulses."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q = a + a.T
    values, vectors = np.linalg.eigh(q)
    total = np.eye(cutoff, dtype=complex)
    cache = {}
    for name, _, params in circuit.gates:
        key = (name, repr(params))
        if key not in cache:
            if name == "cubic_phase":
                u = (vectors * np.exp(1j * params[0] * values**3 / 6)) @ vectors.T
            elif name == "symplectic":
                matrix = params[0]
                if np.allclose(matrix[0], [1, 0], atol=1e-14, rtol=0):
                    u = (vectors * np.exp(0.25j * matrix[1, 0] * values**2)) @ vectors.T
                else:
                    angle = np.arctan2(matrix[1, 0], matrix[0, 0])
                    u = np.diag(np.exp(1j * angle * np.arange(cutoff)))
            elif name == "displace":
                u = gate(basis(1, cutoff), "Displace", params)
            else:
                raise ValueError(name)
            cache[key] = u
        total = cache[key] @ total
    return total


def synthesized_unitary(kind, strength, steps, cutoff, angle=0.0):
    import photographiq as pg

    # Identical slices: evaluate one emitted slice and take its matrix power.
    circuit, _ = (
        pg.synthesis.synthesize_kerr(strength / steps, steps=1)
        if kind == "kerr"
        else pg.synthesis.quadrature_polynomial([(strength / steps, angle, 4)], steps=1)
    )
    return np.linalg.matrix_power(circuit_unitary(circuit, cutoff), steps)


def synthesis_observation(kind, strength, steps, cutoff, source, angle=0.0):
    v = vector(source, cutoff)
    actual = synthesized_unitary(kind, strength, steps, cutoff, angle) @ v
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    x = np.exp(-1j * angle) * a + np.exp(1j * angle) * a.T
    expected = (
        np.exp(1j * strength * np.arange(cutoff) ** 2) * v
        if kind == "kerr"
        else expm(1j * strength * np.linalg.matrix_power(x, 4)) @ v
    )
    overlap = np.vdot(expected, actual)
    distance = np.linalg.norm(actual * np.exp(-1j * np.angle(overlap)) - expected)
    return {
        "kind": kind,
        "strength": strength,
        "steps": steps,
        "cutoff": cutoff,
        "source": source,
        "angle": angle,
        "infidelity": max(0.0, float(1 - abs(overlap) ** 2)),
        "amplitude_error": float(distance),
        "boundary_population": float(np.sum(abs(actual[-2:]) ** 2)),
    }
