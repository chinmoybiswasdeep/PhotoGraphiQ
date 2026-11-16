"""Untruncated position-grid reference for the physical GKP logical-XY instrument.

Independent of the Fock engines and of ``photographiq.encoded``: finite-energy codewords are
written down from their definition on a position grid (no Fock projection), the logical CNOT
is the exact SUM q_a -> q_a + q_d, i.e. psi(x_d, x_a) -> psi(x_d, x_a - x_d), the logical S is
multiplication by exp(i x^2/4) (hbar = 2), the ancilla q homodyne with nearest-cell decoding
is integrated exactly over its decoded cells, and the final X readout integrates the
p-quadrature density over decoded cells. Every repeat-until-success branch is enumerated.

Because the ancilla is measured right after the SUM, the coarse-grained (decoded-bit) data
operator after one round is a Hadamard product O(x, x') -> O(x, x') K_m(x, x') with
K_m(x, x') = int_{cells of parity m} a(y - x) conj(a(y - x')) dy, so the whole protocol is
exact linear algebra on N x N matrices. The map is linear in O, so non-Hermitian codeword
operators |i~><j~| give the instrument's action on entangled inputs (measurement back-action).

What this reference shares with the system under test: the definition of the finite-energy
codewords (sum of Gaussians with an envelope, ``peaks`` cells each side), the protocol's
classical logic and the nearest-cell decoding rule. Everything numerical is separate.
"""

from __future__ import annotations

import numpy as np

SPACING = float(np.sqrt(2 * np.pi))
TOL = 1e-12


def decode_parity(values):
    """Nearest-cell parity with upper-cell ties (the documented decoder rule), vectorised."""
    cells = np.floor(values / SPACING)
    cells = np.where(values >= (cells + 0.5) * SPACING, cells + 1, cells)
    return (cells.astype(int)) % 2


def even_cell_fraction(centers, width):
    """Fraction of [c - w/2, c + w/2] lying in even-parity decoding cells (exact).

    Even cells are [(2k - 1/2) L, (2k + 1/2) L). Integrating the cell indicator exactly over
    each grid cell (instead of sampling it) makes cell integrals second-order accurate.
    """

    def measure(z):  # length of even cells within (-inf, z], up to a constant
        period = np.floor((z + SPACING / 2) / (2 * SPACING))
        return period * SPACING + np.clip(z + SPACING / 2 - 2 * SPACING * period, 0, SPACING)

    return (measure(centers + width / 2) - measure(centers - width / 2)) / width


def quarter_turns(angle, tolerance=TOL):
    reduced = float(np.remainder(angle, 2 * np.pi))
    k = int(np.rint(reduced / (np.pi / 2)))
    return k % 4 if abs(reduced - k * np.pi / 2) <= tolerance else None


class GridModel:
    """Finite-energy square GKP on an untruncated position grid (hbar = 2)."""

    def __init__(self, peak_width, envelope, peaks, dx=0.07):
        self.peak_width, self.envelope, self.peaks, self.dx = peak_width, envelope, peaks, dx
        reach = SPACING * (2 * peaks + 1) + 10 * peak_width
        self.x = np.arange(-reach, reach + dx / 2, dx)
        # ancilla outcome y = x_a + x_d spans twice the range
        self.y = np.arange(-2 * reach, 2 * reach + dx / 2, dx)
        self.p = np.arange(-reach, reach + dx / 2, dx)  # comb in p has the same envelope scale
        self._norms = {}
        self._kernels = {}
        # <p|x> = exp(-i p x / 2) / sqrt(4 pi) for [q, p] = 2i
        self.fourier = np.exp(-0.5j * np.outer(self.p, self.x)) * dx / np.sqrt(4 * np.pi)

    def _comb(self, z, bit):
        centers = SPACING * (2 * np.arange(-self.peaks, self.peaks + 1) + bit)
        return np.sum(
            np.exp(
                -((np.asarray(z)[..., None] - centers) ** 2) / (4 * self.peak_width**2)
                - centers**2 * self.envelope**2 / 4
            ),
            axis=-1,
        )

    def codeword(self, bit, z=None):
        """Normalized finite-energy codeword evaluated at z (default: the data grid)."""
        if bit not in self._norms:
            fine = np.linspace(self.x[0], self.x[-1], 40 * len(self.x))
            values = self._comb(fine, bit)
            self._norms[bit] = np.sqrt(np.trapezoid(values**2, fine))
        return self._comb(self.x if z is None else z, bit) / self._norms[bit]

    def encode(self, c0, c1, z=None):
        """(c0|0~> + c1|1~>)/norm evaluated at z; the norm keeps the codeword overlap."""
        overlap = np.trapezoid(self.codeword(0, self.x) * self.codeword(1, self.x), self.x)
        norm = np.sqrt(abs(c0) ** 2 + abs(c1) ** 2 + 2 * np.real(np.conj(c0) * c1) * overlap)
        return (c0 * self.codeword(0, z) + c1 * self.codeword(1, z)) / norm

    def magic(self, phase, z=None):
        return self.encode(1.0, np.exp(1j * phase), z)

    def kernels(self, phase):
        """K_0, K_1 for an injection round with magic phase ``phase``."""
        key = round(float(np.remainder(phase, 2 * np.pi)), 13)
        if key not in self._kernels:
            # Both grids share dx and y_0 = 2 x_0, so y_i - x_j = x_0 + (i - j) dx: a Toeplitz
            # matrix. Evaluate a(.) once on the 1-D difference grid and index it.
            offsets = np.arange(-(len(self.x) - 1), len(self.y))
            values = self.magic(phase, self.x[0] + offsets * self.dx)
            values[np.abs(values) < 1e-150] = 0  # subnormal Gaussian tails stall BLAS
            index = np.subtract.outer(np.arange(len(self.y)), np.arange(len(self.x)))
            shifted = values[index + len(self.x) - 1]  # a(y - x)
            conjugate = np.ascontiguousarray(shifted.conj().T)
            even = even_cell_fraction(self.y, self.dx)
            pair = []
            for weight in (even * self.dx, (1 - even) * self.dx):
                pair.append(conjugate @ (weight[:, None] * shifted))
            # K[x, x'] = sum_y a(y-x) conj(a(y-x')) -> transpose of conj(M)^T M
            self._kernels[key] = tuple(k.T for k in pair)
        return self._kernels[key]

    def shear(self, operator, turns):
        """Apply S (turns=1) or S^dagger (turns=3): multiplication by exp(+-i x^2/4)."""
        d = np.exp((1 if turns == 1 else -1) * 0.25j * self.x**2)
        return d[:, None] * operator * d.conj()[None, :]

    def x_readout(self, operator):
        """(trace-weight of raw parity 0, raw parity 1) of the p-homodyne X readout."""
        rho_p = np.sum((self.fourier @ operator) * self.fourier.conj(), axis=1)
        even = even_cell_fraction(self.p, self.dx)
        return complex(np.sum(rho_p * even) * self.dx), complex(
            np.sum(rho_p * (1 - even)) * self.dx
        )

    def z_readout(self, operator):
        """(raw parity 0, raw parity 1) weights of the q-homodyne Z readout."""
        rho_q = np.diag(operator)
        even = even_cell_fraction(self.x, self.dx)
        return complex(np.sum(rho_q * even) * self.dx), complex(
            np.sum(rho_q * (1 - even)) * self.dx
        )

    def bloch(self, operator):
        """Decoded (<X>, <Y>, <Z>) of a one-mode operator through the physical readouts.

        X: p cells; Y: Gaussian S^dagger shear then p cells (the Clifford Y readout);
        Z: q cells. Values are trace weights, so divide by the trace for a density.
        """
        x = self.x_readout(operator)
        y = self.x_readout(self.shear(operator, 3))
        z = self.z_readout(operator)
        return tuple(complex(a - b) for a, b in (x, y, z))

    def operator(self, ket, bra=None):
        bra = ket if bra is None else bra
        return np.outer(ket, bra.conj())

    def protocol_map(self, operator, alpha, *, protocol="auto", max_rounds=8):
        """Enumerate every branch; returns rows {bit, success, rounds, path, value}.

        ``value`` is tr[E_bit (branch map)(operator)], a probability for a density operator.
        """
        rows = []

        def final(o, flip, success, path):
            raw = self.x_readout(o)
            for parity in (0, 1):
                rows.append(
                    {
                        "bit": parity ^ flip,
                        "success": success,
                        "rounds": len(path),
                        "path": tuple(path),
                        "value": raw[parity],
                    }
                )

        def recurse(o, phi, flip, path):
            turns = quarter_turns(phi)
            forced = protocol == "injection" and not path
            if turns is not None and not forced:
                if turns == 2:
                    return final(o, flip ^ 1, True, path)
                if turns == 0:
                    return final(o, flip, True, path)
                if protocol == "clifford" or not path:
                    return final(self.shear(o, turns), flip, True, path)
            elif protocol == "clifford":
                raise NotImplementedError("Clifford protocol reached a non-Clifford angle")
            if len(path) >= max_rounds and turns is None:
                return final(o, flip, False, path)
            k0, k1 = self.kernels(phi)
            recurse(o * k0, 0.0, flip, [*path, 0])
            recurse(o * k1, 2 * phi, flip, [*path, 1])

        recurse(operator, -float(alpha), 0, [])  # measuring M(a) = applying D(-a), then X
        return rows

    def outcome_probabilities(self, state, alpha, **options):
        """(P(bit 0), P(bit 1), P(heralded success)) for a pure data wavefunction on the grid."""
        rows = self.protocol_map(self.operator(state), alpha, **options)
        p = [sum(r["value"].real for r in rows if r["bit"] == b) for b in (0, 1)]
        success = sum(r["value"].real for r in rows if r["success"])
        return p[0], p[1], success

    def record_density(self, state, alpha, ancilla_outcomes, data_outcome, *, protocol="auto"):
        """Joint density of a complete analog record, following the protocol's own branching.

        Returns (joint_density, decoded ancilla bits, final logical bit).
        """
        psi = state.astype(complex)
        density, bits, phi, flip = 1.0, [], -float(alpha), 0
        remaining = list(ancilla_outcomes)
        while True:
            turns = quarter_turns(phi)
            forced = protocol == "injection" and not bits
            if turns is not None and not forced:
                if turns == 2:
                    flip ^= 1
                    break
                if turns == 0:
                    break
                if protocol == "clifford" or not bits:
                    psi = psi * np.exp((1 if turns == 1 else -1) * 0.25j * self.x**2)
                    break
            if not remaining:
                raise ValueError("record has fewer ancilla outcomes than the protocol needs")
            y = remaining.pop(0)
            unnormalized = psi * self.magic(phi, y - self.x)
            step = np.trapezoid(abs(unnormalized) ** 2, self.x) / np.trapezoid(
                abs(psi) ** 2, self.x
            )
            density *= step
            psi = unnormalized
            bit = int(decode_parity(np.array([y]))[0])
            bits.append(bit)
            phi = 0.0 if bit == 0 else 2 * phi
        if remaining:
            raise ValueError("record has more ancilla outcomes than the protocol uses")
        psi = psi / np.sqrt(np.trapezoid(abs(psi) ** 2, self.x))
        amplitude = (
            np.sum(np.exp(-0.5j * data_outcome * self.x) * psi) * self.dx / np.sqrt(4 * np.pi)
        )
        density *= abs(amplitude) ** 2
        return float(density), bits, int(decode_parity(np.array([data_outcome]))[0]) ^ flip


def ideal_probabilities(logical_state, alpha):
    """Ideal qubit M(alpha) = cos(alpha) X + sin(alpha) Y outcome probabilities."""
    plus = np.array([1, np.exp(1j * alpha)]) / np.sqrt(2)
    psi = np.asarray(logical_state, complex)
    p0 = abs(np.vdot(plus, psi)) ** 2 / np.vdot(psi, psi).real
    return float(p0), float(1 - p0)


def ideal_partner_bloch(alpha, bit):
    """Partner Bloch vector after M(alpha) on half of (|00> + |11>)/sqrt(2) gives ``bit``."""
    sign = 1 - 2 * bit
    return sign * np.cos(alpha), -sign * np.sin(alpha), 0.0
