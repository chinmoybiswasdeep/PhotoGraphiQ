"""Finite-energy square-lattice GKP resources and elementary shift decoding.

At [q,p]=2i the logical shift spacing is sqrt(2*pi), and stabilizer translations
are twice that spacing. This module does not establish a fault-tolerance threshold.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import expm

from .fock_measurements import wavefunctions
from .states import FockInput

SPACING = float(np.sqrt(2 * np.pi))


def logical_displacement(node, pauli="X"):
    """Return a logical lattice X or Z displacement command, up to global phase.

    X translates q by sqrt(2*pi); Z translates p by that amount. A finite-energy
    envelope is also displaced, so these are not exact finite-codeword permutations.
    """
    from .commands import Displace

    if pauli not in ("X", "Z"):
        raise ValueError("pauli must be X or Z")
    return Displace(node, q=SPACING if pauli == "X" else 0.0, p=SPACING if pauli == "Z" else 0.0)


@dataclass(frozen=True)
class GKPResource:
    """Finite Gaussian-comb logical state projected into a Fock basis.

    Args:
        logical (object): Logical basis bit (0 or 1).
        peak_width (object): Standard deviation of each isolated peak's probability.
        envelope (object): Positive inverse envelope width on peak centers.
        peaks (object): Number of lattice cells on either side of the origin.
        grid_points (object): Odd quadrature grid size, independently converged.

    Raises:
        ValueError: Invalid parameters or an unresolved/truncated resource.

    Example:
        ``state = GKPResource(0, peak_width=0.4, envelope=0.4).fock(48)``.
    """

    logical: int = 0
    peak_width: float = 0.4
    envelope: float = 0.4
    peaks: int = 8
    grid_points: int = 4097

    def __post_init__(self):
        if isinstance(self.logical, bool) or self.logical not in (0, 1):
            raise ValueError("logical must be 0 or 1")
        if (
            not np.isfinite([self.peak_width, self.envelope]).all()
            or min(self.peak_width, self.envelope) <= 0
        ):
            raise ValueError("GKP peak_width and envelope must be finite and positive")
        if isinstance(self.peaks, bool) or not isinstance(self.peaks, int) or self.peaks < 1:
            raise ValueError("peaks must be a positive integer")
        if (
            isinstance(self.grid_points, bool)
            or not isinstance(self.grid_points, int)
            or self.grid_points < 129
            or self.grid_points % 2 == 0
        ):
            raise ValueError("grid_points must be an odd integer >= 129")

    def wavefunction(self):
        """Return normalized real wavefunction and its explicit q grid."""
        centers = SPACING * (2 * np.arange(-self.peaks, self.peaks + 1) + self.logical)
        bound = max(abs(centers)) + 10 * self.peak_width
        q = np.linspace(-bound, bound, self.grid_points)
        if q[1] - q[0] > self.peak_width / 4:
            raise ValueError("GKP peaks are unresolved; increase grid_points")
        values = np.sum(
            np.exp(
                -((q[:, None] - centers) ** 2) / (4 * self.peak_width**2)
                - centers**2 * self.envelope**2 / 4
            ),
            axis=1,
        )
        return q, values / np.sqrt(simpson(values**2, x=q))

    def project(self, cutoff):
        """Return normalized FockInput and pre-normalization captured weight.

        Captured weight diagnoses Fock projection only. Refine peaks and
        grid_points separately; a value near one is not a grid certificate.
        """
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 2:
            raise ValueError("cutoff must be an integer >= 2")
        q, psi = self.wavefunction()
        basis = np.array([wavefunctions(x, cutoff) for x in q])
        vector = simpson(psi[:, None] * basis, x=q, axis=0)
        mass = float(np.vdot(vector, vector).real)
        if not np.isfinite(mass) or mass <= 1e-14 or mass > 1 + 1e-6:
            raise ValueError("Unresolved GKP projection; refine quadrature grid/cutoff")
        return FockInput(tuple(vector / np.sqrt(mass))), mass

    def fock(self, cutoff):
        """Return the normalized finite-Fock projection; inspect project() for mass."""
        return self.project(cutoff)[0]


def superposition(alpha, beta, *, cutoff, **resource_options):
    """Normalize alpha*|0_GKP> + beta*|1_GKP>, retaining finite-state overlap."""
    zero = np.asarray(GKPResource(0, **resource_options).fock(cutoff).amplitudes)
    one = np.asarray(GKPResource(1, **resource_options).fock(cutoff).amplitudes)
    vector = alpha * zero + beta * one
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError("GKP superposition must have finite nonzero norm")
    return FockInput(tuple(vector / norm))


def decode_shift(outcome):
    """Return (nearest-lattice residual, parity) for a real quadrature outcome.

    Exact half-cell ties map to the upper cell. Residual lies in [-L/2,L/2).
    This classical nearest-cell rule is not an optimal finite-energy decoder.
    """
    if not np.isfinite(outcome):
        raise ValueError("Syndrome outcome must be finite")
    # Compare in the original coordinate: dividing a rounded half-cell value
    # can round just below the tie, incorrectly changing the logical parity.
    cell = int(np.floor(outcome / SPACING))
    if outcome >= (cell + 0.5) * SPACING:
        cell += 1
    return float(outcome - cell * SPACING), cell % 2


def stabilizers(state, node=None):
    """Estimate square-lattice stabilizer expectations in a reduced Fock state.

    Displacement matrices are evaluated with padded ladder operators; refine
    Fock cutoff independently, particularly for sharp finite-energy grids.
    """
    if node is None:
        if len(state.nodes) != 1:
            raise ValueError("Choose one mode for GKP stabilizers")
        node = state.nodes[0]
    rho = state.reduced((node,)).density_matrix
    cutoff = len(rho)
    a = np.diag(np.sqrt(np.arange(1, 2 * cutoff + 32)), 1)
    q, p = a + a.T, -1j * (a - a.T)
    return {
        "q_translation": complex(np.trace(rho @ expm(-1j * SPACING * p)[:cutoff, :cutoff])),
        "p_translation": complex(np.trace(rho @ expm(1j * SPACING * q)[:cutoff, :cutoff])),
    }


def correction_pattern(
    *, quadrature="q", resource=None, input_node="in", ancilla="gkp", key="syndrome"
):
    """One finite-ancilla SUM syndrome extraction with modular feed-forward.

    q extraction uses a logical-plus ancilla (all lattice cells). p extraction
    Fourier-conjugates the complete q protocol. The output remains on input_node.
    """
    from .commands import Displace, Entangle, Measure, Output, Prepare, Rotate
    from .expressions import CallableExpression
    from .measurements import Homodyne
    from .pattern import Pattern

    if quadrature not in ("q", "p"):
        raise ValueError("quadrature must be 'q' or 'p'")
    if resource is None:
        raise ValueError("Supply a finite logical-plus FockInput ancilla and test its cutoff")
    pattern = Pattern(inputs=(input_node,))
    if quadrature == "p":
        pattern.append(Rotate(input_node, -np.pi / 2))
    pattern.extend(
        [
            Prepare(ancilla, state=resource),
            Rotate(ancilla, np.pi / 2),
            Entangle(input_node, ancilla, 1.0),
            Rotate(ancilla, -np.pi / 2),
            Measure(ancilla, Homodyne.q(), key),
        ]
    )
    pattern.append(
        Displace(
            input_node,
            q=CallableExpression(lambda records: -decode_shift(records[key])[0], frozenset({key})),
        )
    )
    if quadrature == "p":
        pattern.append(Rotate(input_node, np.pi / 2))
    return pattern.append(Output((input_node,))).validate()
