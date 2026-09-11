"""Explicit ideal references and finite-energy square GKP readout primitives."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import quad_vec

from .gkp import SPACING, GKPResource, decode_shift, logical_displacement, stabilizers
from .instruments import MeasurementInstrument, density_matrix
from .measurements import Homodyne
from .states import FockInput


@dataclass(frozen=True)
class LogicalDecodeResult:
    bit: int
    raw_outcome: float
    cell: int
    residual: float
    decoder: str = "nearest-cell"
    confidence: float | None = None
    probabilities: tuple | None = None
    code_subspace_leakage: float | None = None
    frame_correction: int = 0


class BaseGKPDecoder(ABC):
    @abstractmethod
    def decode(self, outcome) -> LogicalDecodeResult: ...


@dataclass(frozen=True)
class NearestCellDecoder(BaseGKPDecoder):
    """Upper-cell half ties; parity is a decision, not a posterior probability."""

    def decode(self, outcome):
        residual, parity = decode_shift(outcome)
        return LogicalDecodeResult(
            parity, float(outcome), round((outcome - residual) / SPACING), residual
        )


@dataclass(frozen=True)
class LogicalPauliFrame:
    """Ideal logical X^x Z^z frame, modulo global phase; no envelope correction."""

    x_bit: int = 0
    z_bit: int = 0

    def __post_init__(self):
        if any(
            not isinstance(b, (int, np.integer)) or b not in (0, 1)
            for b in (self.x_bit, self.z_bit)
        ):
            raise ValueError("Frame entries must be bits")

    def compose(self, other):
        return LogicalPauliFrame(self.x_bit ^ other.x_bit, self.z_bit ^ other.z_bit)

    def hadamard(self):
        return LogicalPauliFrame(self.z_bit, self.x_bit)

    def phase(self):
        return LogicalPauliFrame(self.x_bit, self.z_bit ^ self.x_bit)

    def cz(self, other):
        return (
            LogicalPauliFrame(self.x_bit, self.z_bit ^ other.x_bit),
            LogicalPauliFrame(other.x_bit, other.z_bit ^ self.x_bit),
        )

    def xy_angle(self, alpha):
        return (-1) ** self.x_bit * alpha + np.pi * self.z_bit

    def correction(self, basis):
        if basis not in ("X", "Y", "Z"):
            raise ValueError("Basis must be X, Y or Z")
        return (
            self.z_bit if basis == "X" else self.x_bit if basis == "Z" else self.x_bit ^ self.z_bit
        )


@dataclass(frozen=True)
class PhysicalGKPReadout:
    """Analog ideal-detector homodyne followed by modular X or Z decoding.

    Postselection always specifies the raw analog value, never the decoded bit.
    Leakage cannot be inferred from one analog outcome and is returned as None.
    """

    basis: str = "Z"
    decoder: BaseGKPDecoder = NearestCellDecoder()
    frame: LogicalPauliFrame = LogicalPauliFrame()
    flip: int = 0

    def __post_init__(self):
        self.validate()

    @property
    def required_capabilities(self):
        return frozenset({"encoded_gkp_xz", "homodyne"})

    def validate(self):
        if self.basis not in ("X", "Z"):
            raise NotImplementedError(
                "Only physical GKP X/Z readout is implemented; Y/XY need validation"
            )
        if (
            not isinstance(self.decoder, BaseGKPDecoder)
            or not isinstance(self.frame, LogicalPauliFrame)
            or not isinstance(self.flip, (int, np.integer))
            or self.flip not in (0, 1)
        ):
            raise ValueError("Supply a GKP decoder and a bit-valued flip")
        return self

    def validate_backend(self, engine):
        engine.require(*self.required_capabilities)

    def execute(self, engine, node, outcome=None):
        from dataclasses import replace

        angle = 0.0 if self.basis == "Z" else np.pi / 2
        raw = engine.measure(node, Homodyne(angle), angle, outcome=outcome)
        decoded = self.decoder.decode(raw)
        correction = self.frame.correction(self.basis) ^ self.flip
        probabilities = decoded.probabilities
        if correction and probabilities is not None:
            probabilities = tuple(reversed(probabilities))
        return replace(
            decoded,
            bit=decoded.bit ^ correction,
            frame_correction=correction,
            probabilities=probabilities,
        )


@dataclass(frozen=True)
class IdealLogicalXYMeasurement:
    """Two-dimensional ideal qubit reference, never a finite oscillator POVM."""

    alpha: float

    def probabilities(self, state):
        if not np.isfinite(self.alpha):
            raise ValueError("Angle must be finite")
        rho = density_matrix(state, 2)
        plus = np.array([1, np.exp(1j * self.alpha)]) / np.sqrt(2)
        p = float(np.vdot(plus, rho @ plus).real)
        return (p, 1 - p)


class LogicalMeasurementSynthesis:
    """Extension point for independently validated logical rotation protocols."""

    def lower(self, code, alpha, **kwargs):
        raise NotImplementedError(
            "No validated magic-state injection or arbitrary logical Rz protocol"
        )


@dataclass(frozen=True)
class GKPCode:
    peak_width: float = 0.4
    envelope: float = 0.4
    cutoff: int = 48
    peaks: int = 8
    grid_points: int = 4097

    def __post_init__(self):
        self.resource(0)
        if isinstance(self.cutoff, bool) or not isinstance(self.cutoff, int) or self.cutoff < 2:
            raise ValueError("cutoff must be an integer >=2")

    def resource(self, bit):
        return GKPResource(bit, self.peak_width, self.envelope, self.peaks, self.grid_points)

    @cached_property
    def _basis(self):
        matrix = np.column_stack([self.resource(b).fock(self.cutoff).amplitudes for b in (0, 1)])
        matrix.setflags(write=False)
        return matrix

    @property
    def gram(self):
        return self._basis.conj().T @ self._basis

    def zero(self):
        return FockInput(tuple(self._basis[:, 0]))

    def one(self):
        return FockInput(tuple(self._basis[:, 1]))

    def encode(self, alpha, beta):
        vector = self._basis @ np.array([alpha, beta], complex)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-14:
            raise ValueError("Superposition must have finite nonzero norm")
        return FockInput(tuple(vector / norm))

    def plus(self):
        return self.encode(1, 1)

    def minus(self):
        return self.encode(1, -1)

    @property
    def code_projector(self):
        if np.linalg.eigvalsh(self.gram).min() < 1e-10:
            raise ValueError("Finite code basis is numerically singular")
        return self._basis @ np.linalg.solve(self.gram, self._basis.conj().T)

    def leakage(self, state):
        rho = density_matrix(state, self.cutoff)
        return float(np.clip(1 - np.trace(self.code_projector @ rho).real, 0, 1))

    def diagnostics(self, state):
        """Subspace leakage and stabilizers; overlap is not discarded."""
        return {
            "gram": self.gram,
            "code_subspace_leakage": self.leakage(state.density_matrix),
            "stabilizers": stabilizers(state),
        }

    def logical_measurement(self, basis="Z", *, alpha=None, **kwargs):
        if basis == "XY":
            if (
                alpha is None
                or not np.isscalar(alpha)
                or not np.isreal(alpha)
                or not np.isfinite(alpha)
            ):
                raise ValueError("XY requires a finite numerical logical angle")
            # Absolute tolerance only; a nearby trainable angle is never rounded to X.
            reduced = float(np.remainder(alpha, 2 * np.pi))
            if min(abs(reduced), abs(reduced - 2 * np.pi)) <= 1e-14:
                return PhysicalGKPReadout("X", **kwargs)
            if abs(reduced - np.pi) <= 1e-14:
                return PhysicalGKPReadout("X", flip=1, **kwargs)
            raise NotImplementedError(
                "Physical logical XY supports only alpha=0 mod pi; no Y or injection"
            )
        if alpha is not None:
            raise ValueError("alpha applies only to XY")
        return PhysicalGKPReadout(basis, **kwargs)

    def logical_gate(self, node, gate):
        """Gaussian ideal-lattice realizations; finite codewords are distorted."""
        from .commands import QuadraticPhase, Rotate

        if gate in ("X", "Z"):
            return logical_displacement(node, gate)
        if gate == "H":
            return Rotate(node, np.pi / 2)
        if gate == "S":
            return QuadraticPhase(node, 1.0)
        raise NotImplementedError("Only ideal-lattice X, Z, H and S realizations exist")

    def logical_cz(self, u, v):
        from .commands import Entangle

        if u == v:
            raise ValueError("CZ needs distinct modes")
        return Entangle(u, v, 1.0)

    def decode(self, outcome, decoder=None):
        return (NearestCellDecoder() if decoder is None else decoder).decode(outcome)

    def discrimination_instrument(self):
        """Unambiguous dual-basis finite POVM with explicit inconclusive outcome.

        Equal reciprocal-vector weights maximize the common success weight.
        This is a mathematical instrument, not a supplied optical synthesis.
        """
        projector = self.code_projector
        dual = self._basis @ np.linalg.inv(self.gram)
        weight = float(np.linalg.eigvalsh(self.gram).min())
        effects = [weight * np.outer(v, v.conj()) for v in dual.T]
        effects.append(projector - sum(effects))
        effects.append(np.eye(self.cutoff) - projector)
        kraus = {}
        for label, effect in zip((0, 1, "inconclusive", "outside-code"), effects, strict=True):
            values, vectors = np.linalg.eigh(effect)
            kraus[label] = ((vectors * np.sqrt(np.maximum(0, values))) @ vectors.conj().T,)
        return MeasurementInstrument(kraus)


def modular_effects(cutoff, basis="Z"):
    """Fock-compressed homodyne cell POVM; integrate complete cells with quadrature.

    A finite window beyond the oscillator turning point is checked against I.
    This coarse-grained POVM discards the analog record and differs from a
    conditional trajectory with a known raw homodyne outcome.
    """
    from .fock_measurements import wavefunctions

    if basis not in ("X", "Z"):
        raise NotImplementedError("Only modular X/Z effects are implemented")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 2:
        raise ValueError("cutoff must be an integer >=2")
    count = int(np.ceil((2 * np.sqrt(cutoff) + 12) / SPACING))
    effects = np.zeros((2, cutoff, cutoff), complex)
    phase = np.exp(-1j * (0 if basis == "Z" else np.pi / 2) * np.arange(cutoff))
    for cell in range(-count, count + 1):

        def integrand(x):
            bra = wavefunctions(x, cutoff) * phase
            return np.outer(bra.conj(), bra)

        value, error = quad_vec(
            integrand, (cell - 0.5) * SPACING, (cell + 0.5) * SPACING, epsabs=1e-10, epsrel=1e-10
        )
        if error > 1e-8:
            raise ArithmeticError("Modular POVM integration failed")
        effects[cell % 2] += value
    if not np.allclose(effects.sum(axis=0), np.eye(cutoff), atol=1e-9, rtol=0):
        raise ArithmeticError("Modular POVM window is incomplete")
    return effects


def multimode_readout(state, bases, *, frames=None):
    """Exact joint modular probabilities within supplied finite Fock density matrix.

    Cost grows quadratically with Fock dimension. Never multiply marginals.
    """
    from itertools import product

    if set(bases) != set(state.nodes):
        raise ValueError("Supply a basis for every state mode")
    frames = {} if frames is None else frames
    if not set(frames) <= set(state.nodes):
        raise ValueError("Frame for unknown mode")
    occupations = np.array(state.basis)
    cutoff = int(occupations.max()) + 1
    cutoff = max(2, cutoff)
    effects = [modular_effects(cutoff, bases[n]) for n in state.nodes]
    rho = state.density_matrix
    joint = {}
    for bits in product((0, 1), repeat=len(state.nodes)):
        effect = np.ones_like(rho)
        for j, (node, bit) in enumerate(zip(state.nodes, bits, strict=True)):
            bit ^= frames.get(node, LogicalPauliFrame()).correction(bases[node])
            effect *= effects[j][bit][occupations[:, j, None], occupations[None, :, j]]
        joint[bits] = float(np.einsum("ij,ji->", rho, effect).real)
    marginals = {
        node: tuple(sum(p for bits, p in joint.items() if bits[j] == bit) for bit in (0, 1))
        for j, node in enumerate(state.nodes)
    }
    return {
        "joint_probabilities": joint,
        "marginal_probabilities": marginals,
        "frames": frames,
        "code_subspace_leakage": None,
        "residuals": None,
        "decoder": "nearest-cell",
    }
