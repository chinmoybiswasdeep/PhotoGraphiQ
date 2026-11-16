"""Explicit ideal references and finite-energy square GKP readout primitives."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from scipy.integrate import quad_vec

from ._checks import is_finite_real_scalar
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


class EncodedGKPMeasurement(ABC):
    """Destructive encoded-qubit instrument executed by a Fock engine on one data mode.

    Fock backends hand the live engine to :meth:`execute`, so an instrument may perform a
    multi-step bosonic protocol (ancilla preparation, Gaussian gates, homodyne detection,
    classical feed-forward) before returning its classical record.
    """

    @abstractmethod
    def execute(self, engine, node, outcome=None, angle=None): ...


@dataclass(frozen=True)
class PhysicalGKPReadout(EncodedGKPMeasurement):
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

    def execute(self, engine, node, outcome=None, angle=None):
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


CLIFFORD_ANGLE_TOLERANCE = 1e-12
XY_PROTOCOLS = ("auto", "clifford", "injection")


def quarter_turns(angle, tolerance=CLIFFORD_ANGLE_TOLERANCE):
    """Return k in 0..3 if angle = k*pi/2 (mod 2pi) within an absolute tolerance, else None."""
    reduced = float(np.remainder(angle, 2 * np.pi))
    k = int(np.rint(reduced / (np.pi / 2)))
    return k % 4 if abs(reduced - k * np.pi / 2) <= tolerance else None


def apply_logical_cnot(engine, control, target):
    """Square-lattice logical CNOT: unit CZ conjugated by quarter turns of the target.

    q_target -> q_target + q_control and p_control -> p_control - p_target up to the lattice
    symmetry R(pi); identical to the SUM used by ``correction_pattern``. Gaussian, so finite
    codewords are distorted (envelope shear), which is part of the finite-energy error.
    """
    engine.rotate(target, np.pi / 2)
    engine.entangle(control, target, 1.0)
    engine.rotate(target, -np.pi / 2)


def apply_logical_phase(engine, node, turns):
    """Apply diag(1, i**turns) for turns in {1, 3} with the Gaussian shear exp(+-i q^2/4).

    QuadraticPhase(+1) maps the finite |+i_L> to |-_L>, i.e. it is the logical S (checked in
    tests/gkp/test_logical_xy.py); turns == 3 uses its inverse. Z (turns == 2) is a Pauli-frame
    outcome flip and is handled by the caller.
    """
    if turns not in (1, 3):
        raise ValueError("apply_logical_phase handles S (1) and S^dagger (3) only")
    engine.quadratic_phase(node, 1.0 if turns == 1 else -1.0)


@dataclass(frozen=True)
class LogicalXYMeasurementResult(LogicalDecodeResult):
    """Classical record of one physical logical-XY measurement trajectory.

    ``raw_outcome``/``residual``/``cell`` refer to the final data-mode p homodyne. ``rounds``
    lists every injection round (target phase, raw ancilla q outcome, decoded bit, residual,
    point density). ``joint_density`` is the product of all homodyne point densities, i.e. the
    likelihood density of the complete continuous record (not a probability). ``bit`` is the
    logical M_L(angle) outcome (0 <-> +1) after the protocol's own Pauli-frame flips.
    ``readout_probabilities`` are the exact (P(bit 0), P(bit 1)) of the final decoded X readout
    given the data state just before it (cell-integrated POVM): averaging them over
    trajectories is a lower-variance (Rao-Blackwell) estimate of the instrument's statistics.
    """

    angle: float = 0.0
    protocol: str = "clifford"
    rounds: tuple = ()
    heralded_success: bool = True
    applied_operations: tuple = ()
    joint_density: float | None = None
    resource: dict | None = None
    retained_norm_min: float | None = None
    cutoff: int | None = None
    ancilla_modes: tuple = ()
    readout_probabilities: tuple | None = None


@dataclass(frozen=True)
class GKPLogicalXYMeasurement(EncodedGKPMeasurement):
    """Physical finite-energy GKP measurement of M_L(a) = cos(a) X_L + sin(a) Y_L.

    Convention: the +1 eigenstate is (|0_L> + e^{ia}|1_L>)/sqrt(2), as in
    :class:`IdealLogicalXYMeasurement` and the logical MuTA. Because the +1 eigenstate is
    D(a)|+> with D(t) = diag(1, e^{it}), P(+) = |<+| D(-a) psi>|^2: measuring M_L(a) equals
    applying D(-a) and then measuring X_L. The instrument physically applies D(-a) to the
    data mode and finishes with the existing p-homodyne X readout:

    * Clifford angles (a = k pi/2): D(-a) is I, S^dagger, Z or S. S/S^dagger are the Gaussian
      shears exp(+-i q^2/4); Z before an X measurement is an exact outcome relabelling
      (Pauli frame). Deterministic, no ancilla.
    * Other angles ("injection"): repeat-until-success gate teleportation with programmable
      ancilla rotations. Round k prepares the finite-energy encoded magic state
      (|0~> + e^{i t_k}|1~>)/norm in a fresh mode, applies the logical CNOT (data control) and
      measures the ancilla q quadrature with nearest-cell decoding. Bit 0 applies D(t_k);
      bit 1 applies D(-t_k), so the next round targets t_{k+1} = 2 t_k. A remaining Clifford
      angle is completed deterministically: pi is a frame flip, +-pi/2 is injected with the
      Y-eigenstate ancilla, whose failure leaves exactly D(pi) = Z (a frame flip). Each
      non-Clifford round is heralded; if ``max_rounds`` is exhausted the record has
      ``heralded_success=False`` (the wrong angle was measured). ``protocol='injection'``
      also realizes Clifford angles (including Y) through ancillas instead of the shear.

    The encoded magic state is a supplied bosonic resource (a normalized finite-Fock vector);
    its preparation is not simulated. Feed-forward uses decoded ancilla bits only, never the
    analog residuals. Optional ``ancilla_transmissivity`` (loss before entangling) and
    ``detector_efficiency``/``homodyne_noise`` (all homodynes) require a mixed Fock backend.
    """

    angle: object = 0.0
    code: "GKPCode | None" = None
    protocol: str = "auto"
    max_rounds: int = 4
    decoder: BaseGKPDecoder = NearestCellDecoder()
    ancilla_transmissivity: float = 1.0
    detector_efficiency: float = 1.0
    homodyne_noise: float = 0.0
    clifford_tolerance: float = CLIFFORD_ANGLE_TOLERANCE

    def __post_init__(self):
        self.validate()

    @property
    def noisy(self):
        return (
            self.ancilla_transmissivity != 1
            or self.detector_efficiency != 1
            or self.homodyne_noise != 0
        )

    @property
    def required_capabilities(self):
        features = {"encoded_gkp_xz", "encoded_gkp_xy", "homodyne", "quadratic_phase"}
        if self.protocol != "clifford":
            features |= {"fock_input", "multimode_fock"}
        if self.noisy:
            features.add("mixed_fock")
        return frozenset(features)

    def validate(self):
        if self.protocol not in XY_PROTOCOLS:
            raise ValueError(f"protocol must be one of {XY_PROTOCOLS}")
        if not isinstance(self.decoder, BaseGKPDecoder):
            raise ValueError("Supply a GKP decoder")
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int):
            raise ValueError("max_rounds must be a nonnegative integer")
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be a nonnegative integer")
        for name in ("ancilla_transmissivity", "detector_efficiency"):
            value = getattr(self, name)
            if not is_finite_real_scalar(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not is_finite_real_scalar(self.homodyne_noise) or self.homodyne_noise < 0:
            raise ValueError("homodyne_noise must be a finite nonnegative variance")
        if not is_finite_real_scalar(self.clifford_tolerance) or self.clifford_tolerance < 0:
            raise ValueError("clifford_tolerance must be finite and nonnegative")
        if self.protocol != "clifford" and not isinstance(self.code, GKPCode):
            raise ValueError("Injection needs the GKPCode that encodes its magic ancillas")
        numeric = is_finite_real_scalar(self.angle)
        if not numeric and not hasattr(self.angle, "evaluate"):
            raise ValueError("angle must be a finite real number or a classical expression")
        if (
            numeric
            and self.protocol == "clifford"
            and quarter_turns(self.angle, self.clifford_tolerance) is None
        ):
            raise NotImplementedError(
                "The Clifford protocol measures only angles k*pi/2; use protocol='injection'"
            )
        return self

    def validate_backend(self, engine):
        engine.require(*self.required_capabilities)
        cutoff = getattr(engine, "cutoff", None)
        if self.code is not None and cutoff is not None and self.code.cutoff > cutoff:
            raise ValueError("Magic-ancilla code cutoff exceeds the engine's total-photon cutoff")

    def resource_parameters(self):
        if self.code is None:
            return None
        return {
            "peak_width": self.code.peak_width,
            "envelope": self.code.envelope,
            "peaks": self.code.peaks,
            "grid_points": self.code.grid_points,
            "cutoff": self.code.cutoff,
        }

    @property
    def _noisy_detection(self):
        return self.detector_efficiency != 1 or self.homodyne_noise != 0

    def _homodyne(self, angle):
        return Homodyne(angle, efficiency=self.detector_efficiency, noise=self.homodyne_noise)

    def execute(self, engine, node, outcome=None, angle=None):
        alpha = self.angle if angle is None else angle
        if not is_finite_real_scalar(alpha):
            raise ValueError("Logical XY angle is unresolved: run the pattern with simulate()")
        alpha = float(alpha)  # type: ignore[arg-type]  # checked finite real just above
        ancilla_outcomes, data_outcome = _split_xy_outcome(outcome)
        start = len(getattr(engine, "diagnostics", ()))
        # phi: phase t of the D(t) still to be applied to the data mode before X readout.
        phi, flip, density = -alpha, 0, 1.0
        rounds: list[dict] = []
        operations: list[str] = []
        ancillas: list = []
        success = True
        while True:
            turns = quarter_turns(phi, self.clifford_tolerance)
            forced = self.protocol == "injection" and not rounds
            if turns is not None and not forced:
                if turns == 2:
                    flip ^= 1
                    operations.append("Z (Pauli-frame outcome flip)")
                    break
                if turns == 0:
                    break
                # S or S^dagger. The Gaussian shear is used on a fresh request; a remainder
                # after injection rounds is completed by injecting the Y eigenstate instead:
                # success applies it, failure leaves D(pi) = Z, an exact frame flip.
                if self.protocol == "clifford" or not rounds:
                    apply_logical_phase(engine, node, turns)
                    operations.append("S" if turns == 1 else "S_dagger")
                    break
            elif self.protocol == "clifford":
                raise NotImplementedError("Clifford protocol reached a non-Clifford angle")
            if len(rounds) >= self.max_rounds and turns is None:
                success = False
                operations.append("round budget exhausted (heralded failure)")
                break
            ancilla = ("gkp-xy-ancilla", node, len(rounds))
            assert self.code is not None  # validate(): every non-Clifford protocol carries a code
            engine.prepare(ancilla, state=self.code.encode(1, np.exp(1j * phi)))
            if self.ancilla_transmissivity != 1:
                engine.loss(ancilla, self.ancilla_transmissivity)
            apply_logical_cnot(engine, node, ancilla)
            given = None
            if ancilla_outcomes is not None:
                if len(rounds) >= len(ancilla_outcomes):
                    raise ValueError("Postselected record has fewer ancilla outcomes than rounds")
                given = ancilla_outcomes[len(rounds)]
            if given is None and not self._noisy_detection:
                given = sample_quadrature(engine, ancilla, 0.0)
            engine.last_measurement = {}
            raw = engine.measure(ancilla, self._homodyne(0.0), 0.0, outcome=given)
            point = engine.last_measurement.get("value")
            density *= point if point is not None else np.nan
            decoded = self.decoder.decode(raw)
            rounds.append(
                {
                    "target_phase": phi,
                    "raw_outcome": float(raw),
                    "bit": int(decoded.bit),
                    "residual": decoded.residual,
                    "cell": decoded.cell,
                    "density": point,
                }
            )
            ancillas.append(ancilla)
            operations.append(f"inject D({phi:.12g}); ancilla bit {decoded.bit}")
            phi = 0.0 if decoded.bit == 0 else 2 * phi
        if ancilla_outcomes is not None and len(ancilla_outcomes) != len(rounds):
            raise ValueError("Postselected record has more ancilla outcomes than rounds")
        readout = _x_readout_probabilities(engine, node) if not self._noisy_detection else None
        if readout is not None and flip:
            readout = readout[::-1]
        if data_outcome is None and not self._noisy_detection:
            data_outcome = sample_quadrature(engine, node, np.pi / 2)
        engine.last_measurement = {}
        raw = engine.measure(node, self._homodyne(np.pi / 2), np.pi / 2, outcome=data_outcome)
        point = engine.last_measurement.get("value")
        density *= point if point is not None else np.nan
        decoded = self.decoder.decode(raw)
        norms = [
            d["retained_norm"]
            for d in getattr(engine, "diagnostics", ())[start:]
            if "retained_norm" in d
        ]
        engine.last_measurement = {
            "kind": "joint_density",
            "value": float(density),
            "rounds": len(rounds),
        }
        return LogicalXYMeasurementResult(
            bit=int(decoded.bit) ^ flip,
            raw_outcome=float(raw),
            cell=decoded.cell,
            residual=decoded.residual,
            decoder=decoded.decoder,
            frame_correction=flip,
            angle=alpha,
            protocol="injection" if rounds else "clifford",
            rounds=tuple(rounds),
            heralded_success=success,
            applied_operations=tuple(operations),
            joint_density=float(density),
            resource=self.resource_parameters(),
            retained_norm_min=min(norms) if norms else None,
            cutoff=getattr(engine, "cutoff", None),
            ancilla_modes=tuple(ancillas),
            readout_probabilities=readout,
        )


SAMPLING_GRID_POINTS = 16385


def sample_quadrature(engine, node, angle, grid_points=SAMPLING_GRID_POINTS):
    """Draw an ideal homodyne outcome from the mode's exact reduced marginal density.

    The density <x_angle|rho_node|x_angle> is tabulated once on a uniform grid spanning the
    oscillator turning point plus a Gaussian margin (the bound used by the generic sampler)
    and inverted with one uniform variate from the engine's generator. The engine then
    projects on that outcome itself, so the conditional state and the recorded density are
    the engine's own. Equivalent in distribution to the generic adaptive-quadrature sampler;
    much faster for repeated GKP readouts.
    """
    state = engine.get_state((node,))
    rho = np.asarray(state.density_matrix)
    occupations = np.asarray(state.basis)[:, 0]
    cutoff = int(occupations.max()) + 1
    bound = 2 * np.sqrt(engine.cutoff) + 10
    x = np.linspace(-bound, bound, grid_points)
    table = np.empty((grid_points, cutoff))
    table[:, 0] = (2 * np.pi) ** (-0.25) * np.exp(-x * x / 4)
    if cutoff > 1:
        table[:, 1] = x * table[:, 0]
    for n in range(2, cutoff):
        table[:, n] = (x * table[:, n - 1] - np.sqrt(n - 1) * table[:, n - 2]) / np.sqrt(n)
    amplitudes = table[:, occupations] * np.exp(-1j * angle * occupations)
    density = np.maximum(np.sum((amplitudes @ rho) * amplitudes.conj(), axis=1).real, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(x))])
    if abs(cdf[-1] - 1) > 1e-6:
        raise ArithmeticError("Tabulated homodyne density is unresolved; refine the cutoff")
    return float(np.interp(float(engine.rng.random()) * cdf[-1], cdf, x))


@lru_cache(maxsize=8)
def _x_effects(cutoff):
    return modular_effects(cutoff, "X")


def _x_readout_probabilities(engine, node):
    """Exact decoded-X probabilities (raw parity 0, 1) of one mode's reduced state."""
    state = engine.get_state((node,))
    occupations = np.asarray(state.basis)[:, 0]
    effects = _x_effects(int(engine.cutoff))
    rho = np.asarray(state.density_matrix)
    values = [
        float(np.einsum("ij,ji->", rho, e[np.ix_(occupations, occupations)]).real) for e in effects
    ]
    return tuple(values)


def _split_xy_outcome(outcome):
    """Postselection record {'ancilla': [q_1, ...], 'data': p} or None (sample)."""
    if outcome is None:
        return None, None
    if not isinstance(outcome, dict) or set(outcome) - {"ancilla", "data"} or "data" not in outcome:
        raise ValueError(
            "Postselect a logical XY measurement with {'ancilla': [q_1, ...], 'data': p}"
        )
    ancilla = tuple(outcome.get("ancilla", ()))
    values = (*ancilla, outcome["data"])
    if not all(is_finite_real_scalar(v) for v in values):
        raise ValueError("Postselected analog outcomes must be finite real scalars")
    return tuple(float(v) for v in ancilla), float(outcome["data"])


class LogicalMeasurementSynthesis:
    """Lower a logical XY angle to a validated physical instrument (see GKPLogicalXYMeasurement)."""

    def __init__(self, protocol=None, **options):
        if protocol is not None and protocol not in XY_PROTOCOLS:
            raise ValueError(f"protocol must be one of {XY_PROTOCOLS}")
        self.protocol, self.options = protocol, options

    def lower(self, code, alpha, **kwargs):
        if self.protocol is None:
            raise NotImplementedError(
                "Choose an explicit validated protocol: LogicalMeasurementSynthesis('clifford' | "
                "'injection' | 'auto'); no implicit rotated-homodyne substitution exists"
            )
        return GKPLogicalXYMeasurement(
            angle=alpha, code=code, protocol=self.protocol, **{**self.options, **kwargs}
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

    def logical_measurement(self, basis="Z", *, alpha=None, protocol=None, **kwargs):
        """Physical logical readout. Without ``protocol`` only X, Z and XY at 0/pi exist.

        ``protocol`` ('clifford', 'injection' or 'auto') selects the validated
        :class:`GKPLogicalXYMeasurement` for basis 'Y' (alpha = pi/2) or 'XY' (any alpha).
        """
        if protocol is not None:
            if basis not in ("XY", "Y"):
                raise ValueError("protocol applies to the Y and XY bases only")
            if basis == "Y":
                if alpha is not None:
                    raise ValueError("Y fixes alpha = pi/2")
                alpha = np.pi / 2
            elif alpha is None or not is_finite_real_scalar(alpha):
                if not hasattr(alpha, "evaluate"):
                    raise ValueError("XY requires a finite numerical logical angle")
            return GKPLogicalXYMeasurement(angle=alpha, code=self, protocol=protocol, **kwargs)
        if basis == "XY":
            if alpha is None or not is_finite_real_scalar(alpha):
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
