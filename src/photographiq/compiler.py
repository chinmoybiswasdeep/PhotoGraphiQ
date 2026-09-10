"""Gaussian circuits compiled to finite-squeezing teleportation patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .commands import Displace, Entangle, Measure, Output, Prepare, QuadraticPhase
from .expressions import Outcome, atan2, sin
from .gaussian import rotation, squeezing, teleportation_matrix
from .measurements import Homodyne
from .pattern import Pattern


def decompose_symplectic(matrix):
    """Return k_j with S=T(k_last)...T(k_first), using four or five steps."""
    s = np.asarray(matrix, dtype=float)
    if (
        s.shape != (2, 2)
        or not np.isfinite(s).all()
        or not np.isclose(np.linalg.det(s), 1, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Expected a real single-mode symplectic matrix")
    prefix = []
    if abs(s[1, 0]) < 1e-8:
        prefix = [0.0]
        s = s @ teleportation_matrix(0).T
    a, c, d = s[0, 0], s[1, 0], s[1, 1]
    ks = prefix + [0.0, (1 - d) / c, c, (1 - a) / c]
    check = np.eye(2)
    for k in ks:
        check = teleportation_matrix(k) @ check
    if not np.allclose(check, matrix, atol=1e-8, rtol=1e-8):
        raise ValueError("Numerically unstable Gaussian decomposition")
    return ks


@dataclass
class Circuit:
    """Ordered photonic gates with Gaussian and resource-injection compilation.

    Args:
        modes (int): Positive number of optical modes, indexed from zero.
        gates (object): Optional legacy gate tuples; fluent construction is preferred.
    """

    modes: int
    gates: list = field(default_factory=list)
    _labels: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if isinstance(self.modes, bool) or not isinstance(self.modes, int) or self.modes < 1:
            raise ValueError("Circuit modes must be a positive integer")

    def _add(self, name, modes, *params):
        if len(set(modes)) != len(modes) or any(
            isinstance(m, bool) or not isinstance(m, int) or m < 0 or m >= self.modes for m in modes
        ):
            raise ValueError(
                f"Circuit gate modes must be distinct integers in range(0, {self.modes})"
            )
        self.gates.append((name, tuple(modes), params))
        return self

    def displace(self, mode, q=0.0, p=0.0):
        """Apply or append quadrature translations q and p in hbar=2 coordinates.

        Args:
            mode (int): Zero-based optical mode index.
            q (float): Position translation; hbar=2 quadrature units.
            p (float): Momentum translation; hbar=2 quadrature units.
        """
        return self._add("displace", (mode,), q, p)

    def rotate(self, mode, angle):
        """Append a rotation by angle radians to this optical circuit.

        Args:
            mode (int): Zero-based optical mode index.
            angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
        """
        self._add("symplectic", (mode,), rotation(angle))
        self._labels.append((len(self.gates) - 1, "R", angle))
        return self

    def squeeze(self, mode, r):
        """Append or apply q squeezing by parameter r.

        Args:
            mode (int): Zero-based optical mode index.
            r (float): Dimensionless squeezing parameter.
        """
        self._add("symplectic", (mode,), squeezing(r))
        self._labels.append((len(self.gates) - 1, "S", r))
        return self

    def gaussian(self, mode, matrix):
        """Compile a real single-mode symplectic matrix through teleportation steps.

        Args:
            mode (int): Zero-based optical mode index.
            matrix (array-like): Matrix in the documented quadrature or occupation basis.
        """
        decompose_symplectic(matrix)
        return self._add("symplectic", (mode,), np.array(matrix, copy=True))

    def cz(self, u, v, weight=1.0):
        """Append or construct a controlled-Z interaction of the specified weight.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            weight (float): Real controlled-Z edge weight.
        """
        return self._add("cz", (u, v), weight)

    def beamsplitter(self, u, v, theta):
        """Mix two optical modes using the package real beamsplitter convention.

        Args:
            u (object): First mode label.
            v (object): Second mode label.
            theta (float): Beamsplitter mixing angle in radians.
        """
        return self._add("beamsplitter", (u, v), theta)

    def identity(self, mode):
        """Construct four Fourier wire steps whose ideal map is identity; finite noise remains.

        Args:
            mode (int): Zero-based optical mode index.
        """
        return self._add("wire", (mode,), [0.0] * 4)

    def cubic_phase(self, mode, gamma):
        """Append CP(gamma)=exp(i gamma q³/6); compile by finite resource injection."""
        return self._add("cubic_phase", (mode,), gamma)

    def kerr(self, mode, kappa):
        """Append exp(i kappa n²); compilation requires approximate cubic synthesis."""
        return self._add("kerr", (mode,), kappa)

    def compile(self, squeezing=1.0, *, return_trace=False, synthesis_steps=8):
        """Return a Pattern, optionally paired with gate-to-command provenance."""
        return compile_circuit(
            self, squeezing=squeezing, return_trace=return_trace, synthesis_steps=synthesis_steps
        )

    def draw(self, **kwargs):
        """Draw optical wires and gates; return a matplotlib Axes."""
        from .visualization import draw_circuit

        return draw_circuit(self, **kwargs)

    def __repr__(self):
        return f"Circuit(modes={self.modes}, gates={len(self.gates)})"


@dataclass(frozen=True)
class CompilationStep:
    """One source gate's generated commands and frontier mapping."""

    gate_index: int
    source_gate: str
    source_modes: tuple
    input_nodes: tuple
    output_nodes: tuple
    resource_nodes: tuple
    command_indices: tuple
    edges: tuple
    measurements: tuple
    corrections: tuple
    source_parameters: tuple = ()


@dataclass(frozen=True)
class CompilationTrace:
    """Ordered provenance for a specific compiled pattern."""

    steps: tuple[CompilationStep, ...]
    pattern_signature: str = field(default="", repr=False)
    source_signature: str = field(default="", repr=False)


def _pattern_signature(pattern):
    """Fingerprint ordinary patterns without excluding runtime-only callables."""
    try:
        return pattern.to_json()
    except TypeError:
        return repr(pattern.inputs) + repr(pattern.commands)


def _circuit_signature(circuit):
    return repr(
        [
            (name, modes, tuple(p.tolist() if isinstance(p, np.ndarray) else p for p in params))
            for name, modes, params in circuit.gates
        ]
    )


class _Builder:
    def __init__(self, modes, r):
        self.pattern = Pattern(inputs=tuple(range(modes)))
        self.frontier = list(range(modes))
        self.next_node = modes
        self.r = r

    def step(self, mode, k):
        old, new = self.frontier[mode], self.next_node
        self.next_node += 1
        angle = atan2(1.0, k)
        self.pattern.extend([Prepare(new, self.r), Entangle(old, new)])
        self.pattern.measure(old, Homodyne(angle))
        self.pattern.displace(new, q=-Outcome(old) / sin(angle))
        self.frontier[mode] = new

    def single(self, mode, matrix):
        for k in decompose_symplectic(matrix):
            self.step(mode, k)

    def cz(self, u, v, weight):
        self.pattern.append(Entangle(self.frontier[u], self.frontier[v], weight))
        for mode in (u, v):
            for _ in range(4):
                self.step(mode, 0.0)

    def sum(self, control, target, weight):
        self.single(target, rotation(np.pi / 2))
        self.cz(control, target, weight)
        self.single(target, rotation(-np.pi / 2))


def compile_circuit(circuit, *, squeezing=1.0, return_trace=False, synthesis_steps=8):
    """Compile gates into finite-resource MBQC; optionally return provenance.

    Kerr lowering is approximate. Increase synthesis_steps and independently
    study resource squeezing and Fock cutoff before interpreting gate accuracy.
    """
    builder = _Builder(circuit.modes, squeezing)
    trace = []

    def lower(name, modes, params):
        if name == "symplectic":
            builder.single(modes[0], params[0])
        elif name == "wire":
            for k in params[0]:
                builder.step(modes[0], k)
        elif name == "displace":
            # Four Fourier teleportations form identity; displacement is an output correction.
            for _ in range(4):
                builder.step(modes[0], 0.0)
            builder.pattern.append(Displace(builder.frontier[modes[0]], *params))
        elif name == "cz":
            builder.cz(modes[0], modes[1], params[0])
        elif name == "beamsplitter":
            # Two half rotations avoid the tan(theta/2) singularity at pi.
            theta = float(params[0])
            if not np.isfinite(theta):
                raise ValueError("Nonfinite beam splitter angle")
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            u, v = modes
            for _ in range(2):
                builder.sum(v, u, -np.tan(theta / 4))
                builder.sum(u, v, np.sin(theta / 2))
                builder.sum(v, u, -np.tan(theta / 4))
        elif name == "cubic_phase":
            from .non_gaussian import cubic_injection

            ancilla = builder.next_node
            builder.next_node += 1
            gadget = cubic_injection(
                params[0],
                squeezing=squeezing,
                input_node=builder.frontier[modes[0]],
                ancilla=ancilla,
                key=("cubic", ancilla),
            )
            builder.pattern.extend(gadget.commands[:-1])
        elif name == "kerr":
            from .synthesis import synthesize_kerr

            expanded, _ = synthesize_kerr(
                params[0], steps=synthesis_steps, modes=circuit.modes, mode=modes[0]
            )
            for gate in expanded.gates:
                lower(*gate)
        else:
            raise NotImplementedError(f"Unsupported circuit gate: {name}")

    labels = {i: label for i, label, _ in circuit._labels}
    for index, (name, modes, params) in enumerate(circuit.gates):
        start = len(builder.pattern.commands)
        before = tuple(builder.frontier[m] for m in modes)
        lower(name, modes, params)
        indices = tuple(range(start, len(builder.pattern.commands)))
        commands = builder.pattern.commands
        trace.append(
            CompilationStep(
                index,
                labels.get(index, name),
                modes,
                before,
                tuple(builder.frontier[m] for m in modes),
                tuple(commands[i].node for i in indices if isinstance(commands[i], Prepare)),
                indices,
                tuple(
                    (commands[i].u, commands[i].v, commands[i].weight)
                    for i in indices
                    if isinstance(commands[i], Entangle)
                ),
                tuple(i for i in indices if isinstance(commands[i], Measure)),
                tuple(i for i in indices if isinstance(commands[i], (Displace, QuadraticPhase))),
                source_parameters=next(
                    ((value,) for i, _, value in circuit._labels if i == index),
                    tuple(
                        tuple(map(tuple, p.tolist())) if isinstance(p, np.ndarray) else p
                        for p in params
                    ),
                ),
            )
        )
    builder.pattern.append(Output(tuple(builder.frontier)))
    pattern = builder.pattern.validate()
    provenance = CompilationTrace(
        tuple(trace), _pattern_signature(pattern), _circuit_signature(circuit)
    )
    pattern._compilation_trace = provenance
    return (pattern, provenance) if return_trace else pattern
