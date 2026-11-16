"""Translations between CV diagrams and PhotoGraphiQ objects.

* :func:`from_pattern` / :func:`to_pattern`: executable ``Pattern`` <-> diagram. Feed-forward
  displacements must be exactly affine in outcomes (``calculus.cv.affine_form``); adaptive
  homodyne angles, nonlinear feed-forward, vector outcomes and photon addition/subtraction
  commands are rejected, never approximated.
* :func:`from_circuit` / :func:`to_circuit`: ``Circuit`` <-> unitary diagram (ideal circuit).
* :func:`from_gaussian_graph` / :func:`to_gaussian_graph`: pure Gaussian states only.
"""

from __future__ import annotations

import numpy as np

from ... import commands as c
from ...expressions import resolve
from ...measurements import Homodyne as HomodyneMeasurement
from ...measurements import PhotonNumber
from ...pattern import Pattern
from ...states import FockInput, GaussianInput
from ..cv import affine_form
from ..gaussian_graph import GaussianGraph
from .ir import (
    CZ,
    BeamSplitter,
    ClassicalSource,
    CVDiagram,
    Discard,
    Displacement,
    Feedforward,
    FockPrepare,
    GaussianLoss,
    Homodyne,
    Kerr,
    PhasePolynomial,
    PhotonCount,
    Prepare,
    PrepareState,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)
from .semantics import evaluate


def _value(x, parameters):
    if getattr(x, "dependencies", frozenset()):
        raise NotImplementedError("Outcome-dependent parameter (adaptive / nonlinear feed-forward)")
    return float(resolve(x, {}, parameters))


def from_pattern(pattern: Pattern, parameters=None) -> CVDiagram:
    """Diagram of an executable pattern (see module docstring for the supported subset)."""
    pattern.validate()
    params = dict(parameters or {})
    boxes = []
    for cmd in pattern.commands:
        if isinstance(cmd, c.Prepare):
            if cmd.state is None:
                boxes.append(Prepare.squeezed(cmd.node, _value(cmd.squeezing, params)))
            elif isinstance(cmd.state, GaussianInput):
                boxes.append(
                    Prepare(
                        cmd.node, tuple(cmd.state.mean), tuple(map(tuple, cmd.state.covariance))
                    )
                )
            elif (
                isinstance(cmd.state, FockInput)
                and sum(abs(a) > 0 for a in cmd.state.amplitudes) == 1
            ):
                boxes.append(
                    FockPrepare(cmd.node, int(np.flatnonzero(np.abs(cmd.state.amplitudes))[0]))
                )
            else:
                raise NotImplementedError(f"Unsupported preparation {type(cmd.state).__name__}")
        elif isinstance(cmd, c.Entangle):
            boxes.append(CZ(cmd.u, cmd.v, _value(cmd.weight, params)))
        elif isinstance(cmd, c.Rotate):
            boxes.append(Rotation(cmd.node, _value(cmd.angle, params)))
        elif isinstance(cmd, c.Squeeze):
            boxes.append(Squeezer(cmd.node, _value(cmd.r, params)))
        elif isinstance(cmd, c.QuadraticPhase):
            boxes.append(Shear(cmd.node, _value(cmd.s, params)))
        elif isinstance(cmd, c.BeamSplitter):
            boxes.append(BeamSplitter(cmd.u, cmd.v, _value(cmd.theta, params)))
        elif isinstance(cmd, c.Loss):
            boxes.append(
                GaussianLoss(
                    cmd.node,
                    _value(cmd.transmissivity, params),
                    _value(cmd.thermal_photons, params),
                )
            )
        elif isinstance(cmd, c.Displace):
            q, p = affine_form(cmd.q, params), affine_form(cmd.p, params)
            if q is None or p is None:
                raise NotImplementedError("Non-affine feed-forward displacement")
            if not q.coefficients and not p.coefficients:
                boxes.append(Displacement(cmd.node, q.constant, p.constant))
                continue
            gains: dict = {}
            for (key, comp), coef in q.coefficients.items():
                if comp is not None:
                    raise NotImplementedError("Vector-outcome feed-forward")
                gains[key] = (gains.get(key, (0.0, 0.0))[0] + coef, gains.get(key, (0.0, 0.0))[1])
            for (key, comp), coef in p.coefficients.items():
                if comp is not None:
                    raise NotImplementedError("Vector-outcome feed-forward")
                gains[key] = (gains.get(key, (0.0, 0.0))[0], gains.get(key, (0.0, 0.0))[1] + coef)
            boxes.append(Feedforward(cmd.node, tuple(gains.items()), (q.constant, p.constant)))
        elif isinstance(cmd, c.Measure):
            m = cmd.measurement
            if isinstance(m, HomodyneMeasurement):
                boxes.append(
                    Homodyne(
                        cmd.node, _value(m.angle, params), cmd.result_key, m.efficiency, m.noise
                    )
                )
            elif isinstance(m, PhotonNumber):
                boxes.append(PhotonCount(cmd.node, cmd.result_key))
            else:
                raise NotImplementedError(f"Unsupported measurement {type(m).__name__}")
        elif isinstance(cmd, c.Kerr):
            boxes.append(Kerr(cmd.node, _value(cmd.kappa, params)))
        elif isinstance(cmd, c.CubicPhase):
            boxes.append(PhasePolynomial(cmd.node, (0.0, 0.0, 0.0, _value(cmd.gamma, params) / 6)))
        elif isinstance(cmd, c.Output):
            continue
        else:
            raise NotImplementedError(f"Unsupported command {type(cmd).__name__}")
    return CVDiagram(pattern.inputs, tuple(boxes), pattern.outputs)


def _single_mode_gates(S):
    """Rotate/Squeeze commands realising a 2x2 symplectic S = R(a) diag(e^-r, e^r) R(b)."""
    u, s, vt = np.linalg.svd(S)
    if np.linalg.det(u) < 0:  # make both factors proper rotations
        u, vt = u @ np.diag([1, -1]), np.diag([1, -1]) @ vt
    a, b = np.arctan2(u[1, 0], u[0, 0]), np.arctan2(vt[1, 0], vt[0, 0])
    r = -np.log(s[0])  # s = (e^{-r}, e^{r}) up to ordering
    return b, r, a


def to_pattern(diagram: CVDiagram) -> Pattern:
    """Executable pattern for the supported subset.

    ``ClassicalSource(key, mean, var)`` has no classical-randomness command in ``Pattern``; it
    is realised exactly by an ancilla prepared with q variance ``var`` (and mean) and measured
    in q, so the recompiled pattern uses one optical mode per source.
    """
    pattern = Pattern(inputs=diagram.inputs)
    for b in diagram.boxes:
        if isinstance(b, Prepare):
            if b.squeezing is not None:
                pattern.append(c.Prepare(b.mode, b.squeezing))
            else:
                pattern.append(
                    c.Prepare(
                        b.mode, state=GaussianInput(tuple(b.mean), tuple(map(tuple, b.covariance)))
                    )
                )
        elif isinstance(b, FockPrepare):
            pattern.append(c.Prepare(b.mode, state=FockInput.number(b.n)))
        elif isinstance(b, CZ):
            pattern.append(c.Entangle(b.u, b.v, b.g))
        elif isinstance(b, Rotation):
            pattern.append(c.Rotate(b.mode, b.theta))
        elif isinstance(b, Squeezer):
            pattern.append(c.Squeeze(b.mode, b.r))
        elif isinstance(b, Shear):
            pattern.append(c.QuadraticPhase(b.mode, b.s))
        elif isinstance(b, BeamSplitter):
            pattern.append(c.BeamSplitter(b.u, b.v, b.theta))
        elif isinstance(b, Displacement):
            pattern.append(c.Displace(b.mode, b.dq, b.dp))
        elif isinstance(b, Symplectic) and len(b.labels) == 1:
            first, r, last = _single_mode_gates(b.S)
            pattern.extend(
                [
                    c.Rotate(b.labels[0], first),
                    c.Squeeze(b.labels[0], r),
                    c.Rotate(b.labels[0], last),
                ]
            )
            if np.any(b.d):
                pattern.append(c.Displace(b.labels[0], *b.d))
        elif isinstance(b, GaussianLoss):
            pattern.append(c.Loss(b.mode, b.eta, b.thermal))
        elif isinstance(b, Homodyne):
            pattern.append(
                c.Measure(b.mode, HomodyneMeasurement(b.theta, b.efficiency, b.noise), b.key)
            )
        elif isinstance(b, PhotonCount):
            pattern.append(c.Measure(b.mode, PhotonNumber(), b.key))
        elif isinstance(b, Feedforward):
            from ...expressions import Outcome

            q = b.constant[0] + sum(a * Outcome(k) for k, (a, _) in b.gains if a)
            p = b.constant[1] + sum(g * Outcome(k) for k, (_, g) in b.gains if g)
            pattern.append(c.Displace(b.mode, q, p))
        elif isinstance(b, ClassicalSource):
            if b.variance <= 0:
                raise NotImplementedError(
                    "A zero-variance source is a deterministic record; Pattern has no command "
                    "for it and an ancilla would need infinite squeezing"
                )
            node = ("source", b.key)
            pattern.append(
                c.Prepare(
                    node,
                    state=GaussianInput((b.mean, 0.0), ((b.variance, 0.0), (0.0, 1 / b.variance))),
                )
            )
            pattern.append(c.Measure(node, HomodyneMeasurement(0.0), b.key))
        elif isinstance(b, Kerr):
            pattern.append(c.Kerr(b.mode, b.kappa))
        elif isinstance(b, PhasePolynomial):
            coeff = list(b.coefficients) + [0.0] * 4
            if len(b.coefficients) > 4:
                raise NotImplementedError("Phase polynomials above degree 3 have no command")
            if coeff[2]:
                pattern.append(c.QuadraticPhase(b.mode, 4 * coeff[2]))
            if coeff[1]:
                pattern.append(c.Displace(b.mode, 0.0, 2 * coeff[1]))
            if coeff[3]:
                pattern.append(c.CubicPhase(b.mode, 6 * coeff[3]))
        elif isinstance(b, Discard):
            raise NotImplementedError("Pattern has no discard command; keep the mode as an output")
        else:
            raise NotImplementedError(f"No executable command for {type(b).__name__}")
    pattern.append(c.Output(diagram.outputs))
    return pattern.validate()


def from_circuit(circuit) -> CVDiagram:
    """Ideal (direct-gate) diagram of a ``photographiq.Circuit`` on modes 0..n-1."""
    labels = {i: (kind, value) for i, kind, value in circuit._labels}
    boxes: list = []
    for index, (name, modes, params) in enumerate(circuit.gates):
        if name == "symplectic":
            kind, value = labels.get(index, (None, None))
            if kind == "R":
                boxes.append(Rotation(modes[0], float(value)))
            elif kind == "S":
                boxes.append(Squeezer(modes[0], float(value)))
            else:
                boxes.append(Symplectic((modes[0],), np.asarray(params[0], float)))
        elif name == "displace":
            boxes.append(Displacement(modes[0], *map(float, params)))
        elif name == "cz":
            boxes.append(CZ(modes[0], modes[1], float(params[0])))
        elif name == "beamsplitter":
            boxes.append(BeamSplitter(modes[0], modes[1], float(params[0])))
        elif name == "wire":
            continue  # the ideal map of an identity wire is the identity
        elif name == "cubic_phase":
            boxes.append(PhasePolynomial(modes[0], (0.0, 0.0, 0.0, float(params[0]) / 6)))
        elif name == "kerr":
            boxes.append(Kerr(modes[0], float(params[0])))
        else:
            raise NotImplementedError(f"Unsupported circuit gate {name}")
    modes = tuple(range(circuit.modes))
    return CVDiagram(modes, tuple(boxes), modes)


def to_circuit(diagram: CVDiagram):
    """``photographiq.Circuit`` of a unitary diagram on wires 0..n-1 (compiles to MBQC)."""
    from ...compiler import Circuit

    n = len(diagram.inputs)
    if tuple(diagram.inputs) != tuple(range(n)) or tuple(diagram.outputs) != tuple(range(n)):
        raise NotImplementedError("to_circuit needs wires 0..n-1 in order")
    circuit = Circuit(n)
    for b in diagram.boxes:
        if isinstance(b, Rotation):
            circuit.rotate(b.mode, b.theta)
        elif isinstance(b, Squeezer):
            circuit.squeeze(b.mode, b.r)
        elif isinstance(b, Shear):
            circuit.gaussian(b.mode, [[1.0, 0.0], [b.s, 1.0]])
        elif isinstance(b, Displacement):
            circuit.displace(b.mode, b.dq, b.dp)
        elif isinstance(b, CZ):
            circuit.cz(b.u, b.v, b.g)
        elif isinstance(b, BeamSplitter):
            circuit.beamsplitter(b.u, b.v, b.theta)
        elif isinstance(b, Symplectic) and len(b.labels) == 1:
            circuit.gaussian(b.labels[0], b.S)
            if np.any(b.d):
                circuit.displace(b.labels[0], *b.d)
        elif isinstance(b, PhasePolynomial) and len(b.coefficients) <= 4:
            co = list(b.coefficients) + [0.0] * 4
            if co[2]:
                circuit.gaussian(b.mode, [[1.0, 0.0], [4 * co[2], 1.0]])
            if co[1]:
                circuit.displace(b.mode, 0.0, 2 * co[1])
            if co[3]:
                circuit.cubic_phase(b.mode, 6 * co[3])
        elif isinstance(b, Kerr):
            circuit.kerr(b.mode, b.kappa)
        else:
            raise NotImplementedError(f"No circuit gate for {type(b).__name__}")
    return circuit


def from_gaussian_graph(graph: GaussianGraph) -> CVDiagram:
    """State-preparation diagram of a pure Gaussian graph state (mean and covariance)."""
    box = PrepareState(graph.nodes, graph.mean, graph.to_covariance())
    return CVDiagram((), (box,), graph.nodes)


def to_gaussian_graph(diagram: CVDiagram, outcomes=None, *, rtol=1e-9) -> GaussianGraph:
    """Graph of the (conditional) output state of an input-free diagram, if pure.

    Records must be fixed by ``outcomes``; averaging over outcomes generally gives a mixed
    state, which has no Gaussian graph (``GaussianGraph.from_covariance`` raises).
    """
    if diagram.inputs:
        raise ValueError("to_gaussian_graph needs a state (no input wires)")
    semantics = evaluate(diagram)
    if semantics.records:
        if outcomes is None:
            raise ValueError("Fix every record with outcomes=...; averaged states are mixed")
        mean, cov, _ = semantics.conditional(outcomes, np.zeros(0), np.zeros((0, 0)))
    else:
        _, cov, mean = semantics.channel()
    return GaussianGraph.from_covariance(cov, diagram.outputs, mean, rtol=rtol)
