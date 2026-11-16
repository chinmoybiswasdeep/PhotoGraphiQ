"""Sound rewrite rules for typed CV diagrams, each with its proof and equivalence class.

Equivalence classes (strongest first):

* ``exact-instrument``: the full conditional instrument is preserved -- for every input, the
  joint law of (output Wigner variables, classical records) is unchanged, hence every record
  density and every conditional state, and the averaged channel. Global or outcome-dependent
  *phases* of a branch are irrelevant at the density-operator level and are recorded in the
  trace when known.
* ``exact-channel``: the outcome-averaged channel is preserved (the diagram has no records
  in the rewritten region).
* ``logical``: the ideal (infinite-squeezing) map is preserved -- for corrected teleportation
  chains the affine part (X, d) of the channel is unchanged at every finite squeezing -- but
  the finite-squeezing noise Y and the record set change. Never treated as physical equality.

Every rule matches *wire-adjacent* boxes (no box in between touches the matched wires), so
moving a box past the boxes in between is the commutation of operations on disjoint wires
(rule ``commute``). Rewrites return a new diagram and append a :class:`RewriteStep`.

Proof sources (full statements in docs/theory/CV_ZX_FOUNDATIONS.md):
rotation/squeezing/shear/displacement fusion -- one-parameter subgroups and the Weyl relation
D(a)D(b) = exp(-i (a_q b_p - a_p b_q)/4) D(a+b) (hbar = 2); symplectic/channel fusion --
composition of Gaussian maps; CZ + q-homodyne -- CZ = exp(i g q_u q_v / 2) commutes with q_u,
and <m|_u CZ = exp(i g m q_v / 2) <m|_u (the q-measurement vertex deletion of Menicucci,
Flammia and van Loock, PRA 83, 042335 (2011), Sec. III); phase-polynomial fusion and
displacement propagation -- the q-spider fusion (Eq. 70) and displacement (Eq. 79) rules of
Nagayoshi et al., PRR 7, 033141 (2025), here as operator identities exp(i f(q)) exp(i g(q)) =
exp(i (f + g)(q)) and exp(i f(q)) X(a) = X(a) exp(i f(q + a)) for X(a) = exp(-i a p / 2);
teleportation resynthesis -- the exact wire channel T(k_L)...T(k_1) of a corrected chain
(``photographiq.gaussian.wire_channel``) and an exact re-decomposition of its product.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .ir import (
    CZ,
    BeamSplitter,
    Box,
    ClassicalSource,
    Displacement,
    Feedforward,
    GaussianChannelBox,
    GaussianLoss,
    Homodyne,
    PhasePolynomial,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)
from .semantics import box_symplectic, compose_channels

TOL = 1e-12


@dataclass(frozen=True)
class RewriteStep:
    rule: str
    affected: tuple
    before: tuple
    after: tuple
    equivalence: str
    assumptions: str
    source: str
    scalar: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "rule": self.rule,
            "affected": list(self.affected),
            "before": list(self.before),
            "after": list(self.after),
            "equivalence": self.equivalence,
            "assumptions": self.assumptions,
            "source": self.source,
            "scalar": self.scalar,
            **self.extra,
        }


def _wires(box):
    return set(box.modes) | set(box.creates) | set(box.consumes)


def next_on(d, i, wire):
    """Index of the next box after i touching bosonic ``wire`` (None if none)."""
    for j in range(i + 1, len(d.boxes)):
        if wire in _wires(d.boxes[j]):
            return j
    return None


def prev_on(d, i, wire):
    for j in range(i - 1, -1, -1):
        if wire in _wires(d.boxes[j]):
            return j
    return None


def _replace_boxes(d, remove, insert_at, new, step, outputs=None):
    boxes = list(d.boxes)
    for k in sorted(remove, reverse=True):
        if k < insert_at:
            insert_at -= 1
        boxes.pop(k)
    boxes[insert_at:insert_at] = list(new)
    return d.with_boxes(boxes, step, outputs)


def _step(name, d, idx, new, equivalence, assumptions, source, scalar=None, **extra):
    return RewriteStep(
        name,
        tuple(idx),
        tuple(d.boxes[i].to_dict() for i in idx),
        tuple(b.to_dict() for b in new),
        equivalence,
        assumptions,
        source,
        scalar,
        extra,
    )


class Rule:
    name = ""
    equivalence = "exact-instrument"
    assumptions = ""
    source = ""

    def find(self, d):  # -> list of match tuples
        raise NotImplementedError

    def apply(self, d, match):
        raise NotImplementedError


# ----------------------------------------------------------------------------- fusion
class _Fuse(Rule):
    kind: Any = Box

    def fuse(self, a, b):
        raise NotImplementedError

    def find(self, d):
        out = []
        for i, b in enumerate(d.boxes):
            if type(b) is self.kind:
                j = next_on(d, i, b.modes[0])
                if j is not None and type(d.boxes[j]) is self.kind and d.boxes[j].modes == b.modes:
                    out.append((i, j))
        return out

    def apply(self, d, match):
        i, j = match
        new, scalar = self.fuse(d.boxes[i], d.boxes[j])
        step = _step(
            self.name, d, match, [new], self.equivalence, self.assumptions, self.source, scalar
        )
        return _replace_boxes(d, [i, j], j, [new], step)


class RotationFusion(_Fuse):
    name, kind = "rotation-fusion", Rotation
    assumptions = "Rotations on the same mode, wire-adjacent."
    source = "R(a) R(b) = R(a + b): one-parameter subgroup generated by the number operator."

    def fuse(self, a, b):
        return Rotation(a.mode, a.theta + b.theta), None


class SqueezingFusion(_Fuse):
    name, kind = "squeezing-fusion", Squeezer
    assumptions = "Same-axis (q) squeezers on the same mode, wire-adjacent."
    source = "S(r1) S(r2) = S(r1 + r2): one-parameter subgroup generated by (qp + pq)."

    def fuse(self, a, b):
        return Squeezer(a.mode, a.r + b.r), None


class ShearFusion(_Fuse):
    name, kind = "shear-fusion", Shear
    assumptions = "Quadratic phases on the same mode, wire-adjacent."
    source = "exp(i s1 q^2/4) exp(i s2 q^2/4) = exp(i (s1 + s2) q^2/4) (commuting functions of q)."

    def fuse(self, a, b):
        return Shear(a.mode, a.s + b.s), None


class DisplacementFusion(_Fuse):
    name, kind = "displacement-fusion", Displacement
    assumptions = "Constant displacements on the same mode, wire-adjacent."
    source = "Weyl relation D(b) D(a) = exp(i (a_q b_p - a_p b_q)/4) D(a + b) (hbar = 2)."

    def fuse(self, a, b):
        phase = (a.dq * b.dp - a.dp * b.dq) / 4
        return Displacement(
            a.mode, a.dq + b.dq, a.dp + b.dp
        ), f"global phase exp(i*{float(phase)!r})"


class FeedforwardMerge(_Fuse):
    name, kind = "feedforward-merge", Feedforward
    assumptions = "Outcome-affine displacements on the same mode, wire-adjacent."
    source = (
        "Displacements commute up to a phase; with outcome-dependent gains the phase is "
        "outcome dependent, i.e. a branch-global phase, invisible at the density level."
    )
    scalar = "branch-dependent global phase (density operators unchanged)"

    def fuse(self, a, b):
        gains = dict(a.gains)
        for k, (x, y) in b.gains:
            g = gains.get(k, (0.0, 0.0))
            gains[k] = (g[0] + x, g[1] + y)
        const = (a.constant[0] + b.constant[0], a.constant[1] + b.constant[1])
        return Feedforward(a.mode, tuple(gains.items()), const), self.scalar


class PhaseFusion(_Fuse):
    name, kind = "phase-polynomial-fusion", PhasePolynomial
    assumptions = (
        "Two exp(i f(q)) gates on the same mode, wire-adjacent; f, g real "
        "polynomials. Exact on the common invariant domain (Schwartz space)."
    )
    source = "Nagayoshi et al. PRR 7, 033141 (2025) q-spider fusion Eq. (70), 1->1 case."

    def fuse(self, a, b):
        n = max(len(a.coefficients), len(b.coefficients))
        ca = list(a.coefficients) + [0.0] * (n - len(a.coefficients))
        cb = list(b.coefficients) + [0.0] * (n - len(b.coefficients))
        return PhasePolynomial(a.mode, tuple(x + y for x, y in zip(ca, cb))), None


# ----------------------------------------------------------------------------- identities
def _is_identity(b):
    two_pi = lambda t: abs(np.remainder(t + np.pi, 2 * np.pi) - np.pi) < TOL  # noqa: E731
    if isinstance(b, Rotation):
        return two_pi(b.theta)
    if isinstance(b, Squeezer):
        return abs(b.r) < TOL
    if isinstance(b, Shear):
        return abs(b.s) < TOL
    if isinstance(b, Displacement):
        return abs(b.dq) < TOL and abs(b.dp) < TOL
    if isinstance(b, CZ):
        return abs(b.g) < TOL
    if isinstance(b, BeamSplitter):
        return two_pi(b.theta)
    if isinstance(b, Symplectic):
        return np.allclose(b.S, np.eye(len(b.S)), atol=TOL, rtol=0) and not np.any(
            np.abs(b.d) > TOL
        )
    if isinstance(b, PhasePolynomial):
        return len(b.coefficients) <= 1
    if isinstance(b, Feedforward):
        return (
            all(abs(x) < TOL and abs(y) < TOL for _, (x, y) in b.gains)
            and abs(b.constant[0]) < TOL
            and abs(b.constant[1]) < TOL
        )
    if isinstance(b, GaussianLoss):
        return b.eta == 1
    return False


class IdentityRemoval(Rule):
    name = "identity-removal"
    assumptions = "Box parameters equal the identity (angle 0 mod 2 pi, r = s = g = 0, ...)."
    source = "Each generator at its identity parameter is the identity operator/channel."

    def find(self, d):
        return [(i,) for i, b in enumerate(d.boxes) if _is_identity(b)]

    def apply(self, d, match):
        b = d.boxes[match[0]]
        scalar = (
            f"global phase exp(i*{float(b.coefficients[0])!r})"
            if isinstance(b, PhasePolynomial) and b.coefficients
            else None
        )
        step = _step(
            self.name, d, match, [], self.equivalence, self.assumptions, self.source, scalar
        )
        return _replace_boxes(d, [match[0]], match[0], [], step)


class Commute(Rule):
    name = "commute"
    assumptions = (
        "Adjacent boxes on disjoint bosonic wires; the second reads no record of the first."
    )
    source = "Operations on disjoint tensor factors commute; classical dependencies preserved."

    def find(self, d):
        out = []
        for i in range(len(d.boxes) - 1):
            a, b = d.boxes[i], d.boxes[i + 1]
            if (
                not (_wires(a) & _wires(b))
                and not (set(a.writes) & set(b.reads))
                and not (set(b.writes) & set(a.reads))
            ):
                out.append((i, i + 1))
        return out

    def apply(self, d, match):
        i, j = match
        step = _step(
            self.name,
            d,
            match,
            [d.boxes[j], d.boxes[i]],
            self.equivalence,
            self.assumptions,
            self.source,
        )
        boxes = list(d.boxes)
        boxes[i], boxes[j] = boxes[j], boxes[i]
        return d.with_boxes(boxes, step)


# ----------------------------------------------------------------------------- Gaussian fusion
class SymplecticFusion(Rule):
    name = "symplectic-fusion"
    assumptions = "Two Gaussian unitary boxes, the second the next box touching the first's modes."
    source = "Composition of Gaussian unitaries: (S2, d2) o (S1, d1) = (S2 S1, S2 d1 + d2)."

    def find(self, d):
        out = []
        for i, a in enumerate(d.boxes):
            if not (a.unitary and a.gaussian) or isinstance(a, Feedforward):
                continue
            nxt = [next_on(d, i, m) for m in a.modes]
            if None in nxt or len(set(nxt)) != 1:
                continue
            j = nxt[0]
            b = d.boxes[j]
            if b.unitary and b.gaussian and not isinstance(b, Feedforward):
                out.append((i, j))
        return out

    def apply(self, d, match):
        from ... import gaussian as gm

        i, j = match
        a, b = d.boxes[i], d.boxes[j]
        labels = tuple(dict.fromkeys(b.modes + a.modes))
        n = len(labels)

        def lift(box):
            S, t = box_symplectic(box)
            idx = [2 * labels.index(m) + k for m in box.modes for k in (0, 1)]
            full, shift = gm.embed(S, idx, 2 * n), np.zeros(2 * n)
            shift[idx] = t
            return full, shift

        S1, d1 = lift(a)
        S2, d2 = lift(b)
        new = Symplectic(labels, S2 @ S1, S2 @ d1 + d2)
        step = _step(
            self.name,
            d,
            match,
            [new],
            self.equivalence,
            self.assumptions,
            self.source,
            "global phase (unrecorded) from displacement ordering",
        )
        return _replace_boxes(d, [i, j], j, [new], step)


class ChannelFusion(Rule):
    name = "channel-fusion"
    equivalence = "exact-channel"
    assumptions = "Two single-mode Gaussian channels (loss, channel or unitary), wire-adjacent, one non-unitary."
    source = "Gaussian channel composition (X2 X1, X2 Y1 X2^T + Y2, X2 d1 + d2)."

    @staticmethod
    def _xyd(b):
        if isinstance(b, GaussianLoss):
            return (
                np.sqrt(b.eta) * np.eye(2),
                (1 - b.eta) * (2 * b.thermal + 1) * np.eye(2),
                np.zeros(2),
            )
        if isinstance(b, GaussianChannelBox):
            return b.X, b.Y, b.d
        S, t = box_symplectic(b)
        return S, np.zeros((2, 2)), t

    def _channel(self, b):
        return (
            isinstance(b, (GaussianLoss, GaussianChannelBox))
            or (b.unitary and b.gaussian and not isinstance(b, Feedforward))
        ) and len(b.modes) == 1

    def find(self, d):
        out = []
        for i, a in enumerate(d.boxes):
            if self._channel(a):
                j = next_on(d, i, a.modes[0])
                if (
                    j is not None
                    and self._channel(d.boxes[j])
                    and d.boxes[j].modes == a.modes
                    and (not a.unitary or not d.boxes[j].unitary)
                ):
                    out.append((i, j))
        return out

    def apply(self, d, match):
        i, j = match
        X, Y, t = compose_channels(self._xyd(d.boxes[i]), self._xyd(d.boxes[j]))
        new = GaussianChannelBox(d.boxes[i].modes, X, (Y + Y.T) / 2, t)
        step = _step(self.name, d, match, [new], self.equivalence, self.assumptions, self.source)
        return _replace_boxes(d, [i, j], j, [new], step)


# ----------------------------------------------------------------------------- measurements
def _q_sign(theta):
    if abs(np.sin(theta)) > TOL:
        return None
    return float(np.sign(np.cos(theta)))


class CZMeasurementToFeedforward(Rule):
    name = "cz-q-measurement-to-feedforward"
    assumptions = (
        "CZ(u, v, g) followed on u (wire-adjacent) by an ideal homodyne of q_u "
        "(theta = 0 or pi); any state of u and v."
    )
    source = (
        "CZ = exp(i g q_u q_v / 2) commutes with q_u and <m|_u CZ = exp(i g m q_v / 2) <m|_u "
        "(MFvL PRA 83, 042335, Sec. III: q measurement deletes the vertex)."
    )

    def find(self, d):
        out = []
        for i, b in enumerate(d.boxes):
            if isinstance(b, CZ):
                for k, other in ((b.u, b.v), (b.v, b.u)):
                    j = next_on(d, i, k)
                    if (
                        j is not None
                        and isinstance(d.boxes[j], Homodyne)
                        and d.boxes[j].ideal
                        and _q_sign(d.boxes[j].theta) is not None
                    ):
                        out.append((i, j, k, other))
        return out

    def apply(self, d, match):
        i, j, k, other = match
        cz, hom = d.boxes[i], d.boxes[j]
        sign = _q_sign(hom.theta)
        new = [hom, Feedforward(other, ((hom.key, (0.0, cz.g * sign)),))]
        step = _step(self.name, d, (i, j), new, self.equivalence, self.assumptions, self.source)
        return _replace_boxes(d, [i, j], i, new, step)


class PrepareMeasureToSource(Rule):
    name = "prepare-measure-to-source"
    assumptions = "A single-mode Gaussian preparation measured next by a homodyne (any efficiency)."
    source = (
        "The record of a homodyne on a Gaussian state (mean m, covariance V) is "
        "N(h.m, h V h + (1-eta)/eta + noise); no quantum wire survives."
    )

    def find(self, d):
        out = []
        for i, b in enumerate(d.boxes):
            if isinstance(b, Prepare):
                j = next_on(d, i, b.mode)
                if j is not None and isinstance(d.boxes[j], Homodyne):
                    out.append((i, j))
        return out

    def apply(self, d, match):
        i, j = match
        prep, hom = d.boxes[i], d.boxes[j]
        h = np.array([np.cos(hom.theta), np.sin(hom.theta)])
        var = float(
            h @ np.asarray(prep.covariance) @ h + (1 - hom.efficiency) / hom.efficiency + hom.noise
        )
        new = ClassicalSource(hom.key, float(h @ np.asarray(prep.mean)), var)
        step = _step(self.name, d, match, [new], self.equivalence, self.assumptions, self.source)
        return _replace_boxes(d, [i, j], j, [new], step)


class DisplacementPropagation(Rule):
    """Move a q displacement before exp(i f(q)): exp(i f(q)) X(a) = X(a) exp(i f(q + a))."""

    name = "displacement-through-phase-polynomial"
    assumptions = (
        "Displacement(dq=a, dp=0) followed (wire-adjacent) by exp(i f(q)), f real "
        "polynomial; exact operator identity on the Schwartz space."
    )
    source = (
        "X(a)^dagger f(q) X(a) = f(q + a); Nagayoshi et al. PRR 7, 033141 (2025) "
        "displacement rule Eq. (79), 1->1 case."
    )

    def find(self, d):
        out = []
        for i, b in enumerate(d.boxes):
            if isinstance(b, PhasePolynomial):
                j = prev_on(d, i, b.mode)
                if (
                    j is not None
                    and isinstance(d.boxes[j], Displacement)
                    and d.boxes[j].dp == 0
                    and d.boxes[j].dq != 0
                ):
                    out.append((j, i))
        return out

    def apply(self, d, match):
        from math import comb

        j, i = match
        disp, phase = d.boxes[j], d.boxes[i]
        a, c = disp.dq, phase.coefficients
        shifted = [
            sum(c[n] * comb(n, k) * a ** (n - k) for n in range(k, len(c))) for k in range(len(c))
        ]
        new = [PhasePolynomial(phase.mode, tuple(shifted)), disp]
        # order: exp(i f(q)) after X(a)  ==  exp(i f(q + a)) before X(a)
        step = _step(self.name, d, match, new, self.equivalence, self.assumptions, self.source)
        return _replace_boxes(d, [j, i], j, new, step)


# ----------------------------------------------------------------------------- MBQC (logical)
def _teleport_matrix(k):
    return np.array([[-k, -1.0], [1.0, 0.0]])


def teleport_steps(d):
    """Standard corrected teleportation steps (compile_circuit wire steps) in a diagram.

    Step: Prepare.squeezed(b, r); CZ(a, b, 1); Homodyne(a, theta, key) ideal;
    Feedforward(b, {key: (-1/sin theta, 0)}), all wire-adjacent; shear k = cot(theta).
    """
    steps = []
    for j, hom in enumerate(d.boxes):
        if not isinstance(hom, Homodyne) or not hom.ideal or abs(np.sin(hom.theta)) < 1e-9:
            continue
        a = hom.mode
        i = prev_on(d, j, a)
        if i is None or not isinstance(d.boxes[i], CZ) or abs(d.boxes[i].g - 1) > TOL:
            continue
        cz = d.boxes[i]
        b = cz.v if cz.u == a else cz.u
        p = prev_on(d, i, b)
        f = next_on(d, i, b)
        if (
            p is None
            or f is None
            or not isinstance(d.boxes[p], Prepare)
            or d.boxes[p].squeezing is None
        ):
            continue
        ff = d.boxes[f]
        expected = -1 / np.sin(hom.theta)
        if (
            not isinstance(ff, Feedforward)
            or len(ff.gains) != 1
            or ff.gains[0][0] != hom.key
            or abs(ff.gains[0][1][0] - expected) > 1e-9 * max(1, abs(expected))
            or abs(ff.gains[0][1][1]) > TOL
            or any(abs(x) > TOL for x in ff.constant)
        ):
            continue
        readers = [m for m, box in enumerate(d.boxes) if hom.key in box.reads and m != f]
        if readers:
            continue
        steps.append(
            {
                "input": a,
                "output": b,
                "k": np.cos(hom.theta) / np.sin(hom.theta),
                "r": d.boxes[p].squeezing,
                "boxes": (p, i, j, f),
                "key": hom.key,
            }
        )
    return steps


def _short_decomposition(S, atol=1e-9):
    """Fewest shears k with T(k_L)...T(k_1) = S, trying L = 0, 1, 2, 3, then the compiler's."""
    from ...compiler import decompose_symplectic

    if np.allclose(S, np.eye(2), atol=atol):
        return []
    candidates = []
    if abs(S[1, 1]) < atol and abs(S[0, 1] + 1) < atol and abs(S[1, 0] - 1) < atol:
        candidates.append([-S[0, 0]])
    if abs(S[1, 1] + 1) < atol:
        candidates.append([-S[1, 0], S[0, 1]])
    if abs(S[1, 1]) > atol:
        k2 = S[1, 1]
        candidates.append([(S[1, 0] + 1) / k2, k2, (1 - S[0, 1]) / k2])
    candidates.append(list(decompose_symplectic(S)))
    for ks in sorted(candidates, key=len):
        P = np.eye(2)
        for k in ks:
            P = _teleport_matrix(k) @ P
        if np.allclose(P, S, atol=atol * max(1, np.abs(S).max()), rtol=0):
            return [float(k) for k in ks]
    raise ArithmeticError("No verified teleportation decomposition")


class TeleportResynthesis(Rule):
    """Replace a chain of corrected teleportation steps by the fewest steps with the same map."""

    name = "teleport-chain-resynthesis"
    equivalence = "logical"
    assumptions = (
        "A maximal chain of standard corrected wire steps on one logical wire, equal "
        "resource squeezing, records read only by their own corrections. The ideal "
        "map T(k_L)...T(k_1) and the displacement are preserved exactly; the finite-"
        "squeezing noise Y and the record set change (reported by the verifier)."
    )
    source = (
        "Exact unconditional wire channel (S = prod T(k_i), Y = sum of propagated "
        "diag(0, e^{-2r})) and a verified exact re-decomposition of S."
    )

    def chains(self, d):
        steps = teleport_steps(d)
        by_input = {s["input"]: s for s in steps}
        heads = [s for s in steps if not any(t["output"] == s["input"] for t in steps)]
        chains = []
        for head in heads:
            chain = [head]
            while chain[-1]["output"] in by_input:
                nxt = by_input[chain[-1]["output"]]
                if nxt["boxes"][1] != next_on(d, chain[-1]["boxes"][3], chain[-1]["output"]):
                    break  # something else acts on the wire between the steps
                chain.append(nxt)
            if len({s["r"] for s in chain}) == 1:
                chains.append(chain)
        return chains

    def find(self, d):
        out = []
        for chain in self.chains(d):
            S = np.eye(2)
            for s in chain:
                S = _teleport_matrix(s["k"]) @ S
            ks = _short_decomposition(S)
            if len(ks) < len(chain):
                out.append((tuple(s["boxes"] for s in chain), tuple(ks)))
        return out

    def apply(self, d, match):
        chain_boxes, ks = match
        steps = {s["boxes"]: s for s in teleport_steps(d)}
        chain = [steps[b] for b in chain_boxes]
        r = chain[0]["r"]
        a0, outputs = chain[0]["input"], [s["output"] for s in chain]
        L, Lp = len(chain), len(ks)
        new = []
        nodes = [a0] + outputs[L - Lp :]
        keys = [chain[0]["key"]] + [chain[m]["key"] for m in range(L - Lp + 1, L)]
        for t, k in enumerate(ks):
            a, b = nodes[t], nodes[t + 1]
            theta = float(np.arctan2(1.0, k))
            new += [
                Prepare.squeezed(b, r),
                CZ(a, b, 1.0),
                Homodyne(a, theta, keys[t]),
                Feedforward(b, ((keys[t], (-1 / np.sin(theta), 0.0)),)),
            ]
        remove = sorted(i for boxes in chain_boxes for i in boxes)
        step = _step(
            self.name,
            d,
            remove,
            new,
            self.equivalence,
            self.assumptions,
            self.source,
            None,
            steps_before=L,
            steps_after=Lp,
        )
        # insert where the chain first touches its input wire (its first CZ): the input is
        # alive there and nothing between touches the chain wires or reads its records
        anchor = chain_boxes[0][1]
        boxes = [b for i, b in enumerate(d.boxes) if i not in remove]
        at = anchor - sum(1 for i in remove if i < anchor)
        boxes[at:at] = new
        new_outputs: tuple = d.outputs
        if Lp == 0:  # identity chain: the input wire continues under its own label
            last = chain[-1]["output"]
            boxes = [_relabel(x, last, a0) for x in boxes]
            new_outputs = tuple(a0 if o == last else o for o in new_outputs)
        return d.with_boxes(boxes, step, new_outputs)


def _relabel(box, old, new):
    changes = {}
    for f in ("mode", "u", "v"):
        if getattr(box, f, None) == old:
            changes[f] = new
    if hasattr(box, "labels") and old in box.labels:
        changes["labels"] = tuple(new if x == old else x for x in box.labels)
    return replace(box, **changes) if changes else box


GAUSSIAN_RULES = (
    RotationFusion(),
    SqueezingFusion(),
    ShearFusion(),
    DisplacementFusion(),
    IdentityRemoval(),
    SymplecticFusion(),
    ChannelFusion(),
    FeedforwardMerge(),
    CZMeasurementToFeedforward(),
    PrepareMeasureToSource(),
    Commute(),
)
NON_GAUSSIAN_RULES = (PhaseFusion(), DisplacementPropagation())
MBQC_LOGICAL_RULES = (TeleportResynthesis(),)
RULES = {r.name: r for r in GAUSSIAN_RULES + NON_GAUSSIAN_RULES + MBQC_LOGICAL_RULES}


def apply_rewrite(diagram, rule, match=None):
    """Apply ``rule`` (name or Rule) at ``match`` (default: its first match); returns a new diagram."""
    rule = RULES[rule] if isinstance(rule, str) else rule
    matches = rule.find(diagram)
    if match is None:
        if not matches:
            raise ValueError(f"Rule {rule.name} does not match")
        match = matches[0]
    elif match not in matches:
        raise ValueError(f"{match!r} is not a match of {rule.name}")
    return rule.apply(diagram, match)
