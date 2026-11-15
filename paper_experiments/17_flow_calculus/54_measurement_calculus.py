"""R54: Qubit measurement calculus (DKP) with full branch semantics; CV command view only.

Scientific question: do the measurement-calculus rewrites of ``photographiq.calculus.qubit``
(standardization to NEMC form, signal shifting) preserve the full process semantics, branch
by branch including every measurement outcome, and are adaptive measurements, corrections and
classical dependencies tracked correctly? For CV patterns: are commands classified correctly
and displacement corrections extracted as exact affine forms of the records? (No complete CV
measurement calculus is implemented or claimed.)

Qubit semantics (DKP, Danos-Kashefi-Panangaden 2007): N |+>, E CZ, M^{s,t}(plane, a) measures
the observable Z^t X^s O(plane, a) X^s Z^t with s, t the parities of its domains (outcome 0 =
+1 eigenvalue; XY: cos X + sin Y, XZ: cos Z + sin X, YZ: cos Z + sin Y), X/Z apply iff
their domain parity is 1, S flips a stored signal. Each branch is keyed by the FINAL signals of
the measured qubits (signal shifting relabels raw outcomes and restores the signals with S).

References computed in this script:

1. Independent dense simulator: Choi vector of each branch (one reference qubit per input in a
   Bell pair), measuring the conjugated observable built as a matrix (not via an angle table)
   and projecting on its (-1)^r eigenvector. Branches compared up to a branch phase (the
   eigenvector phase is a convention); probabilities ||v_b||^2 compared without any freedom.
2. Graphix 0.4 statevector simulator with ``FixedBranchSelector`` on the original patterns
   (angles in units of pi; same plane conventions), random normalized input state.
3. Library ``run_branch`` operators, compared with (1).

Patterns: seeded random valid non-standard patterns (interleaved N/E/M/X/Z, random planes,
angles and s/t domains: adaptive measurements and corrections), and flow-derived standard
patterns (``pattern_from_flow`` from causal flow / gflow / Pauli flow certificates).

CV part: ``photographiq.calculus.cv.command_view`` on CV-flow corrected patterns: role
classification counts against the command list, and every affine correction form evaluated on
random records against direct evaluation of the original expression; a deliberately nonlinear
feed-forward must be reported as non-affine.

Seed 20261154. Limitations: dense simulation (<= 12 simulated qubits); random patterns have 2
inputs, 3 ancillas, 2 outputs; Graphix comparison only where its commands coincide (no S).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

common.require_module("graphix", "R54 compares branch semantics against Graphix 0.4")
from graphix import command as gc  # noqa: E402
from graphix.branch_selector import FixedBranchSelector  # noqa: E402
from graphix.measurements import Measurement  # noqa: E402
from graphix.pattern import Pattern as GPattern  # noqa: E402
from graphix.sim.statevec import Statevector  # noqa: E402

import photographiq as pg  # noqa: E402
from photographiq.calculus import cv as cvcalc  # noqa: E402
from photographiq.calculus import qubit as q  # noqa: E402
from photographiq.flows import find_cv_flow, find_causal_flow, find_gflow, find_pauli_flow, resource_graph  # noqa: E402
from photographiq.flows.cv import corrected_pattern  # noqa: E402
from photographiq.ir import HomodyneSpec, Model, OpenGraph, Plane, QubitMeasurement  # noqa: E402

SEED = 20261154
N_RANDOM = 30  # quick; --full: 120
PLANES = (Plane.XY, Plane.YZ, Plane.XZ)

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_BRANCH = 1e-10  # branch Choi vectors / Kraus maps up to phase (float64, <= 12 qubits)
TOL_PROB = 1e-12  # branch probabilities (no phase freedom)
TOL_GRAPHIX = 1e-10  # 1 - |<graphix|ours>| on normalized branch states
TOL_AFFINE = 1e-12
# Sensitivity: dropping one output Pauli correction must change some branch by > 1e-3 in at
# least 80% of patterns (it is invisible only if the output is an eigenstate of that Pauli).
NEG_CONTROL_MIN_DEV, NEG_CONTROL_MIN_FRACTION = 1e-3, 0.8  # affine form vs direct expression evaluation, relative to max(1, |value|)

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0 + 0j, -1.0])


# ----- patterns -----------------------------------------------------------------------------
def subset(rng, items):
    return frozenset(x for x in items if rng.random() < 0.5)


def random_pattern(rng, n_in=2, n_aux=3, n_out=2, steps=18):
    inputs = list(range(n_in))
    live, measured, cmds = list(inputs), [], []
    fresh, remaining = iter(range(n_in, n_in + n_aux)), n_aux
    for _ in range(steps):
        kind = rng.choice(["N", "E", "E", "M", "X", "Z"])
        if kind == "N" and remaining:
            node = next(fresh)
            remaining -= 1
            cmds.append(q.N(node))
            live.append(node)
        elif kind == "E" and len(live) >= 2:
            u, v = rng.choice(len(live), 2, replace=False)
            cmds.append(q.E(live[u], live[v]))
        elif kind == "M" and live and len(live) + remaining > n_out:
            node = live.pop(int(rng.integers(len(live))))
            plane = PLANES[int(rng.integers(3))]
            cmds.append(q.M(node, plane, float(rng.uniform(-math.pi, math.pi)), subset(rng, measured), subset(rng, measured)))
            measured.append(node)
        elif kind in "XZ" and measured and live:
            cls = q.X if kind == "X" else q.Z
            cmds.append(cls(live[int(rng.integers(len(live)))], subset(rng, measured) or {measured[-1]}))
    for _ in range(remaining):
        node = next(fresh)
        cmds.append(q.N(node))
        live.append(node)
    while len(live) > n_out:
        node = live.pop(0)
        cmds.append(q.M(node, PLANES[int(rng.integers(3))], float(rng.uniform(-3, 3)), subset(rng, measured)))
        measured.append(node)
    return q.QubitPattern(inputs, cmds, tuple(live))


def flow_patterns():
    def og(edges, inputs, outputs, specs=None):
        g = nx.Graph(list(edges))
        specs = specs or {}
        meas = {n: specs.get(n, QubitMeasurement(Plane.XY, 0.3 + 0.17 * i)) for i, n in enumerate(g.nodes) if n not in outputs}
        return OpenGraph(Model.QUBIT, g, inputs, outputs, meas)

    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3), ordering="sorted")
    cases = {
        "flow_line5_causal": (og(nx.path_graph(5).edges, (0,), (4,)), find_causal_flow),
        "flow_cluster2x3_causal": (og(grid.edges, (0, 3), (2, 5)), find_causal_flow),
        "flow_gflow_not_causal": (og([(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4)], (3, 2), (0, 4)), find_gflow),
        "flow_xz_yz_gflow": (og(list(nx.path_graph(4).edges) + [(2, 5)], (), (3,), {0: QubitMeasurement(Plane.XZ, 0.4), 5: QubitMeasurement(Plane.YZ, 0.9)}), find_gflow),
        "flow_pauli_y_z": (og([(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (2, 5)], (0,), (4, 5),
                              {1: QubitMeasurement(Plane.YZ, math.pi / 2, "Y"), 2: QubitMeasurement(Plane.XZ, 0.0, "Z"),
                               3: QubitMeasurement(Plane.XY, 0.9)}), find_pauli_flow),
    }
    out = {}
    for name, (graph, finder) in cases.items():
        result = finder(graph)
        assert result.certified, name
        out[name] = q.pattern_from_flow(graph, result)
    return out


# ----- independent simulator ----------------------------------------------------------------
def _apply(state, op, axes):
    k = len(axes)
    out = np.tensordot(op.reshape((2,) * 2 * k), state, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes)


def observable(plane, a):
    a1, a2 = {Plane.XY: (_X, _Y), Plane.XZ: (_Z, _X), Plane.YZ: (_Z, _Y)}[Plane(plane)]
    return math.cos(a) * a1 + math.sin(a) * a2


def parity(domain, signals):
    return sum(signals[j] for j in domain) % 2


def branches(pattern):
    """{final-signal tuple over sorted measured nodes: Choi vector (outputs, refs)}."""
    measured = [c.node for c in pattern.commands if isinstance(c, q.M)]
    key_nodes = sorted(measured, key=repr)
    out = {}
    for raw in itertools.product((0, 1), repeat=len(measured)):
        outcome = dict(zip(measured, raw))
        alive = list(pattern.inputs) + [("ref", n) for n in pattern.inputs]
        state = np.zeros((2,) * len(alive), complex)
        state[(0,) * len(alive)] = 1
        h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        cnot = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
        for n in pattern.inputs:
            state = _apply(state, h, [alive.index(n)])
            state = _apply(state, cnot, [alive.index(n), alive.index(("ref", n))])
        signals = {}
        for c in pattern.commands:
            if isinstance(c, q.N):
                state = np.multiply.outer(state, np.array([1, 1]) / math.sqrt(2))
                alive.append(c.node)
            elif isinstance(c, q.E):
                state = _apply(state, np.diag([1, 1, 1, -1]).astype(complex), [alive.index(c.u), alive.index(c.v)])
            elif isinstance(c, q.M):
                s, t = parity(c.s_domain, signals), parity(c.t_domain, signals)
                frame = np.linalg.matrix_power(_X, s) @ np.linalg.matrix_power(_Z, t)
                obs = frame.conj().T @ observable(c.plane, c.angle) @ frame
                vals, vecs = np.linalg.eigh(obs)
                r = outcome[c.node]
                phi = vecs[:, int(np.argmin(vals))] if r else vecs[:, int(np.argmax(vals))]
                state = np.tensordot(phi.conj(), state, axes=([0], [alive.index(c.node)]))
                alive.remove(c.node)
                signals[c.node] = r
            elif isinstance(c, (q.X, q.Z)):
                if parity(c.domain, signals):
                    state = _apply(state, _X if isinstance(c, q.X) else _Z, [alive.index(c.node)])
            elif isinstance(c, q.S):
                signals[c.node] ^= parity(c.domain, signals)
        final = list(pattern.outputs) + [("ref", n) for n in pattern.inputs]
        vec = np.transpose(state, [alive.index(n) for n in final]).reshape(-1)
        out[tuple(signals[n] for n in key_nodes)] = vec
    return out


def phase_dev(a, b):
    ov = np.vdot(b, a)
    ph = ov / abs(ov) if abs(ov) > 1e-300 else 1.0
    return float(np.abs(a - ph * b).max(initial=0.0))


def compare_branches(ba, bb):
    dev = prob = 0.0
    for k, v in ba.items():
        dev = max(dev, phase_dev(v, bb[k]))
        prob = max(prob, abs(np.vdot(v, v).real - np.vdot(bb[k], bb[k]).real))
    return dev, prob


def to_graphix(pattern):
    planes = {Plane.XY: Measurement.XY, Plane.YZ: Measurement.YZ, Plane.XZ: Measurement.XZ}
    g = GPattern(input_nodes=list(pattern.inputs))
    for c in pattern.commands:
        if isinstance(c, q.N):
            g.add(gc.N(c.node))
        elif isinstance(c, q.E):
            g.add(gc.E((c.u, c.v)))
        elif isinstance(c, q.M):
            g.add(gc.M(c.node, planes[c.plane](c.angle / math.pi), set(c.s_domain), set(c.t_domain)))
        elif isinstance(c, q.X):
            g.add(gc.X(c.node, set(c.domain)))
        elif isinstance(c, q.Z):
            g.add(gc.Z(c.node, set(c.domain)))
        else:
            raise TypeError(c)
    g.reorder_output_nodes(list(pattern.outputs))
    return g


def graphix_check(pattern, rng):
    """1 - |<graphix branch|ours branch>| over all branches with p > 1e-6, random input."""
    g = to_graphix(pattern)
    n_in = len(pattern.inputs)
    psi = rng.normal(size=2**n_in) + 1j * rng.normal(size=2**n_in)
    psi /= np.linalg.norm(psi)
    worst, checked = 0.0, 0
    for b in q.enumerate_branches(pattern):
        ours = b.operator @ psi
        if np.linalg.norm(ours) < 1e-3:
            continue
        ours /= np.linalg.norm(ours)
        sel = FixedBranchSelector({n: bool(r) for n, r in b.outcomes.items()})
        theirs = np.asarray(g.simulate(input_state=Statevector(data=psi), branch_selector=sel).flatten())
        worst = max(worst, abs(abs(np.vdot(theirs, ours)) - 1))
        checked += 1
    return worst, checked


def dependency_ok(pattern):
    """Edges j -> i iff j in the s/t domain of M_i; schedule layers depend only on earlier layers."""
    dg = pattern.dependency_graph()
    expected = {(j, c.node) for c in pattern.commands if isinstance(c, q.M) for j in c.s_domain | c.t_domain}
    layers = pattern.schedule()
    seen, ok = set(), set(dg.edges) == expected
    for layer in layers:
        for n in layer:
            ok &= all(j in seen for j, _ in dg.in_edges(n))
        seen |= set(layer)
    return ok, len(layers)


def qubit_row(name, pattern, rng, kind):
    trace = []
    std = q.standardize(pattern, trace=trace)
    shifted = q.signal_shift(std)
    b0, b1, b2 = branches(pattern), branches(std), branches(shifted)
    d_std, p_std = compare_branches(b0, b1)
    d_shift, p_shift = compare_branches(b0, b2)
    # library run_branch vs independent simulator (Choi vector = K / sqrt(2^n_in))
    lib = q.branch_operators(pattern, sorted(pattern.measured, key=repr))
    scale = math.sqrt(2 ** len(pattern.inputs))
    d_lib = max(phase_dev(lib[k].reshape(-1), v.reshape(len(lib[k]), -1).reshape(-1) * scale) for k, v in b0.items())
    gx_dev, gx_n = graphix_check(pattern, rng)
    dep_ok, depth = dependency_ok(pattern)
    dep_std_ok, depth_std = dependency_ok(std)
    dep_shift_ok, depth_shift = dependency_ok(shifted)
    m = len(pattern.measured)
    probs = [float(np.vdot(v, v).real) for v in b0.values()]
    ref = next(iter(b0.values()))
    det = max(max(abs(p - 2.0**-m) for p in probs),
              max(phase_dev(v / np.linalg.norm(v), ref / np.linalg.norm(ref)) for v in b0.values()))
    rules = Counter(r for r, _ in trace)
    # negative control: drop the last Pauli correction of the standard form; the simulator must see it
    corr_idx = [i for i, c in enumerate(std.commands) if isinstance(c, (q.X, q.Z))]
    control = None
    if corr_idx:
        mutated = std.copy([c for i, c in enumerate(std.commands) if i != corr_idx[-1]])
        control = compare_branches(b0, branches(mutated))[0]
    return {
        "pattern": name,
        "kind": kind,
        "commands": len(pattern.commands),
        "measurements": m,
        "adaptive_measurements": sum(1 for c in pattern.commands if isinstance(c, q.M) and (c.s_domain or c.t_domain)),
        "corrections": sum(isinstance(c, (q.X, q.Z)) for c in pattern.commands),
        "was_standard": pattern.is_standard(),
        "standardized_is_standard": std.is_standard(),
        "rewrite_steps": len(trace),
        "rules": "; ".join(f"{k}:{v}" for k, v in sorted(rules.items())),
        "branches": 2**m,
        "std_branch_dev": d_std,
        "std_prob_dev": p_std,
        "shift_branch_dev": d_shift,
        "shift_prob_dev": p_shift,
        "library_vs_independent": d_lib,
        "graphix_dev": gx_dev,
        "graphix_branches_checked": gx_n,
        "dependency_tracking_ok": dep_ok and dep_std_ok and dep_shift_ok,
        "depth_original": depth,
        "depth_standardized": depth_std,
        "depth_signal_shifted": depth_shift,
        "strong_determinism_dev": det,
        "library_is_deterministic": bool(q.is_deterministic(pattern).deterministic),
        "negative_control_dev": control,
    }


# ----- CV command view --------------------------------------------------------------------
def cv_rows(rng):
    def og(edges, inputs, outputs, shears):
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        return OpenGraph(Model.CV_GAUSSIAN, g, inputs, outputs,
                         {n: HomodyneSpec(shears.get(n, 0.0)) for n in g if n not in outputs}, {n: 1.0 for n in g})

    graphs = {
        "cv_wire4": og([(0, 1, 1.0), (1, 2, 0.8), (2, 3, 1.2)], (0,), (3,), {0: 0.2, 1: -0.4, 2: 0.7}),
        "cv_cluster2x3": og([((0, 0), (0, 1), 1.1), ((0, 1), (0, 2), 0.7), ((1, 0), (1, 1), 1.3), ((1, 1), (1, 2), 0.9),
                             ((0, 0), (1, 0), 1.0), ((0, 1), (1, 1), 0.8), ((0, 2), (1, 2), 1.2)],
                            ((0, 0), (1, 0)), ((0, 2), (1, 2)), {(0, 0): 0.3, (1, 1): -0.2}),
    }
    rows = []
    for name, graph in graphs.items():
        flow = find_cv_flow(graph, verify=False)
        pattern = corrected_pattern(resource_graph(graph), flow.measurement_order, flow.correction_map, flow.certificate["shears"])
        rows.append(cv_view_row(name, pattern, rng, expect_affine=True))
    bad = pg.Pattern(inputs=(0,))
    from photographiq.commands import Entangle, Prepare
    bad.extend([Prepare(1, 1.0), Entangle(0, 1)])
    bad.measure(0, pg.Homodyne(0.4))
    bad.displace(1, q=pg.Outcome(0) * pg.Outcome(0), p=0.5 * pg.Outcome(0) + 0.1)
    rows.append(cv_view_row("cv_nonlinear_feedforward", bad.validate(), rng, expect_affine=False))
    return rows


def cv_view_row(name, pattern, rng, expect_affine):
    from photographiq.commands import Displace, Measure, Signal
    from photographiq.expressions import Expr
    view = cvcalc.command_view(pattern)
    roles = Counter(v.role.value for v in view.commands)
    n_meas = sum(isinstance(c, Measure) for c in pattern.commands)
    n_ff = sum(isinstance(c, Displace) and bool(getattr(c.q, "dependencies", ()) or getattr(c.p, "dependencies", ()))
               for c in pattern.commands)
    n_cls = sum(isinstance(c, Signal) for c in pattern.commands)
    keys = [o.key for o in view.outcomes]
    worst, affine, nonaffine = 0.0, 0, 0
    for corr in view.corrections:
        cmd = pattern.commands[corr.index]
        for comp, raw in (("q", cmd.q), ("p", cmd.p)):
            form = getattr(corr, comp)
            if not isinstance(form, cvcalc.AffineForm):
                nonaffine += 1
                continue
            affine += 1
            for _ in range(5):
                records = {k: float(rng.normal(0, 3)) for k in keys}
                direct = raw.evaluate(records, {}) if isinstance(raw, Expr) else float(raw)
                worst = max(worst, abs(form.evaluate(records) - direct) / max(1.0, abs(direct)))
    return {
        "pattern": name,
        "kind": "cv-command-view",
        "commands": len(pattern.commands),
        "measurements": n_meas,
        "role_measurement": roles.get("measurement", 0),
        "role_feedforward": roles.get("feedforward", 0),
        "role_classical": roles.get("classical", 0),
        "role_quantum": roles.get("quantum", 0),
        "expected_feedforward": n_ff,
        "expected_classical": n_cls,
        "outcome_variables": len(view.outcomes),
        "continuous_outcomes": sum(o.continuous for o in view.outcomes),
        "affine_components": affine,
        "nonaffine_components": nonaffine,
        "affine_eval_max_rel_error": worst,
        "expect_affine": expect_affine,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="120 random patterns")
    args = parser.parse_args()
    n_random = 120 if args.full else N_RANDOM
    plt = common.setup_style()
    rng = common.rng(SEED)
    rows = []
    for i in range(n_random):
        rows.append(qubit_row(f"random_{i:03d}", random_pattern(rng), rng, "random non-standard"))
    for name, pattern in flow_patterns().items():
        rows.append(qubit_row(name, pattern, rng, "flow-derived"))
    cvr = cv_rows(rng)

    failures = []
    for r in rows:
        p = []
        if max(r["std_branch_dev"], r["shift_branch_dev"]) > TOL_BRANCH:
            p.append("rewrite changed a branch map")
        if max(r["std_prob_dev"], r["shift_prob_dev"]) > TOL_PROB:
            p.append("rewrite changed a branch probability")
        if r["library_vs_independent"] > TOL_BRANCH:
            p.append("library branch operator differs from independent simulator")
        if r["graphix_dev"] > TOL_GRAPHIX or r["graphix_branches_checked"] == 0:
            p.append("Graphix branch states differ")
        if not r["standardized_is_standard"] or not r["dependency_tracking_ok"]:
            p.append("not standard / dependency tracking wrong")
        if r["kind"] == "flow-derived" and (r["strong_determinism_dev"] > TOL_BRANCH or not r["library_is_deterministic"]):
            p.append("flow-derived pattern not strongly deterministic")
        if r["kind"] != "flow-derived" and r["library_is_deterministic"] != (r["strong_determinism_dev"] <= TOL_BRANCH):
            p.append("library determinism verdict disagrees with independent simulator")
        if p:
            failures.append({**r, "problems": "; ".join(p)})
    for r in cvr:
        p = []
        if r["role_measurement"] != r["measurements"] or r["role_feedforward"] != r["expected_feedforward"] \
                or r["role_classical"] != r["expected_classical"] or r["continuous_outcomes"] != r["outcome_variables"]:
            p.append("command classification mismatch")
        if r["affine_eval_max_rel_error"] > TOL_AFFINE:
            p.append("affine form disagrees with expression")
        if r["expect_affine"] and r["nonaffine_components"]:
            p.append("affine correction reported non-affine")
        if not r["expect_affine"] and r["nonaffine_components"] == 0:
            p.append("nonlinear feed-forward reported as affine")
        if p:
            failures.append({**r, "problems": "; ".join(p)})

    controls = [r["negative_control_dev"] for r in rows if r["negative_control_dev"] is not None]
    detected = sum(d > NEG_CONTROL_MIN_DEV for d in controls)
    if detected < NEG_CONTROL_MIN_FRACTION * len(controls):
        failures.append({"pattern": "negative_control", "problems": f"only {detected}/{len(controls)} mutations detected"})
    worst = {k: max(r[k] for r in rows) for k in ("std_branch_dev", "shift_branch_dev", "std_prob_dev", "shift_prob_dev",
                                                  "library_vs_independent", "graphix_dev")}
    total_branches = sum(r["branches"] for r in rows)
    gx_branches = sum(r["graphix_branches_checked"] for r in rows)
    rule_totals = Counter()
    for r in rows:
        for item in filter(None, r["rules"].split("; ")):
            k, v = item.split(":")
            rule_totals[k] += int(v)
    rule_totals = dict(rule_totals)
    common.save_result(rows + cvr, "R54_measurement_calculus", extra={
        "seed": SEED,
        "backend": "independent dense Choi-vector simulator (NumPy); graphix 0.4 statevector; photographiq.calculus.qubit",
        "precision": "float64",
        "parameters": {"n_random": n_random, "random_pattern": "2 inputs, 3 ancillas, 2 outputs, 18 steps"},
        "acceptance": {"TOL_BRANCH": TOL_BRANCH, "TOL_PROB": TOL_PROB, "TOL_GRAPHIX": TOL_GRAPHIX, "TOL_AFFINE": TOL_AFFINE},
        "summary": {"qubit_patterns": len(rows), "branches_compared": total_branches, "graphix_branches": gx_branches,
                    "worst": worst, "rule_counts": rule_totals,
                    "negative_control_detected": f"{detected}/{len(controls)}"},
        "scope_note": "CV: command classification and affine correction extraction only; no CV measurement calculus.",
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    idx = np.arange(len(rows))
    floor = 1e-17
    for k, (key, lab, mk) in enumerate((("std_branch_dev", "standardize vs original", "o"), ("shift_branch_dev", "signal shift vs original", "s"),
                         ("library_vs_independent", "library vs independent sim.", "^"), ("graphix_dev", "Graphix vs ours", "x"))):
        axes[0].scatter(idx + (k - 1.5) * 0.18, [max(r[key], floor) for r in rows], s=12, marker=mk, label=lab)
    axes[0].axhline(TOL_BRANCH, color="k", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("pattern index (random, then flow-derived)")
    axes[0].set_ylabel("max branch deviation (floored 1e-17)")
    axes[0].set_title("(a) branch-wise process semantics")
    axes[0].legend(frameon=False, fontsize=7, loc="center left")
    d0 = np.array([r["depth_original"] for r in rows])
    d2 = np.array([r["depth_signal_shifted"] for r in rows])
    pairs = Counter(zip(d0, d2))
    axes[1].scatter([a for a, _ in pairs], [b for _, b in pairs], s=[25 * c for c in pairs.values()], alpha=0.6)
    lim = max(d0.max(), d2.max()) + 0.5
    axes[1].plot([0, lim], [0, lim], "k--", lw=1)
    axes[1].set_xlabel("measurement-dependency depth, original")
    axes[1].set_ylabel("depth after standardize + signal shift")
    axes[1].set_title("(b) adaptive depth (marker area = count)")
    names = sorted(rule_totals)
    axes[2].bar(names, [rule_totals[n] for n in names], color="#5e35b1")
    axes[2].set_ylabel("applications (all patterns)")
    axes[2].set_title("(c) DKP rewrite rules applied")
    axes[2].tick_params(axis="x", rotation=30)
    common.save_figure(fig, "R54_measurement_calculus")
    plt.close("all")
    common.save_caption("R54_measurement_calculus", f"""
**R54 - qubit measurement calculus preserves the full branch semantics.** {n_random} seeded random
valid non-standard DKP patterns (2 inputs, 3 ancillas, 2 outputs; interleaved N/E/M/X/Z, random
planes/angles, random s/t domains) and {len(rows) - n_random} flow-derived standard patterns (causal flow,
gflow incl. XZ/YZ planes, Pauli flow). (a) For each of the {total_branches} outcome branches, Choi
vectors of the original, standardized (NEMC) and signal-shifted patterns from an independent dense
simulator (measuring Z^t X^s O X^s Z^t), compared up to a branch phase: worst {worst['std_branch_dev']:.1e}
(standardize), {worst['shift_branch_dev']:.1e} (signal shift); branch probabilities differ by at most
{max(worst['std_prob_dev'], worst['shift_prob_dev']):.1e}. Library `run_branch` vs independent simulator:
{worst['library_vs_independent']:.1e}; Graphix 0.4 statevector on {gx_branches} branches with a random input:
1 - |overlap| <= {worst['graphix_dev']:.1e}. Dashed: acceptance {TOL_BRANCH:g}. Negative control: dropping one output
correction is detected (branch deviation > {NEG_CONTROL_MIN_DEV:g}) in {detected}/{len(controls)} patterns. (b) Dependency depth of the
measurement schedule before and after signal shifting. (c) Rewrite-rule applications during
standardization. CV patterns: only command classification and exact affine correction extraction
are validated ({sum(r['affine_components'] for r in cvr)} affine components, worst relative error
{max(r['affine_eval_max_rel_error'] for r in cvr):.1e}; nonlinear feed-forward reported as non-affine);
no CV measurement calculus is claimed. Source: R54.
""")
    if failures:
        common.save_csv(failures, "R54_measurement_calculus_FAILURES")
        raise AssertionError(f"{len(failures)} R54 check(s) failed; see FAILURES csv")
    common.print_summary("R54 measurement calculus", qubit_patterns=len(rows), branches=total_branches,
                         graphix_branches=gx_branches, worst={k: f"{v:.1e}" for k, v in worst.items()},
                         rules=rule_totals, cv_rows=len(cvr), negative_control=f"{detected}/{len(controls)}")


if __name__ == "__main__":
    main()
