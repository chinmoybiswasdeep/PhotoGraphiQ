"""R53: Qubit causal flow, gflow and Pauli flow against Graphix 0.4 and the defining conditions.

Scientific question: do ``photographiq.flows.find_causal_flow`` / ``find_gflow`` /
``find_pauli_flow`` agree with Graphix 0.4 on flow existence where the two use the same
semantics (qubit open graphs, planes XY/XZ/YZ, Pauli labels X/Y/Z, identical observable
conventions: XY cos X + sin Y, XZ cos Z + sin X, YZ cos Z + sin Y), and do the returned
flows -- ours AND Graphix's -- satisfy the defining conditions and give strongly
deterministic patterns?

Definitions checked by an independent checker written in this script (plain set algebra;
the library verifiers are not used), with the strict order u < v iff layer(u) < layer(v) and
outputs after every layer:

* causal flow (Danos-Kashefi 2006): f(i) ~ i, i < f(i), every neighbour k != i of f(i) has
  i < k; f injective into non-inputs; XY measurements only;
* gflow (Browne et al. 2007, three-plane form of Backens et al. 2021): G1 j in g(i)\\{i} =>
  i < j; G2 j in Odd(g(i))\\{i} => i < j; G3 XY: i notin g(i), i in Odd; XZ: i in g, i in Odd;
  YZ: i in g, i notin Odd;
* Pauli flow (Simmons 2021, Def. 2.4): P1-P3 relaxations for X/Y/Z-labelled nodes, P4-P9
  self conditions.

Determinism: an independent dense statevector simulator (Choi vector with one reference qubit
per input; projective measurement on the eigenvector of the measured observable; X/Z
corrections X_{p(u)\\{u}}, Z_{Odd(p(u))\\{u}} on nodes still alive) checks that every one of
the 2^m branches has probability 2^-m and the same output state up to phase.

Graphs: textbook successes (line, 2x3 cluster), gflow without causal flow, YZ/XZ planes,
Pauli flow needed (X measurements labelled in the XZ plane), Y/Z Pauli measurements, and
failures (|I| > |O|, XZ node inside an input line, XZ labels without Pauli relaxation),
plus seeded random open graphs (n = 3..8). CV graphs are never given qubit certificates:
``analyze(cv_graph, 'gflow')`` must be ``unsupported``.

Limitations: Graphix's correction sets need not equal ours (flows are not unique), so sets are
compared only as "both valid"; depth is compared (ours is maximally delayed). Statevector
determinism is limited to <= 12 simulated qubits. Seed 20261153.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

common.require_module("graphix", "R53 compares against Graphix 0.4")
from graphix.fundamentals import Axis  # noqa: E402
from graphix.fundamentals import Plane as GPlane  # noqa: E402
from graphix.opengraph import OpenGraph as GOpenGraph  # noqa: E402

from photographiq.flows import analyze, find_causal_flow, find_gflow, find_pauli_flow  # noqa: E402
from photographiq.ir import HomodyneSpec, Model, OpenGraph, Plane, QubitMeasurement  # noqa: E402

SEED = 20261153
N_RANDOM = 80  # per notion; --full uses 300
MAX_SIM_QUBITS = 12

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_DETERMINISM = 1e-12  # 1 - |overlap| and |p - 2^-m|: exact 0, float64 dense statevector
REQUIRE_EXISTENCE_AGREEMENT = 1.0  # fraction of graphs where existence agrees with Graphix
EXPECTED = {  # (notion, expected certified) for named cases, derived by hand / from the literature
    "line_5_xy": {"causal": True, "gflow": True, "pauli": True},
    "cluster_2x3_xy": {"causal": True, "gflow": True, "pauli": True},
    "gflow_not_causal": {"causal": False, "gflow": True, "pauli": True},
    "yz_state_prep": {"causal": False, "gflow": True, "pauli": True},
    "xz_line_start": {"causal": False, "gflow": True, "pauli": True},
    "xz_inside_input_line": {"causal": False, "gflow": False, "pauli": False},
    "x_paulis_in_xz_plane": {"causal": False, "gflow": False, "pauli": True},
    "y_z_paulis": {"pauli": True},
    "more_inputs_than_outputs": {"causal": False, "gflow": False, "pauli": False},
}
PAULI_ANGLE = {"X": {"XY": 0.0, "XZ": math.pi / 2}, "Y": {"XY": math.pi / 2, "YZ": math.pi / 2},
               "Z": {"XZ": 0.0, "YZ": 0.0}}


def pauli(label, plane):
    return QubitMeasurement(Plane(plane), PAULI_ANGLE[label][plane], label)


def qubit_graph(edges, inputs, outputs, specs=None, nodes=()):
    graph = nx.Graph(list(edges))
    graph.add_nodes_from(nodes)
    specs = specs or {}
    meas = {n: specs.get(n, QubitMeasurement(Plane.XY, 0.3 + 0.1 * i))
            for i, n in enumerate(graph.nodes) if n not in outputs}
    return OpenGraph(Model.QUBIT, graph, inputs, outputs, meas)


def named_cases():
    line = list(nx.path_graph(4).edges)
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3), ordering="sorted")
    return {
        "line_5_xy": qubit_graph(nx.path_graph(5).edges, (0,), (4,)),
        "cluster_2x3_xy": qubit_graph(grid.edges, (0, 3), (2, 5)),
        "gflow_not_causal": qubit_graph([(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4)], (3, 2), (0, 4)),
        "yz_state_prep": qubit_graph([(0, 1)], (), (1,), {0: QubitMeasurement(Plane.YZ, 0.7)}),
        "xz_line_start": qubit_graph(line, (), (3,), {0: QubitMeasurement(Plane.XZ, 0.4)}),
        "xz_inside_input_line": qubit_graph(line, (0,), (3,), {1: QubitMeasurement(Plane.XZ, 0.4)}),
        "x_paulis_in_xz_plane": qubit_graph([(0, 1), (1, 2)], (0,), (2,), {0: pauli("X", "XZ"), 1: pauli("X", "XZ")}),
        "y_z_paulis": qubit_graph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 4), (2, 5)], (0,), (4, 5),
                                  {1: pauli("Y", "YZ"), 2: pauli("Z", "XZ"), 3: QubitMeasurement(Plane.XY, 0.9)}),
        "more_inputs_than_outputs": qubit_graph([(0, 1), (0, 2), (1, 2)], (0, 1), (2,)),
    }


def random_open_graph(rng, n, pauli_fraction, planes):
    graph = nx.gnp_random_graph(n, rng.uniform(0.25, 0.6), seed=int(rng.integers(1 << 30)))
    nodes = [int(v) for v in rng.permutation(n)]
    k = int(rng.integers(max(1, n // 3), n // 2 + 2))
    outputs = tuple(nodes[:k])
    inputs = tuple(int(v) for v in rng.choice(nodes, int(rng.integers(0, k + 1)), replace=False))
    specs = {}
    for v in graph.nodes:
        if v in outputs:
            continue
        if rng.random() < pauli_fraction:
            label = str(rng.choice(["X", "Y", "Z"]))
            specs[v] = pauli(label, str(rng.choice(sorted(PAULI_ANGLE[label]))))
        else:
            specs[v] = QubitMeasurement(Plane(str(rng.choice(planes))), float(rng.uniform(0.1, 1.4)))
    return OpenGraph(Model.QUBIT, graph, inputs, outputs, specs)


# ----- Graphix bridge -------------------------------------------------------------------------
def to_graphix(og, pauli_axes):
    def label(m):
        return getattr(Axis, m.pauli) if pauli_axes and m.pauli else getattr(GPlane, m.plane.value)
    return GOpenGraph(og.graph, list(og.inputs), list(og.outputs), {n: label(m) for n, m in og.measurements.items()})


def graphix_flow(og, notion):
    g = to_graphix(og, pauli_axes=notion == "pauli")
    res = {"causal": g.to_causalflow_or_none, "gflow": g.to_gflow_or_none, "pauli": g.to_pauliflow_or_none}[notion]()
    if res is None:
        return None
    layers = [set(layer) for layer in res.partial_order_layers]  # outputs first in Graphix
    layers = tuple(tuple(sorted(layer)) for layer in reversed(layers[1:] if og.outputs else layers))
    return {u: set(p) for u, p in res.correction_function.items()}, layers


# ----- independent checkers -------------------------------------------------------------------
def odd(og, k):
    return {v for v in og.graph.nodes if len(set(og.graph[v]) & set(k)) % 2 == 1}


def ranks(og, layers):
    rank = {n: i for i, layer in enumerate(layers) for n in layer}
    for o in og.outputs:
        rank[o] = len(layers)
    return rank


def labels_of(og, notion):
    out = {}
    for n, m in og.measurements.items():
        out[n] = m.pauli if (notion == "pauli" and m.pauli) else m.plane.value
    return out


def check(og, notion, sets, layers):
    """Return list of violated conditions (empty = valid)."""
    rank = ranks(og, layers)
    measured = set(og.measured)
    bad = []
    if sorted(map(repr, rank)) != sorted(map(repr, og.nodes)) or set(n for layer in layers for n in layer) != measured:
        bad.append("layers do not partition measured nodes")
    if set(sets) != measured:
        bad.append("domain")
    lab = labels_of(og, notion)
    before = lambda a, b: rank.get(a, -1) < rank.get(b, -1)  # noqa: E731
    for u, p in sets.items():
        p = set(p)
        if p & set(og.inputs):
            bad.append(f"{u}: correction set meets inputs")
        o = odd(og, p)
        if notion == "causal":
            if len(p) != 1:
                bad.append(f"{u}: causal flow needs a single successor")
                continue
            (f,) = p
            if lab[u] != "XY":
                bad.append(f"{u}: causal flow needs XY")
            if not og.graph.has_edge(u, f):
                bad.append(f"{u}: F1")
            if not before(u, f):
                bad.append(f"{u}: F2")
            if any(k != u and not before(u, k) for k in og.graph[f]):
                bad.append(f"{u}: F3")
            continue
        for v in p - {u}:
            if (notion == "gflow" or lab.get(v) not in ("X", "Y")) and not before(u, v):
                bad.append(f"{u}: G1/P1 at {v}")
        for v in o - {u}:
            if (notion == "gflow" or lab.get(v) not in ("Y", "Z")) and not before(u, v):
                bad.append(f"{u}: G2/P2 at {v}")
        if notion == "pauli":
            for v, lv in lab.items():
                if lv == "Y" and v != u and not before(u, v) and ((v in p) != (v in o)):
                    bad.append(f"{u}: P3 at {v}")
        inp, ino = u in p, u in o
        ok = {"XY": not inp and ino, "XZ": inp and ino, "YZ": inp and not ino,
              "X": ino, "Z": inp, "Y": inp != ino}[lab[u]]
        if not ok:
            bad.append(f"{u}: self condition {lab[u]}")
    if notion == "causal":
        succ = [next(iter(p)) for p in sets.values() if len(p) == 1]
        if len(set(succ)) != len(succ):
            bad.append("causal flow not injective")
    return bad


# ----- independent statevector determinism ----------------------------------------------------
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0 + 0j, -1.0])


def _apply(state, op, axes):
    k = len(axes)
    out = np.tensordot(op.reshape((2,) * 2 * k), state, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes)


def branch_vector(og, order, sets, outcomes):
    nodes = list(og.nodes)
    alive = nodes + [("ref", n) for n in og.inputs]
    state = np.zeros((2,) * len(alive), complex)
    state[(0,) * len(alive)] = 1
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    for i in range(len(nodes)):
        state = _apply(state, h, [i])
    cnot = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    for n in og.inputs:  # Bell pair with a reference qubit: the Choi vector of the pattern
        state = _apply(state, cnot, [alive.index(n), alive.index(("ref", n))])
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    for u, v in og.graph.edges:
        state = _apply(state, cz, [alive.index(u), alive.index(v)])
    for u, s in zip(order, outcomes):
        m = og.measurements[u]
        a, b = {"XY": (_X, _Y), "XZ": (_Z, _X), "YZ": (_Z, _Y)}[m.plane.value]
        obs = math.cos(m.angle) * a + math.sin(m.angle) * b
        values, vectors = np.linalg.eigh(obs)
        phi = vectors[:, int(np.argmin(values))] if s else vectors[:, int(np.argmax(values))]
        state = np.tensordot(phi.conj(), state, axes=([0], [alive.index(u)]))
        alive.remove(u)
        if s:
            p = set(sets[u])
            for gate, targets in ((_X, p - {u}), (_Z, odd(og, p) - {u})):
                for t in targets:
                    if t in alive:  # components on earlier Pauli nodes act as phases (Pauli flow)
                        state = _apply(state, gate, [alive.index(t)])
    final = list(og.outputs) + [("ref", n) for n in og.inputs]
    return np.transpose(state, [alive.index(n) for n in final]).reshape(-1)


def determinism(og, sets, layers):
    order = [n for layer in layers for n in layer]
    if len(og.nodes) + len(og.inputs) > MAX_SIM_QUBITS:
        return None
    m = len(order)
    ref = branch_vector(og, order, sets, (0,) * m)
    ref = ref / np.linalg.norm(ref)
    worst = 0.0
    for outcomes in itertools.product((0, 1), repeat=m):
        vec = branch_vector(og, order, sets, outcomes)
        prob = float(np.linalg.norm(vec) ** 2)
        worst = max(worst, abs(prob - 2.0 ** -m), 1 - abs(np.vdot(ref, vec)) / math.sqrt(prob))
    return worst


FINDERS = {"causal": find_causal_flow, "gflow": find_gflow, "pauli": find_pauli_flow}


def ours(og, notion):
    r = FINDERS[notion](og)
    if not r.certified:
        return r, None
    sets = {u: {v} for u, v in r.certificate["flow"].items()} if notion == "causal" else r.certificate["correction_sets"]
    return r, ({u: set(p) for u, p in sets.items()}, r.layers)


def compare(name, og, notion, origin):
    r, mine = ours(og, notion)
    theirs = graphix_flow(og, notion)
    row = {
        "graph": name, "origin": origin, "notion": notion,
        "nodes": og.graph.number_of_nodes(), "edges": og.graph.number_of_edges(),
        "inputs": len(og.inputs), "outputs": len(og.outputs),
        "ours_certified": mine is not None, "graphix_certified": theirs is not None,
        "ours_status": r.status, "complete": r.diagnostics.get("complete"),
    }
    row["existence_agrees"] = row["ours_certified"] == row["graphix_certified"]
    if mine:
        row["ours_violations"] = "; ".join(check(og, notion, *mine))
        row["ours_depth"] = len(mine[1])
        row["ours_order"] = str([n for layer in mine[1] for n in layer])
        row["ours_determinism_dev"] = determinism(og, *mine)
    if theirs:
        row["graphix_violations"] = "; ".join(check(og, notion, *theirs))
        row["graphix_depth"] = len(theirs[1])
        row["graphix_determinism_dev"] = determinism(og, *theirs)
    if mine and theirs:
        row["same_layers"] = [set(x) for x in mine[1]] == [set(x) for x in theirs[1]]
        row["same_correction_sets"] = mine[0] == theirs[0]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="300 random graphs per notion")
    args = parser.parse_args()
    n_random = 300 if args.full else N_RANDOM
    plt = common.setup_style()
    rng = common.rng(SEED)
    rows = []
    for name, og in named_cases().items():
        for notion in ("causal", "gflow", "pauli"):
            rows.append(compare(name, og, notion, "named"))
    planes = {"causal": ("XY",), "gflow": ("XY", "XZ", "YZ"), "pauli": ("XY", "XZ", "YZ")}
    fraction = {"causal": 0.0, "gflow": 0.0, "pauli": 0.4}
    for notion in ("causal", "gflow", "pauli"):
        for i in range(n_random):
            og = random_open_graph(rng, int(rng.integers(3, 9)), fraction[notion], planes[notion])
            rows.append(compare(f"random_{notion}_{i:03d}", og, notion, "random"))

    # CV guard: qubit certificates are never applied to CV graphs.
    cv = OpenGraph(Model.CV_GAUSSIAN, nx.path_graph(3), (0,), (2,), {0: HomodyneSpec(), 1: HomodyneSpec()})
    guard = {n: analyze(cv, n).status for n in ("causal", "gflow", "pauli")}

    failures = []
    for r in rows:
        p = []
        if not r["existence_agrees"]:
            p.append("existence disagrees with Graphix")
        exp = EXPECTED.get(r["graph"], {}).get(r["notion"])
        if exp is not None and r["ours_certified"] != exp:
            p.append(f"expected certified={exp}")
        if r["ours_certified"] is False and r["complete"] is not True:
            p.append("negative answer not from a complete algorithm")
        for who in ("ours", "graphix"):
            if r.get(f"{who}_violations"):
                p.append(f"{who} flow violates: {r[who + '_violations']}")
            dev = r.get(f"{who}_determinism_dev")
            if dev is not None and dev > TOL_DETERMINISM:
                p.append(f"{who} flow pattern not strongly deterministic ({dev:.1e})")
        if r.get("ours_depth") and r.get("graphix_depth") and r["ours_depth"] > r["graphix_depth"]:
            p.append("our flow is deeper than Graphix's (should be maximally delayed)")
        if p:
            failures.append({**r, "problems": "; ".join(p)})
    if set(guard.values()) != {"unsupported"}:
        failures.append({"graph": "cv_guard", "problems": f"qubit flow applied to CV graph: {guard}"})

    summary = {}
    for notion in ("causal", "gflow", "pauli"):
        sub = [r for r in rows if r["notion"] == notion]
        both = [r for r in sub if r["ours_certified"] and r["graphix_certified"]]
        dets = [r[k] for r in sub for k in ("ours_determinism_dev", "graphix_determinism_dev") if r.get(k) is not None]
        summary[notion] = {
            "graphs": len(sub),
            "agree": sum(r["existence_agrees"] for r in sub),
            "positive": sum(r["ours_certified"] for r in sub),
            "negative": sum(not r["ours_certified"] for r in sub),
            "same_layers": sum(bool(r.get("same_layers")) for r in both),
            "same_correction_sets": sum(bool(r.get("same_correction_sets")) for r in both),
            "ours_shallower": sum(r["ours_depth"] < r["graphix_depth"] for r in both),
            "determinism_checked": len(dets),
            "worst_determinism_dev": max(dets, default=0.0),
        }
    agreement = sum(r["existence_agrees"] for r in rows) / len(rows)
    if agreement < REQUIRE_EXISTENCE_AGREEMENT:
        failures.append({"graph": "all", "problems": f"agreement {agreement}"})

    common.save_result(rows, "R53_qubit_flows_graphix", extra={
        "seed": SEED,
        "backend": "photographiq.flows (qubit) vs graphix 0.4 OpenGraph.to_*flow_or_none; independent set-algebra checker; independent dense statevector",
        "precision": "exact set algebra; float64 statevector",
        "parameters": {"n_random_per_notion": n_random, "max_sim_qubits": MAX_SIM_QUBITS, "full": args.full},
        "acceptance": {"TOL_DETERMINISM": TOL_DETERMINISM, "REQUIRE_EXISTENCE_AGREEMENT": REQUIRE_EXISTENCE_AGREEMENT, "EXPECTED": EXPECTED},
        "summary": summary,
        "cv_guard_status": guard,
    })
    lines = ["| notion | graphs | agree with Graphix | positive | negative | same layers | same sets | ours shallower | worst determinism dev |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        lines.append(f"| {k} | {s['graphs']} | {s['agree']} | {s['positive']} | {s['negative']} | {s['same_layers']} | "
                     f"{s['same_correction_sets']} | {s['ours_shallower']} | {s['worst_determinism_dev']:.1e} |")
    (common.TABLES / "R53_qubit_flows_graphix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    notions = list(summary)
    x = np.arange(len(notions))
    ax1.bar(x - 0.2, [summary[n]["positive"] for n in notions], 0.4, color="#2e7d32", label="flow exists (both)")
    ax1.bar(x + 0.2, [summary[n]["negative"] for n in notions], 0.4, color="#c62828", label="no flow (both)")
    for i, n in enumerate(notions):
        ax1.text(i, max(summary[n]["positive"], summary[n]["negative"]) + 2,
                 f"agree {summary[n]['agree']}/{summary[n]['graphs']}", ha="center", fontsize=8)
    ax1.set_xticks(x, notions)
    ax1.set_ylabel("graphs")
    ax1.set_ylim(0, max(max(s["positive"], s["negative"]) for s in summary.values()) * 1.35)
    ax1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax1.set_title("(a) existence: PhotoGraphiQ vs Graphix 0.4")
    keys = [("jointly certified", lambda n: summary[n]["positive"]), ("same layers", lambda n: summary[n]["same_layers"]),
            ("same correction sets", lambda n: summary[n]["same_correction_sets"]),
            ("statevector-checked (ours + Graphix)", lambda n: summary[n]["determinism_checked"])]
    for k, (label, f) in enumerate(keys):
        vals = [f(n) for n in notions]
        bars = ax2.bar(x + (k - 1.5) * 0.2, vals, 0.2, label=label)
        ax2.bar_label(bars, fontsize=7)
    ax2.set_xticks(x, notions)
    ax2.set_ylabel("count")
    ax2.set_ylim(0, max(s["determinism_checked"] for s in summary.values()) * 1.3)
    ax2.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax2.set_title("(b) certified flows: ours vs Graphix certificates")
    common.save_figure(fig, "R53_qubit_flows_graphix")
    plt.close("all")
    dets = max(s["worst_determinism_dev"] for s in summary.values())
    ndet = sum(s["determinism_checked"] for s in summary.values())
    common.save_caption("R53_qubit_flows_graphix", f"""
**R53 - qubit causal flow, gflow and Pauli flow against Graphix 0.4.** {len(named_cases())} named open
graphs (textbook successes and failures) and {n_random} seeded random open graphs per notion
(n = 3..8; planes XY only for causal flow, XY/XZ/YZ for gflow, 40% Pauli labels for Pauli flow).
(a) Existence decisions agree with Graphix's `to_*flow_or_none` on
{sum(s['agree'] for s in summary.values())}/{sum(s['graphs'] for s in summary.values())} (graph, notion) pairs;
negative answers come from complete layered algorithms. Every returned flow (ours and Graphix's)
satisfies the defining conditions (F1-F3, G1-G3, P1-P9) re-checked by an independent set-algebra
checker, and {ndet} flow-derived patterns were simulated branch by branch by an independent dense
statevector code: all 2^m branches equiprobable with identical output up to phase (worst
deviation {dets:.1e}). (b) Jointly certified flows: identical layer partitions in
{sum(s['same_layers'] for s in summary.values())} cases, ours never deeper (maximally delayed).
Correction sets need not coincide (flows are not unique): identical sets in
{sum(s['same_correction_sets'] for s in summary.values())} of the jointly certified cases. Qubit flow
notions are never applied to CV graphs (`analyze(cv, 'gflow')` returns `{guard['gflow']}`). Source: R53.
""")
    if failures:
        common.save_csv(failures, "R53_qubit_flows_graphix_FAILURES")
        raise AssertionError(f"{len(failures)} R53 check(s) failed; see FAILURES csv")
    common.print_summary("R53 qubit flows vs Graphix", **{k: v for k, v in summary.items()}, cv_guard=guard)


if __name__ == "__main__":
    main()
