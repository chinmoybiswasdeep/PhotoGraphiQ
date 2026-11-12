"""Qubit-MBQC benchmark: PhotoGraphiQ (flows + calculus.qubit) versus Graphix 0.4.

Scope: qubit measurement-based quantum computing only. Graphix is not a CV or GBS simulator,
so nothing here says anything about PhotoGraphiQ's CV/Gaussian/Fock engines. Only tasks with
identical inputs and identical mathematical outputs are timed, and every timed instance is
first checked for agreement:

1. ``causal``   causal-flow finding (XY open graphs). Same question: does a causal flow exist,
                and if so return one with its layered partial order.
2. ``gflow``    plane-aware gflow finding (XY/XZ/YZ labels).
3. ``pauli``    Pauli-flow finding (XY/XZ/YZ plus Pauli X/Y/Z labels).
   For 1-3, PhotoGraphiQ's public finders (``find_*``) always run their independent verifier
   before returning, Graphix's ``to_*_or_none`` do not. The like-for-like pair is therefore
   PhotoGraphiQ ``find_*`` versus Graphix ``to_*_or_none`` + ``check_well_formed``; Graphix
   search-only is reported as a separate column, not as the comparison.
   Each certificate is checked by BOTH verifiers (PhotoGraphiQ ``verify_*`` and Graphix
   ``check_well_formed``), including cross-checks of the other package's certificate.
4. ``standardize`` / ``signal_shift`` on one common input pattern: Graphix's flow-derived
   pattern ``XZCorrections.to_pattern`` (non-standard: corrections interleaved), converted
   command by command to a ``QubitPattern``. Signal shifting takes the Graphix standard form
   as common input. Graphix's ``shift_signals`` returns the shift as a dict, PhotoGraphiQ
   appends explicit ``S`` commands: the quantum commands must coincide and the S domains must
   equal the dict. Map equivalence is checked on small sizes by exact branch-by-branch dense
   simulation (``photographiq.calculus.qubit.equivalent``).
5. ``statevector``: one fixed outcome branch of the same standardized pattern, input |+> on
   every input qubit, output = normalised output state. PhotoGraphiQ ``run_branch`` versus
   Graphix ``Pattern.simulate(backend="statevector", FixedBranchSelector)``; inputs are turned
   into ``N`` commands for PhotoGraphiQ so both compute one state (``run_branch`` would
   otherwise compute the whole Kraus map, which is NOT the same task).

Timings exclude graph/pattern construction and conversion. Every call is warmed up first;
``first_call_s`` is the very first call on that instance. Graphix's numba kernels carry
explicit signatures and are compiled (or cache-loaded) eagerly at import, so no JIT cost lands
in any timed call; it is reported separately as ``environment.setup_import_s``. For calls
faster than 2 ms, each sample is the mean of ``inner_loops`` back-to-back calls, so the IQR of
those rows is the spread of batch means, not of single calls. No machine-load check is made:
run on an otherwise idle machine.

Run:  python benchmarks/graphix/bench_graphix_mbqc.py           (fast, ~minutes)
      python benchmarks/graphix/bench_graphix_mbqc.py --full    (larger sizes, much longer)
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import importlib.metadata as md
import json
import math
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import networkx as nx
import numpy as np
from graphix import optimization as gopt
from graphix.branch_selector import FixedBranchSelector
from graphix.command import CommandKind
from graphix.flow.core import CausalFlow, GFlow, PauliFlow
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph as GOpenGraph

from photographiq.calculus import qubit as q
from photographiq.flows import (
    find_causal_flow,
    find_gflow,
    find_pauli_flow,
    verify_causal_flow,
    verify_gflow,
    verify_pauli_flow,
)
from photographiq.ir import Model, OpenGraph, Plane, QubitMeasurement

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bench_common import cpu_model, git_state  # noqa: E402  (shared metadata helpers)

RESULTS = HERE / "results"
PAULI = {"X": (Plane.XY, 0.0), "Y": (Plane.XY, math.pi / 2), "Z": (Plane.YZ, 0.0)}

FAST = {
    "flow_sizes": {
        "linear": [8, 16, 32, 64, 128, 256, 512],
        "grid3": [4, 8, 16, 32, 64],
        "brickwork4": [9, 17, 33, 65],
        "lc_random": [4, 8, 16, 32, 64],
    },
    "pattern_sizes": {
        "linear": [4, 8, 16, 64, 128],
        "grid3": [2, 3, 4, 8, 16],
        "brickwork4": [3, 9, 17],
        "lc_random": [3, 4, 5, 8, 16],
    },
    "sv_sizes": {"linear": [4, 8, 12, 16], "grid3": [2, 3, 4, 5], "lc_random": [3, 4, 5, 6]},
    "agreement_instances": 120,
    "cap_s": 0.2,
    "warmup": 1,
    "repeats": 5,
}
FULL = {
    "flow_sizes": {
        "linear": [8, 32, 128, 512, 1024, 2048],
        "grid3": [4, 16, 64, 128, 256],
        "brickwork4": [9, 33, 65, 129, 257],
        "lc_random": [4, 8, 16, 32, 64, 128, 256],
    },
    "pattern_sizes": {
        "linear": [4, 8, 16, 64, 128, 256, 512],
        "grid3": [2, 3, 4, 8, 16, 32, 64],
        "brickwork4": [3, 9, 17, 33, 65],
        "lc_random": [3, 4, 5, 8, 16, 32, 64],
    },
    "sv_sizes": {"linear": [4, 8, 12, 16, 19], "grid3": [2, 3, 4, 5, 6], "lc_random": [3, 4, 5, 6]},
    "agreement_instances": 500,
    "cap_s": 5.0,
    "warmup": 3,
    "repeats": 15,
}
EQUIV_MAX_MEASURED, EQUIV_MAX_QUBITS = 8, 16  # exact branch enumeration only up to these


# ----------------------------------------------------------------------------- instances


_LC_SELF = {"XY": "XZ", "XZ": "XY", "YZ": "YZ"}  # Backens et al., Quantum 5, 421 (2021):
_LC_NEIGHBOUR = {"XY": "XY", "XZ": "YZ", "YZ": "XZ"}  # local complementation keeps gflow
_PAULI_OK = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}  # Pauli relaxing each plane


def family(name: str, size: int, rng: np.random.Generator):
    """(graph, inputs, outputs, plane labels of measured nodes); integer node labels.

    linear / grid3 / brickwork4: XY clusters with causal flow (inputs left, outputs right).
    lc_random: 3 x size grid, then |measured|//2 random local complementations at measured
    non-input nodes with the plane update of Backens et al. 2021 -> random graph with mixed
    planes that keeps a gflow. er: Erdos-Renyi open graph with random planes (flow may or may
    not exist; used for the existence-agreement sweep only).
    """
    if name == "er":
        g = nx.gnp_random_graph(size, rng.uniform(0.25, 0.6), seed=int(rng.integers(1 << 30)))
        nodes = [int(v) for v in rng.permutation(size)]
        k = int(rng.integers(max(1, size // 3), size // 2 + 2))
        outs = tuple(nodes[:k])
        ins = tuple(int(v) for v in rng.choice(nodes, int(rng.integers(0, k + 1)), replace=False))
        planes = {v: str(rng.choice(["XY", "XZ", "YZ"])) for v in g.nodes if v not in outs}
        return g, ins, outs, planes
    if name == "linear":
        g, ins, outs = nx.path_graph(size), (0,), (size - 1,)
    elif name in ("grid3", "brickwork4", "lc_random"):
        rows = 4 if name == "brickwork4" else 3
        g = nx.Graph()
        idx = lambda r, c: r * size + c  # noqa: E731
        g.add_nodes_from(range(rows * size))
        for r in range(rows):
            g.add_edges_from((idx(r, c), idx(r, c + 1)) for c in range(size - 1))
        for c in range(size):
            for r in range(rows - 1):
                # grid: every rung; brickwork (Broadbent-Fitzsimons-Kashefi 2009 layout):
                # rungs at columns 2,4 (mod 8) between rows (0,1),(2,3), at 6,0 between (1,2)
                brick = (c % 8 in (2, 4) and r % 2 == 0) or (c % 8 in (6, 0) and c and r % 2)
                if name != "brickwork4" or brick:
                    g.add_edge(idx(r, c), idx(r + 1, c))
        ins = tuple(idx(r, 0) for r in range(rows))
        outs = tuple(idx(r, size - 1) for r in range(rows))
    else:
        raise ValueError(name)
    planes = {v: "XY" for v in g.nodes if v not in outs}
    if name == "lc_random":
        candidates = [v for v in planes if v not in ins]
        for _ in range(len(planes) // 2 if candidates else 0):
            u = int(rng.choice(candidates))
            nb = list(g.neighbors(u))
            for i, a in enumerate(nb):
                for c in nb[i + 1 :]:
                    g.remove_edge(a, c) if g.has_edge(a, c) else g.add_edge(a, c)
            planes[u] = _LC_SELF[planes[u]]
            for v in nb:
                if v in planes:
                    planes[v] = _LC_NEIGHBOUR[planes[v]]
    return g, ins, outs, planes


def labels_for(task: str, planes: dict, rng) -> dict:
    """causal: all XY; gflow: the family's planes; pauli: 30% of nodes relabelled to a Pauli
    that relaxes their plane (so a gflow stays a Pauli flow)."""
    if task == "causal":
        return {n: "XY" for n in planes}
    if task == "gflow":
        return dict(planes)
    return {
        n: str(rng.choice(_PAULI_OK[p])) if rng.random() < 0.3 else p for n, p in planes.items()
    }


def make_open_graphs(graph, inputs, outputs, labels, rng):
    """The same open graph for both packages (angles: radians here, units of pi in Graphix)."""
    pm, gm = {}, {}
    for n, lab in labels.items():
        if lab in PAULI:
            plane, a = PAULI[lab]
            pm[n] = QubitMeasurement(plane, a, lab)
            gm[n] = getattr(Measurement, lab)  # +X = XY(0), +Y = XY(pi/2), +Z = YZ(0)
        else:
            a = float(rng.uniform(0.1, 1.4))  # never a Pauli angle
            pm[n] = QubitMeasurement(Plane(lab), a)
            gm[n] = getattr(Measurement, lab)(a / math.pi)
    pog = OpenGraph(Model.QUBIT, graph, inputs, outputs, pm)
    gog = GOpenGraph(nx.Graph(graph), list(inputs), list(outputs), gm)
    return pog, gog


def instance(fam: str, size: int, task: str, seed: int):
    rng = np.random.default_rng(seed)
    graph, ins, outs, planes = family(fam, size, rng)
    return make_open_graphs(graph, ins, outs, labels_for(task, planes, rng), rng)


# ----------------------------------------------------------------------------- adapters

PG_FIND = {"causal": find_causal_flow, "gflow": find_gflow, "pauli": find_pauli_flow}
GX_FIND = {
    "causal": lambda g: g.to_causalflow_or_none(),
    "gflow": lambda g: g.to_gflow_or_none(),
    "pauli": lambda g: g.to_pauliflow_or_none(),
}
GX_CLASS = {"causal": CausalFlow, "gflow": GFlow, "pauli": PauliFlow}


def gx_layers_to_pg(flow, outputs) -> tuple:
    """Graphix layers (layer 0 = outputs, last = earliest) -> PhotoGraphiQ (earliest first)."""
    layers = [tuple(sorted(set(layer) - set(outputs))) for layer in flow.partial_order_layers]
    return tuple(layer for layer in reversed(layers) if layer)


def pg_verify(task, pog, sets, layers) -> list:
    if task == "causal":
        return verify_causal_flow(pog, {u: next(iter(s)) for u, s in sets.items()}, layers)
    return (verify_gflow if task == "gflow" else verify_pauli_flow)(pog, sets, layers)


def gx_verify(task, gog, sets, layers, outputs) -> bool:
    """Graphix's own verifier on a certificate given in PhotoGraphiQ layer form."""
    order = ([frozenset(outputs)] if outputs else []) + [frozenset(lay) for lay in reversed(layers)]
    flow = GX_CLASS[task](gog, {u: frozenset(s) for u, s in sets.items()}, order)
    return flow.is_well_formed()


def pg_sets(task, result) -> dict:
    if task == "causal":
        return {u: {v} for u, v in result.certificate["flow"].items()}
    return {u: set(s) for u, s in result.certificate["correction_sets"].items()}


def from_graphix(gp, prepare_inputs: bool = False) -> q.QubitPattern:
    """Command-by-command copy of a Graphix pattern (angles x pi). ``prepare_inputs`` turns
    the inputs into N (|+>) commands, i.e. Graphix's default input state."""
    cmds = [q.N(n) for n in gp.input_nodes] if prepare_inputs else []
    for c in gp:
        if c.kind == CommandKind.N:
            cmds.append(q.N(c.node))
        elif c.kind == CommandKind.E:
            cmds.append(q.E(*c.nodes))
        elif c.kind == CommandKind.M:
            b = c.measurement.to_bloch()
            cmds.append(q.M(c.node, b.plane.name, float(b.angle) * math.pi, c.s_domain, c.t_domain))
        elif c.kind == CommandKind.X:
            cmds.append(q.X(c.node, c.domain))
        elif c.kind == CommandKind.Z:
            cmds.append(q.Z(c.node, c.domain))
        else:
            raise TypeError(f"unsupported Graphix command {c!r}")
    return q.QubitPattern(() if prepare_inputs else gp.input_nodes, cmds, gp.output_nodes)


def counts(cmds) -> dict:
    out = dict(collections.Counter(type(c).__name__ for c in cmds))
    out["domain_entries"] = sum(
        len(d)
        for c in cmds
        for d in (
            (c.s_domain, c.t_domain) if type(c).__name__ == "M" else (getattr(c, "domain", ()),)
        )
    )
    out["total"] = len(cmds)
    return out


def semantic_summary(cmds, outputs):
    """N set, CZ set, measurement (plane, angle mod 2pi, s, t), per-output XOR-ed X/Z frames."""
    preps, edges, meas = set(), set(), {}
    frames = {o: [frozenset(), frozenset()] for o in outputs}
    for c in cmds:
        name = type(c).__name__
        if name == "N":
            preps.add(c.node)
        elif name == "E":
            edges ^= {frozenset(c.nodes)}
        elif name == "M":
            if isinstance(c, q.M):
                plane, angle = c.plane.value, c.angle
            else:
                b = c.measurement.to_bloch()
                plane, angle = b.plane.name, float(b.angle) * math.pi
            meas[c.node] = (
                plane,
                round(angle % (2 * math.pi), 9),
                frozenset(c.s_domain),
                frozenset(c.t_domain),
            )
        elif name in ("X", "Z"):
            frames[c.node][name == "Z"] ^= frozenset(c.domain)
    return preps, edges, meas, {o: tuple(f) for o, f in frames.items()}


# ----------------------------------------------------------------------------- timing


def measure(call, *, warmup: int, repeats: int, setup=None, min_sample: float = 2e-3) -> dict:
    """Median/IQR/min per call. With ``setup`` (destructive calls) the setup runs untimed
    before every call; otherwise fast calls are looped ``inner`` times per sample."""

    def once() -> float:
        arg = setup() if setup else None
        start = time.perf_counter()
        call(arg) if setup else call()
        return time.perf_counter() - start

    first = once()
    warm = first
    for _ in range(warmup):
        warm = once()
    inner = (
        1 if setup or warm >= min_sample else min(20000, max(1, int(min_sample / max(warm, 1e-7))))
    )
    samples = []
    for _ in range(repeats):
        if inner == 1:
            samples.append(once())
        else:
            start = time.perf_counter()
            for _ in range(inner):
                call()
            samples.append((time.perf_counter() - start) / inner)
    arg = setup() if setup else None
    tracemalloc.start()
    call(arg) if setup else call()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    return {
        "first_call_s": first,
        "median_s": float(med),
        "q1_s": float(q1),
        "q3_s": float(q3),
        "iqr_s": float(q3 - q1),
        "min_s": float(min(samples)),
        "max_s": float(max(samples)),
        "warmup": warmup,
        "repeats": repeats,
        "inner_loops": inner,
        "tracemalloc_peak_bytes": int(peak),
    }


# ----------------------------------------------------------------------------- tasks


def check_flow(task, pog, gog) -> dict:
    """Existence agreement and all four verifier checks; no timing."""
    mine, theirs = PG_FIND[task](pog), GX_FIND[task](gog)
    row = {"pg_exists": mine.certified, "gx_exists": theirs is not None}
    row["agree"] = row["pg_exists"] == row["gx_exists"]
    if mine.certified:
        sets = pg_sets(task, mine)
        row["pg_cert_pg_verifier_ok"] = pg_verify(task, pog, sets, mine.layers) == []
        row["pg_cert_gx_verifier_ok"] = gx_verify(task, gog, sets, mine.layers, pog.outputs)
        row["pg_depth"] = len(mine.layers)
    if theirs is not None:
        layers = gx_layers_to_pg(theirs, pog.outputs)
        sets = {u: set(s) for u, s in theirs.correction_function.items()}
        row["gx_cert_pg_verifier_ok"] = pg_verify(task, pog, sets, layers) == []
        row["gx_cert_gx_verifier_ok"] = theirs.is_well_formed()
        row["gx_depth"] = len(layers)
    row["all_checks_ok"] = row["agree"] and all(v for k, v in row.items() if k.endswith("_ok"))
    return row


def agreement_sweep(n_instances: int, seed: int) -> dict:
    """Existence agreement on many small random open graphs (4-12 nodes), per task."""
    rng = np.random.default_rng(seed)
    out = {}
    for task in PG_FIND:
        tally = collections.Counter()
        for _ in range(n_instances):
            size = int(rng.integers(4, 13))
            pog, gog = instance("er", size, task, int(rng.integers(2**31)))
            row = check_flow(task, pog, gog)
            tally["instances"] += 1
            tally["flow_exists"] += row["pg_exists"]
            tally["existence_agree"] += row["agree"]
            tally["all_checks_ok"] += row["all_checks_ok"]
            tally["pg_depth_le_gx_depth"] += row.get("pg_depth", 0) <= row.get("gx_depth", 0)
        out[task] = dict(tally)
    return out


def bench_flows(cfg, seed, log) -> list:
    rows = []
    for task in PG_FIND:
        for fam, sizes in cfg["flow_sizes"].items():
            for size in sizes:
                pog, gog = instance(fam, size, task, seed + size)
                row = {
                    "task": task,
                    "family": fam,
                    "size_param": size,
                    "nodes": pog.graph.number_of_nodes(),
                    "edges": pog.graph.number_of_edges(),
                    **check_flow(task, pog, gog),
                }
                kw = {"warmup": cfg["warmup"], "repeats": cfg["repeats"]}
                row["photographiq_find_with_builtin_verify"] = measure(
                    lambda: PG_FIND[task](pog), **kw
                )
                row["graphix_find_only"] = measure(lambda: GX_FIND[task](gog), **kw)

                def gx_both():
                    f = GX_FIND[task](gog)
                    if f is not None:
                        f.check_well_formed()

                row["graphix_find_plus_check_well_formed"] = measure(gx_both, **kw)
                a = row["photographiq_find_with_builtin_verify"]["median_s"]
                b = row["graphix_find_plus_check_well_formed"]["median_s"]
                row["ratio_pg_over_gx_search_plus_verify"] = a / b
                rows.append(row)
                log(
                    f"flow {task:6s} {fam:10s} n={row['nodes']:5d} exists={row['pg_exists']!s:5s} "
                    f"ok={row['all_checks_ok']} pg={a:.3e}s gx(find+check)={b:.3e}s "
                    f"gx(find)={row['graphix_find_only']['median_s']:.3e}s"
                )
                if max(a, b) > cfg["cap_s"]:
                    break
    return rows


def pattern_instance(fam, size, seed):
    """Graph with a gflow (causal for XY families) and its Graphix flow-derived pattern."""
    task = "gflow" if fam == "lc_random" else "causal"
    pog, gog = instance(fam, size, task, seed)
    flow = GX_FIND[task](gog)
    if flow is None:
        return None
    return pog, flow.to_xzcorrections().to_pattern()


def bench_patterns(cfg, seed, log) -> list:
    rows = []
    kw = {"warmup": cfg["warmup"], "repeats": cfg["repeats"]}
    for fam, sizes in cfg["pattern_sizes"].items():
        for size in sizes:
            made = pattern_instance(fam, size, seed + size)
            if made is None:
                log(f"pattern {fam} size={size}: no flow, skipped")
                continue
            pog, gp0 = made
            p0 = from_graphix(gp0)
            ours_std, gx_std = q.standardize(p0), gopt.standardize(gp0)
            p_std = from_graphix(gx_std)  # common input of the signal-shift comparison
            ours_shift = q.signal_shift(p_std)
            gx_shift = gx_std.copy()
            signal_dict = gx_shift.shift_signals(method="direct")
            quantum = [c for c in ours_shift.commands if not isinstance(c, q.S)]
            ours_signals = {
                c.node: frozenset(c.domain) for c in ours_shift.commands if isinstance(c, q.S)
            }
            row = {
                "family": fam,
                "size_param": size,
                "nodes": pog.graph.number_of_nodes(),
                "measured": len(p0.measured),
                "inputs": len(p0.inputs),
                "counts_input": counts(p0.commands),
                "counts_std_photographiq": counts(ours_std.commands),
                "counts_std_graphix": counts(list(gx_std)),
                "counts_shift_photographiq": counts(ours_shift.commands),
                "counts_shift_graphix": counts(list(gx_shift)),
                "both_standard": ours_std.is_standard() and gx_std.is_standard(),
                "std_same_semantic_form": semantic_summary(ours_std.commands, p0.outputs)
                == semantic_summary(list(gx_std), p0.outputs),
                "shift_same_quantum_commands": semantic_summary(quantum, p0.outputs)
                == semantic_summary(list(gx_shift), p0.outputs),
                "shift_same_signal_map": ours_signals
                == {n: frozenset(d) for n, d in signal_dict.items() if d},
            }
            if (
                len(p0.measured) <= EQUIV_MAX_MEASURED
                and pog.graph.number_of_nodes() + len(p0.inputs) <= EQUIV_MAX_QUBITS
            ):
                tail = [
                    q.S(n, frozenset(signal_dict[n]) - {n})
                    for n in reversed(p_std.measured)
                    if signal_dict.get(n)
                ]
                gx_shift_q = from_graphix(gx_shift)
                gx_shift_q = gx_shift_q.copy(gx_shift_q.commands + tail)
                reports = {
                    "input_deterministic": q.is_deterministic(p0),
                    "std_photographiq_vs_input": q.equivalent(p0, ours_std),
                    "std_graphix_vs_input": q.equivalent(p0, p_std),
                    "shift_photographiq_vs_std": q.equivalent(p_std, ours_shift),
                    "shift_graphix_plus_S_vs_std": q.equivalent(p_std, gx_shift_q),
                }
                row["map_equivalence"] = {
                    k: {
                        "ok": bool(
                            r.deterministic if hasattr(r, "deterministic") else r.equivalent
                        ),
                        "max_deviation": float(r.max_deviation),
                    }
                    for k, r in reports.items()
                }
            else:
                row["map_equivalence"] = "not run (exact branch enumeration too large)"
            row["photographiq_standardize"] = measure(lambda: q.standardize(p0), **kw)
            row["graphix_standardize"] = measure(lambda: gopt.standardize(gp0), **kw)
            row["photographiq_signal_shift"] = measure(lambda: q.signal_shift(p_std), **kw)
            row["graphix_shift_signals_direct"] = measure(
                lambda p: p.shift_signals(method="direct"), setup=gx_std.copy, **kw
            )
            rows.append(row)
            st = row["photographiq_standardize"]["median_s"], row["graphix_standardize"]["median_s"]
            sh = (
                row["photographiq_signal_shift"]["median_s"],
                row["graphix_shift_signals_direct"]["median_s"],
            )
            eq = row["map_equivalence"]
            eq = all(v["ok"] for v in eq.values()) if isinstance(eq, dict) else "n/a"
            log(
                f"pattern {fam:10s} n={row['nodes']:4d} std pg={st[0]:.3e} gx={st[1]:.3e} | "
                f"shift pg={sh[0]:.3e} gx={sh[1]:.3e} | same_form={row['std_same_semantic_form']} "
                f"shift_same={row['shift_same_quantum_commands'] and row['shift_same_signal_map']} "
                f"maps_equiv={eq}"
            )
            if max(*st, *sh) > cfg["cap_s"]:
                break
    return rows


def bench_statevector(cfg, seed, log) -> list:
    rows = []
    kw = {"warmup": cfg["warmup"], "repeats": cfg["repeats"]}
    for fam, sizes in cfg["sv_sizes"].items():
        for size in sizes:
            made = pattern_instance(fam, size, seed + size)
            if made is None:
                continue
            pog, gp0 = made
            gx_std = gopt.standardize(gp0)
            pq = from_graphix(gx_std, prepare_inputs=True)
            if pog.graph.number_of_nodes() > q.MAX_QUBITS:
                break
            rng = np.random.default_rng(seed + 7 * size)
            outcomes = {n: int(rng.integers(2)) for n in pq.measured}
            selector = FixedBranchSelector(results=dict(outcomes))

            def ours():
                psi = q.run_branch(pq, outcomes).operator[:, 0]
                return psi / np.linalg.norm(psi)

            def theirs():
                return gx_std.simulate(backend="statevector", branch_selector=selector)

            a = ours()
            b = np.asarray(theirs().flatten()).ravel()
            row = {
                "family": fam,
                "size_param": size,
                "nodes": pog.graph.number_of_nodes(),
                "inputs": len(gx_std.input_nodes),
                "outputs": len(gx_std.output_nodes),
                "branch_seed": seed + 7 * size,
                "fidelity_abs_overlap": float(abs(np.vdot(a, b))),
                "abs_overlap_error": float(abs(abs(np.vdot(a, b)) - 1)),
                "photographiq_run_branch": measure(ours, **kw),
                "graphix_statevector_fixed_branch": measure(theirs, **kw),
            }
            rows.append(row)
            m1 = row["photographiq_run_branch"]["median_s"]
            m2 = row["graphix_statevector_fixed_branch"]["median_s"]
            log(
                f"statevector {fam:8s} n={row['nodes']:3d} pg={m1:.3e} gx={m2:.3e} "
                f"|1-|<a|b>||={row['abs_overlap_error']:.1e}"
            )
            if max(m1, m2) > cfg["cap_s"]:
                break
    return rows


# ----------------------------------------------------------------------------- output


def environment() -> dict:
    from photographiq import native

    def version(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        sampling_threads = native.require("sampling").max_threads()
    except Exception as exc:  # noqa: BLE001 - metadata only
        sampling_threads = f"unavailable: {exc}"
    import numba

    def import_seconds(stmt: str) -> float:
        """Fresh-interpreter import time (setup cost incl. eager numba compilation/cache load)."""
        code = f"import time; t = time.perf_counter(); {stmt}; print(time.perf_counter() - t)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return float(out.stdout.strip() or "nan")

    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in ("photographiq", "graphix", "numpy", "networkx", "numba", "piquasso", "scipy")
        },
        "env_threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
        "photographiq_native_components": list(native.components()),
        "photographiq_native_sampling_max_threads": sampling_threads,
        "native_code_used_by_this_benchmark": "none on the PhotoGraphiQ side (flows and "
        "calculus.qubit are pure Python/NumPy); Graphix uses numba-JIT GF(2) and statevector "
        "kernels",
        "precision": "complex128 statevectors; GF(2) uint8 linear algebra",
        "git": git_state(),
        "timer": "time.perf_counter (Python GC left enabled)",
        "setup_import_s": {
            "photographiq": import_seconds(
                "import photographiq.flows, photographiq.calculus.qubit"
            ),
            "graphix": import_seconds(
                "import graphix.opengraph, graphix.pattern, graphix.sim.statevec, "
                "graphix.optimization"
            ),
        },
        "memory_note": "tracemalloc_peak_bytes = peak Python/NumPy heap during one warm call; it "
        "does not capture numba NRT allocations or other C-level allocations",
    }


def plot(results: dict, stem: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"PhotoGraphiQ": "#2a6fdb", "Graphix": "#d9822b", "Graphix (search only)": "#d9822b"}
    panels = []
    for task in PG_FIND:
        rows = [r for r in results["flows"] if r["task"] == task]
        panels.append(
            (
                f"{task} flow finding",
                rows,
                [
                    ("PhotoGraphiQ", "photographiq_find_with_builtin_verify", "-"),
                    ("Graphix", "graphix_find_plus_check_well_formed", "-"),
                    ("Graphix (search only)", "graphix_find_only", ":"),
                ],
            )
        )
    pat = results["patterns"]
    panels.append(
        (
            "standardization",
            pat,
            [
                ("PhotoGraphiQ", "photographiq_standardize", "-"),
                ("Graphix", "graphix_standardize", "-"),
            ],
        )
    )
    panels.append(
        (
            "signal shifting",
            pat,
            [
                ("PhotoGraphiQ", "photographiq_signal_shift", "-"),
                ("Graphix", "graphix_shift_signals_direct", "-"),
            ],
        )
    )
    panels.append(
        (
            "statevector, one fixed branch",
            results["statevector"],
            [
                ("PhotoGraphiQ", "photographiq_run_branch", "-"),
                ("Graphix", "graphix_statevector_fixed_branch", "-"),
            ],
        )
    )
    markers = {"linear": "o", "grid3": "s", "brickwork4": "^", "lc_random": "D"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (title, rows, series) in zip(axes.flat, panels):
        for fam, marker in markers.items():
            sub = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["nodes"])
            if not sub:
                continue
            x = [r["nodes"] for r in sub]
            for label, key, style in series:
                y = [r[key]["median_s"] for r in sub]
                err = [
                    [r[key]["median_s"] - r[key]["q1_s"] for r in sub],
                    [r[key]["q3_s"] - r[key]["median_s"] for r in sub],
                ]
                ax.errorbar(
                    x,
                    y,
                    yerr=err,
                    color=colors[label],
                    linestyle=style,
                    marker=marker,
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                    label=f"{label}, {fam}",
                    markerfacecolor="white" if style == ":" else None,
                )
        ax.set(
            xscale="log",
            yscale="log",
            title=title,
            xlabel="graph nodes",
            ylabel="median time per call [s] (bars: IQR)",
        )
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    env = results["environment"]
    fig.suptitle(
        f"Qubit MBQC: PhotoGraphiQ {env['packages']['photographiq']} vs Graphix "
        f"{env['packages']['graphix']} | {env['cpu']} | Python {env['python']}",
        fontsize=10,
    )
    paths = []
    for ext in ("svg", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def run(
    cfg: dict, *, seed: int = 2026, out_dir: Path = RESULTS, plots: bool = True, log=print
) -> dict:
    t0 = time.perf_counter()
    results = {
        "benchmark": "graphix_mbqc",
        "scope": "qubit MBQC only (flows, measurement calculus, small statevector); Graphix is "
        "not a CV/GBS simulator and no CV claim is made",
        "config": {k: v for k, v in cfg.items()},
        "seed": seed,
        "environment": environment(),
        "semantics": {
            "flows": "same open graph (graph, ordered inputs/outputs, plane/Pauli labels); "
            "output: existence + a flow with layered order. Ratio column compares PhotoGraphiQ "
            "find_* (search + built-in verify) with Graphix to_*_or_none + check_well_formed.",
            "standardize": "same non-standard input (Graphix XZCorrections.to_pattern of the "
            "Graphix flow); output: an equivalent NEMC standard pattern.",
            "signal_shift": "same standard input (Graphix standard form); PhotoGraphiQ appends "
            "S commands, Graphix returns the signal dict (same information).",
            "statevector": "same standardized pattern, same fixed branch, input |+>^k; output: "
            "normalized output state. PhotoGraphiQ run_branch with inputs as N commands.",
            "not_applicable": [
                "CV/Gaussian/GBS comparisons (Graphix has none)",
                "PhotoGraphiQ full Kraus map (run_branch with inputs) vs Graphix "
                "single state: different tasks, not compared",
                "Graphix tensor-network / density-matrix backends: no PhotoGraphiQ "
                "qubit counterpart",
            ],
        },
    }
    log("agreement sweep ...")
    results["agreement_random_small"] = agreement_sweep(cfg["agreement_instances"], seed)
    log(json.dumps(results["agreement_random_small"]))
    results["flows"] = bench_flows(cfg, seed, log)
    results["patterns"] = bench_patterns(cfg, seed, log)
    results["statevector"] = bench_statevector(cfg, seed, log)
    results["wall_time_s"] = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        out_dir / f"graphix_mbqc_{'full' if cfg is FULL else 'fast'}_"
        f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    if plots:
        results["plots"] = plot(results, stem)
    stem.with_suffix(".json").write_text(json.dumps(results, indent=1, default=str))
    log(f"wrote {stem.with_suffix('.json')} ({results['wall_time_s']:.1f} s)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full", action="store_true", help="larger sizes (long)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument(
        "--cap", type=float, help="stop growing a family once a median exceeds this"
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=RESULTS, help="directory for JSON and plots"
    )
    args = parser.parse_args()
    cfg = FULL if args.full else FAST
    for key, val in (("repeats", args.repeats), ("warmup", args.warmup), ("cap_s", args.cap)):
        if val is not None:
            cfg[key] = val
    run(cfg, seed=args.seed, out_dir=args.out_dir, plots=not args.no_plots)


if __name__ == "__main__":
    main()
