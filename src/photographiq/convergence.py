"""Reproducible cutoff comparisons with explicit branch comparability."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .backends.fock import FockResultState
from .fock_analysis import fidelity, trace_distance
from .simulator import simulate


@dataclass
class CutoffStudy:
    rows: list[dict]
    results: list


def cutoff_convergence(pattern, cutoffs, *, seed=0, **kwargs) -> CutoffStudy:
    """Run a pattern (or cutoff -> pattern factory) at strictly increasing cutoffs.

    Use measurement_outcomes to compare the same conditional branch. Equal seeds
    do not imply equal outcomes: differing branches are marked incomparable.
    A pattern factory regenerates resources whose definition depends on cutoff.
    Native unitary norm conservation is not a convergence certificate.
    """
    cutoffs = tuple(cutoffs)
    if (
        len(cutoffs) < 2
        or any(not isinstance(c, int) or isinstance(c, bool) or c < 2 for c in cutoffs)
        or any(b <= a for a, b in zip(cutoffs, cutoffs[1:]))
    ):
        raise ValueError("Provide at least two strictly increasing integer cutoffs >=2")
    if "backend" in kwargs or "cutoff" in kwargs:
        raise ValueError("The convergence helper controls backend and cutoff")
    results: list = []
    rows: list[dict] = []
    for cutoff in cutoffs:
        current = pattern(cutoff) if callable(pattern) else pattern
        start = perf_counter()
        result = simulate(current, backend="piquasso-fock", cutoff=cutoff, seed=seed, **kwargs)
        elapsed = perf_counter() - start
        state = result.state
        if not isinstance(state, FockResultState):
            raise TypeError("Cutoff studies require Fock outputs")
        row = {
            "cutoff": cutoff,
            "seconds": elapsed,
            "norm": state.norm,
            "minimum_retained_norm": min(state.retained_norms, default=1),
            "maximum_boundary_population": max(
                (d["boundary_population"] for d in state.diagnostics), default=0
            ),
            "peak_dimension": max((d["dimension"] for d in state.diagnostics), default=1),
            "outcomes": result.outcomes,
            "photon_numbers": {n: state.photon_number(n) for n in state.nodes},
            "quadratures": {
                n: {"q": state.quadrature(n), "p": state.quadrature(n, np.pi / 2)}
                for n in state.nodes
            },
            "fidelity_to_previous": None,
            "trace_distance_to_previous": None,
            "probability_l1_to_previous": None,
            "comparable_to_previous": False,
        }
        if results:
            previous = results[-1]
            comparable = (
                previous.outcomes == result.outcomes and previous.state.nodes == state.nodes
            )
            row["comparable_to_previous"] = comparable
            if comparable:
                row["fidelity_to_previous"] = fidelity(previous.state, state)
                row["trace_distance_to_previous"] = trace_distance(previous.state, state)
                p, q = previous.state.probabilities, state.probabilities
                row["probability_l1_to_previous"] = sum(
                    abs(p.get(b, 0) - q.get(b, 0)) for b in set(p) | set(q)
                )
        rows.append(row)
        results.append(result)
    return CutoffStudy(rows, results)
