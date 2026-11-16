"""Finite-Fock (truncated) semantics of CV diagrams on PhotoGraphiQ's native Fock engines.

This is *numerical* semantics: an exclusive total-photon cutoff, renormalisation after every
operation (the engine refuses truncation beyond its norm tolerance and records retained
norms). Agreement at a finite cutoff is evidence, never a proof of an operator identity on
the infinite-dimensional space; ``cutoff_study`` reports convergence explicitly.

Records: photon counts are probabilities, homodyne records are densities. ``Postselect``
boxes require the corresponding outcome to be supplied. ``FockRun.probability`` is the product
of the probabilities of *all* discrete records of the run: when every photon count is fixed by
``outcomes`` (as post-selection requires) it is the heralding probability of that branch;
when counts are sampled it is the probability of the sampled branch. It is reported
separately from the normalised conditional output state.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ...measurements import Homodyne as HomodyneMeasurement
from ...measurements import PhotonNumber
from ...registry import create_backend
from ...states import FockInput, GaussianInput
from .ir import (
    CZ,
    BeamSplitter,
    ClassicalSource,
    Discard,
    Displacement,
    Feedforward,
    FockPrepare,
    GaussianLoss,
    GKPPrepare,
    Homodyne,
    Kerr,
    ModularDecode,
    PhasePolynomial,
    PhotonCount,
    Postselect,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)


@dataclass
class FockRun:
    state: object  # reduced to the diagram outputs
    records: dict
    probability: float  # product of discrete-record probabilities (heralding)
    density: float  # product of homodyne record densities
    retained_norm: float  # smallest retained norm of any operation


def run_fock(diagram, cutoff, *, inputs=None, outcomes=None, seed=None, backend=None):
    """Execute ``diagram`` once on a Fock engine; ``outcomes`` postselect records by key."""
    outcomes = dict(outcomes or {})
    mixed = any(isinstance(b, GaussianLoss) for b in diagram.boxes)
    engine = create_backend(backend or ("native-mixed-fock" if mixed else "native-fock"), cutoff)
    engine.reset(seed)
    inputs = dict(inputs or {})
    for w in diagram.inputs:
        engine.prepare(w, squeezing=0, state=inputs.get(w, GaussianInput()))
    records, probability, density, discarded = {}, 1.0, 1.0, []
    classical_rng = np.random.default_rng(seed)  # one stream: independent classical sources

    def measure(node, measurement, angle, key):
        nonlocal probability, density
        given = outcomes.get(key)
        engine.last_measurement = {}
        record = (
            engine.measure(node, measurement, angle, outcome=given)
            if given is not None
            else engine.measure(node, measurement, angle)
        )
        stat = engine.last_measurement or {}
        if stat.get("kind") == "probability":
            probability *= stat["value"]
        elif stat.get("kind") == "density":
            density *= stat["value"]
        records[key] = record

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in diagram.boxes:
            if isinstance(b, Prepare):
                engine.prepare(
                    b.mode,
                    squeezing=0,
                    state=GaussianInput(tuple(b.mean), tuple(map(tuple, b.covariance))),
                )
            elif isinstance(b, FockPrepare):
                engine.prepare(b.mode, squeezing=0, state=FockInput.number(b.n))
            elif isinstance(b, GKPPrepare):
                engine.prepare(b.mode, squeezing=0, state=b.code.encode(b.c0, b.c1))
            elif isinstance(b, Rotation):
                engine.rotate(b.mode, b.theta)
            elif isinstance(b, Squeezer):
                engine.squeeze(b.mode, b.r)
            elif isinstance(b, Shear):
                engine.quadratic_phase(b.mode, b.s)
            elif isinstance(b, Displacement):
                engine.displace(b.mode, b.dq, b.dp)
            elif isinstance(b, BeamSplitter):
                engine.beamsplitter(b.u, b.v, b.theta)
            elif isinstance(b, CZ):
                engine.entangle(b.u, b.v, b.g)
            elif isinstance(b, Symplectic) and len(b.labels) == 1:
                from .translate import _single_mode_gates

                first, r, last = _single_mode_gates(b.S)
                engine.rotate(b.labels[0], first)
                engine.squeeze(b.labels[0], r)
                engine.rotate(b.labels[0], last)
                if np.any(b.d):
                    engine.displace(b.labels[0], *b.d)
            elif isinstance(b, PhasePolynomial):
                c = list(b.coefficients) + [0.0] * 4
                if len(b.coefficients) > 4:
                    raise NotImplementedError("Phase polynomials above degree 3")
                # commuting functions of q: order irrelevant; c0 is a global phase
                if c[2]:
                    engine.quadratic_phase(b.mode, 4 * c[2])
                if c[1]:
                    engine.displace(b.mode, 0.0, 2 * c[1])
                if c[3]:
                    engine.cubic_phase(b.mode, 6 * c[3])
            elif isinstance(b, Kerr):
                engine.kerr(b.mode, b.kappa)
            elif isinstance(b, GaussianLoss):
                engine.loss(b.mode, b.eta, b.thermal)
            elif isinstance(b, Homodyne):
                measure(b.mode, HomodyneMeasurement(b.theta, b.efficiency, b.noise), b.theta, b.key)
            elif isinstance(b, PhotonCount):
                measure(b.mode, PhotonNumber(), 0.0, b.key)
            elif isinstance(b, Postselect):
                if b.key not in outcomes or outcomes[b.key] != b.value:
                    raise ValueError(
                        f"Postselect({b.key!r} = {b.value!r}) needs outcomes[{b.key!r}] = {b.value!r}"
                    )
            elif isinstance(b, Feedforward):
                q = b.constant[0] + sum(a * float(records[k]) for k, (a, _) in b.gains)
                p = b.constant[1] + sum(g * float(records[k]) for k, (_, g) in b.gains)
                engine.displace(b.mode, q, p)
            elif isinstance(b, ModularDecode):
                records[b.key] = int(np.rint(float(records[b.source]) / b.spacing)) % 2
            elif isinstance(b, ClassicalSource):
                records[b.key] = outcomes.get(
                    b.key, classical_rng.normal(b.mean, np.sqrt(b.variance))
                )
            elif isinstance(b, Discard):
                discarded.append(b.mode)
            else:
                raise NotImplementedError(f"No Fock execution for {type(b).__name__}")
    state = engine.get_state(diagram.outputs) if diagram.outputs else None
    norms = getattr(engine, "retained_norms", None) or [1.0]
    return FockRun(state, records, probability, density, float(min(norms)))


def cutoff_study(diagram, cutoffs, observable, **kwargs):
    """Evaluate ``observable(FockRun)`` at each cutoff; refused cutoffs are reported, not hidden."""
    rows = []
    for cutoff in cutoffs:
        try:
            run = run_fock(diagram, cutoff, **kwargs)
            rows.append(
                {
                    "cutoff": cutoff,
                    "value": observable(run),
                    "retained_norm": run.retained_norm,
                    "probability": run.probability,
                }
            )
        except ValueError as exc:
            rows.append({"cutoff": cutoff, "value": None, "refused": str(exc)})
    return rows
