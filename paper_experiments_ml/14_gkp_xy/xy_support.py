"""Shared helpers for the physical logical-XY GKP studies R60-R67 (extension catalogue).

Three independent levels are kept apart in every study:

* ideal logical qubit (``ideal_probabilities``, ``ideal_partner_bloch``);
* untruncated finite-energy reference (``GridModel`` in tests/validation/gkp_xy_reference.py:
  position grid, exact SUM/shear, exact cell integrals; no Fock truncation);
* the system under test: ``pg.GKPLogicalXYMeasurement`` executed by the native Fock engines.

So ideal-vs-reference is the physical finite-energy error, reference-vs-engine is Fock
truncation, and engine shot spread is sampling error.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "tests" / "validation"))

import common  # noqa: E402
from gkp_xy_reference import (  # noqa: E402
    GridModel,
    ideal_partner_bloch,
    ideal_probabilities,
)

import photographiq as pg  # noqa: E402
from photographiq.encoded import _x_readout_probabilities  # noqa: E402

# Representative resource qualities (peak width = envelope); cutoffs chosen from the
# measured retained norms (two live modes, up to four injection rounds).
RESOURCES = {
    "broad": {"peak_width": 0.6, "envelope": 0.6, "peaks": 6, "cutoff": 64},
    "medium": {"peak_width": 0.45, "envelope": 0.45, "peaks": 6, "cutoff": 80},
    "sharp": {"peak_width": 0.3, "envelope": 0.3, "peaks": 8, "cutoff": 112},
}
CARDINAL = {"0": (1, 0), "1": (0, 1), "+": (1, 1), "-": (1, -1), "+i": (1, 1j), "-i": (1, -1j)}
GENERIC = (float(np.cos(0.4)), complex(np.exp(0.9j) * np.sin(0.4)))  # breaks a -> -a symmetry
REQUIRED_ANGLES = (0.0, np.pi / 8, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4, np.pi)
ROUNDS = 4

_grids = {}


def grid(name, dx=0.07):
    key = (name, dx)
    if key not in _grids:
        r = RESOURCES[name]
        _grids[key] = GridModel(r["peak_width"], r["envelope"], r["peaks"], dx=dx)
    return _grids[key]


def code(name, cutoff=None):
    r = RESOURCES[name]
    return pg.GKPCode(
        cutoff=cutoff or r["cutoff"],
        peak_width=r["peak_width"],
        envelope=r["envelope"],
        peaks=r["peaks"],
        grid_points=2049,
    )


def single_mode_pattern(instrument):
    pattern = pg.Pattern(inputs=("d",))
    pattern.append(pg.Measure("d", instrument, "m"))
    return pattern.append(pg.Output(()))


def run_single(name, alpha, logical, *, protocol="auto", shots=1, seed=0, cutoff=None,
               backend="native-fock", outcomes=None, **options):
    """Engine records for one data mode; ``outcomes`` postselects a complete analog record."""
    gkp = code(name, cutoff)
    instrument = gkp.logical_measurement(
        "XY", alpha=alpha, protocol=protocol, max_rounds=options.pop("max_rounds", ROUNDS), **options
    )
    pattern = single_mode_pattern(instrument)
    kwargs = {"backend": backend, "cutoff": gkp.cutoff, "inputs": {"d": gkp.encode(*logical)}}
    records, seconds = [], []
    seeds = np.random.SeedSequence(seed).spawn(shots)
    for child in seeds:
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            extra = {"measurement_outcomes": {"m": outcomes}} if outcomes is not None else {}
            records.append(pg.simulate(pattern, seed=child, **kwargs, **extra).records["m"])
        seconds.append(time.perf_counter() - start)
    return records, seconds


def bell_resource(gkp):
    """Two-mode finite-energy Bell state (|0~0~> + |1~1~>)/norm as a FockSuperposition."""
    zero, one = (np.asarray(gkp.resource(b).fock(gkp.cutoff).amplitudes) for b in (0, 1))
    amplitudes = {}
    for m in range(gkp.cutoff):
        for n in range(gkp.cutoff - m):
            value = zero[m] * zero[n] + one[m] * one[n]
            if abs(value) > 1e-15:
                amplitudes[(m, n)] = value
    norm = np.sqrt(sum(abs(v) ** 2 for v in amplitudes.values()))
    return pg.FockSuperposition(tuple(amplitudes), tuple(v / norm for v in amplitudes.values()))


def partner_bloch(state, node, cutoff):
    """Decoded (<X>, <Y>, <Z>) of one mode via the physical readouts (Y = S^dagger then X)."""
    from photographiq.registry import create_backend

    effects = pg.modular_effects(cutoff, "Z")
    reduced = state.reduced((node,))
    occupations = np.asarray(reduced.basis)[:, 0]
    rho = np.asarray(reduced.density_matrix)
    z = [float(np.einsum("ij,ji->", rho, e[np.ix_(occupations, occupations)]).real) for e in effects]
    engine = create_backend("native-mixed-fock", cutoff)
    engine.reset(0)
    rho = rho / np.trace(rho).real
    engine.prepare_resource(("p",), pg.FockDensityMatrix(rho, tuple((int(n),) for n in occupations)))
    x = _x_readout_probabilities(engine, "p")
    engine.quadratic_phase("p", -1.0)
    y = _x_readout_probabilities(engine, "p")
    return x[0] - x[1], y[0] - y[1], z[0] - z[1]


def plt():
    return common.setup_style()


def save(rows, name, extra, figure=None):
    common.save_result(rows, name, extra=extra, meta_extra={"experiment_id": name.split("_")[0]})
    if figure is not None:
        common.save_figure(figure, name)


__all__ = [
    "CARDINAL",
    "GENERIC",
    "REQUIRED_ANGLES",
    "RESOURCES",
    "ROUNDS",
    "GridModel",
    "bell_resource",
    "code",
    "common",
    "grid",
    "ideal_partner_bloch",
    "ideal_probabilities",
    "partner_bloch",
    "pg",
    "plt",
    "run_single",
    "save",
    "single_mode_pattern",
]
