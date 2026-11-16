"""Tutorial 9: finite-energy GKP readout and Z-syndrome extraction as typed diagrams.

``GKPPrepare`` prepares the finite-energy codeword c0|0~> + c1|1~> of a ``pg.GKPCode`` (Fock
semantics only); ``ModularDecode`` is the classical nearest-cell parity
round(m / sqrt(2 pi)) mod 2 of a homodyne record. Probabilities of decoded bits are obtained by
integrating the postselected record density with a midpoint rule aligned to the decoding cells
(a quadrature rule, so the result carries a discretisation error as well as the cutoff error).

Key claims checked:
* the readout diagram (prepare -> q homodyne -> parity) reproduces PhotoGraphiQ's existing
  Z readout ``pg.multimode_readout`` within 2e-3, and the record density integrates to 1;
* R(pi/2) CZ R(-pi/2) on (data, ancilla) is the SUM gate q_a -> q_a + q_d (Gaussian part);
* Z-syndrome extraction onto a |0~> ancilla decodes the data's Z value, but with a larger
  finite-energy error than the direct readout (the ancilla's own peak width adds to the
  record); the gap is printed, not hidden. This is a finite-energy, finite-cutoff numerical
  statement, not a fault-tolerance claim.
"""

import warnings
from time import perf_counter

import numpy as np

import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    CZ,
    CVDiagram,
    GKPPrepare,
    Homodyne,
    ModularDecode,
    Rotation,
    run_fock,
    unitary_semantics,
)

warnings.simplefilter("ignore", RuntimeWarning)
start = perf_counter()
CUTOFF = 40
code = pg.GKPCode(cutoff=CUTOFF, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
logical = (np.cos(0.4), np.sin(0.4))  # cos 0.4 |0~> + sin 0.4 |1~>


def p_bit_zero(diagram, per_cell, cells=5):
    """P(bit = 0) and total probability: midpoint rule on cells [(k - 1/2) s, (k + 1/2) s].

    Aligning the grid with the cells avoids misclassifying points at the cell boundaries.
    """
    s = np.sqrt(2 * np.pi)
    h = s / per_cell
    p0 = total = 0.0
    for k in range(-cells, cells + 1):
        for j in range(per_cell):
            run = run_fock(diagram, CUTOFF, outcomes={"m": float((k - 0.5) * s + (j + 0.5) * h)})
            total += run.density * h
            p0 += run.density * h * (run.records["bit"] == 0)
    return p0, total


# 1. Direct Z readout.
readout = CVDiagram(
    (), (GKPPrepare("g", code, *logical), Homodyne("g", 0.0, "m"), ModularDecode("m", "bit")), ()
)
p_direct, total_direct = p_bit_zero(readout, 32)
state = run_fock(CVDiagram((), (GKPPrepare("g", code, *logical),), ("g",)), CUTOFF).state
existing = pg.multimode_readout(state, {"g": "Z"})["joint_probabilities"][(0,)]
print(
    f"direct readout: P(bit=0) = {p_direct:.5f} (integrated density {total_direct:.6f}); "
    f"multimode_readout {existing:.5f}; ideal cos^2(0.4) = {logical[0] ** 2:.5f}"
)

# 2. The SUM gate built from generators, checked on its Gaussian semantics.
S, _ = unitary_semantics(
    CVDiagram(
        ("d", "a"),
        (Rotation("a", np.pi / 2), CZ("d", "a", 1.0), Rotation("a", -np.pi / 2)),
        ("d", "a"),
    )
)
print("SUM symplectic (d, a interleaved):", np.round(S, 12).tolist())

# 3. Z-syndrome extraction: SUM from the data onto a |0~> ancilla, read the ancilla.
syndrome = CVDiagram(
    (),
    (
        GKPPrepare("d", code, *logical),
        GKPPrepare("a", code, 1.0, 0.0),
        Rotation("a", np.pi / 2),
        CZ("d", "a", 1.0),
        Rotation("a", -np.pi / 2),
        Homodyne("a", 0.0, "m"),
        ModularDecode("m", "bit"),
    ),
    ("d",),
)
p_syndrome, total_syndrome = p_bit_zero(syndrome, 16)
print(
    f"syndrome readout: P(bit=0) = {p_syndrome:.5f} (integrated density "
    f"{total_syndrome:.6f}); gap to direct readout {p_direct - p_syndrome:+.4f}"
)
print("diagram is Gaussian:", syndrome.is_gaussian, "| live outputs:", syndrome.outputs)
print(f"runtime: {perf_counter() - start:.1f} s")

assert abs(total_direct - 1) < 1e-3 and abs(total_syndrome - 1) < 1e-3
assert abs(p_direct - existing) < 2e-3
expected_sum = np.array([[1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 1, 0], [0, 0, 0, 1]])
assert np.allclose(S, expected_sum, atol=1e-12)  # q_a += q_d, p_d -= p_a
assert 0 < p_direct - p_syndrome < 0.1  # extra finite-energy error from the ancilla
