"""Exact rational linear algebra for CV-flow rank decisions (stdlib ``fractions``).

Every float64 is a dyadic rational, so ``Fraction(x)`` is exact and the ranks and
solutions computed here are those of the *given* weights with no rounding at all: a
rank decision made here is a proof for that data, not an estimate. This is stronger
than any fixed-precision (e.g. mpmath) re-check and needs no optional dependency. Cost
grows with the bit size of the eliminated entries, so callers restrict it to small
graphs unless asked explicitly.
"""

from __future__ import annotations

from fractions import Fraction

Matrix = list[list[Fraction]]


def exact(matrix) -> Matrix:
    return [[Fraction(float(x)) for x in row] for row in matrix]


def _rref(rows: Matrix, pivot_columns: int) -> tuple[Matrix, list[int]]:
    """Gauss-Jordan elimination pivoting on the first ``pivot_columns`` columns."""
    rows = [row[:] for row in rows]
    pivots: list[int] = []
    for col in range(pivot_columns):
        r = len(pivots)
        pivot = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][col]
        rows[r] = [x / lead for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
    return rows, pivots


def _dot(a, b) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def rank(matrix: Matrix) -> int:
    return len(_rref(matrix, len(matrix[0]) if matrix else 0)[1])


def solve_round(m: Matrix) -> tuple[int, dict, dict]:
    """Decide e_u in range(M) for every row u of M exactly; solve where possible.

    Returns (rank, solutions, witnesses): ``solutions[u]`` is
    the minimum-norm c with M c = e_u, and ``witnesses[u]`` a left-null vector y
    (y^T M = 0) with y_u != 0, which proves that M c = e_u has no solution.
    """
    rows, cols = len(m), len(m[0]) if m else 0
    identity = [[Fraction(int(i == j)) for j in range(rows)] for i in range(rows)]
    reduced, pivots = _rref([m[i] + identity[i] for i in range(rows)], cols)
    null = [row[cols:] for row in reduced[len(pivots) :]]  # y^T M = 0 for each
    witnesses = {u: next(y for y in null if y[u]) for u in range(rows) if any(y[u] for y in null)}
    # Minimum norm: c = M^T y with (M M^T) y = e_u (consistent iff e_u in range M).
    gram = [[_dot(m[i], m[j]) for j in range(rows)] for i in range(rows)]
    reduced, pivots = _rref([gram[i] + identity[i] for i in range(rows)], rows)
    solutions = {}
    for u in range(rows):
        if u in witnesses:
            continue
        y = [Fraction(0)] * rows
        for k, p in enumerate(pivots):
            y[p] = reduced[k][rows + u]
        solutions[u] = [_dot([m[i][j] for i in range(rows)], y) for j in range(cols)]
    return len(pivots), solutions, witnesses
