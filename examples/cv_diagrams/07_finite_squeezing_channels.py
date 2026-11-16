"""Tutorial 7: finite squeezing and loss as Gaussian channels (X, Y, d).

At finite resource squeezing r a corrected teleportation step is the channel
X = T(k), Y = diag(0, e^{-2r}) (vacuum units, hbar = 2). A compiled pattern is a product of
such steps, so X is r-independent while Y(r) = e^{-2r} Y_1. A logical rewrite changes Y_1, so
the noise *ratio* optimised/original does not depend on r. Loss and general Gaussian channels
add noise; their composition is exact (rule ``channel-fusion``, class exact-channel).

Key claims checked: Y e^{2r} is r-independent for the compiled and the optimised pattern and
the ratio is constant; X is the same at every r; channel fusion of loss + squeezing is exact
and completely positive; a non-CP GaussianChannelBox is rejected.
"""

from time import perf_counter

import numpy as np

import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    CVDiagram,
    DiagramTypeError,
    GaussianChannelBox,
    GaussianLoss,
    Squeezer,
    apply_rewrite,
    compare,
    compose_channels,
    evaluate,
    is_completely_positive,
    optimize_cv_pattern,
    resource_cost,
)

start = perf_counter()
circuit = pg.Circuit(1).rotate(0, 0.7).squeeze(0, 0.3).rotate(0, -0.4).squeeze(0, -0.2)

rows = []
for r in (0.5, 1.0, 1.5, 2.0):
    pattern = circuit.compile(squeezing=r)
    result = optimize_cv_pattern(pattern, equivalence="logical")
    X0, Y0, _ = evaluate(result.original).channel()
    X1, Y1, _ = evaluate(result.best).channel()
    c0, c1 = result.original_cost, result.best_cost
    rows.append((r, X0, Y0, Y1, c0, c1))
    print(
        f"r={r:.1f}: modes {c0['resource_modes']}->{c1['resource_modes']}, added noise "
        f"{c0['added_noise']:.5f}->{c1['added_noise']:.5f} (ratio "
        f"{c1['added_noise'] / c0['added_noise']:.6f}), squeezing energy "
        f"{c0['squeezing_energy']:.3f}->{c1['squeezing_energy']:.3f}"
    )

scaled0 = [Y0 * np.exp(2 * r) for r, _, Y0, _, _, _ in rows]
scaled1 = [Y1 * np.exp(2 * r) for r, _, _, Y1, _, _ in rows]
spread = max(np.abs(s - scaled0[0]).max() for s in scaled0)
spread += max(np.abs(s - scaled1[0]).max() for s in scaled1)
x_spread = max(np.abs(row[1] - rows[0][1]).max() for row in rows)
print(f"max spread of Y e^(2r) over r: {spread:.1e}; of X over r: {x_spread:.1e}")

# Loss then squeezing: channel fusion into one GaussianChannelBox.
lossy = CVDiagram((0,), (GaussianLoss(0, 0.8, 0.1), Squeezer(0, 0.4)), (0,))
fused = apply_rewrite(lossy, "channel-fusion")
X, Y, d = evaluate(fused).channel()
loss = (np.sqrt(0.8) * np.eye(2), 0.2 * 1.2 * np.eye(2), np.zeros(2))
squeeze = (pg.gaussian.squeezing(0.4), np.zeros((2, 2)), np.zeros(2))
Xc, Yc, _ = compose_channels(loss, squeeze)
fuse_gap = max(np.abs(X - Xc).max(), np.abs(Y - Yc).max())
print(
    f"fused channel Y = {np.round(Y, 5).tolist()}; |fused - compose_channels| = {fuse_gap:.1e}; "
    f"CP: {is_completely_positive(X, Y)}; physical equivalent: "
    f"{compare(lossy, fused).physical_equivalent}"
)
print(f"added noise of the lossy diagram: {resource_cost(lossy)['added_noise']:.5f}")

try:  # amplification without the noise it requires is not a channel
    GaussianChannelBox((0,), 2 * np.eye(2), np.zeros((2, 2)))
    rejected = False
except DiagramTypeError as exc:
    rejected = True
    print("rejected:", exc)
print(f"runtime: {perf_counter() - start:.1f} s")

ratios = [c1["added_noise"] / c0["added_noise"] for *_, c0, c1 in rows]
assert spread < 1e-9 and x_spread < 1e-10
assert max(ratios) - min(ratios) < 1e-9
assert fuse_gap < 1e-14 and is_completely_positive(X, Y) and rejected
