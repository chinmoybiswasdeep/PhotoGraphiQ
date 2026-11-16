"""Tutorial 3: verify rewrites against independent references, and catch a false rule.

Three independent checks of rewrites:
* ``compare``: exact semantic matrices, with logical / physical / conditional layers reported
  separately;
* ``analysis.gaussian_channel`` of the recompiled executable Pattern (a different code path);
* the Weyl phase recorded by displacement fusion against dense matrix exponentials in a
  truncated Fock space (finite-cutoff agreement is evidence, not proof).

Key claims checked: all exact rewrites of a random diagram are physically and conditionally
equivalent; the CZ + q-homodyne rewrite matches a postselected Gaussian-backend trajectory of
the original pattern; a false "rule" (swapping a rotation and a squeezer on one mode) is
rejected; a logical rewrite is logically but not physically equivalent.
"""

from time import perf_counter

import numpy as np
from scipy.linalg import expm

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import (
    CZ,
    BeamSplitter,
    CVDiagram,
    Displacement,
    Feedforward,
    Homodyne,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    apply_rewrite,
    compare,
    evaluate,
    from_pattern,
    to_pattern,
)
from photographiq.calculus.cv_diagrams.rules import GAUSSIAN_RULES

start = perf_counter()
rng = np.random.default_rng(3)

# 1. Every exact Gaussian rule, at every match, on a random two-mode diagram.
boxes = []
for _ in range(12):
    m = int(rng.integers(2))
    kind = rng.integers(5)
    boxes.append(
        [
            Rotation(m, rng.normal()),
            Squeezer(m, 0.3 * rng.normal()),
            Shear(m, rng.normal()),
            Displacement(m, rng.normal(), rng.normal()),
            BeamSplitter(m, 1 - m, rng.normal()),
        ][kind]
    )
boxes += [Rotation(0, 0.4), Rotation(0, -0.4), Displacement(1, 0.2, 0.1), Displacement(1, 0.3, 0.0)]
d = CVDiagram((0, 1), tuple(boxes), (0, 1))
worst, checked, executed = 0.0, 0, 0
for rule in GAUSSIAN_RULES:
    for match in rule.find(d):
        out = rule.apply(d, match)
        rep = compare(d, out)
        assert rep.physical_equivalent and rep.conditional_equivalent, rule.name
        worst = max(worst, rep.channel_X_residual, rep.displacement_residual)
        checked += 1
        try:  # independent path: executable pattern -> analysis.gaussian_channel
            ref = gaussian_channel(to_pattern(out))
        except NotImplementedError:  # e.g. multi-mode Symplectic boxes have no command
            continue
        X, _, _ = evaluate(d).channel()
        worst = max(worst, np.abs(ref.matrix - X).max())
        executed += 1
print(f"{checked} exact rewrites verified ({executed} also executed); worst residual {worst:.1e}")

# 2. Measurement rewrite vs a postselected trajectory of the *original* executable pattern.
base = CVDiagram(
    (0,),
    (
        Prepare.squeezed("a", 0.4),
        CZ(0, "a", 0.8),
        Homodyne("a", 0.0, "m"),
        Feedforward(0, (("m", (0.3, 0.0)),)),
    ),
    (0,),
)
rewritten = apply_rewrite(
    apply_rewrite(base, "cz-q-measurement-to-feedforward"), "prepare-measure-to-source"
)
mean_in, cov_in = np.array([0.5, -0.1]), np.array([[1.2, 0.1], [0.1, 1.0]])
mean, cov, density = evaluate(rewritten).conditional({"m": -0.6}, mean_in, cov_in)
pattern = to_pattern(base)
run = pg.simulate(
    pattern,
    backend="gaussian",
    measurement_outcomes={"m": -0.6},
    initial_state=pg.GaussianState(mean_in, cov_in, pattern.inputs),
)
traj_gap = max(
    np.abs(run.state.mean - mean).max(),
    np.abs(run.state.covariance - cov).max(),
    abs(np.exp(run.log_likelihood) - density),
)
print(f"ancilla eliminated; |rewritten conditional - original trajectory| = {traj_gap:.1e}")

# 3. Weyl phase of displacement fusion vs dense matrices (hbar = 2, q = a + a^dag).
a_vec, b_vec = (0.3, -0.2), (0.1, 0.5)
fused = apply_rewrite(
    CVDiagram((0,), (Displacement(0, *a_vec), Displacement(0, *b_vec)), (0,)),
    "displacement-fusion",
)
phase = float(fused.trace[-1].scalar.split("i*")[1].rstrip(")"))
N = 100
a = np.diag(np.sqrt(np.arange(1, N)), 1)
q, p = a + a.T, -1j * (a - a.T)


def D(dq, dp):  # Heisenberg q -> q + dq, p -> p + dp
    return expm(0.5j * (dp * q - dq * p))


lhs = D(*b_vec) @ D(*a_vec)  # a first, then b
rhs = np.exp(1j * phase) * D(a_vec[0] + b_vec[0], a_vec[1] + b_vec[1])
phase_gap = np.abs(lhs - rhs)[:20, :20].max()
print(f"recorded Weyl phase {phase:.6f}; dense-matrix residual (cutoff {N}) {phase_gap:.1e}")

# 4. A false rule: swap a rotation and a squeezer on the same wire.
right = CVDiagram((0,), (Rotation(0, 0.7), Squeezer(0, 0.4)), (0,))
wrong = CVDiagram((0,), (Squeezer(0, 0.4), Rotation(0, 0.7)), (0,))
false = compare(right, wrong)
print(f"false swap: logical {false.logical_equivalent}, X residual {false.channel_X_residual:.3f}")

# 5. A logical rewrite: same ideal map, different finite-squeezing noise and records.
chain = from_pattern(pg.Circuit(1).rotate(0, 0.3).rotate(0, 0.5).compile(squeezing=1.0))
short = apply_rewrite(chain, "teleport-chain-resynthesis")
rep = compare(chain, short)
print(
    f"resynthesis: {short.trace[-1].extra['steps_before']} -> "
    f"{short.trace[-1].extra['steps_after']} steps; logical {rep.logical_equivalent}, "
    f"physical {rep.physical_equivalent} (Y residual {rep.channel_Y_residual:.3f}); "
    f"{rep.measurement_record_mapping}"
)
print(f"runtime: {perf_counter() - start:.1f} s")

assert checked >= 5 and executed >= 3 and worst < 1e-10
assert traj_gap < 1e-9 and phase_gap < 1e-10
assert not false.logical_equivalent
assert rep.logical_equivalent and not rep.physical_equivalent and rep.conditional_equivalent is None
