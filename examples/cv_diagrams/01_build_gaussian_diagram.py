"""Tutorial 1: build a typed Gaussian CV diagram and read its three semantics.

The diagram is one corrected teleportation step: a p-squeezed resource mode "a" (squeezing r),
CZ(0, a), a p-homodyne of the input mode 0 (record "m") and the outcome-affine correction
q_a -= m. Wires are typed: modes are bosonic (linear, never copied), "m" is a real record.

Key claims checked:
* the averaged channel is X = [[0, -1], [1, 0]], Y = diag(0, e^{-2r}), d = 0, and it equals
  ``analysis.gaussian_channel`` of the executable Pattern (an independent implementation);
* the conditional instrument (state given m, and the record density) equals a postselected
  Gaussian-backend trajectory of the same pattern;
* ill-typed diagrams raise DiagramTypeError.
"""

import json
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus.cv_diagrams import (
    CZ,
    CVDiagram,
    DiagramTypeError,
    Feedforward,
    Homodyne,
    Prepare,
    Rotation,
    Squeezer,
    draw,
    evaluate,
    to_pattern,
    unitary_semantics,
)

matplotlib.use("Agg")
start = perf_counter()
r = 0.5

step = CVDiagram(
    inputs=(0,),
    boxes=(
        Prepare.squeezed("a", r),  # covariance diag(e^{2r}, e^{-2r})
        CZ(0, "a", 1.0),
        Homodyne(0, np.pi / 2, "m"),  # p-homodyne: consumes mode 0, writes record "m"
        Feedforward("a", (("m", (-1.0, 0.0)),)),  # q_a += -1 * m
    ),
    outputs=("a",),
)
print(step, "records:", step.records, "gaussian:", step.is_gaussian)

# 1. Averaged channel x_out = X x_in + d + noise(Y).
semantics = evaluate(step)
X, Y, d = semantics.channel()
print("X =", np.round(X, 12).tolist())
print("Y =", np.round(Y, 6).tolist(), " e^{-2r} =", round(np.exp(-2 * r), 6))
reference = gaussian_channel(to_pattern(step))
gap = max(
    np.abs(X - reference.matrix).max(),
    np.abs(Y - reference.noise).max(),
    np.abs(d - reference.displacement).max(),
)
print(f"max |diagram - gaussian_channel(Pattern)| = {gap:.1e}")

# 2. Conditional instrument for a coherent-like input and a fixed record.
mean_in, cov_in = np.array([0.4, -0.2]), np.array([[1.3, 0.2], [0.2, 0.9]])
mean, cov, density = semantics.conditional({"m": 0.7}, mean_in, cov_in)
pattern = to_pattern(step)
trajectory = pg.simulate(
    pattern,
    backend="gaussian",
    measurement_outcomes={"m": 0.7},
    initial_state=pg.GaussianState(mean_in, cov_in, pattern.inputs),
)
print("conditional mean:", np.round(mean, 6).tolist(), " record density:", round(density, 6))
cond_gap = max(
    np.abs(trajectory.state.mean - mean).max(),
    np.abs(trajectory.state.covariance - cov).max(),
    abs(np.exp(trajectory.log_likelihood) - density),
)
print(f"max |diagram - postselected trajectory| = {cond_gap:.1e}")

# 3. A unitary diagram has a symplectic semantics.
S, shift = unitary_semantics(CVDiagram((0,), (Rotation(0, 0.3), Squeezer(0, 0.2)), (0,)))
print("R(0.3) then S(0.2): S =", np.round(S, 6).tolist())

# 4. Typing: a record cannot feed a bosonic port; a consumed mode cannot be reused.
errors = []
for boxes, outputs in (
    ((Homodyne(0, 0.0, "m"), Rotation("m", 0.1)), ()),
    ((Homodyne(0, 0.0, "m"), Rotation(0, 0.1)), ()),
    ((Feedforward(0, (("k", (1.0, 0.0)),)),), (0,)),
):
    try:
        CVDiagram((0,), boxes, outputs)
    except DiagramTypeError as exc:
        errors.append(str(exc))
print(*("DiagramTypeError: " + e for e in errors), sep="\n")

# 5. Serialisation, fingerprint and a static drawing.
again = CVDiagram.from_dict(json.loads(step.to_json()))
print(
    "fingerprint:",
    step.fingerprint(),
    "round trip equal:",
    again.fingerprint() == step.fingerprint(),
)
figure = Path(tempfile.gettempdir()) / "cv_diagram_teleport_step.png"
draw(step, title="one teleportation step").figure.savefig(figure, dpi=150)
print("drawing saved to the temp directory as", figure.name)
print(f"runtime: {perf_counter() - start:.1f} s")

assert np.allclose(X, [[0, -1], [1, 0]], atol=1e-12) and np.allclose(d, 0)
assert np.allclose(Y, np.diag([0.0, np.exp(-2 * r)]), atol=1e-12)
assert gap < 1e-12 and cond_gap < 1e-9
assert np.allclose(S @ pg.gaussian.omega(1) @ S.T, pg.gaussian.omega(1))
assert len(errors) == 3 and again.fingerprint() == step.fingerprint()
