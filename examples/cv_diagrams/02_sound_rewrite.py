"""Tutorial 2: apply sound rewrite rules and read the rewrite trace.

Each rule declares an equivalence class: ``exact-instrument`` (joint law of outputs and
records unchanged), ``exact-channel`` (outcome-averaged channel unchanged, no records in the
region) or ``logical`` (ideal map only). Rules match wire-adjacent boxes only, never mutate
their input and append a RewriteStep to the new diagram's trace.

Key claims checked: every exact-instrument rewrite below leaves the averaged channel and the
conditional instrument unchanged (``compare``); the ancilla-elimination pair removes a
resource mode; channel fusion preserves the averaged channel; the original is never mutated.
"""

import json
from time import perf_counter

import numpy as np

from photographiq.calculus.cv_diagrams import (
    CZ,
    RULES,
    CVDiagram,
    Displacement,
    Feedforward,
    GaussianLoss,
    Homodyne,
    Prepare,
    Rotation,
    Squeezer,
    apply_rewrite,
    compare,
    trace_json,
)

start = perf_counter()
print("rules:", ", ".join(f"{name} [{rule.equivalence}]" for name, rule in RULES.items()))

# 1. Fusion needs wire adjacency: the squeezer on mode 0 blocks the rotations ...
blocked = CVDiagram((0,), (Rotation(0, 0.2), Squeezer(0, 0.1), Rotation(0, 0.3)), (0,))
try:
    apply_rewrite(blocked, "rotation-fusion")
except ValueError as exc:
    print("blocked:", exc)
# ... but a box on another wire does not (it commutes, so it is not "between").
d = CVDiagram(
    (0, 1),
    (
        Rotation(0, 0.2),
        Squeezer(1, 0.1),
        Rotation(0, 0.3),
        Displacement(1, 0.2, 0.1),
        Displacement(1, -0.1, 0.4),
    ),
    (0, 1),
)
snapshot = d.to_dict()
fused = apply_rewrite(d, "rotation-fusion")
fused = apply_rewrite(fused, "displacement-fusion")
for step in fused.trace:
    print(f"{step.rule}: boxes {step.affected} -> {step.after[0]['box']}, scalar {step.scalar}")
report = compare(d, fused)
print(
    f"boxes {len(d.boxes)} -> {len(fused.boxes)}; physical {report.physical_equivalent}, "
    f"conditional {report.conditional_equivalent}, X residual {report.channel_X_residual:.1e}"
)

# 2. Ancilla elimination: CZ + ideal q-homodyne -> feed-forward (MFvL vertex deletion), then
#    the measured preparation becomes a classical Gaussian source.
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
step1 = apply_rewrite(base, "cz-q-measurement-to-feedforward")
step2 = apply_rewrite(step1, "prepare-measure-to-source")
print("after elimination:", [type(b).__name__ for b in step2.boxes])
for out in (step1, step2):
    rep = compare(base, out)
    print(
        f"  {out.trace[-1].rule}: conditional {rep.conditional_equivalent} "
        f"(instrument residual {rep.instrument_residual:.1e})"
    )
merged = apply_rewrite(step2, "feedforward-merge")
print("merged feed-forward:", merged.boxes[-1].gains, "|", merged.trace[-1].scalar)

# 3. Channel fusion: loss followed by a squeezer -> one GaussianChannelBox (exact-channel).
lossy = CVDiagram((0,), (GaussianLoss(0, 0.7, 0.1), Squeezer(0, 0.3)), (0,))
channel = apply_rewrite(lossy, "channel-fusion")
rep = compare(lossy, channel)
print(
    f"channel fusion [{channel.trace[-1].equivalence}]: physical {rep.physical_equivalent}, "
    f"Y residual {rep.channel_Y_residual:.1e}"
)

trace = json.loads(trace_json(merged, note="tutorial 2"))
print("trace rules:", [s["rule"] for s in trace["trace"]])
print(f"runtime: {perf_counter() - start:.1f} s")

assert d.to_dict() == snapshot and len(d.boxes) == 5  # rewrites never mutate
assert report.physical_equivalent and report.conditional_equivalent
assert all(compare(base, out).conditional_equivalent for out in (step1, step2, merged))
assert not any(isinstance(b, Prepare) for b in step2.boxes)
assert rep.physical_equivalent and channel.trace[-1].equivalence == "exact-channel"
weyl = float(fused.trace[-1].scalar.split("i*")[1].rstrip(")"))
assert np.isclose(weyl, (0.2 * 0.4 - 0.1 * -0.1) / 4)  # (a_q b_p - a_p b_q) / 4, a first
assert np.isclose(fused.boxes[-1].dq, 0.1) and np.isclose(fused.boxes[-1].dp, 0.5)
