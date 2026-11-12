# Issues found while building the manuscript experiment suite

Per the task's instructions, nothing here was silently patched. These are
documented observations with minimal reproducers; `src/photographiqml` was
not modified. None of these were treated as reasons to weaken an
experiment's acceptance condition after the fact — where an experiment's
original design assumption turned out to be wrong, the design was corrected
(and the correction is noted in the relevant script's docstring), not the
threshold.

## 1. MentPy's `restrict_trainable` never actually fixes the Ment angle (upstream, documented, not a defect)

`R10_trainability_audit` (`02_mentpy_validation/10_trainability_audit.py`)
quantitatively confirms `docs/research/mentpy-audit.md`'s existing claim: for
every one of 8 tested `(n_wires, n_layers, one_column)` configurations,
`mentpy.templates.muta(..., restrict_trainable=True)`'s restricted column-3
`Ment` objects report `angle=None` (i.e. `Ment.is_trainable()` is still
`True`) — the restriction only ever removes the node from the
`trainable_nodes` *list*, and even that survives only for the single
unstacked case (`n_wires=1, n_layers=1`, where neither `add_edge` nor
`hstack` is called after the list is set).

Minimal reproducer (see `R10_trainability_audit.json`):

```python
import mentpy as mp

g = mp.templates.muta(2, 1, one_column=True, restrict_trainable=True)
col3_node = ...  # a column-3 node, semantically mapped
assert (
    col3_node not in g.trainable_nodes
)  # list omission: also fails except at n_wires=1,n_layers=1
assert g[col3_node].angle is None  # angle never actually fixed, in EVERY configuration tested
```

**This is not a PhotoGraphiQML defect.** PhotoGraphiQML never relies on
MentPy's list convention; `ansatz/muta.py` freezes column-3 parameters
directly via `ParameterStore.freeze`, verified robust in every configuration
by `R10`'s own `photographiqml_robust` check (100% pass). Affects MentPy
`0.1.0a15` (commit `63c3d83`). Does not block any PhotoGraphiQML manuscript
claim; it is exactly the caveat the release documentation already states.

## 2. MentPy's raw `Flow.correction_op` includes a formal self-correction on the measured node (upstream convention difference, not a defect)

`R8_flow_dependency_order.py` found that `mentpy`'s
`Flow.correction_op(node)` reports a Z-component on the just-measured `node`
itself (since `node` is a graph-neighbor of its own flow successor, it
appears in `odd_neighborhood({successor})`), whereas
`photographiqml.ansatz.muta`'s own `corrections` map explicitly excludes the
measured node (`z = frozenset(graph.neighbors(successor)) - {node}` in
`ansatz/muta.py`), since a Z correction on an already destructively-measured
qubit has no physical effect on the surviving register.

Minimal reproducer:

```python
# For source=(0,0) in a 2-wire MuTA(2,one_column=True):
# model.corrections[source] = (successor=(0,1), z_targets={(1,0),(0,2),(1,2)})
# but the MAPPED mentpy correction_op's raw Z-part decodes to
# {(1,0),(0,2),(1,2),(0,0)} -- includes (0,0), the source itself.
```

**Not a defect in either package** — both are internally consistent; they
differ only in whether the formal frame bookkeeping includes a physically
inert self-term. `R8` subtracts the source node from MentPy's mapped Z-set
before comparing (see `self_correction_dropped` field in
`R8_flow_dependency_order.json`), documented rather than silently absorbed.
Does not block any manuscript claim.

## 3. Composite two-feature MuTAClassifier boundary did not train reliably in a moderate epoch budget (PhotoGraphiQML ansatz/optimizer, scientific finding)

While designing `R23_classifier_verification`, a two-wire
`MuTAClassifier(MuTA(2, one_column=True), trainer=Trainer("adam"))` was
first tried on the synthetic boundary `y = 1[cos(x0)+cos(x1) > 0]` (raw
2-D features, first-wire-Z logistic head). Across 5 stratified splits at 80
and again at 200 Adam epochs (`learning_rate` 0.05 and 0.1), training loss
plateaued near `log(2) ≈ 0.693` (chance level) and held-out accuracy stayed
at 0.50–0.78 (median 0.64), well below a reasonable separability bar.

Minimal reproducer:

```python
import numpy as np
from photographiqml import MuTA, MuTAClassifier, Trainer

X = np.random.default_rng(0).uniform(0, np.pi, size=(60, 2))
y = (np.cos(X[:, 0]) + np.cos(X[:, 1]) > 0).astype(int)
clf = MuTAClassifier(
    MuTA(2, one_column=True), trainer=Trainer("adam", epochs=200, learning_rate=0.1), seed=0
)
clf.fit(X[:42], y[:42])
print(clf.history.losses[-1])  # stays near 0.69 (chance), does not converge
```

**Scientific impact:** this does not by itself indicate a code defect (the
8-parameter, one-layer ansatz combined with a first-wire-only Z readout may
simply lack the capacity/trainability for this composite two-feature
boundary within this epoch budget, or central-difference Adam may be
struggling with a flat optimization landscape here — R22's sensitivity
sweep shows the *same* ansatz reliably learns single-target-unitary gate
objectives). It does mean: **do not cite `MuTAClassifier` as demonstrated
to learn arbitrary composite multi-feature boundaries** without further
investigation; `R23` was rescoped to the single-feature threshold task the
package's own README already demonstrates working (median accuracy 0.98
across 5 splits), which is what is actually validated in this suite. This
finding does not block the manuscript's classifier-wrapper *correctness*
claim (which `R23` in its final, rescoped form supports), only a broader
*expressivity* claim that was never made in the source documentation to
begin with.

## Not filed as issues (environment/tooling notes, no scientific content)

- The active `.venv` initially resolved `photographiqml` to the unrelated
  sibling `cv-mb-qrc` checkout rather than this repository. This caused all
  direct manuscript-script invocations to fail before experiment execution;
  it is a reproducibility defect, not a scientific result. `common.py` now
  puts this checkout's `src/` first before importing the public API, and the
  affected scripts are rerun with the resolved module path recorded in their
  metadata.

- Two-mode raw-Piquasso CZ on **pure** `|mu>|nu>` basis codewords (R40)
  needed a materially higher cutoff (48+) than the 1-wire single-mode
  preparation case (R39, cutoff 16 sufficient) or than typical
  superposition-state execution (cutoff 24, per `docs/physical/evidence.json`)
  to clear PhotoGraphiQ's retained-norm guard. This is consistent with,
  and a quantitative extension of, the cutoff-sensitivity behavior the
  release report already documents for two-mode operations — not a new
  finding.
