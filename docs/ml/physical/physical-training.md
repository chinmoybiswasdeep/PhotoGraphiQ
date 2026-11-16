# Physical training

Two kinds of physical training exist. `DiscreteSearch` serves the default signed-X family
(`measurement_family="X"`) and is unchanged. Continuous tools serve
`measurement_family="XY"` ([below](#continuous-training-xy-family)).

## Discrete search (signed-X family)

The default `PhysicalMuTA` declares a categorical X family. Its initializer samples exact 0/pi
values and rejects a continuous initialization scale. `DiscreteSearch` performs
bounded coordinate flips on that family. Initial values must be exact categorical
choices; a continuously trained solution is never snapped to the allowed set.

```python
from photographiqml import PhysicalMuTA, GKPPhysicalConfig, MuTAClassifier
from photographiqml.physical_training import DiscreteSearch
model = PhysicalMuTA(physical_config=GKPPhysicalConfig(
    cutoff=24, peak_width=.9, envelope=.9, peaks=4, grid_points=1025))
classifier = MuTAClassifier(model, trainer=DiscreteSearch(sweeps=1),
    physical_options={"mode": "physical-shots", "shots": 2, "seed": 7})
classifier.fit([[0], [.2], [2.9], [3.141592653589793]], [0, 0, 1, 1])
```

For each candidate, decoded first-wire Z features feed a classical two-parameter
head fit by BFGS with a small L2 penalty. This continuous fit optimizes only the
classical head. Physical measurement choices remain discrete. The wrapper stores
resource diagnostics and uncertified convergence status for its selected candidate.
`MuTARegressor` uses squared error; `MuTAClassifier` uses logistic loss. Explicit
shot count and seed are required. Conditional postselection is not accepted as
a supervised sampling distribution.

The signed-X family is highly restricted: changing 0 to pi changes a measurement's
label, rather than supplying a continuously expressive logical gate. A discrete
search can exercise the physical pipeline but does not demonstrate the paper's
continuous trainability claims. At small shot counts its selections may reflect
noise. Reusing training seeds facilitates comparisons; validation must use fresh
trajectories and adequately refined resources.

`experiments/restricted_physical.py` records two seeded fits and predictions on the
same training inputs with independent validation trajectories. Those accuracies
are execution evidence, not held-out generalization estimates or advantage claims.
The saved evidence includes configuration, seeds, shots, search evaluations and
resource/convergence diagnostics.

At two shots per input the recorded fresh-trajectory accuracies were 0/4 and 4/4
for training seeds 7 and 19. This instability is direct evidence that this small
demonstration does not support a robust classifier-performance claim. The selected
patterns must be reevaluated with substantially more shots and independent data
before any such claim. Cutoff 40 was used for these runs after a separate 16-shot
study at cutoff 24 triggered the strict retained-norm guard.

## Continuous training (XY family)

`PhysicalMuTA(measurement_family="XY")` initialises continuous angles (scale 0.1 by
default) in the angle space [0, 2 pi). Three tools in `photographiqml.physical_training`
(also exported from `photographiqml`) optimise them on the physical simulator. All three
require an XY-family `PhysicalMuTA` and raise `ValueError` otherwise.

- `physical_loss(model, input_state, parameters, weights, *, shots=16, seed=0,
  output_basis="Z")` returns a `PhysicalLossEstimate(value, standard_error, per_shot,
  probabilities, shots, seed)`. `weights` maps output bit tuples to real weights (missing
  labels weigh zero), so the loss is sum_b w_b P(b) over the decoded output distribution.
  Each trajectory's conditional joint POVM row (from
  `diagnostics["per_shot_joint_probabilities"]`) gives one per-shot value; the standard
  error is their spread over sqrt(shots). At least two shots are required.
- `physical_gradient(model, input_state, parameters, name, weights, *, step=0.1, shots=16,
  seed=0)` returns `(gradient, standard_error)` from a central finite difference in one
  angle. Both points use the same seed (common random numbers), so the per-shot
  differences are paired and their spread is the gradient's standard error. It always
  uses `output_basis="Z"`.
- `ContinuousPhysicalSearch(shots=16, seed=0, maxiter=12, xatol=0.05).fit(model,
  input_state, parameters, name, weights, bounds=(0, 2 pi))` minimises the loss over one
  angle with the others fixed, using SciPy's bounded Brent method on the sampled objective
  with a fixed seed. It returns `ContinuousSearchResult(angle, loss, evaluations)`.

```python
import numpy as np
from photographiqml import (ContinuousPhysicalSearch, GKPPhysicalConfig, PhysicalMuTA,
                            physical_gradient, physical_loss)
config = GKPPhysicalConfig(cutoff=56, backend="native-fock", xy_protocol="auto",
                           max_dimension=200_000)
model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
names = list(model.parameters())
angles = dict(zip(names, (0.0, 0.7, 0.0, 0.0)))
weights = {(1,): 1.0}                       # loss = P(decoded output bit 1)
estimate = physical_loss(model, [1, 0], angles, weights, shots=2, seed=3)
gradient, se = physical_gradient(model, [1, 0], angles, names[1], weights, shots=2, seed=3)
found = ContinuousPhysicalSearch(shots=2, seed=3, maxiter=3).fit(
    model, [1, 0], angles, names[1], weights)
print(estimate.value, estimate.standard_error, gradient, se, found.angle)
```

The shot counts and `maxiter` above only keep the snippet short (it still takes a few
minutes); useful estimates need many more shots and search iterations.

**No parameter-shift rule is assumed.** The finite-energy instrument's dependence on the
angle is not the ideal qubit's (injection rounds, heralding, envelope distortion), so an
ideal-qubit shift rule has no justification here. The gradient is a finite difference of
the sampled physical loss and carries both sampling error (reported) and finite-difference
bias (not reported; it depends on `step`).

Common random numbers make nearby evaluations comparable but do not remove estimator
noise: the angle found by `ContinuousPhysicalSearch` is an optimum for that seed, shot
count and resource. Revalidate it at independent seeds. A sampled finite-energy loss is
never the ideal logical loss, and optimisation success is not evidence of a quantum
learning advantage. `MuTAClassifier` and `MuTARegressor` still accept only `DiscreteSearch`
for physical models, so they are not connected to these continuous tools. Experiment R66
(`paper_experiments_ml/14_gkp_xy/66_physical_muta_learning.py`) compares a physically
optimised angle with the logical optimum; see the
[final report](../../development/GKP_ARBITRARY_ANGLE_FINAL_REPORT.md).

## Preserved boundaries

Logical Adam/SGD/L-BFGS training and the Eq. 5 `MuTAKernel` remain ideal logical
features. Eq. 5 needs arbitrary angles and has no physical counterpart here.
`QuantumInstrumentModel` remains logical and explicitly rejects physical models:
a decoded distribution is insufficient to supply a quantum-output instrument.
Training a logical model first is valid; its physical validation must audit the
result and reject unsupported angles without modification. CVMuTA is a separate
future derivation and is not inferred from these finite GKP experiments.
