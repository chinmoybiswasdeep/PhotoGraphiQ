# Discrete physical training

`PhysicalMuTA` declares a categorical X family. Its initializer samples exact 0/pi
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

## Preserved boundaries

Logical Adam/SGD/L-BFGS training and the Eq. 5 `MuTAKernel` remain ideal logical
features. Eq. 5 needs arbitrary angles and has no physical counterpart here.
`QuantumInstrumentModel` remains logical and explicitly rejects physical models:
a decoded distribution is insufficient to supply a quantum-output instrument.
Training a logical model first is valid; its physical validation must audit the
result and reject unsupported angles without modification. CVMuTA is a separate
future derivation and is not inferred from these finite GKP experiments.
