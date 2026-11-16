# CV-native learning (`photographiqml.cv`)

`photographiqml.cv` is a third model family next to the ideal logical-qubit MuTA and the
restricted finite-energy GKP physical MuTA. It learns directly with continuous-variable
(CV) optical states. Classical data is encoded into Gaussian input states. Three things can
process those states: a trainable Gaussian program, its compiled finite-squeezing CV-MBQC
channel, or a directly trainable weighted cluster with trainable homodyne angles. An
experimental non-Gaussian program on the Fock engine is also available. The output is read
by homodyne statistics and passed to an explicit classical head.

The family does not change `MuTA`, `PhysicalMuTA` or any other existing model. `CVCluster`
is a directly trainable weighted grid cluster. It is not the CVMuTA proposal on the
roadmap, which remains planned.

## What is quantum and what is classical

| Stage | Quantum (physically executed) | Classical |
| --- | --- | --- |
| Encoding | Product Gaussian input state per sample (`CVFeatureMap`): displacement, squeezing or rotation | Choice of features and scale; any preprocessing |
| Processing | `GaussianProgram` (direct circuit or its compiled CV-MBQC pattern), `CVCluster` (weighted cluster, homodyne angles, CV-flow feedforward), `NonGaussianProgram` (Kerr, cubic phase; experimental) | Feedforward arithmetic (displacements proportional to outcomes) |
| Readout | Homodyne statistics: output means, q variances, sampled homodyne outcomes, Born probability of a binned homodyne outcome (`head="homodyne"`) | Sample means from S shots |
| Head and training | (none) | Affine, logistic or softmax head; loss; optimiser (`Trainer`: Adam, SGD, L-BFGS); gradients |

The simulator computes exact Gaussian moments with NumPy. The executable
`photographiq.Pattern` behind every Gaussian model is also available and is checked
against those moments. Gaussian models are efficiently classically simulable. No page
in this section claims a quantum advantage.

## Three program paths

| Path | Object | Execution | Trainable quantum parameters |
| --- | --- | --- | --- |
| A | `GaussianProgram`, `layered_program` | `mode="ideal"`: direct circuit (symplectic S, shift t). `mode="physical"`: unconditional channel of the pattern from `photographiq.compile_circuit` at resource squeezing r | Named gate arguments: rotation, squeezing (optionally bounded), shear, displacement, beam splitter, CZ weight |
| B | `CVCluster` | Exact unconditional channel of a w x c weighted grid cluster with CV-flow corrections; `mode="ideal"` drops the finite-squeezing noise | Homodyne angles (bounded or direct), optionally CZ rung weights |
| C | `NonGaussianProgram` (experimental) | Native Fock engine with a required cutoff; retained norm reported | Gate arguments including Kerr kappa and cubic gamma |

## Pages

- [Architecture](architecture.md): pipeline, modules, data flow.
- [Mathematical model](mathematical-model.md): conventions, maps, channels, readouts.
- [Gaussian layers](gaussian-layers.md): `GaussianProgram` and `layered_program`.
- [Feature encoding](feature-encoding.md): `CVFeatureMap`, `Feature`, re-uploading.
- [Direct MBQC (CVCluster)](direct-mbqc.md): trainable homodyne angles on a weighted cluster.
- [Compilation](compilation.md): ideal circuit versus compiled CV-MBQC channel.
- [Training](training.md): `CVRegressor`, `CVClassifier`, readouts, heads, shots, penalties.
- [Gradients](gradients.md): complex step, central differences, user callables.
- [Kernels](kernels.md): `CVKernel` overlap, normalized and fidelity kernels.
- [Finite squeezing](finite-squeezing.md): what finite resources change and what they do not.
- [Non-Gaussian](non-gaussian.md): Kerr and cubic-phase programs on the Fock engine.
- [Resource analysis](resource-analysis.md): `resource_report`, `pattern_resources`.
- [Validation](validation.md): tests and experiments R68 to R75 for each capability.
- [Examples](examples.md): eight runnable tutorials with their printed output.

## Minimal example

```python
import numpy as np
from photographiqml.cv import CVFeatureMap, CVRegressor, layered_program

X = np.random.default_rng(0).uniform(-1, 1, (40, 2))
y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2
model = CVRegressor(layered_program(2, 1), CVFeatureMap(2), readout="q", epochs=80)
model.fit(X, y)
print(model.score(X, y), model.resource_report(X)["measurements"])
```

## Scope and limitations

- Only Gaussian programs have exact moment execution, complex-step gradients, shot
  estimators and kernels. Non-Gaussian programs are experimental, use central
  differences, and need a Fock cutoff whose convergence you must check.
- With displacement encoding and a linear mean readout, a Gaussian model is affine in x.
  Nonlinear decision functions come from an identified source: classical preprocessing,
  input-dependent Gaussian gates, covariance observables, a nonlinear classical head, or a
  non-Gaussian gate ([training](training.md), R71).
- Loss, detector inefficiency and mode mismatch are not modelled. Finite squeezing of the
  resource modes is the only physical imperfection in the Gaussian channel.
- Inputs are product states, one feature per mode.
