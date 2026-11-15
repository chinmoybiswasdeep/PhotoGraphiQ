# Models, backends and validation

This page summarises what PhotoGraphiQML computes, which backend executes it, and how
each layer is validated in the unified PhotoGraphiQ repository. The detailed pages are
linked from each section.

## Two layers

| Layer | Object | What it is | Reference used to validate it |
| --- | --- | --- | --- |
| Ideal logical | `MuTA`, `TriangleNeuron`, `MuTAKernel`, `QuantumInstrumentModel`, `MuTAClassifier`, `MuTARegressor` | Qubit measurement-based model: open graph, flow, adaptive XY measurements, exact branch contraction | Analytic identities (Euler rotations, exp(i phi XX/2)), independent contraction code, MentPy 0.1.0a15 |
| Restricted physical | `PhysicalMuTA`, `GKPBridge.run`, `lower_muta_to_gkp` | Finite-energy GKP modes in a truncated Fock space; signed-X measurements (angle 0 or pi only), homodyne + nearest-cell decoding, classical Pauli frame | Independently assembled PhotoGraphiQ patterns, raw Piquasso, a second Fock engine (`native-fock` vs `piquasso-fock`), the logical layer for decoded statistics |

MentPy validates the logical layer only. It is never an oracle for GKP or Fock results.

## Logical MuTA models

A MuTA is a stack of triangle neurons on `n_wires` wires. Each measured vertex carries a
trainable XY angle; flow corrections make the output independent of the measurement
branch, so `MuTA.run` returns one deterministic logical state. `MuTA.unitary` gives the
implemented two-level-per-wire unitary. Parameter names, ordering, freezing and
persistence are listed in the [input/output contract](api.md). The paper mapping and
its limits are in [MuTA mapping](research/muta-mapping.md).

## Kernels, training and gradients

- **Quantum kernel.** `MuTAKernel` uses the paper's fixed two-feature measurement map
  (Eq. 5) and returns the fidelity Gram matrix. It is logical only; there is no physical
  MuTA kernel. Gram matrices are symmetric and positive semidefinite to roundoff.
- **Training.** `Trainer` runs Adam, SGD or L-BFGS on a deterministic scalar objective.
  `MuTAClassifier` (logistic head) and `MuTARegressor` (affine head) read the first-wire
  Z expectation. They are verified on simple tasks; a composite two-feature boundary did
  not train reliably (manuscript-suite issue 3).
- **Gradients.** The default gradient is a central finite difference of the logical
  objective. It agrees with the analytic parameter-shift rule to about 1e-10 in the
  manuscript suite. Stochastic score-function gradients and automatic differentiation of
  the physical path are not implemented.
- **Physical training.** Only a discrete search over the 0/pi family exists
  ([physical training](physical/physical-training.md)). It is a pipeline exercise, not a
  demonstration of continuous trainability.

## Physical GKP lowering and restricted execution

`lower_muta_to_gkp` audits every bound angle before any Fock allocation, then builds a
public PhotoGraphiQ `Pattern`: finite GKP |+> preparations, one unit-gain CZ per graph
edge, a p-homodyne with nearest-cell decoding per measured vertex, a classical
signal carrying the interpreted bit, and the outputs. See [lowering](physical/lowering.md),
[supported measurements](physical/supported-measurements.md) and
[Pauli frames](physical/pauli-frames.md).

Finite GKP codewords are not orthogonal. The physical instrument is therefore not the
logical unitary, even with exact numerics. The gap between decoded statistics and the
logical target is a property of the finite resource (peak width, envelope, peaks), not an
engine or truncation error. It shrinks as the resource narrows.

## Backend selection

`GKPPhysicalConfig(backend=...)` accepts two engines:

| Backend | Engine | When to use |
| --- | --- | --- |
| `piquasso-fock` (default) | Piquasso 8.0.1 Fock simulator; CZ applied as Euler factors, truncated after each | Reproducing recorded manuscript evidence; independent reference |
| `native-fock` | PhotoGraphiQ's compiled C++ Fock engine (`photographiq._native`, component `fock`); CZ applied as the exact projected operator P U P | Faster execution, especially for two or more wires; requires the compiled extension |

`native-mixed-fock` and other backends are rejected before allocation. Both engines
share the Python preparation, homodyne readout, renormalisation, decoding and frame code,
so decoded bits, sampled analog outcomes and sampled output bits are identical for the
same seed. The engines differ only by CZ truncation; the difference is of the size of each
engine's own truncation error and vanishes as the cutoff grows. Measured values are in
[physical architecture](physical/architecture.md) and in core manuscript experiment R59
(`paper_experiments/20_photographiqml/59_photographiqml_physical.py`).

Measured in core experiment R59 (2026-11-13; timings provisional, the machine was shared):

| Setting | max \|ΔP\| decoded, native vs Piquasso | 1 − F final state | Each engine vs native at a high cutoff |
| --- | --- | --- | --- |
| wide GKP (width/envelope 0.9, 4 peaks), cutoff 32, 1 and 2 wires | 1.2e-6 to 9.0e-5 | ≤ 1.5e-7 | 2 wires, native: 1.1e-4 (cutoff 32 vs 48) |
| default GKP (0.6, 6 peaks), 1 wire, cutoff 32 / 40 / 48 | 4.6e-4 / 3.3e-4 / 9.0e-5 | 1.3e-5 / 1.5e-6 / 1.8e-7 | vs native cutoff 64: native 6.7e-5 / 6.7e-6 / 1.6e-5; Piquasso 2.7e-4 / 7.8e-5 / 3.3e-5 |

For the default resource, Piquasso's per-factor CZ truncation dominates the difference.
Use a cutoff of about 48 when the two engines must agree to 1e-4. Decoded bits, sampled
output bits and empirical frequencies were identical in every case. With 8 shots,
Rao-Blackwell probabilities differed by 1.7e-4. Per conditional run, a 1-wire model took 3
to 12 s on native-fock and 4 to 19 s on piquasso-fock; a 2-wire model at Hilbert dimension
5984 took 5.3 s on native-fock and 241 s on piquasso-fock. The manuscript scripts PGML-R38, R41, R43
and R45 were also rerun with `backend="native-fock"` (scratch copies). They passed their
own checks: the native Pattern comparison error is 0.0. Their decoded quantities differ from
the recorded `piquasso-fock` values by at most 4.6e-5 (R38/R41), 9.5e-5 (R43 observable
difference) and 1.2e-4 (R45 pair correlation).

Near PhotoGraphiQ's strict retained-norm guard one engine can stop with
"increase cutoff" while the other continues. Raise the cutoff until neither is near the
guard.

## Measurement outcomes

- `physical-conditional` conditions on one explicit real homodyne outcome per measured
  vertex. It is a postselected branch with a conditional density, not an event with a
  finite probability.
- `physical-shots` samples complete trajectories with seeded child streams and then
  samples output bits from each trajectory's joint POVM.
- Decoded results keep raw outcome, cell, residual, decoded bit and frame correction.
  The nearest-cell decoder reports no confidence or posterior.

See [execution modes](physical/execution-modes.md) and [decoding](physical/decoding.md).

## Statistical uncertainty

`decoded_joint_probabilities` is the Rao-Blackwell mean of the conditional joint POVM
probabilities over trajectories, with standard error = sample standard deviation /
sqrt(shots). `empirical_probabilities` are sampled-bit frequencies with plug-in
binomial errors. Small-sample plug-in errors can be zero; neither is a confidence
interval and neither certifies resource accuracy.

## Physical versus logical validation

| Question | Check | Where |
| --- | --- | --- |
| Is the logical model right? | Analytic identities, independent contraction, MentPy | PGML-R1 to R13, `tests/ml/mentpy_reference` |
| Does lowering preserve the open graph? | Vertex, edge, input/output and dimension counts | PGML-R35, R36 |
| Are Pauli frames right? | All logical branches contracted independently | PGML-R37 |
| Does the physical pattern run as intended? | Independently assembled public Pattern; raw Piquasso state preparation and CZ | PGML-R38 to R41 |
| Do two Fock engines agree? | `native-fock` vs `piquasso-fock`: decoded probabilities, final states, records, shots | Core R59, `tests/ml/cross_layer/test_native_backend.py` |
| Is the result converged? | Cutoff, grid, peaks (numerical) separately from width and envelope (physical) | PGML-R47, R56, R58, core R59 |
| How far is physics from logic? | Decoded TV distance and observable differences; no decoded-state fidelity is reported | PGML-R43, R56, core R59 |

`convergence.certified` is always `False`: a sweep is evidence, not a certificate.

## Limitations

- Physical measurements are signed X only (angles 0 or pi). Arbitrary-angle physical
  MuTA, physical kernels and continuous physical training are not supported.
- No loss, detector inefficiency or soft adaptive decoding in the physical path.
- Entangled logical inputs are rejected; product inputs only.
- No decoded-state fidelity is computed; only decoded outcome statistics are compared.
- Nothing here is a fault-tolerance, threshold, universality or quantum-advantage claim.
- Timings in the manuscript suites are single-machine observations.
