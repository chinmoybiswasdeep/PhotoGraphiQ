# Mathematical reference (moved)

The derivations that used to live on this single page were split into topic pages under
[Theory](theory/index.md) and updated for version 0.3.1 (native C++ engines, mixed-Fock
backends, flows and calculus). Nothing was dropped; each former section now lives here:

| Former section | New location |
|---|---|
| Canonical coordinates | [Gaussian states and channels](theory/gaussian.md#quadratures-and-canonical-commutation-relations) |
| Canonical weighted graph states | [CV cluster states](theory/cluster-states.md#weighted-graphs-and-canonical-cv-cluster-states) |
| Destructive Gaussian measurements | [Gaussian measurements](theory/gaussian.md#gaussian-measurements-homodyne-heterodyne-general-dyne) |
| One-step teleportation and finite-resource noise | [One teleportation step](theory/cluster-states.md#one-teleportation-step) |
| Gaussian compiler | [Compilation](theory/compilation.md) |
| CV-flow versus causal scheduling | [Flow and determinism](theory/flow.md) |
| Displacement frame and safe rewriting | [Displacement frame](theory/gaussian.md#displacement-frame), [pattern rewriting](theory/compilation.md#pattern-rewriting) |
| Fock truncation and observables | [Fock truncation](theory/non-gaussian.md#fock-truncation), [Gaussian observables](theory/gaussian.md#gaussian-states-and-the-covariance-matrix) |

Two statements of the old page are no longer true and were corrected in the new pages:
pure Fock vectors are no longer "evolved by Piquasso" (the default Fock engines are the native
C++ `native-fock` and `native-mixed-fock`; Piquasso is optional), and noisy Fock homodyne,
mixed (density-matrix) inputs and finite-energy GKP states are now implemented (mixed-Fock
backends and the GKP modules).
