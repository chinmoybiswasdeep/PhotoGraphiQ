# Theory and mathematical reference

These pages state the physics and mathematics that PhotoGraphiQ implements, in the
conventions the code uses, with short executable checks that tie each formula to the
implementation. Read [conventions](../getting-started/conventions.md) first.

## Conventions at a glance

| Quantity | Convention |
|---|---|
| Units | $\hbar=2$: $q=a+a^\dagger$, $p=-i(a-a^\dagger)$, $[q,p]=2i$ |
| Ordering | interleaved $(q_0,p_0,q_1,p_1,\dots)$ |
| Covariance | statistical, $V_{ij}=\tfrac12\langle\{\Delta R_i,\Delta R_j\}\rangle$; vacuum $V=I$ |
| CZ$(g)$ | $p_u\mapsto p_u+g\,q_v$, $p_v\mapsto p_v+g\,q_u$ |
| Homodyne at $\theta$ | measures $q\cos\theta+p\sin\theta$ (radians) |
| Resource squeezing $r>0$ | $p$-squeezed, $V=\mathrm{diag}(e^{2r},e^{-2r})$ |
| Fock truncation | total photon number $\sum_in_i<\text{cutoff}$ |

## Ideal versus finite

Two error sources must never be mixed:

- **Finite squeezing is physical.** It turns every measurement-based gate into a Gaussian
  channel: an intended map plus added noise, reported separately. The ideal
  infinite-squeezing relations (zero nullifiers, unitary teleportation, CV flow
  determinism) are limits, not states that can be simulated.
- **Fock truncation is numerical.** It is controlled by the cutoff and diagnosed by the
  retained norm, and observables must be converged with the cutoff.

## Pages

| Topic | Page |
|---|---|
| Quadratures, CCR, Gaussian states, covariance, symplectic maps, Gaussian channels, homodyne/heterodyne/general-dyne, displacement frame | [Gaussian states and channels](gaussian.md) |
| CV cluster states, weighted graphs, nullifiers, teleportation step, feed-forward, finite-squeezing noise | [CV cluster states](cluster-states.md) |
| Shear decomposition, two-mode gates, channel extraction, pattern rewriting | [Compilation](compilation.md) |
| Fock truncation, non-Gaussian gates and measurements, heralding | [Fock and non-Gaussian](non-gaussian.md) |
| Density matrices, loss, Kraus instruments | [Mixed-state conditioning](mixed-state-conditioning.md) |
| Cubic/Kerr/quartic synthesis and error budgets | [Nonlinear synthesis](nonlinear-synthesis.md) |
| Finite-energy GKP codewords, encoded measurement, decoding | [GKP](gkp.md), [GKP measurement](gkp-measurement.md), [GKP decoding](gkp-decoding.md) |
| Causal flow, gflow, Pauli flow, CV flow, determinism | [Flow and determinism](flow.md) |
| Measurement calculus, local Cliffords, ZX (qubit subset), Gaussian graphical calculus | [Calculus](calculus.md) |
| Permanents, hafnians, torontonians, BS/GBS sampling | [Boson sampling and GBS](sampling.md) |
| Fixed-branch gradients | [Gradient estimators](gradient-estimators.md) |

Practical usage is in the [user guide](../user-guide/backends.md); evidence for each formula's
implementation is in the [validation](../validation/index.md) section.
