# Changelog

## 0.2.0 — experimental non-Gaussian MBQC

- Add sparse correlated Fock resources and public native pure-state import.
- Add capability preflight and explicit resource descriptions for cats and finite
  cubic-phase states, preserving CP(gamma)=exp(i gamma q³/6) at hbar=2.
- Add native Kerr/quadratic gates and mathematical photon addition/subtraction
  with square-root occupation factors and explicit normalization diagnostics.
- Add conditional ideal Fock homodyne, explicit postselection, probability/density
  records, and reverse exponentiation for count-dependent expressions.
- Add physical vacuum-tap subtraction, general/cat resource filters and finite
  cubic injection with nonlinear Gaussian feedforward.
- Add occupation-aligned fidelity/trace distance, quadrature moments, Wigner
  marginals, finite-grid negativity and branch-aware cutoff studies.
- Add norm/boundary warnings and Fock-dimension allocation guards.
- Preserve Gaussian compiler, flow and channel-analysis behavior; add quadratic
  phase execution to Gaussian backends. The affine channel analyzer still rejects
  this newly introduced command until its separate analysis rule is implemented.
- Extend analytical, independent Fock, native Piquasso and Graphix structural
  validation; update tutorials, experiments and the Quantum-style manuscript.

Fock loss/noisy measurements, mixed-state inputs, dynamic representation switching,
GKP decoding and universal non-Gaussian compilation remain unsupported.
The revised PDF is `paper/PhotoGraphiQ-v0.2.pdf`; `paper/PhotoGraphiQ.pdf` retains
the prior version because the open PDF viewer locked it during the update.
