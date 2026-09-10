# Changelog

### v0.3 scientific hardening

- Correct JAX beamsplitter boundary ordering and package phase convention.
- Reject unresolved JAX tensor/addition normalization; expose retained masses
  and boundary diagnostics, with explicit eager/traced invalid-branch behavior.
- Canonicalize mixed-state occupation integers for Windows/Numba preparation.
- Correct GKP floating-point half-cell tie parity.
- Preserve nonlinear approximation metadata and warn during Kerr compilation.
- Add independent channel, GKP, gradient and synthesis tests and measured evidence.
- Restore missing tutorial transcripts to clean documentation builds; withhold
  the coverage badge until Codecov receives a verified main-branch upload.

Changes follow semantic versioning. Historical entries below are preserved.

## Unreleased — 0.3.0 development

### Added

- MkDocs Material website, source API reference, input/output guides, glossary,
  24 executable tutorials, 10 demo projects and five curated notebooks.
- Circuit drawings, richer MBQC diagrams and compilation provenance with SVG,
  PNG and PDF export.
- Experimental mixed-Fock density matrices, thermal attenuation and conditional
  counting/homodyne; calibrated inefficient/noisy detection.
- Cubic-resource compilation and approximate quartic/Kerr synthesis over a
  Gaussian-plus-cubic target gate set, with explicit refinement controls.
- Finite-energy square-lattice GKP resources, stabilizer diagnostics and elementary
  modular syndrome correction infrastructure.
- Optional JAX finite-Fock execution, fixed-branch gradients, Gaussian sampling
  reparameterization and a discrete likelihood-ratio estimator primitive.
- Documentation CI/Pages configuration, Codecov uploads and community templates.

### Changed

- Source installation paths, metadata, contribution guidance and result explanations.
- Cutoff studies may select the mixed-Fock backend.
- New research paths remain experimental; no publication or remote deployment is
  implied by this development version. Existing v0.2 evidence remains historical.

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


## v0.2 validation hardening

- Retain the original failing injection cases while separating finite-Fock,
  Gaussian-decomposition and independent quadrature error budgets.
- Reject invalid/insufficient resources before earlier native preparations and
  prevent slightly mixed Gaussian inputs from being silently treated as pure.
- Add optional raw quadrature moments through order four and photon moments;
  avoid dense allocation for pure norms and pure/pure state metrics.
- Check partial homodyne CDF integration errors; expand analytical density,
  Wigner, mixed-state, sensitivity, heralding and label-aware Graphix validation.
- Run every release gate on Python 3.11?3.14, retain per-version artifacts and
  disable fail-fast cancellations. See docs/release-hardening-report.md for
  actual outcomes; a configured workflow alone is not a release certification.
