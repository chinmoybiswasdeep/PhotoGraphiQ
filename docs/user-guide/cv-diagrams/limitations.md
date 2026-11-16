# Limitations

This page lists what `photographiq.calculus.cv_diagrams` does **not** do. The theory note
keeps the authoritative list in
[§6, "Not supported"](../../theory/CV_ZX_FOUNDATIONS.md).

## Scope of the calculus

- **Not a complete CV-ZX calculus.** Completeness is neither claimed nor tested. Only the
  rules in the [rule table](rewrite-rules.md#rule-table) exist.
- **No claims of universality, fault tolerance or quantum advantage.**
- **No relational (infinite-squeezing) semantics.** The complete equational theories of
  Booth, Carette and Comfort (arXiv:2403.10479, arXiv:2401.07914) are cited, not implemented.
- **Equivalence is judged at the density-operator level.** Global and branch phases are
  recorded in the trace when known (`displacement-fusion`, `identity-removal` of a constant
  phase polynomial), and are otherwise noted as unrecorded (`symplectic-fusion`).

## Unsupported generators

| Not available | Comment |
|---|---|
| Multi-leg q- or p-spiders, Fourier boxes (Nagayoshi et al.) | Only the 1 → 1 q-spider `PhasePolynomial` exists |
| p-spiders exp(i f(p)) | Conjugate a `PhasePolynomial` by quarter rotations |
| Adaptive (outcome-dependent) homodyne angles | `from_pattern` rejects them |
| Nonlinear or vector-record feed-forward | `Feedforward` is exactly affine in scalar records |
| Photon addition or subtraction commands | Use the heralded diagram (beam splitter, `PhotonCount`, `Postselect`) |
| Threshold detectors, heterodyne, general POVMs | Not generators |
| Fock superposition preparations | `FockPrepare` is a single number state; use a `Pattern` input for others |
| Mixed GKP preparations, GKP logical gates | Only `GKPPrepare` (finite-energy codeword) and `ModularDecode` (nearest-cell parity) |

## Unsupported rewrites

- Nagayoshi's bialgebra, copy, antipode, Fourier, squeezing, quadratic and inversion rules.
- Exact two-node wire shortening on finite-squeezing clusters. It holds only in the ideal
  limit (R62, see [GaussianGraph integration](gaussian-graph-integration.md)). The exact
  replacement is `teleport-chain-resynthesis`, and it is **logical** only.
- Measurement-reordering rules. `commute` never changes the measurement depth. Depth
  reductions come from rebuilding patterns from a layered flow certificate, outside the rule
  set.
- Rewrites that change heralding probabilities. Heralded diagrams are evaluated, not rewritten.
- Rules for Kerr, photon counting, post-selection or GKP boxes.
- Lowering or optimising Kerr gates in MBQC, and general non-Gaussian MBQC optimisation.

## Semantics

- **Gaussian semantics** (`evaluate`, `compare`, `resource_cost["added_noise"]`) need a
  Gaussian diagram. Non-Gaussian boxes raise `NotImplementedError`.
- **Fock semantics** (`run_fock`) is truncated. Finite-cutoff agreement is evidence, never a
  proof. `run_fock` does not support `PrepareState`, multi-mode `Symplectic` or
  `GaussianChannelBox`.
- **Approximate optimisation** bounds max\|ΔY\| entrywise, not a diamond norm or any other
  channel distance.
- **Costs** have different units. `weighted_cost` weights are user choices. Use
  `pareto_front` to avoid choosing them.
- **Timings** in R66 are provisional (one machine).

## Translation gaps

- `to_pattern`: no `Discard`, multi-mode `Symplectic`, `GaussianChannelBox`, `PrepareState`,
  `Postselect`, `GKPPrepare`, `ModularDecode`, or phase polynomials above degree 3. Each
  `ClassicalSource` costs one optical mode in the recompiled pattern.
- `to_circuit`: wires must be 0..n−1 in order, and the diagram may contain only gates (no
  preparations, measurements, feed-forward or channels).
- `pattern_flow_report`: only CZ graphs with squeezed preparations and homodynes at
  sin θ ≠ 0. Any other box gives `"unsupported"`.
- `CVDiagram.from_dict` cannot rebuild `GKPPrepare`.

## Resolved implementation issues

A documentation review found five issues. All were fixed, and regression tests cover them
(`tests/calculus/test_cv_diagrams_optimize.py::test_review_fixes_gaussian_phase_polynomial_sources_and_strategy`):

- `PhasePolynomial` of degree ≤ 2 is now a Gaussian *unitary* box. `evaluate`, `compare`,
  `resource_cost` and the fusion rules accept it.
- `to_pattern` raises `NotImplementedError` for a zero-variance `ClassicalSource`. Such a
  source is a deterministic record, and an ancilla realising it would need infinite squeezing.
- `run_fock` draws all unfixed `ClassicalSource` records from one seeded stream, so they are
  independent.
- `optimize_cv_pattern` rejects any `strategy` other than `"greedy"` or `"beam"`.
- `draw` no longer has the unused `max_columns` argument.
- `FockRun.probability` is the product of the probabilities of *all* discrete records in the
  run. It is the heralding probability when every count is fixed by `outcomes`; the docstring
  now says this.
