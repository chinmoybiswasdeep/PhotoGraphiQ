# CV diagrammatic calculus and CV-MBQC optimisation: final implementation report

## Summary

`photographiq.calculus.cv_diagrams` adds a **typed Gaussian process-diagram calculus** to
PhotoGraphiQ. It has three parts:

1. **Hybrid classical–quantum Gaussian semantics.** Each diagram has a unitary, an
   outcome-averaged channel and a full conditional instrument.
2. **Rewrites with proofs.** Fourteen rewrite rules, each proved and labelled with its
   equivalence class.
3. **CV-MBQC tooling.** Translation to and from executable `Pattern`s, a resource-cost model,
   CV-flow certification, and a verified, deterministic optimiser in *physical*, *logical* and
   *approximate* modes.

A restricted non-Gaussian layer adopts the 1 → 1 q-spider fragment of Nagayoshi et al. (Phys.
Rev. Research 7, 033141 (2025)), together with its fusion and displacement rules as operator
identities. This layer also executes Kerr, Fock, photon-counting, post-selection and
finite-energy GKP diagrams on the native Fock engines.

**Not claimed:**

- a complete or universal CV-ZX calculus;
- universal non-Gaussian optimisation;
- fault tolerance or quantum advantage.

**Headline evidence** (experiments R61–R68):

| Experiment | Result |
|---|---|
| R61 | Every Gaussian rule is exact against independent references (worst residual 8.8e-16). |
| R63 | Logical-mode optimisation of 54 random compiled circuits: resource modes 3537 → 2819, measurement depth 2163 → 1731. The logical map is exact, flow is certified and execution is verified. |
| R63, R65 | The finite-squeezing noise is reported, and it *rose* in 3 of 54 and 1 of 24 programs respectively. |
| R64 | Exact feed-forward merging (390 → 120 commands). Flow-layered scheduling cuts depth (e.g. 40 → 20) with the conditional instrument preserved. |
| R67 | End to end, 38 → 18 resource modes. Both patterns are verified by 2000 executed trajectories each. |
| R68 | Non-Gaussian rules converge in Fock against a dense `expm` reference while a false rule does not. The GKP readout and SUM-syndrome diagrams agree with untruncated references. |

## 1. Initial revision and working-tree status

- **Branch and revision:** branch `upgraded`, HEAD `c8d053a8c388d075086a7b642d52eddf3b318568`.
- **Pre-existing state:** the working tree already held uncommitted changes from the previous
  (CV-native ML) task; they are listed in the audit. Nothing was staged, committed, pushed,
  reset or switched, and the earlier changes were preserved.
- **End state** (read-only checks after all runs): still on `upgraded`, HEAD unchanged, and
  `git diff --cached` is empty.
  - This task modified three tracked files: `README.md`, `docs/COMPLETE_FUNCTIONALITY.md` and
    `mkdocs.yml`. The other modified tracked files belong to the previous task.
  - All new files (§21) and the R61–R68 results and figures are untracked.

## 2. Existing functionality reused

| Reused | Role here |
|---|---|
| `GaussianGraph` (MFvL) | Interoperability and measurement-rule references |
| `calculus.cv.affine_form` | Feed-forward translation |
| `analysis.gaussian_channel` | Independent channel reference |
| Gaussian backends | Conditional and trajectory references |
| `flows.find_cv_flow`, `flows.cv.corrected_pattern`, `cvflow.flow_pattern` | Flow certificates and layered schedules |
| `compiler.compile_circuit`, `decompose_symplectic` | Compilation and resynthesis fallback |
| Native Fock engines | Non-Gaussian and GKP semantics |
| `GKPCode` and `multimode_readout` | GKP preparation and existing readout |
| `tests/validation/gkp_xy_reference.py`, `tests/validation/cv_fock_reference.py` | Untruncated grid and dense Fock references |
| `paper_experiments/common.py` | Results and figures |

The audit is in [CV_ZX_IMPLEMENTATION_AUDIT.md](CV_ZX_IMPLEMENTATION_AUDIT.md).

## 3. Formalisms implemented

A typed process calculus with:

- linear bosonic wires and copyable real-record wires;
- hybrid Gaussian semantics (theory note §3);
- the restricted Nagayoshi q-spider fragment.

The ħ = 1 → 2 conversion and the rotation-sign difference are documented. Theory, literature
and proofs are in
[docs/theory/CV_ZX_FOUNDATIONS.md](../theory/CV_ZX_FOUNDATIONS.md).

## 4. Generators

**Gaussian:**

- `Prepare` (with `.squeezed` and `.vacuum`), `PrepareState`
- `Rotation`, `Squeezer`, `Shear`, `Displacement`, `BeamSplitter`, `CZ`, `Symplectic`
- `GaussianLoss`, `GaussianChannelBox` (complete positivity checked)
- `Homodyne` (efficiency, added noise), `Feedforward` (exact affine in records)
- `ClassicalSource`, `Discard`

**Non-Gaussian (Fock semantics):**

- `PhasePolynomial` (exp(i f(q))), `Kerr`
- `FockPrepare`, `PhotonCount`, `Postselect`
- `GKPPrepare`, `ModularDecode`

Every generator declares its ports. Parameters are validated. All generators serialise except
`GKPPrepare`, which holds a code object.

## 5. Rewrite-rule inventory

| Rule | Class | Source / proof |
|---|---|---|
| rotation-, squeezing-, shear-fusion | exact instrument | one-parameter subgroups (§4.1–4.3) |
| displacement-fusion | exact instrument + recorded phase | Weyl relation, ħ = 2 (§4.4); phase checked against dense `expm` |
| identity-removal | exact instrument | §4.5 (constant phase recorded) |
| commute | exact instrument | disjoint tensor factors (§4.7) |
| symplectic-fusion | exact instrument | Gaussian unitary composition (§4.6) |
| channel-fusion | exact channel | Gaussian channel composition; complete positivity preserved (§4.8) |
| feedforward-merge | exact instrument (branch phase) | §4.11 |
| cz-q-measurement-to-feedforward | exact instrument (ideal detector only) | MFvL vertex deletion, process form (§4.9) |
| prepare-measure-to-source | exact instrument | homodyne record law (§4.10) |
| phase-polynomial-fusion | exact (operator identity) | Nagayoshi et al. Eq. (70), 1 → 1 (§4.12) |
| displacement-through-phase-polynomial | exact (operator identity) | Nagayoshi et al. Eq. (79), 1 → 1 (§4.13) |
| teleport-chain-resynthesis | **logical** | exact wire channel and a verified re-decomposition (§4.14) |

Flow-layered scheduling (R64) is a pattern transformation, not a box rule. Its equivalence is
measured, not assumed.

## 6. Semantic contracts

**Diagram contracts:**

- Diagrams are immutable, and every rewrite returns a new diagram with a `RewriteStep` trace.
  The trace records rule, affected boxes, boxes before and after, equivalence, assumptions,
  source and scalar.
- `evaluate` raises on non-Gaussian boxes.
- `to_gaussian_graph` raises on mixed states.
- Translation raises on adaptive angles, non-affine feed-forward, vector-record feed-forward
  and unsupported commands.

**Optimiser contracts:**

- `optimize_cv_pattern` compares every candidate with the *original*, so errors cannot drift
  across steps. In executable mode each candidate must also recompile.
- `compare` reports logical, physical and conditional equivalence separately, each with
  residuals and a stated tolerance.

## 7. Gaussian process verification (R61; tests)

**R61.** Ten rules, 40 random instances each. Each rewritten diagram's semantics is compared
with an independent reference built from the *original* diagram:

| Check | Reference | Worst residual |
|---|---|---|
| Averaged channel | `gaussian_channel` of its Pattern | 8.8e-16 |
| Conditional state | postselected Gaussian-backend trajectories | 8.8e-16 |
| Record density | postselected Gaussian-backend trajectories | 8.8e-16 |
| Displacement-fusion phase | dense Fock `expm` (cutoff 120) | ~1e-16 |

**Tests (`tests/calculus/test_cv_diagrams.py`):**

- generators against `photographiq.gaussian`;
- associativity, identity, tensor products and permutations;
- the channel with loss and an inefficient homodyne against `gaussian_channel`;
- the conditional state and density against postselection;
- random exact rewrites, checked by instrument residuals.

## 8. Conditional measurement verification

**Agreement with trajectories.** Conditional semantics agree with postselected trajectories,
in both the state and the density (tests; R61). Executed trajectories agree with the averaged
channel for both compiled and optimised patterns (R67: max |z| = 1.16 and 2.00 over 2000
shots each).

**GaussianGraph cross-check.** The q and p homodyne graph rules of `GaussianGraph` match
diagram conditioning to 2.4e-12 (R62; tests).

**Two-node wire shortening.** On a 5-node path, p-measuring two adjacent interior nodes gives
an outer edge of −0.5, −0.953, −0.9975 and −0.9999 at r = 0, 0.75, 1.5 and 2.25: an exact
real CZ weight of −1 only in the ideal limit. A single interior p measurement instead gives
the imaginary entry e^{2r} (R62).

## 9. Circuit and pattern translation coverage

**Supported:**

- every Gaussian command of `Pattern` with constant parameters: Prepare (squeezed or
  GaussianInput), Entangle, Rotate, Squeeze, QuadraticPhase, BeamSplitter, Loss, Displace
  (affine in records), Measure (Homodyne, PhotonNumber);
- Kerr and CubicPhase;
- the `Circuit` gates: symplectic (R/S/general), displace, cz, beamsplitter, identity wire,
  cubic phase, kerr.

**Rejected:**

- adaptive angles;
- non-affine or vector feed-forward;
- photon addition and subtraction commands (heralded subtraction is expressed as a diagram);
- Fock superposition preparations other than number states;
- multi-mode `Symplectic` in `to_pattern`.

**Round trips.** `from_pattern` → `to_pattern` reproduces `gaussian_channel` to 1e-13. The
compiled logical map equals the ideal circuit diagram to 1e-12.

## 10. MBQC optimisation results

**Physical mode.** No exact rule applies to `compile_circuit` patterns, so physical mode
changes none of them. This is a negative result, reported in R63 and in the tests. On
flow-built clusters, physical mode merges feed-forward commands exactly: 390 → 120 across 15
grids, with record dependencies unchanged (R64).

**Logical mode** (54 random 1–3-mode circuits, R63):

| Quantity | Compiled | Optimised |
|---|---|---|
| Resource modes | 3537 | 2819 |
| Measurement depth | 2163 | 1731 |

- The logical residual is at most 3.4e-14, and the residual of the executed recompiled
  pattern is at most 3.0e-14.
- CV flow is certified for all 47 patterns checked.
- The largest gains are on single-mode chains: R62 shortened 890 steps to 180 over 60
  single-mode programs. Beam-splitter-heavy programs gain least, because their sum gadgets
  interrupt the chains.

**End to end (R67).** Compiled R, S, R, CZ, S, R, D, R:

| Pattern | Resource modes | Measurement depth | Added noise | Executed |
|---|---|---|---|---|
| Compiled | 38 | 26 | 1.49 | yes |
| Optimised | 18 | 15 | 0.82 | yes |

Both are certified and both are executed. Each matches its own channel, and the mean
predictions agree to 1e-9.

## 11. Physical versus ideal equivalence

**Logical rewrites change the noise.** Logical rewrites preserve (X, d) exactly at every
squeezing, but they change Y. Both versions scale exactly as e^{−2r}, so the noise ratio is
independent of r to 1e-9 (R65). The noise ratio is below 1 in most cases but not all:

| Experiment | Median ratio | Range | Programs where noise increased |
|---|---|---|---|
| R63 | 0.62 | 0 to 6.3 | 3 of 54 |
| R65 | 0.66 | max 6.7 | — |
| R62 | — | — | 4 of 60 |

In R65 the fewest-mode variant was not the quietest in 1 of 24 programs, so the Pareto front
over chain subsets is reported. Exact ancilla elimination (CZ + q homodyne, then
preparation + homodyne) removes a mode with the instrument unchanged (R65 control).

**Approximate mode** accepts logical rewrites only within an entrywise max|ΔY| budget. This is
not a diamond-norm bound.

## 12. CV-flow certification

`pattern_flow_report` reads the cluster from a diagram: CZ weights, squeezing, homodyne shears
and inputs/outputs. It then runs `find_cv_flow`. Original and optimised patterns were
certified in every case checked (R63, R67; tests). Frame shifts (constant displacements,
feed-forward and classical sources) are ignored because they do not alter the graph.

Flow certification is the dominant cost of verification. It takes 0.13–16 s for 9–72-node
clusters (R66; a concurrently loaded first run measured up to 45 s), whereas the semantic check
takes ≤ 0.016 s on 576 boxes. The exact rational
fallback makes clusters above about 80 nodes expensive.

## 13. Resource-cost comparisons

`resource_cost` records:

- resource modes, ancilla modes and classical sources;
- entangling edges, measurements and measurement depth;
- feed-forward commands, feed-forward dependencies and maximum gain;
- squeezing energy (Σ sinh² r, in photons) and optical energy (Σ mean photon number);
- added noise.

`weighted_cost` uses explicit weights; the units differ, so these are user choices, not a
universal cost. `pareto_front` avoids choosing weights. All comparisons in R62–R67 use the
same model.

## 14. Non-Gaussian capabilities (R68; tests)

**Phase-polynomial rules.** The phase-polynomial fusion and displacement-propagation rules
converge in Fock to a dense `expm` reference (cutoff 150). The original diagram reaches
5e-11 infidelity at cutoff 70. The fused diagram, which has a larger cubic coefficient,
converges more slowly: 2e-8 at cutoff 70 for input 1.5, down more than 10⁵ from cutoff 15.
The false-rule control, which commutes exp(i f(q)) with a q displacement, stays at
2e-2 to 4e-2 at every cutoff.

The first run had used a p displacement as the control. That one *commutes* with
exp(i f(q)), and the run correctly found it converging. The control was corrected and the
change disclosed.

**Heralded photon subtraction.** For α = 0.8 and θ = 0.3, the herald probability is
μe^{−μ} = 0.0529 to ≤ 1e-16, and the output mean photon number is α² cos² θ to 1e-16.

**Kerr.** Kerr and cubic-phase diagrams agree with the dense reference (tests). No rewrite
changes a heralding probability; heralded diagrams are evaluated, never rewritten.

**Finite-cutoff evidence.** Finite-cutoff agreement is used as evidence alongside the
analytic proofs, never as a proof.

## 15. GKP integration (R68; tests)

The diagram GKPPrepare (width = envelope = 0.6) → q homodyne → nearest-cell parity gives:

| Source | P(Z = 0) |
|---|---|
| Diagram | 0.78686 |
| Existing `multimode_readout` | 0.78684 |
| Untruncated grid model | 0.78675 |
| Ideal cos² 0.4 | 0.848 (gap = finite energy) |

The first run disagreed by 1.7e-3 because of a decoding-cell-unaligned outcome quadrature. The
cell-aligned midpoint grid fixed it; this is disclosed.

**Z-syndrome extraction.** SUM onto a |0~⟩ ancilla, then ancilla readout: the diagram at
cutoff 64 gives 0.72310 and an independent untruncated grid computation of the same protocol
gives 0.72310. The gap to the direct readout is the finite-energy SUM error at width 0.6.

**Scope.** No new GKP operations, logical rewrites or arbitrary-angle measurements were
introduced; the GKP diagrams only execute existing primitives.

## 16. Numerical convergence

| Quantity | Behaviour |
|---|---|
| Fock cutoff sweeps (R68) | Exponential convergence of true rules; false rule flat |
| Herald probability | Exact from cutoff 16 |
| GKP readout | Cutoff-independent between 40 and 48 (two-mode syndrome at 64) |
| Gaussian semantics | Exact linear algebra; residuals at machine precision |

## 17. Performance (provisional, R66; one Windows machine, not idle)

| Operation | Size | Time |
|---|---|---|
| Diagram semantics | up to 576 boxes | 0.7 ms → 16 ms |
| `gaussian_channel` | up to 576 boxes | 1.6 ms → 0.33 s |
| Optimiser | — | greedy ≤ 0.15 s, beam ≤ 0.23 s; 1–5 candidates on these programs (beam: up to 10) |
| `find_cv_flow` | 9 → 72 nodes | 0.13 s → 16 s |

- Beam search was never worse than greedy on these programs.
- The first evaluator implementation used dense O(n³) feed-forward updates. Rank-one updates
  replaced them, so a 796-box pattern evaluates in 0.03 s.

## 18. Reproducibility outputs

Each of R61–R68 writes:

- `paper_experiments/results/{csv,json}/R6x_cvzx_*`, where the JSON also carries
  `configuration`, `validation_report` and `source_revision`;
- a `.metadata` environment companion;
- `paper_experiments/figures/{pdf,png,svg}/R6x_cvzx_*` and a caption.

Seeds are fixed in every script.

## 19. Tests passed, failed and skipped

All counts are from local runs in this working tree (Python 3.12.14, Windows 11). Remote CI
was not run.

| Run | Result |
|---|---|
| `tests/calculus/test_cv_diagrams.py` + `test_cv_diagrams_optimize.py` (new) | 29 passed, 0 failed, 0 skipped |
| `tests/calculus` (all, including the existing ZX, qubit, Clifford, CV and GaussianGraph tests) | 303 passed |
| `tests/flow` + `tests/analytical` | 94 passed |
| `experiments/validate_release.py` | **All release gates passed** |
| └ pytest, whole suite | 1815 passed, 13 skipped, 0 failed (18 min) |
| └ coverage | 92.82% (floor 90%) |
| └ ruff check, ruff format --check, mypy, build, physics-evidence | pass |
| `scripts/check_public_api.py` | pass |
| `mkdocs build --strict` | exit 0 |
| Tutorials `examples/cv_diagrams/01…09` | all pass their asserts |
| Experiments R61–R68 | all pass their declared criteria (revisions disclosed) |

**Test added after the gate run.** One translation round-trip test was added after the release
gate (making 29 new tests). It raised the coverage of `translate.py` from 54.7% to 87%. In the
targeted run, the new modules are covered as follows:

| Module | Coverage |
|---|---|
| `semantics` | 95% |
| `ir` | 94% |
| `optimize` | 93% |
| `rules` | 93% |
| `translate` | 87% |
| `visualization` | 84% |
| `fock` | 83% |

**Skips.** None of the 13 skips are in the new code:

- MentPy (2);
- Playwright (1);
- CUDA-Q, Linux only (1);
- cross-backend capability skips (7);
- Piquasso cross-Kerr (2).

**Python versions.** Python 3.11, 3.13 and 3.14 were not available, so they were not tested.

## 20. Known limitations

- **Completeness:** not a complete CV-ZX calculus. Only the listed rules are implemented. Most
  Nagayoshi rules (multi-leg spiders, Fourier, bialgebra, copy, antipode, quadratic,
  inversion) and the relational semantics of Booth–Carette–Comfort are not implemented.
- **Physical mode:** cannot simplify `compile_circuit` patterns, because no exact rule applies.
- **Logical rewrites:** can raise the finite-squeezing noise. Use the Pareto front or the
  approximate mode budget.
- **ClassicalSource:** recompiles to an ancilla, because `Pattern` has no classical-randomness
  command.
- **Flow certification:** dominates verification cost and is slow above about 80 nodes.
- **Non-Gaussian:** no Kerr lowering, no non-Gaussian MBQC optimisation and no
  heralding-changing rewrites. Fock semantics are truncated.
- **Approximate mode:** the noise budget is not a diamond-norm bound.
- **Timings:** provisional.

**Post-implementation review.** Writing and running the user documentation exposed five
issues. All are fixed, with a regression test, and none affects R61–R68:

1. A degree ≤ 2 `PhasePolynomial` lacked Gaussian semantics. It is now a unitary box.
2. A zero-variance `ClassicalSource` divided by zero in `to_pattern`. It now raises a clear
   error.
3. Seeded `run_fock` re-seeded each `ClassicalSource`, which correlated the sources. One
   stream is now used.
4. `strategy` was unvalidated, and the `draw` argument `max_columns` was ignored. Both are
   fixed.
5. The `FockRun.probability` docstring was wrong. It is corrected.

## 21. Changed files

**New:**

- `src/photographiq/calculus/cv_diagrams/`: `__init__`, `ir`, `semantics`, `rules`,
  `translate`, `optimize`, `fock`, `visualization`
- `tests/calculus/test_cv_diagrams.py`, `tests/calculus/test_cv_diagrams_optimize.py`
- `paper_experiments/22_cv_diagrams/` (README, `cvzx_support.py`, R61–R68) and their results
  and figures
- `docs/theory/CV_ZX_FOUNDATIONS.md`
- `docs/development/CV_ZX_IMPLEMENTATION_AUDIT.md` and this report
- `docs/user-guide/cv-diagrams/`
- `examples/cv_diagrams/`

**Modified:** `README.md`, `docs/COMPLETE_FUNCTIONALITY.md`, `mkdocs.yml`.

No existing source module, test or experiment was modified. In particular these are unchanged:
`calculus/zx.py`, `gaussian_graph.py`, `cv.py`, `calculus/__init__.py`, the flow modules,
the backends and the PhotoGraphiQML stack.

## 22. Remaining priorities

- Nagayoshi Fourier and multi-leg spider rules with exact scalar bookkeeping, then their
  one-mode Gaussian completeness fragment.
- A classical-randomness command in `Pattern`, so that eliminated ancillas stay eliminated.
- Gadget-level rewrites inside beam-splitter sum chains.
- Noise-aware objectives directly in the search.
- A faster flow-certification path for large clusters.
- Heralding-aware non-Gaussian rewrites with instrument-level proofs.

## Capability matrix

| Capability | Status | Evidence |
|---|---|---|
| Typed diagrams, wire typing, immutability, serialisation | Independently validated | Tests; R61 |
| Gaussian unitary / channel / conditional semantics | Independently validated | `gaussian_channel`, postselected trajectories, `GaussianGraph` |
| Gaussian rewrite rules (11 rules: 10 instrument-exact, 1 channel-exact) | Independently validated | R61 (40 instances each), tests |
| GaussianGraph interoperability (pure states) | Independently validated | Round trips; MFvL measurement rules (R62) |
| Pattern / Circuit ↔ diagram translation | Independently validated | Channel round trips; executed trajectories (R67) |
| Resource-cost model, Pareto front | Implemented with regression tests | Tests; R63–R66 |
| CV-flow certification of candidates | Independently validated (uses existing certified solver) | R63, R67; tests |
| Physical-mode optimisation | Independently validated (limited scope) | R64 (feed-forward merge exact); no change on compiled patterns |
| Logical-mode optimisation (teleport-chain resynthesis) | Independently validated | R62, R63, R67 (logical map exact, executed) |
| Approximate-mode optimisation | Implemented with regression tests | Entrywise noise budget only |
| Flow-layered scheduling | Experimental | R64 (equivalence measured: instrument preserved on 15 grids) |
| Non-Gaussian phase-polynomial rules | Experimental | Analytic proofs; Fock convergence vs `expm`; false-rule control (R68) |
| Kerr, Fock, photon counting, post-selection diagrams | Experimental | Closed forms and dense reference (R68; tests) |
| GKP preparation, readout, SUM-syndrome diagrams | Experimental | Existing readout and untruncated grid (R68; tests) |
| Nagayoshi multi-leg spiders, Fourier, bialgebra and other rules | Unsupported | Not implemented |
| Complete CV-ZX calculus | Unsupported | Not claimed |
| Adaptive-angle or non-affine feed-forward translation | Unsupported | Rejected by translation |
| Non-Gaussian MBQC optimisation, Kerr lowering | Unsupported | Not implemented |
| Fault tolerance, quantum advantage | Unsupported | Not claimed |
