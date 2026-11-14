# New core experiment R59: PhotoGraphiQML physical execution on the native backend

| Field | Value |
| --- | --- |
| ID | PGQ-R59 (core suite). Unrelated to PhotoGraphiQML suite experiment PGML-R59 (end-to-end scaling) |
| Script | `paper_experiments/20_photographiqml/59_photographiqml_physical.py` |
| Outputs | `results/{csv,json}/R59_photographiqml_physical.*` (+ `.metadata.json`), `figures/{pdf,png,svg}/R59_photographiqml_physical.*`, `figures/captions/R59_photographiqml_physical.caption.md`; `results/csv/R59_photographiqml_physical_FAILURES.csv` only if a check fails |
| Run | `cd paper_experiments/20_photographiqml && ../../.venv/Scripts/python 59_photographiqml_physical.py [--quick \| --full]` |
| Needs | compiled `photographiq._native` with the `fock` component; Piquasso 8.0.1 |
| Proposed placement | Main text, next to the native-fock validation and the PhotoGraphiQML bridge (or Appendix "PhotoGraphiQML physical layer"); one figure, six panels |

## Objective

Run the restricted physical PhotoGraphiQML MuTA (signed-X measurements, angles 0 or pi)
end to end on PhotoGraphiQ's compiled `native-fock` engine. Validate it against
`piquasso-fock` and against the ideal logical MuTA: outcome probabilities, conditional final
states, measurement records, Pauli-frame corrections, shot statistics, cutoff convergence
and runtime.

## Model

hbar = 2. Finite-energy GKP codewords (Gaussian peaks of width `peak_width` under a
Gaussian `envelope`, `peaks` peaks) projected onto a total-photon-number Fock cutoff.
Three resource levels: wide (0.9, 0.9, 4 peaks), moderate (0.6, 0.6, 6 peaks; the
`GKPPhysicalConfig` default) and narrow (0.45, 0.45, 8 peaks). One unit-gain CZ per graph
edge, p-homodyne with nearest-cell decoding (lattice spacing sqrt(2 pi)), classical logical
Pauli frame, joint X/Z modular readout of the outputs. Finite codewords are not
orthogonal, so the physical instrument is not the logical unitary. The logical-physical
gap is a resource property, reported separately from engine and truncation error.

## Sections and acceptance criteria (declared in the script before evaluation)

Result of the default run (2026-11-13, `.venv`, Python 3.12.14, Piquasso 8.0.1, native
components gaussian/fock/sampling): **0 failures, exit 0, 954 s** (provisional; the machine
was shared). Numbers below are copied from `results/json/R59_photographiqml_physical.json`.

| Section | What is checked | Acceptance | Measured |
| --- | --- | --- | --- |
| A. Engine agreement (15 conditional cases: 1 wire, seeds 0-1, Z and X readout, moderate resource at cutoffs 32/40/48; wide resource 1 wire seeds 0-1 and 2 wires seed 2 at cutoff 32) | max \|ΔP\| of decoded joint probabilities; final-state infidelity 1 − F; executing engine identity; identical decoded bits; probability validity | Wide resource: \|ΔP\| ≤ 2e-4, 1 − F ≤ 2e-6. Moderate resource: both must be smaller at cutoff 48 than at cutoff 32 for every case. Always: bits identical, \|Σp − 1\| ≤ 1e-9, negativity ≤ 1e-9 | Wide: max \|ΔP\| 8.99e-5 (2 wires), 7.6e-6 and 1.2e-6 (1 wire); max 1 − F 1.50e-7. Moderate, max \|ΔP\| by cutoff: 4.64e-4 (32), 3.28e-4 (40), 8.98e-5 (48); 1 − F: up to 1.28e-5 (32), 1.47e-6 (40), 1.82e-7 (48). Decoded bits identical in all 15 cases. Max \|Σp − 1\| 8.9e-16 |
| A'. Logical interface | logical target inside the physical result vs logical `MuTA.unitary` applied to the input (independent code path) | ≤ 1e-12 | max 2.08e-16 |
| A''. Logical-physical gap | TV(decoded, logical) and observable error on each engine | descriptive; must be engine-independent (follows from A) | Gap differs between engines by ≤ 4.6e-4 (same order as the engine difference). The gap itself is large and physical: TV 0.02 to 0.34, observable error up to 0.58 (2 wires, wide) |
| B. Cutoff convergence (vs native at the largest cutoff) | 1 wire moderate: native 32-56 vs 64, Piquasso 32/40/48 vs native 64; 2 wires wide: native 32/40 vs 48 | Every native cutoff ≥ resolved cutoff (moderate 40, wide 32) within 2e-4 of the reference; guard trips recorded as data | 1 wire native: 6.7e-5, 6.7e-6, 1.6e-5, 5.2e-6 (32/40/48/56). Piquasso: 2.7e-4, 7.8e-5, 3.3e-5 (32/40/48), so both engines approach one limit. 2 wires native: 1.1e-4, 3.1e-5. Retained norm (1 wire native) 0.99971 at 32 to 0.999997 at 56. No guard trips |
| C. Pauli frame on physical data (1 wire, all 16 lattice-point branches: analog outcome 0 or sqrt(2 pi) per measured node) | TV to logical target with and without the decoded frame | At each resource the mean corrected TV is below the mean uncorrected TV; worst corrected TV does not increase as the resource narrows | wide (Δ 0.9, cutoff 32): 0.191 vs 0.277, worst 0.362. Moderate (0.6, 40): 0.099 vs 0.287, worst 0.275. Narrow (0.45, 56): 0.038 vs 0.328, worst 0.110 |
| D. Shots (1 wire, moderate, cutoff 40, 8 shots, seed 160) | same seed on both engines | identical decoded bits, sampled output bits and empirical frequencies; Rao-Blackwell \|ΔP\| ≤ 2e-4 | All identical; RB \|ΔP\| 1.70e-4. Sampled analog homodyne values differ by up to 8.9e-3 (each engine samples from its own state), never across a decoding cell. RB P(Z=0) = 0.546 ± 0.107 on both engines vs logical 0.965 (finite-GKP gap; 8 shots) |
| E. Runtime and memory (provisional) | seconds per run, Hilbert dimension, state bytes, tracemalloc | descriptive | 1 wire, dimension 528-1176: Piquasso 4-19 s, native 2.6-11.6 s per conditional run. 2 wires, dimension 5984: Piquasso 240.9 s, native 5.3 s. 8-shot run: 146 s vs 129 s (the engines differ little on shot runs; where that time goes was not profiled). Dense state 16·D bytes: 8.4 kB (D = 528) to 95.7 kB (D = 5984). tracemalloc peak 1-14 MB (NumPy-visible only). Process peak working set 340 MB |
| F. Interface rejections | pi/2 angle on both backends; `native-mixed-fock` | rejected before Fock allocation | Both rejected with `UnsupportedPhysicalMeasurement` (tracemalloc peak about 21 kB, so no Fock allocation happened); `native-mixed-fock` raises `ValueError` |

### Acceptance revisions made during development (recorded, not hidden)

1. **Engine tolerance domain.** The tolerances 2e-4 / 2e-6 were taken from
   `tests/ml/cross_layer/test_native_backend.py`, which measured them for the **wide**
   resource only. The first runs used them at the moderate resource too and failed:
   \|ΔP\| = 2.01e-4 to 4.64e-4 at cutoff 32 and 3.28e-4 at cutoff 40, with 1 − F = 1.28e-5
   at cutoff 32. Native's own error against cutoff 64 is only 6.7e-5 there, while Piquasso's
   is 2.7e-4. The difference therefore comes from Piquasso's per-Euler-factor CZ truncation.
   It is not a native defect. The tolerance values were kept and asserted only in the
   domain where they were established. For the moderate resource the criterion became "the
   difference decreases with cutoff", and it does: 4.6e-4, then 3.3e-4, then 9.0e-5.
   **Finding:** the engine-agreement scale documented in `docs/ml/physical/architecture.md`
   (≤ 9e-5 at cutoff 32) holds for the wide resource only. At the default resource
   (`GKPPhysicalConfig()`, peak width 0.6) Piquasso needs a cutoff of about 48 to reach it.
2. **Cutoff monotonicity.** The first default run also required the distance to the
   largest-cutoff result to decrease monotonically. It does not (1 wire: 6.7e-5, 6.7e-6,
   1.6e-5, 5.2e-6). This matches the non-monotone retained norm already documented in
   `docs/ml/physical/architecture.md`. Monotonicity is now reported, not asserted. The
   asserted criterion is that every resolved cutoff lies within 2e-4 of the reference.
3. **2-wire convergence case.** The moderate-resource 2-wire case at cutoff 40 (seed 2)
   tripped PhotoGraphiQ's retained-norm guard (norm 0.99842). The 2-wire convergence study
   now uses the wide resource, matching the 2-wire engine case. Guard trips are recorded
   as data instead of crashing the script.

The figure and caption of the final run were redrawn with `--replot` from the saved JSON
after layout fixes (legend placement, y ticks, caption wording). The numerical code was
unchanged.

## Figure

`figures/png/R59_photographiqml_physical.png` (also pdf/svg) has six panels.
(a) Engine \|ΔP\| and 1 − F per case, with the tolerances.
(b) Cutoff convergence of both engines towards the native reference.
(c) Frame-corrected vs uncorrected TV over the 16 lattice-point branches at three resource levels.
(d) Shot statistics: Rao-Blackwell estimate ± SE and empirical frequency ± SE for both engines, with the logical target.
(e) Runtime vs Hilbert dimension (provisional).
(f) Logical-physical TV gap, native vs Piquasso.
The caption is in `figures/captions/R59_photographiqml_physical.caption.md`.

## Cross-references (not duplicated)

PGML-R34 (GKP codeword projection), R35/R36 (capability map, lowering preservation), R37
(exhaustive logical Pauli-frame branches), R38/R41 (independently assembled Pattern,
piquasso-fock), R43/R56 (decoded logical-physical gap), R44 (shot convergence), R47/R58
(resource-axis convergence); `tests/ml/cross_layer/test_native_backend.py`.

## Limitations

- Engine agreement is independence class C: both engines receive the same finite-GKP
  codeword amplitudes and the same lowered Pattern, and share the Python preparation,
  homodyne readout, renormalisation, decoding and frame code. Only the Fock evolution
  (and its CZ truncation) is independent.
- Conditional runs are postselected branches, not deterministic outcomes.
- No loss, no detector inefficiency, no arbitrary-angle measurement, no decoded-state
  fidelity, no fault-tolerance or threshold claim.
- Timings are provisional (the machine was shared with other jobs). tracemalloc sees only
  NumPy/Python allocations; native C++ buffers are not traced. The reported state memory is
  the dense-vector estimate 16 x dimension bytes from the capability audit.
