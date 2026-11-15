# New manuscript experiments R43-R50: native engines and photonic sampling

These eight experiments validate the compiled engines (`photographiq._native`: `gaussian`, `fock`,
`sampling`) and the boson-sampling / GBS layer (`photographiq.sampling`). Each script is directly
executable from `paper_experiments/` (`python 14_native_gaussian/43_native_gaussian_validation.py`),
declares its acceptance constants before evaluation, raises `AssertionError` on failure (after
writing `results/csv/<ID>_FAILURES.csv`), and saves CSV/JSON/metadata through `common.save_result`
and figures plus captions through `common.save_figure` / `common.save_caption`. Caption numbers are
formatted from the computed results.

Existing evidence code is reused rather than duplicated:
`tests/validation/test_val_gaussian.py` (generator symplectics, dilation/precision-matrix conditioning,
adaptive wire), `tests/validation/test_val_fock.py` (enlarged-space `expm`, mpmath closed forms,
Hermite quadrature of the projected cubic phase), `benchmarks/gaussian_boson_sampling/bench_gbs.py`
(`_pq_state`) and `threshold_reference_matrix.py` (case builder, `run_case`, 40-digit reference).

All results below are from the `--full` (publication) configuration, run on 2026-11-13 on the
shared development machine (i5-12450HX, 12 logical CPUs, RTX 4050, Windows 11, Python 3.12.14,
Piquasso 8.0.1, MSVC Release build with OpenMP). **All timings are provisional**: other jobs
shared the CPU (load 100 % for most of the session) and must be rerun on an idle machine.

| ID | Script | Default runtime (this machine, loaded) | `--full` runtime | Outcome | Proposed `run_all_safe` class |
|----|--------|------|------|---------|------|
| R43 | `14_native_gaussian/43_native_gaussian_validation.py` | 15 s | 9 min (loaded machine) | passed (0 failures) | safe (default); `--full` is a long run |
| R44 | `14_native_gaussian/44_native_gaussian_performance.py` | 3 min | 24 min | passed (accuracy checks) | long benchmark (repeated timing, subprocess per configuration) |
| R45 | `15_native_fock/45_native_fock_accuracy.py` | 30 s | 40 s | passed | safe (uses Piquasso, as other safe scripts do) |
| R46 | `15_native_fock/46_fock_space_optimization.py` | 46 s | 40 s | passed | long benchmark (repeated timing; GPU variant runs only if CuPy sees a device) |
| R47 | `15_native_fock/47_native_mixed_fock_validation.py` | 6 min (loaded) | 7.5 min | passed | large Fock (density matrices up to cutoff 48; Piquasso mixed Fock) |
| R48 | `16_sampling/48_boson_sampling.py` | 1 min | 46 s-1 min | passed | safe |
| R49 | `16_sampling/49_gaussian_boson_sampling.py` | 10 s | 15 s | passed | safe |
| R50 | `16_sampling/50_threshold_gbs.py` | 1.6 min (8 processes) | 4-7.5 min (8 processes) | passed | long benchmark (multiprocess 40-digit mpmath reference) |

---

## R43 - Native Gaussian backend validation

**Scientific question.** Does `native-gaussian` (in-place O(n k) C++ local updates) reproduce the exact
Gaussian phase-space evolution (states, gates, loss, homodyne and heterodyne conditioning, adaptive
feed-forward) at every mode count and depth we use, and does it agree trajectory-by-trajectory with the
NumPy `gaussian` backend?

**Assumptions / model.** hbar = 2, interleaved (q, p), vacuum V = I, statistical covariance. Random
mixed initial states V = S diag(nu) S^T (generator scaled by 1/sqrt(4n)); seeded random circuits of
rotations, squeezers, beam splitters, CZ, shears, displacements and thermal loss (eta in [0.6, 1],
n_th <= 0.2). Active gate parameters shrink as min(1, sqrt(40 n / D)) for depth D so that the
accumulated random-walk squeezing stays O(1): the study measures rounding growth with depth, not the
conditioning of ever more squeezed states (see "Defects / notes" for what happens without this).

**Backend.** `native-gaussian` vs NumPy `gaussian` (float64).

**Seed.** 20261143.

**Parameters.** n = 1, 2, 4, 8, 16, 32, 64, 128; D = 10, 40, 160, 640, 2560 (default: n <= 32,
D <= 640); "unitary" and "noisy" families; conditioning on n = 2, 4, 8 random mixed states (ideal
homodyne at two angles, eta = 0.7 / noise 0.15 homodyne, heterodyne); 2-step adaptive wire
(outcome-dependent angle) and a 24-step sheared Fourier wire, 20 seeds each.

**Reference.** Closed-form single-mode states; dense first-principles propagation with
S = expm(2 Omega M) of each gate's quadratic generator and loss X = sqrt(eta) I,
Y = (1 - eta)(2 n_th + 1) I; dilation + precision-matrix conditioning (not the Schur form the
backends use); hand-propagated teleportation steps (`tests/validation/test_val_gaussian.py`).
The composed native symplectic S is extracted column by column from the native mean update.

**Acceptance criteria.** states <= 1e-13; circuit mean/covariance relative error <= 64 D eps
(heuristic forward-error model, declared, not proven); symplectic residual
||S Omega S^T - Omega||_max / ||S||_2^2 <= 64 D eps; physicality min eig(V + i Omega) >= -1e-12 ||V||;
conditional moments <= 1e-10 (relative to max(1, ||V||)); outcome density <= 1e-10 relative;
native vs NumPy same seed <= 1e-11; feed-forward vs hand <= 1e-11.

**Actual results (full).** 80 circuits, 0 failures. Worst covariance relative error 1.6e-12 (unitary
family, n = 1, D = 2560, where the expm reference itself contributes equally - NumPy shows 1.45e-12);
noisy family worst 3.4e-14. Worst symplectic residual of the native S 2.0e-15. Worst physicality margin
-1.3e-15 (pure states sit at the boundary). Closed-form states exact (0.0). Conditioning: worst
conditional-moment error 3.6e-16, worst density relative error 1.6e-15. Adaptive wire vs hand
propagation 4.0e-15; native vs NumPy with the same seed: outcomes and final moments agree to 1.8e-15.

**Figures.** `figures/{pdf,png,svg}/R43_native_gaussian_validation.*` (+ caption).

**Limitations.** float64 only; the depth bound is a heuristic model; covariance-scale-limited
squeezing; no batched trajectory runner exists (conditioning tested per trajectory).

**Proposed placement.** Appendix "Native Gaussian engine validation" (panel b/c could join the main-text
software-validation figure).

---

## R44 - Native Gaussian performance (with accuracy)

**Scientific question.** What does the native Gaussian backend buy over the NumPy backend and over
independent implementations, versus modes, gate count, number of conditional measurements, batch
size and thread count - and at what accuracy?

**Model.** R43's noisy random circuits on random mixed states; sequences of postselected homodyne
measurements (theta = 0.3, outcome 0.2); batched beam splitter + homodyne conditioning on n = 16 states.

**Backends / implementations.** `native-gaussian`; NumPy `gaussian` backend (dense S V S^T plus
per-gate validation); a dense NumPy symplectic propagation written in the script; Piquasso 8.0.1
`GaussianSimulator` running the whole circuit as one program; for batches the native OpenMP kernels
(`batch_apply_symplectic`, `batch_condition_homodyne`) vs a vectorised NumPy dense update/Schur formula.

**Seed.** 20261144.

**Parameters (full).** modes 4-256 at 200 gates; gates 50-3200 at n = 32; 1-31 measurements at n = 32;
batch 1-4096; OMP_NUM_THREADS 1, 2, 4, 8, 12; 2 warm-ups, 11 repeats (default: smaller grid, 5 repeats).

**Method.** Each configuration runs in a fresh subprocess (OMP_NUM_THREADS takes effect; it was
checked that `_native.gaussian.max_threads` follows it; it also limits NumPy's BLAS in that process).
Runtime: `common.benchmark`, median + IQR. Memory: `psutil.Process().memory_info().peak_wset`
(Windows peak working set; `ru_maxrss` elsewhere), reported as absolute peak and increase over the
pre-run peak. tracemalloc is not used (it misses C++ allocations). Gate timings exclude the O(n^3)
validation of the start state (an unvalidated copy of a validated state is loaded).

**Reference / acceptance.** Every implementation vs the R43 dense first-principles reference:
max relative error <= 1e-10; native batch kernels vs vectorised NumPy <= 1e-12.

**Actual results (full, PROVISIONAL timings).** 69 configurations, 0 accuracy failures; worst accuracy
3.4e-14; batch agreement 5.6e-16. At 200 gates: native 2.1 ms (n = 4) -> 10.6 ms (n = 256); NumPy
backend 0.11 s -> 44.3 s (4160x slower at n = 256); Piquasso 7.0 s at n = 128 (1660x slower than native;
not run at 256); dense NumPy 1.9 s at n = 256. Batch 4096 (n = 16): native 60 ms on 1 thread, 31 ms on
12 threads (vectorised NumPy 0.12-0.21 s); the thread curve is non-monotonic on the loaded machine
(4 threads: 96 ms). Peak-memory increases are below the import-time peak for most native
configurations (e.g. +0.2 MiB at n = 128 vs +4.8 MiB NumPy backend, +14.8 MiB Piquasso); the method
cannot resolve allocations smaller than the interpreter's import peak.

**Figures.** `R44_native_gaussian_performance.*`.

**Limitations.** Shared, loaded machine: all timings must be rerun idle (`--full --repeats 15`). The
dense script propagation shares gate matrices with the reference, so its ~1e-16 accuracy is a
consistency floor. Single-trajectory native kernels are serial (only batch kernels use OpenMP).

**Proposed placement.** Main-text performance figure (panels a, d) after the idle rerun; rest in the
performance appendix.

---

## R45 - Native Fock accuracy and cutoff convergence

**Scientific question.** How accurate is `native-fock` (total-photon truncation, P U P with
renormalisation and recorded retained norm) for Fock inputs, displacement, squeezing, beam splitters,
Kerr, cubic phase and conditional photon counting, and how do state, observable and norm errors
converge with the cutoff? Where does Piquasso 8.0.1 agree?

**Model.** One- and two-mode systems: D(alpha)|0> (|alpha| = 1.30, 4), D(4)|cutoff/2>, S(0.8)|0>,
Kerr(0.35) on a coherent state, cubic phase (gamma = 0.1, 0.3, both `projected` and
`truncated_generator` definitions), 50:50 splitter on |1,1>, |2,1>, |3,3>, |4,2>, two squeezers
(r = 0.6) + 50:50 splitter (TMSV) with photon counting k = 0..3.

**Backend.** `native-fock` (norm guard opened to 1 - 1e-12 so truncation is studied, never rejected);
`piquasso-fock` at cutoffs <= 48.

**Seed.** 20261145 (deterministic cases).

**Parameters.** cutoffs 6, 8, 12, 16, 24, 32, 48, 64, 96 (default up to 48).

**Reference.** Kept in two classes. Exact-in-truncation: P U psi / ||P U psi|| from 30-digit closed forms
(coherent, squeezed), an exact 50-digit associated-Laguerre sum for <m|D|n>, exact Kerr phases,
enlarged-space `expm` (beam splitter), Hermite quadrature of the projected cubic phase, exact TMSV
truncation (conditional state |k> exactly when 2k < cutoff). Untruncated physics: <n> = |alpha|^2,
sinh^2 r; Kerr <a> from 400 levels; cubic phase p -> p + gamma q^2 (hbar = 2) giving <p> = gamma,
<p^2> = 1 + 3 gamma^2; P(k) = (1 - l^2) l^(2k).

**Acceptance criteria.** exact-in-truncation state and retained norm <= 1e-12 (projected cubic <= 1e-10,
quadrature floor); HOM P(1,1) <= 1e-15; normalisation <= 1e-12; untruncated observables at the top
cutoff must be explained by the recorded leaked norm, |err| <= 10 * cutoff * (1 - retained) + 1e-12
(Gaussian-tailed cases); cubic phase: monotone convergence over the last three cutoffs.

**Actual results (full).** 252 rows, 0 failures. Worst exact-in-truncation error 1.7e-15 over all
cutoffs up to 96 (incl. |alpha| = 4). HOM P(1,1) = 0. Projected cubic (gamma = 0.3) <p^2> error at cutoff
96: 9.2e-12. Native vs Piquasso on D(4)|cutoff/2>: Piquasso error grows to 3.9e-6 at cutoff 48 (native
7.2e-16); on D(4)|0>, squeezing and the truncated-generator cubic phase both agree to rounding.

**Figures.** `R45_native_fock_accuracy.*`.

**Limitations.** one/two modes (multimode cost is R46); float64. During development the cubic
reference was first written with hbar = 1 formulas (gamma/2, 3 gamma^2/4); the check failed and the
formula was corrected to hbar = 2 - the engine was right.

**Proposed placement.** Appendix "Fock engine accuracy"; panel (d) supports a one-line main-text remark on
large-cutoff stability vs Piquasso.

---

## R46 - Fock-space optimisations (only those implemented)

**Scientific question.** What do the implemented Fock-space optimisations buy in runtime and memory, at
identical physics and identical output?

**Implemented optimisations (from `cpp/src/fock/{basis,kernels,gates}.cpp`,
`backends/native_fock.py`, `gpu/fock.py`).** (1) total-photon basis C(m + c - 1, m) instead of c^m
(a model choice: reported as size only); (2) fibre-local gate application (no global operator; P U P
per fibre); (3) passive gates conserve total photon number, so `gaussian_unitary` builds only the
sector blocks (block-diagonal local matrix), and with total-photon truncation a passive gate is exactly
unitary on the truncated space; (4) diagonal gates as phase multiplication; (5) caches (basis tables,
cubic-phase matrix, sparse gate structure).

**Compared variants (same P U P model).** native fibre kernel `apply_pure`; global sparse gate G
from `photographiq.gpu.fock` with scipy.sparse (CPU) and cuSPARSE (GPU, incl. transfers;
`Execution.device` recorded = `cuda:0`); dense D x D matrix (D <= 4000).

**Seed.** 20261146. **Parameters (full).** (m, c) = (2,30), (3,16), (4,12), (5,10), (6,8), (8,6),
(4,20), (6,12), (8,9), (10,7); batch 1 and 32; sector study at cutoffs 6-64; 9 repeats.

**Acceptance.** variants agree <= 1e-12; passive-gate output norm 1 to 1e-13; phase vs diag-matrix
<= 1e-14; off-sector entries of the passive local matrix exactly 0; GPU variant must report a CUDA device.

**Actual results (full, PROVISIONAL timings).** 216 rows, 0 failures; all variants agree to 1.7e-16.
Gate representation: native local matrix 32 KB vs CSR 2.7 MB at (8, 9), D = 12870; dense G 63 MB at
D = 2002. One beam splitter, B = 1: native 0.64 ms vs scipy 0.77 ms vs cuSPARSE 0.52 ms at D = 12870;
for B = 32 the native path (a Python loop over states) is slower than the batched sparse products
(25 ms vs 13 ms scipy, 10.5 ms GPU). Passive sectors: off-sector entries exactly 0, 97.9 % zeros at
cutoff 64; `gaussian_unitary` for the beam splitter is 3.7x faster than for an active CZ of the same
local size at cutoff 64 (observation, not ablation). Phase multiplication vs diagonal-matrix
application: median speedup 1.0x (no measurable gain at these sizes). Caches: basis table 600x,
projected cubic matrix (cutoff 60) ~3e5x (0.18 s -> 0.6 us), sparse structure 5.6x.

**Figures.** `R46_fock_space_optimization.*`.

**Limitations.** Only pure-state kernels are timed (mixed kernels: see R47 defect note); the passive
skip cannot be disabled from Python; B = 32 native uses a Python loop (there is no batched native
pure kernel); provisional timings.

**Proposed placement.** Appendix "Fock-space engineering"; panel (a) and the passive-sector statement can
support the methods section.

---

## R47 - Native mixed-Fock validation

**Scientific question.** Does `native-mixed-fock` produce physical density matrices that reproduce
mixed-state physics (thermal/mixed Gaussian inputs, loss, partial trace, conditional measurement), and
how does it converge with the cutoff?

**Model.** Thermal inputs (nbar = 0.3, 1.0); displaced squeezed thermal input (r = 0.4, theta = 0.8,
beta = 0.6 + 0.3i, nbar = 0.3); pure loss (eta = 0.3, 0.7) on |1.5>; thermal loss (n_th = 0.4); TMSV
(r = 0.6) partial trace; lossy TMSV (eta = 0.8) with photon counting k = 0..2 and homodyne conditioning
(x = 0.7); 21-operation 3-mode sequence (BS, CZ, Kerr, cubic, squeeze, displace, thermal loss) x 3.

**Backend.** `native-mixed-fock` (OpenMP pinned to 1 thread, see notes); `piquasso-mixed-fock` at
cutoff <= 8; NumPy `gaussian` for Gaussian conditioning; `native-fock` for consistency.

**Seed.** 20261147. **Parameters (full).** cutoffs 8, 12, 16, 24, 32, 48; sequence cutoffs 6, 8, 10, 12.

**Reference.** exact truncated density matrices (geometric thermal, TMSV reduced, 90-level dense expm for
the displaced squeezed thermal input); invariants (coherent output fidelity and purity, <n> laws,
analytic thermal marginal P(k)); Gaussian conditional moments and density; pure-engine outer product.

**Acceptance.** exact-in-truncation <= 1e-12; Hermiticity <= 1e-13, |tr - 1| <= 1e-12,
min eigenvalue >= -1e-12 after every operation; mixed vs pure <= 1e-12; observables explained by the
leaked norm (K = 10, as R45); Gaussian-moment comparison <= 1e-6 at the top cutoff.

**Actual results (full).** 84 rows + 84 sequence steps, 0 failures. Worst exact-in-truncation 4.4e-16;
pure loss -> |sqrt(eta) alpha> to 2.2e-16 at cutoff 48; lossy-TMSV homodyne conditioning vs Gaussian
3.1e-12 at cutoff 48 (5.3e-8 at 32); worst Hermiticity/trace residual 1.3e-15, smallest eigenvalue
-8.8e-16; mixed vs pure 5.0e-16. Native vs Piquasso mixed Fock on the 21-operation sequence: 1.0e-3
(c = 6), 5.5e-4 (c = 8) - truncation-level differences (Piquasso's CZ is a truncated Euler decomposition).

**Figures.** `R47_native_mixed_fock_validation.*`.

**Limitations.** <= 3 modes; Piquasso comparison only at cutoff <= 8 (its mixed-state program validation
took 378 s for one cutoff-16 pass here); float64.

**Proposed placement.** Appendix "Mixed-state Fock engine".

---

## R48 - Standard (Fock-state) boson sampling

**Scientific question.** Are boson-sampling probabilities |Perm(U_{t,s})|^2 / (prod s! prod t!) correct
for collision-free and collision events, normalised, consistent with an independent Fock-space
evolution, and are the exact samplers distributed correctly? How do the permanent kernels scale?

**Model.** scipy.stats `unitary_group` Haar unitaries (fixed seeds), Fock inputs incl. bunched ones,
DFT interferometers.

**Backend.** `photographiq.sampling` (native Gray-code Ryser, NumPy multiplicity-aware Ryser).

**Seed.** 20261148. **Parameters (full).** permanent sizes 1-14, 16, 18, 20 (3 matrices each);
10 distribution cases (m = 4-12, up to 5 photons, incl. (2,2,0..), (3,2,0..)); DFT m = 3-7; 20000-shot
samplers (Clifford-Clifford and enumeration).

**Reference.** permutation-sum definition (n <= 8) and Glynn's formula (this script); native Fock-space
evolution of the m-mode passive unitary (sector recurrence, cutoff n + 1); Piquasso `PureFockSimulator`
(`NumberState` + `Interferometer`, same convention, checked); DFT suppression law; Pearson chi-square.

**Acceptance.** permanent error / prod_i sum_j |a_ij| <= 1e-13; |sum P - 1| <= 1e-12; vs Fock-space and
Piquasso <= 1e-12; suppressed DFT mass <= 1e-14; chi-square p >= 1e-4.

**Actual results (full).** 0 failures. Permanent relative error <= 2.7e-11 at n = 20 (vs float64 Glynn;
<= 1e-13 for n <= 8 vs the definition); scaled error 1.5e-16. Normalisation 1.2e-15; vs Fock-space
evolution 5.0e-16; vs Piquasso 5.3e-16; suppressed DFT mass <= 6.7e-27; sampler p-values 0.63, 0.30.
Timing (provisional): native at n = 20 18.5 ms with 12 threads, 82 ms with 1 thread; NumPy Ryser
~0.4 s at n = 18; the threaded kernel was slower than 1 thread for n <= 8.

**Figures.** `R48_boson_sampling.*`.

**Limitations.** Glynn reference is itself float64 for n > 8; distributions only up to C(n+m-1, n)
enumerable sizes.

**Proposed placement.** Appendix "Sampling layer"; one sentence in the main text.

---

## R49 - Gaussian boson sampling with PNR detectors

**Scientific question.** Are PNR probabilities P(n) = p_0 lhaf(A_{n+n}; gamma)/prod n! correct (hafnian,
loop hafnian, pure and mixed formulas) for squeezed states through Haar interferometers with loss and
displacement, across modes, squeezing, photon number and cutoff?

**Model.** `squeezed_interferometer_state`: q-squeezed vacua (r = 0.3-1.0), scipy Haar U, loss
eta = 0.7-1, displacement |alpha| <= 0.4.

**Backend.** `photographiq.sampling.pnr_probability` (native and NumPy kernels).

**Seed.** 20261149 (one unitary and displacement per (m, r, displaced) family, so the cutoff sweep is a
convergence study of one state).

**Parameters (full).** Fock-reference families m = 2-5, cutoffs 6-16 (undisplaced), 8-40 (displaced);
Piquasso families m = 4-12, up to 12 photons; hafnian definition checks for sizes 1-12; patterns with
more than 12 photons are not evaluated.

**Reference.** perfect-matching definitions (this script); Fock-space evolution (product of single-mode
squeezed columns, total-photon truncation, m-mode passive Fock matrix) - exact for every pattern below
the cutoff when undisplaced and lossless; Piquasso `get_particle_detection_probability` on the identical
program (`bench_gbs._pq_state`); sum rule: hafnian mass below the cutoff = exact retained Fock mass.

**Acceptance.** hafnian / matching sum <= 1e-12 (relative to sum |terms|); undisplaced vs Fock <= 1e-13
absolute; pure vs mixed formula <= 1e-13; vs Piquasso <= 1e-9 relative (P > 1e-12); displaced vs Fock at
the top cutoff <= 1e-10 (|n| <= 4); mass rule <= 1e-12.

**Actual results (full).** 5117 compared patterns, 0 failures. Undisplaced vs Fock worst 5.6e-16;
displaced at the top cutoff 2.2e-16 (convergence from 4e-5 at cutoff 8); vs Piquasso worst relative
6.3e-12; hafnian definitions 2.9e-16; mass rule 1.7e-15.

**Figures.** `R49_gaussian_boson_sampling.*`.

**Limitations.** <= 12 photons per pattern; Fock reference limited to <= 5 modes; displaced references
converge (not exact) because displacement acts after truncation.

**Proposed placement.** Appendix "Sampling layer" (with R48, R50).

---

## R50 - Threshold-detector GBS: cancellation and precision

**Scientific question.** How accurate are threshold (click) probabilities in double precision (native and
Python), is the reported rounding bound B reliable, where does the alternating inclusion-exclusion sum
lose accuracy, and do `precision=<digits>` / `precision="auto"` recover it? How does Piquasso compare?

**Model.** `threshold_reference_matrix.build` cases: Haar interferometer, r in {0.05, 0.2, 0.5, 1.0, 1.5},
eta in {1.0, 0.9, 0.5}, half displaced (alpha_j = 0.4 u e^{2 pi i v}), random k-click patterns, plus
"zero" cases (untouched vacuum mode that clicks).

**Backend.** `threshold_probability`: native and Python double sums, `precision=50`, `precision="auto"`
(`ThresholdInfo`), and Piquasso 8.0.1 `get_threshold_detection_probability` (scored against a reference
for its own stored state).

**Seed.** 20261150. **Parameters (full).** modes 4, 8, 12, 16, 20; k in {1, 2, 3, 4, 6, 8, 10, 12};
185 cases (170 random + 15 zero); precision sweep 16-60 digits on the 8 highest-cancellation cases;
full click distributions for m = 2-8.

**Reference.** The 40-digit mpmath reference of `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py`
(rerun here; uncertainty 10^(3-40) sum|terms|), evaluated on each library's stored state. The 2026-11-11
382-case matrix (docs/validation/gbs-threshold-accuracy.md) is cited for comparison only; no number from it
is presented as new.

**Acceptance.** double error <= B + reference uncertainty in every case (native and Python); a
`ThresholdPrecisionWarning` whenever B/|P| > 1e-6; `precision="auto"` within rtol 1e-6 of the reference
(cases with reliable reference); `precision=50` within 10 x (reference uncertainty + its own reported
error); zero cases: |P_ref(stored state)| <= 1e-14 and warned; click distributions sum to 1 within 1e-10.

**Actual results (full).** 0 failures. Bound held in 185/185 cases (smallest B/actual 6.2, median 330).
Double precision is NOT full precision here: native relative error > 1e-6 in 20/170 random cases
(Python 20, Piquasso 21); worst native 2.3e7 (m = 12/16, k = 12, r = 0.05, P ~ 1e-22); median native error
by k: 1.2e-15 (k = 1) ... 2e-9 (k = 8) ... 1e-8 (k = 12), maximum 4.8e-1 at k = 8, 7.8e4 at k = 10. The
double error tracks eps x cancellation sum|terms|/|P| (cancellation up to 1e26). `precision="auto"` used
mpmath in 42 cases (26-51 digits, <= 2.1 s each) and stayed within 1.9e-8 of the reference everywhere;
`precision=50` matched the reference to float64 output. Click distributions sum to 1 within 2.2e-15.
Zero cases: the reference of the stored state lies between -5.5e-17 and +1.9e-40 (14 of 15 negative: the
stored covariance is physical only to rounding) and every zero case warned.

**Figures.** `R50_threshold_gbs.*` (error vs probability, error vs cancellation, B/actual histogram,
error vs working digits).

**Limitations.** clicks <= 12 in this run (16-click cases are in the cited 2026-11-11 matrix); the
reference is 40 digits, so agreement beyond 40 digits is only to float64 output.

**Proposed placement.** Main text (one panel: error vs cancellation) or Appendix "Threshold GBS precision";
the bound and `precision="auto"` behaviour belong in the numerical-methods section.

---

## Defects / notes found outside this scope (not fixed here)

1. **Eigen GEMM nested OpenMP makes small `apply_mixed` calls 30-300x slower on a loaded machine**
   (`cpp/src/fock/kernels.cpp`, `multiply_fibres`). When the outer fibre loop runs serially
   (`gate_work * dim <= 2^22`), the per-fibre products `u * block` are Eigen GEMMs, which Eigen
   parallelises with OpenMP. Reproduction (`.venv/Scripts/python`):
   ```python
   import time, numpy as np
   from photographiq import native
   E = native.require("fock"); c, s = np.cos(np.pi/4), np.sin(np.pi/4)
   bs = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
   for cut in (12, 16, 20):
       u = E.gaussian_unitary(bs, np.zeros(4), cut); d = E.dimension(2, cut)
       rho = np.eye(d, dtype=complex) / d; t = time.perf_counter()
       E.apply_mixed(rho, 2, cut, [0, 1], u); print(cut, time.perf_counter() - t)
   ```
   Measured: 0.24-0.48 s (cutoff 12), 0.55-2.3 s (cutoff 16), 0.1 s (cutoff 20, outer loop parallel)
   with the default 12 threads; 1.4 ms / 8 ms / 50 ms with `OMP_NUM_THREADS=1`. Proposed fix: compile with
   `EIGEN_DONT_PARALLELIZE` (or `Eigen::setNbThreads(1)` in the kernels), since the kernels parallelise
   over fibres themselves. R47 and R49 pin the runtime to 1 thread
   (`native.require("sampling").set_num_threads(1)`) for that reason.
2. **OpenMP spin-wait overhead in the sampling kernels on a loaded machine.** An aborted R49 run
   accumulated ~11,000 CPU-seconds; the threaded permanent was slower than 1 thread for n <= 8 (R48).
   Proposal: a minimum-work threshold for parallel regions (as `kParallelWork` in the Fock kernels) and/or
   `OMP_WAIT_POLICY=passive` guidance in docs/performance.md.
3. **`pnr_distribution(state, cutoff)` is impractical for cutoff >~ 25** (hafnian cost 2^(N/2) N^4 for
   every pattern up to cutoff - 1 photons). Not a bug; docs could state the cost. R49 evaluates only
   patterns with <= 12 photons.
4. **`GaussianState.validate` uses an absolute tolerance** (`src/photographiq/states.py:180`,
   `min eig(V + i Omega) < -1e-7`). Extremely squeezed pure states (||V|| ~ 4e14 after an unscaled
   2560-gate random circuit on 2 modes, seed 0 of R43's generator) violate it purely by rounding
   (relative violation ~1e-16), so `get_state()` raises. Proposal: compare with `-tol * max(1, ||V||_2)`.
   Low priority (states this squeezed are unphysical); R43 bounds the accumulated squeezing instead.
5. **Piquasso 8.0.1 mixed Fock is very slow through the adapter** (378 s for one R47 cutoff-16 pass,
   dominated by Piquasso's program validation in `MixedFockBackend._gaussian_density` /
   `_density`). Informational.

## Not done

* Timings (R44, R46, R48 runtime panel) were taken on a loaded shared machine and are provisional.
* R50 default configuration was last run before the zero-case criterion correction (it then reported the
  2 zero cases whose stored-state reference is -3.5e-17 / -1.8e-21 as failures under the earlier, wrongly
  specified criterion "|value| <= B"); the corrected criterion passed in the `--full` run (which includes
  those cases). Default configurations were not rerun at the end so that the saved outputs are the
  `--full` results.
* No `run_all_safe.py`, `EXPERIMENT_INDEX.md`, `MANUSCRIPT_MAP.md` or `STATUS.md` changes (main agent).
