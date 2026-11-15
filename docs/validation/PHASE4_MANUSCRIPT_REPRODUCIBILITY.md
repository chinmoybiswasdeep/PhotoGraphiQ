# Phase 4: Performance, benchmarks and manuscript reproducibility

Target manuscript: *PhotoGraphiQ: A photonic continuous-variable software platform to perform
measurement-based quantum computing* (Quantum). Every number below comes from the 2026-11-12
reproduction run in `benchmarks/results_20261112/`
unless stated otherwise. The run used HEAD `f30321f` plus uncommitted local changes, with
`summary.json` recording the commands, exit codes and environment. Timings on this laptop vary
by about ±30 % between runs for small cases; accuracy fields are deterministic.

## 4.1 Contribution matrix

Novelty is claimed only where it is not a reimplementation. The comparisons with prior
literature below are the auditors' assessment and should be confirmed by the authors' own
literature review before submission.

| Contribution | Class | Relation to existing work |
| --- | --- | --- |
| CV-MBQC pattern layer: weighted cluster graphs, patterns, outcome-dependent feed-forward, circuit-to-pattern compilation, finite-squeezing channels | New software architecture / scientific integration | The theory is established (Menicucci et al. 2006; Gu et al. 2009). Graphix covers qubit MBQC only. The authors should compare the CV-MBQC support of general CV simulators (Piquasso, Strawberry Fields, including its time-domain programs) themselves. |
| Native C++20 Gaussian, pure-Fock and mixed-Fock engines behind one backend contract | New numerical implementation | Standard symplectic and Fock methods (cf. Piquasso, The Walrus). The one-mode R·S·R·D factorisation with a Jacobi closed-form squeeze factor and a tail bound is an implementation choice, not a new algorithm. |
| Capability registry and pre-allocation feature check | New software architecture | Engineering contribution |
| Boson-sampling and GBS kernels (Ryser/Glynn permanents, hafnians, loop hafnians, torontonians, Clifford & Clifford sampling) | Reimplementation of existing techniques | Björklund et al., Quesada et al., Clifford & Clifford, Piquasso, The Walrus |
| Threshold probabilities in a dark-mode-conditioned `expm1` form, with a rounding bound, warnings and adaptive mpmath precision | New numerical implementation and validation methodology | The torontonian formula is established (Quesada et al. 2018). The reformulation, the per-call error bound (rigorous for conditioning, calibrated for the alternating sum) and the 382-case reference matrix are implementation contributions. |
| Qubit flows (causal flow, gflow, Pauli flow), measurement calculus, local Cliffords, ZX subset | Reimplementation of existing techniques | Danos–Kashefi–Panangaden, Mhalla–Perdrix, Browne et al., Simmons; cross-checked against Graphix |
| CV flow with exact rational impossibility witnesses, numerical-uncertainty classification and regime-labelled certificates | New validation methodology on an existing algorithm | CV flow condition from Booth & Markham (Quantum 7, 1146, 2023) |
| Gaussian-pure-state graphical calculus | Reimplementation | Menicucci, Flammia & van Loock, PRA 83, 042335 (2011), integrated with the CV-MBQC patterns |
| Interactive construct → flow → corrections → execution views driven by real simulation traces | New visualization capability | n/a |
| Independent-reference validation matrix, cross-backend agreement, manuscript reproduction with recorded-value comparison | New validation methodology (software practice) | n/a |
| PhotoGraphiQML: MuTA logical models, logical-to-physical GKP lowering on the CV backends | New scientific integration | MuTA (Mantilla Calderón et al., PRA 2026); logical results checked against MentPy |
| Optional CuPy GPU kernels (batched Gaussian; batched Fock P U P) | New numerical implementation | n/a |
| CUDA-Q Dynamics adapter | Software integration, execution unverified | n/a |

## 4.2 Experiment-to-claim mapping and reproduction

The suites carry their own claim maps: `paper_experiments/EXPERIMENT_INDEX.md` and
`paper_experiments/MANUSCRIPT_MAP.md` (PGQ-R1–R42), and `paper_experiments_ml/EXPERIMENT_INDEX.md`
(PGML-R1–R59). This audit re-ran them in copies, so the recorded results are unchanged, and
compared every regenerated value with the recorded one using `scripts/compare_recorded_results.py`.

| Suite | Re-run | Result | Comparison with recorded values |
| --- | --- | --- | --- |
| PhotoGraphiQ core, safe subset | 24 run, 18 skipped by the runner (long benchmarks, JAX) | 24 passed | **Every value present in both files is reproduced** (exactly, or within 1e-12 absolute for 8 rounding-level error metrics in R15). 15/25 files agree completely. 10 show schema or configuration drift: the recorded files came from a different version of the scripts (extra or renamed fields; R1 now uses 4000 shots instead of 500; R2 uses 13 rows instead of 7). R13 records two error messages that are now more precise ("Self CZ is invalid…", "Command 0 (Output) repeats a mode…" instead of "Cyclic command dependencies"), the intended validation fix. |
| PhotoGraphiQML, safe subset (project venv, no MentPy) | 37 run, 22 skipped by the runner | 28 passed, 9 failed (MentPy not installed) | 28/28 regenerated files identical |
| PhotoGraphiQML with MentPy (isolated venv, MentPy at CI commit `63c3d83e`) | 24 experiments, including R7–R13, R20, R21 | 24 passed | every non-timing value bit-identical ([evidence](photographiqml-mentpy-validation.md)) |

**Recommendation.** Before submission, regenerate the recorded core results with the current
scripts so that recorded and reproducible values have the same schema. That is the
maintainer's decision; this audit did not overwrite them.

**Heavy experiments not re-run** in this audit (runner-skipped): PGQ R17, R18, R20–R23, R25–R28,
R32–R34, R36–R38, R41 and R42; 22 PGML moderate or heavy runs. Their recorded results remain the
historical evidence. `python scripts/reproduce_manuscript.py --mode full` runs only the safe
subsets; each heavy script runs individually from a copy.

## 4.3–4.8 Benchmark methodology

- **Same work, same inputs.** Every comparison runs the identical task on identical inputs. Outputs are checked for agreement before timing, and a mismatch aborts the run.
- **Timing method.** Setup is timed separately. Medians and IQR use `time.perf_counter` with warm-up, and cold first calls are reported separately.
- **Hardware.** Windows 11; i5-12450HX (12 logical CPUs, OpenMP 12 threads); RTX 4050 Laptop GPU (CuPy 14.2.0, CUDA 13.2); Python 3.12.14; NumPy 2.5.3; SciPy 1.18.1; Piquasso 8.0.1 with numba 0.67.0; Graphix 0.4. Libraries use their own threading, so ratios are not single-core.
- **Truncation.** Fock comparisons use the same total-photon cutoff on both engines. The CZ uses a different truncation scheme in Piquasso, and that tolerance is empirical and recorded per row.

## Results (manuscript-ready tables)

### Native Gaussian engine vs NumPy (`bench-native-gaussian.json`)

| Task | n = 8 | n = 64 | n = 256 | Comparison |
| --- | --- | --- | --- | --- |
| single beam-splitter gate | 43× | 329× | 7506× | dense NumPy backend (O(n³) plus re-validation) ÷ native O(nk) update |
| single homodyne | 11× | 27× | 133× | dense NumPy backend ÷ native |
| batched gates | 1.2× | 2.5× | 2.6× | vectorized NumPy ÷ native batch (like-for-like) |
| batched homodyne | 3.8× | 10× | 5.1× | vectorized NumPy ÷ native batch (like-for-like) |

The like-for-like batched gain is 1.2–10×. The large single-state factors come from the
algorithmic O(n³) → O(nk) change, not from C++ alone.

### Native Fock vs Piquasso (`bench-fock.json`)

- **Per gate:** 73 matched cases. Native is a median 44× faster, ranging from 1.2× to 4.3×10⁴.
- **Whole patterns:**

  | Pattern | Native pure | Piquasso pure | Native mixed | Piquasso mixed |
  |---|---|---|---|---|
  | wire, 4 modes, cutoff 16 | 2.6 ms | 51 ms | 33 ms | 1.21 s |
  | wire, 8 modes, cutoff 16 | 4.8 ms | 90 ms | 58 ms | 2.31 s |
  | network, 3 modes, cutoff 10 | 1.7 ms | 40 ms | 43 ms | 0.89 s |

- **Accuracy vs cutoff** (two-mode Gaussian circuit against the exact Gaussian photon numbers; both engines identical): 9.8e-4 at cutoff 8, 7.8e-7 at 16, 5.0e-10 at 24, 2.7e-13 at 32.

### GPU (CuPy, RTX 4050)

- **Batched Gaussian** (`bench-gpu-gaussian.json`), n = 64, B = 256, float64:

  | Task | NumPy | Native C++ | GPU kernel | GPU end to end |
  |---|---|---|---|---|
  | dense symplectic | 433 ms | 214 ms | 31 ms | 46 ms |
  | homodyne | 113 ms | 4.9 ms | 4.0 ms | 15.9 ms |
  | local beam splitter | 10.3 ms | 10.4 ms | 1.6 ms | 13.9 ms |

  On homodyne the GPU kernel only ties native C++, and end to end it is slower. float64 runs at 1/64 of the float32 rate on this GeForce GPU.

- **Batched Fock** (`bench-gpu-fock.json`), complex128, native-loop time ÷ GPU time:

  | Case | GPU kernel | GPU end to end |
  |---|---|---|
  | pure, B = 1 | 0.1–15× | 0.05–4.8× |
  | pure, B = 32 | 1.8–150× | 1.3–6.6× |
  | pure, B = 256 | 3.6–200× | 2.7–7.5× |
  | mixed, B = 1–8 | 0.9–5.2× | 0.7–4.3× |

  Single small states are slower on the GPU.

### Boson sampling and GBS vs Piquasso (`bench-boson-sampling.json`, `bench-gbs.json`)

| Task | PhotoGraphiQ native | Piquasso |
|---|---|---|
| permanent, n = 20 | 8.6 ms | 8.2 ms |
| hafnian, n = 24 | 23 ms | 15 ms |
| loop hafnian, n = 24 | 25 ms | 18 ms |
| torontonian, n = 14 | 15 ms | 8.7 ms |
| exact sampler, 16 modes / 6 photons | 0.19 s | 0.77 s |
| full output distribution, 16 modes / 6 photons | 6.1 s | 1.0 s |

PhotoGraphiQ's matrix-function kernels are comparable to or slower than Piquasso's. The
2026-11-11 run had the permanent at 6.5 ms vs 8.5 ms, within run-to-run variation. The exact
sampler is faster; the full distribution is slower because it loops over patterns in Python.
PNR probabilities agree with Piquasso to ≤ 5.7e-12 relative.

### Threshold-probability accuracy against a 50-digit reference (`threshold_mp_reference.json`)

| Modes, clicks (r = 0.5, η = 0.9) | P | PhotoGraphiQ native | PhotoGraphiQ Python | Piquasso | Native bound |
|---|---|---|---|---|---|
| 24, 12 | 2.7e-9 | 1.7e-7 | 1.7e-7 | 2.2e-8 | 1.4e-4 |
| 28, 14 | 6.1e-11 | 2.8e-5 | 2.5e-5 | 2.2e-6 | 2.0e-2 |
| 32, 16 | 9.0e-13 | 7.5e-3 | 2.9e-3 | 3.0e-4 | 3.6 |

In these cases Piquasso is about 10× more accurate in double precision. PhotoGraphiQ reports a
bound that holds (see the [382-case matrix](gbs-threshold-accuracy.md)) and offers
`precision="auto"`.

### Qubit MBQC vs Graphix 0.4 (`graphix/`)

- **Agreement:** 360/360 random instances (causal flow, gflow, Pauli flow) agree on flow existence, and every certificate passes both verifiers.
- **Speed:** PhotoGraphiQ's search-with-built-in-verification ÷ Graphix search + `check_well_formed` is a median 1.6× for causal flow (0.7–5×) and a median 18× for gflow and Pauli flow (4–160×). Standardisation is about 29× slower, and branch simulation about 7.5× slower at 18 qubits.
- **Verdict:** a correctness-checked reimplementation, not a performance claim.

## 4.9 Figures and their provenance

| Figure | Source | Data |
| --- | --- | --- |
| Architecture A–H | `docs/architecture/figures.json` → `scripts/generate_architecture_figures.py` | 532 code references validated, 7.0 in print check, byte-deterministic |
| Native Gaussian runtime vs modes | `benchmarks/plot_results.py` | `results_20261112/figures_bench-native-gaussian/` |
| Fock per-gate runtime, patterns and memory, thread scaling, accuracy vs cutoff | same | `results_20261112/figures_bench-fock/` |
| CPU vs GPU runtime vs batch | same | `results_20261112/figures_bench-gpu-gaussian/` |
| BS/GBS kernels, probabilities, sampling, memory | the benchmark scripts | `results_20261112/figures_default/` |
| Graphix comparison | the benchmark script | `results_20261112/graphix/*.svg|pdf` |

## 4.10 Notebook validation

`examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb` (14 sections) executed with 0
errors and 12 figures in a clean Python 3.13.15 venv: a non-editable install with
`[dev,autodiff]` plus `nbclient`/`ipykernel`. It runs the small `notebook` configuration by
default (`FULL = False`); large runs need an explicit opt-in. It was then re-executed in place in
the project venv, 0 errors, so the stored outputs match the current source.

## Known limitations of the evidence

- **Single machine:** Windows laptop only; no Linux or other GPUs. The remote CI was not re-run, because the changes are unstaged.
- **CUDA-Q:** no CUDA-Q timing or accuracy exists (execution unverified).
- **GPU float64:** consumer GPU float64 throughput limits the float64 speedups.
- **Heavy experiments:** not re-run (listed in 4.2).

## Reproduction commands

```bash
python scripts/reproduce_manuscript.py --list                 # what runs
python scripts/reproduce_manuscript.py                        # fast validation (minutes)
python scripts/reproduce_manuscript.py --mode full            # this audit's run (~45 min here)
python scripts/compare_recorded_results.py RERUN_JSON_DIR paper_experiments/results/json --atol 1e-12
python benchmarks/plot_results.py --gaussian ... --fock ... --gpu ...
```
