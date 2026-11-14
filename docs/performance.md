# Performance and allocation

This page gives the cost model of each representation and the current measured numbers.
All measurements come from the 2026-11-12 reproduction run in
`benchmarks/results_20261112/` (one Windows 11 laptop: Intel i5-12450HX, 12 logical CPUs,
OpenMP 12 threads; RTX 4050 Laptop GPU with CuPy 14.2.0; Python 3.12.14). Every comparison
runs the same task on the same inputs and checks that the outputs agree before timing.
Timings of small cases vary by about ±30 % between runs; accuracy fields are deterministic.
Full tables and methodology: [Phase 4 report](validation/PHASE4_MANUSCRIPT_REPRODUCIBILITY.md).

## Cost model

**Gaussian.** Covariance storage is $O(n^{2})$ for $n$ modes. The NumPy `gaussian` backend
forms the full symplectic matrix and re-validates the state ($O(n^{3})$ per gate); the native
`native-gaussian` engine updates only the rows of the $k$ affected modes ($O(nk)$). Homodyne
conditioning is a rank-one Schur update. Physicality checks and dense factorizations are
$O(n^{3})$.

**Fock.** For $m$ modes and exclusive total-photon cutoff $c$ the dimension is
$D=\binom{m+c-1}{m}$, not $c^{m}$: two modes at $c=20$ give $D=210$, four modes $D=8855$. A
complex128 vector needs $16D$ bytes; a density matrix needs $16D^{2}$ bytes before gate
workspaces. Default guards: `max_dimension` $=10^{6}$ (pure) and `max_matrix_bytes`
$=256$ MB (mixed). Observables, partial traces and differentiable dense evolution allocate
additional matrices.

**Compilation.** Each compiled one-mode Gaussian gate uses four or five teleportation steps;
product-formula Kerr synthesis can create many Gaussian steps and nonlinear injections for a
single logical gate.

## Measured: native engines

| Comparison | Result |
|---|---|
| Native Gaussian vs NumPy, single beam splitter, $n=8/64/256$ | 43× / 329× / 7506× (algorithmic $O(n^{3})\to O(nk)$, not C++ alone) |
| Native Gaussian vs NumPy, single homodyne, $n=8/64/256$ | 11× / 27× / 133× |
| Native Gaussian vs vectorized NumPy, batched gates / homodyne | 1.2–2.6× / 3.8–10× (like-for-like) |
| Native Fock vs Piquasso, per gate (73 matched cases) | median 44×, range 1.2× to 4.3×10⁴ |

| Pattern | Native pure | Piquasso pure | Native mixed | Piquasso mixed |
|---|---|---|---|---|
| wire, 4 modes, cutoff 16 | 2.6 ms | 51 ms | 33 ms | 1.21 s |
| wire, 8 modes, cutoff 16 | 4.8 ms | 90 ms | 58 ms | 2.31 s |
| network, 3 modes, cutoff 10 | 1.7 ms | 40 ms | 43 ms | 0.89 s |

Accuracy versus cutoff (two-mode Gaussian circuit against exact Gaussian photon numbers;
both engines identical): 9.8e-4 at cutoff 8, 7.8e-7 at 16, 5.0e-10 at 24, 2.7e-13 at 32.

## Measured: GPU (CuPy)

| Batched Gaussian, $n=64$, $B=256$, float64 | NumPy | Native C++ | GPU kernel | GPU end to end |
|---|---|---|---|---|
| dense symplectic | 433 ms | 214 ms | 31 ms | 46 ms |
| homodyne | 113 ms | 4.9 ms | 4.0 ms | 15.9 ms |
| local beam splitter | 10.3 ms | 10.4 ms | 1.6 ms | 13.9 ms |

Batched Fock gates (complex128, native loop ÷ GPU): 1.8–150× (kernel) and 1.3–6.6× (end to
end) at $B=32$; single small states are slower on the GPU (down to 0.05×). GeForce GPUs run
float64 at 1/64 of the float32 rate, which caps these speedups. Details: [GPU](gpu.md).

## Measured: boson sampling, GBS and flows

| Task | PhotoGraphiQ native | Piquasso |
|---|---|---|
| permanent, $n=20$ | 8.6 ms | 8.2 ms |
| hafnian, $n=24$ | 23 ms | 15 ms |
| torontonian, $n=14$ | 15 ms | 8.7 ms |
| exact sampler, 16 modes / 6 photons | 0.19 s | 0.77 s |
| full output distribution, 16 modes / 6 photons | 6.1 s | 1.0 s |

The matrix-function kernels are comparable to or slower than Piquasso's. Qubit flow search is
a correctness-checked reimplementation (360/360 agreement with Graphix), not a speed claim.

## Practical advice

- Use a Gaussian backend whenever the pattern is Gaussian; prefer `native-gaussian` beyond a
  few tens of modes.
- For Fock work start with one- or two-mode examples, inspect energy, boundary population and
  retained norm, and raise the cutoff until the reported observables converge.
- Use the GPU only for batches of states or very large single Fock states.
- Record the elapsed time and environment with your results (`paper_experiments/common.py`
  writes a provenance record); do not transfer single-machine timings to other hardware.

## Re-running

```sh
python scripts/reproduce_manuscript.py --mode full     # all benchmarks + safe experiment subsets
python benchmarks/gaussian/bench_native_gaussian.py    # one benchmark; see benchmarks/README.md
```
