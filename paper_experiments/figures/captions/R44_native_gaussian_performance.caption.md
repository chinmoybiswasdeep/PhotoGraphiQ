**R44 - native Gaussian performance with accuracy.** Timings: final (idle machine declared via PGQ_IDLE_MACHINE=1).
Median runtime (error bars: IQR/2; 2 warm-up runs excluded, 11 repeats) of seeded random
noisy Gaussian circuits (R43 gate set incl. thermal loss) on a random mixed n-mode state for the
compiled `native-gaussian` backend, the NumPy `gaussian` backend (dense S V S^T and per-gate
validation), a dense NumPy symplectic propagation written in the script, and Piquasso 8.0.1's
GaussianSimulator executing the whole circuit as one program. (a) n = 4..256 at 200 gates:
native is 3607x faster than the NumPy backend at n = 256 and 2936x faster than
Piquasso at n = 128. (b) gate count at n = 32. (c) sequential postselected homodyne measurements at
n = 32. (d) accuracy of the same runs against the first-principles reference (worst 3.4e-14;
acceptance 1e-10; the script's dense propagation uses the same gate matrices as the
reference, so its ~1e-16 is a consistency floor, not independent evidence). (e) batched beam splitter + homodyne conditioning on n = 16 states
(native OpenMP kernels vs vectorised NumPy; agreement 5.6e-16), one thread. (f) thread count
(OMP_NUM_THREADS, applied to the whole process incl. NumPy BLAS): native batch 4096 takes 2.84e-02 s on
1 and 1.74e-02 s on 12 threads. Single-trajectory native kernels are serial.
Peak memory per configuration (psutil peak working set in a fresh subprocess) is in the CSV.
Seed 20261144. Source: R44 (14_native_gaussian/44_native_gaussian_performance.py). Limitation: timings
are single-machine observations.
