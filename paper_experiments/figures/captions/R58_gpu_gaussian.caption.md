**R58 - batched Gaussian kernels on CPU (NumPy, native C++/OpenMP) and GPU (CuPy).** Timings:
final (idle machine declared via PGQ_IDLE_MACHINE=1) (median of 21 calls after 2 warm-ups). Device: NVIDIA GeForce RTX 4050 Laptop GPU (CC 8.9, CuPy 14.2.0, CUDA runtime 13.2); GPU execution verified for every GPU row (execution record
'cuda:0' and device-memory pointer check; no CPU fallback counted). Inputs: B copies of an n-mode
squeezed CZ chain with thermal loss (modes [4, 16, 32, 64], B [64, 1024, 16384]); ops: beam splitter on two modes,
a different random symplectic per state on all modes, homodyne conditioning on a given outcome
(angle 0.4, efficiency 0.9), purity. (a) Largest configuration: host data preparation (shared by
all engines, not included in any engine time), NumPy and native host times, and the GPU stages host->device copy, kernel (CUDA events,
device-resident data, including validity-check syncs) and device->host copy, GPU end-to-end (NumPy in/out), and the explicit float32 kernel. (b) NumPy time over GPU kernel time versus batch size (float64); end-to-end
ratios including transfers are in the CSV. Accuracy vs the NumPy float64 reference: float64
engines <= 3.8e-16, float32 GPU <= 7.0e-07 (relative to max |entry|). Source: R58.
