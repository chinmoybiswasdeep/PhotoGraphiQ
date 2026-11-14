**R36: Gaussian engine runtime.** Median wall time (error bars: half the IQR) of one `simulate` call on an N-mode line
resource (r = 1) for N = [10, 20, 50, 100, 200]: preparation + CZ entangling only (circles) and with N - 1 q-homodynes and
outcome-dependent displacements (squares; the difference estimates measurement + feed-forward cost), on NumPy
`gaussian`, C++ `native-gaussian` and Piquasso `piquasso`. Before timing, each engine reproduced the NumPy records and
output moments for the same seed (max difference 1.8e-15). At
N = 200: preparation 15.2 s (gaussian), 0.144 s
(native-gaussian), 88 s (piquasso); with measurements
18.5 s, 0.0742 s,
119 s. Timings are final (idle machine declared via PGQ_IDLE_MACHINE=1) (full configuration,
12 logical CPUs, OMP_NUM_THREADS=None, native build
MSVC 19.50.35729.0, OpenMP True); local observations only. Source:
`paper_experiments/11_performance/36_gaussian_runtime.py`.
