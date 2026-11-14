**R38: PhotoGraphiQ Gaussian engines versus raw Piquasso.** Left: median wall time of an N-mode Gaussian workload
(per-mode squeeze and rotation, nearest-neighbour CZ chain; N = [5, 10, 20, 40]) as one raw Piquasso program with a single
`GaussianSimulator` call, and through `photographiq.simulate` on the `piquasso` backend (one Piquasso program per gate),
NumPy `gaussian` and C++ `native-gaussian`; right: ratio to raw Piquasso (dotted line: parity). Every engine first
reproduced the raw Piquasso covariance to 1e-09. At N = 40: raw Piquasso
0.0095 s, piquasso backend 0.252 s (x26.51),
gaussian 0.0849 s (x8.94), native-gaussian
0.00581 s (x0.61). Timings are
final (idle machine declared via PGQ_IDLE_MACHINE=1) (full configuration, 12 logical CPUs, native build
MSVC 19.50.35729.0, OpenMP True); local observations, not cross-machine
claims. Source: `paper_experiments/11_performance/38_piquasso_overhead.py`.
