**Native backend execution.** The compiled two-mode pattern of demo 04 executed by the C++
`native-gaussian` backend and by the NumPy `gaussian` backend with identical seeds and
coherent inputs. (a) The execution stage of one native trajectory (recorded outcomes `m`).
(b) Largest per-measurement outcome difference over 20 seeded runs: 8.9e-16
(identical trajectories up to float rounding; points at 1e-18 are exact zeros). (c) Median
wall time per `simulate` call: native 19.4 ms, NumPy
51.3 ms. Timings are provisional (machine shared with other jobs,
20 runs, includes Python dispatch). Native components available:
gaussian, fock, sampling.

Source: `scripts/generate_demo_figures.py` (`10_native_backend_execution`).
