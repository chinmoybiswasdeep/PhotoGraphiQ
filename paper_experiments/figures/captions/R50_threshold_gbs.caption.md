**R50 - threshold-detector GBS: cancellation and precision.** 170 random cases (modes
[4, 8, 12, 16, 20], clicks k <= 12, squeezing r in [0.05, 0.2, 0.5, 1.0, 1.5], transmissivity in
[1.0, 0.9, 0.5], half displaced) plus 15 zero-probability cases, all scored against a 40-digit mpmath reference
of the stored state (benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py, rerun; cases with
reference uncertainty > 1e-08 excluded from scoring: 0). (a) Relative error vs
probability: double precision (native, Python, Piquasso 8.0.1) degrades for small P - native errors above 1e-6 in
20 / Python 20 / Piquasso 21 of 170 cases (worst native 2.3e+07); `precision="auto"` (mpmath
in 32 cases) stays within 1.9e-08 of the reference everywhere (rtol 1e-06). (b) The double error tracks eps x
cancellation sum|terms|/|P|, largest for weak squeezing and many clicks: double precision is NOT full precision
here. (c) The reported rounding bound B held in 185/185 cases (native and Python); smallest B/actual
6.2. (d) Error vs mpmath working digits for the 8 highest-cancellation cases. Full click
distributions (m <= 8) sum to 1 within 2.2e-15. Seed 20261150. Source: R50
(16_sampling/50_threshold_gbs.py).
