# GBS threshold-probability accuracy

This page records how accurate `photographiq.sampling.threshold_probability` is in double
precision. It covers the native and Python backends, the rounding bound `B` that the function
reports, the opt-in `precision="auto"` mode, and Piquasso 8.0.1 on the same states. Data:
`benchmarks/gaussian_boson_sampling/results/threshold_reference_matrix_20261111-233720.json`
(seed 2026, i5-12450HX, 12 logical CPUs, Windows, total run time 25 min).

## Method

`benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py` builds the grid below.

* Modes {4, 8, 12, 16, 20, 24}. Clicks k = 1 .. min(modes, 16).
* For every (modes, k), all five squeezings r in {0.05, 0.2, 0.5, 1.0, 1.5}. Each case gets a
  seeded random transmissivity in {1.0, 0.9, 0.5}, displacement on or off
  (alpha_j = 0.4 u e^{2 pi i v}), a seeded Haar interferometer and a random k-click pattern.
  This gives 360 cases.
* 22 zero-probability cases. One input mode is vacuum, the interferometer does not touch it,
  and it clicks. These cases use k in {1, 3, min(m, 8), min(m, 12)}.

The reference is a 40-digit mpmath sum of the stored double-precision covariance and mean,
taken exactly. It conditions on the dark modes (a Schur complement in mpmath) and then
evaluates `sum_T (-1)^|T| 2^|T| exp(-e_T^T C_T^-1 e_T / 2) / sqrt(det C_T)`. A depth-first walk
over the 2^k subsets extends one Cholesky factor per added mode. This code is separate from
PhotoGraphiQ's own `precision=` path. Piquasso computes its own covariance, which differs from
PhotoGraphiQ's by up to 1.1e-14, so it is compared with a reference for its own state. The
reference uncertainty is estimated as `10^(3-40) sum|terms| / |P|`. It is below 1e-10 in
356 of the 360 random cases. It is 3e-8 to 8e-8 in three cases and 3e-2 in one case
(20 modes, 16 clicks, r = 0.05, P = 2e-31, cancellation 3e35). In those four cases the
reference is less accurate than the 45- to 51-digit `auto` results, and none of them changes
a conclusion below.

The double-precision bound `B` is the absolute bound of the `photographiq.sampling.gbs`
module doc. The bound **holds** when `|double - ref| <= B + ref uncertainty`.

## Results by modes and clicks

Each cell shows the maximum relative error for native / Python / Piquasso. The number in
brackets counts the cases whose native error is above 1e-6.

| modes | k=1-4 | k=5-8 | k=9-12 | k=13-16 |
|---|---|---|---|---|
| 4 | 2e-11 / 5e-12 / 6e-10 (0/20) | - | - | - |
| 8 | 2e-11 / 3e-11 / 6e-10 (0/20) | 2e-05 / 4e-05 / 5e-04 (3/20) | - | - |
| 12 | 1e-10 / 2e-10 / 3e-09 (0/20) | 2e-06 / 6e-06 / 1e-05 (1/20) | 2e+08 / 3e+07 / 3e+04 (7/20) | - |
| 16 | 2e-09 / 4e-09 / 4e-08 (0/20) | 6e-03 / 6e-04 / 4e-02 (2/20) | 4e+08 / 8e+07 / 1e+06 (8/20) | 4e+09 / 1e+11 / 3e+11 (10/20) |
| 20 | 1e-10 / 3e-11 / 6e-09 (0/20) | 3e-02 / 5e-03 / 4e-02 (3/20) | 3e+07 / 8e+06 / 2e+02 (8/20) | 3e+16 / 3e+15 / 3e+09 (12/20) |
| 24 | 3e-08 / 1e-09 / 7e-08 (0/20) | 2e-02 / 4e-03 / 5e-01 (4/20) | 4e+05 / 6e+04 / 2e+07 (10/20) | 3e+11 / 1e+11 / 9e+11 (12/20) |

| clicks k | cases | median P | native max rel | Python max rel | Piquasso max rel | native / Python / Piquasso > 1e-6 | median B/actual (native) | min B/actual (native, Python) |
|---|---|---|---|---|---|---|---|---|
| 1 | 30 | 1.5e-03 | 1.0e-14 | 1.0e-14 | 6.9e-13 | 0 / 0 / 0 | 4.8e+02 | 28, 34 |
| 2 | 30 | 2.7e-04 | 9.1e-13 | 4.8e-13 | 9.5e-12 | 0 / 0 / 0 | 1.3e+03 | 5.2, 12 |
| 3 | 30 | 5.8e-05 | 1.4e-10 | 3.4e-11 | 6.4e-09 | 0 / 0 / 0 | 3.8e+02 | 18, 23 |
| 4 | 30 | 1.2e-05 | 3.1e-08 | 3.6e-09 | 7.1e-08 | 0 / 0 / 0 | 4.9e+02 | 15, 14 |
| 5 | 25 | 6.0e-07 | 3.2e-07 | 8.5e-07 | 1.5e-05 | 0 / 0 / 3 | 1.1e+03 | 6.6, 14 |
| 6 | 25 | 3.7e-07 | 1.0e-05 | 1.4e-05 | 1.3e-04 | 1 / 1 / 3 | 8.1e+02 | 8.4, 23 |
| 7 | 25 | 5.8e-08 | 8.6e-04 | 2.5e-04 | 3.0e-03 | 4 / 2 / 7 | 1.7e+02 | 14, 16 |
| 8 | 25 | 2.1e-08 | 2.5e-02 | 4.7e-03 | 4.8e-01 | 8 / 7 / 8 | 1.8e+02 | 8.9, 21 |
| 9 | 20 | 6.8e-09 | 2.0e+03 | 7.8e+02 | 1.5e+04 | 6 / 6 / 7 | 3.0e+02 | 12, 18 |
| 10 | 20 | 1.2e-08 | 3.7e+05 | 6.1e+04 | 1.2e+06 | 9 / 7 / 8 | 1.7e+02 | 9.9, 35 |
| 11 | 20 | 1.6e-09 | 3.0e+07 | 7.9e+06 | 5.1e+05 | 9 / 8 / 8 | 4.8e+02 | 20, 45 |
| 12 | 20 | 3.3e-09 | 4.4e+08 | 7.9e+07 | 1.7e+07 | 9 / 9 / 9 | 6.9e+02 | 26, 190 |
| 13 | 15 | 3.1e-09 | 2.1e+11 | 4.3e+10 | 4.3e+11 | 8 / 7 / 7 | 6.0e+02 | 47, 220 |
| 14 | 15 | 2.9e-09 | 3.3e+10 | 2.2e+09 | 4.4e+10 | 8 / 8 / 7 | 1.5e+03 | 76, 400 |
| 15 | 15 | 4.1e-09 | 2.6e+11 | 1.1e+11 | 8.7e+11 | 9 / 8 / 7 | 1.4e+03 | 68, 130 |
| 16 | 15 | 4.7e-10 | 3.1e+16 | 2.9e+15 | 9.6e+02 | 9 / 9 / 8 | 3.0e+03 | 150, 1100 |

## Results by squeezing

| squeezing r | k=1-4 | k=5-8 | k=9-12 | k=13-16 | median native / Python / Piquasso | median cancellation |
|---|---|---|---|---|---|---|
| 0.05 | 3e-08 / 4e-09 / 7e-08 (0/24) | 3e-02 / 5e-03 / 5e-01 (9/20) | 4e+08 / 8e+07 / 2e+07 (16/16) | 3e+16 / 3e+15 / 9e+11 (12/12) | 2.9e-06 / 1.5e-06 / 3.3e-05 | 4.0e+12 |
| 0.2 | 1e-10 / 2e-10 / 4e-10 (0/24) | 2e-05 / 4e-06 / 9e-06 (4/20) | 1e+00 / 1e-01 / 5e-01 (14/16) | 6e+05 / 9e+04 / 2e+05 (12/12) | 1.6e-08 / 1.1e-08 / 7.3e-08 | 3.7e+09 |
| 0.5 | 1e-11 / 5e-12 / 2e-11 (0/24) | 2e-08 / 8e-10 / 4e-09 (0/20) | 4e-06 / 2e-06 / 6e-06 (3/16) | 7e-03 / 9e-03 / 6e-04 (10/12) | 5.4e-11 / 2.1e-11 / 2.6e-11 | 3.2e+06 |
| 1.0 | 2e-13 / 9e-14 / 9e-14 (0/24) | 2e-11 / 2e-11 / 3e-12 (0/20) | 9e-10 / 2e-10 / 1e-10 (0/16) | 1e-08 / 1e-08 / 5e-09 (0/12) | 3.2e-13 / 2.4e-13 / 8.2e-14 | 1.2e+04 |
| 1.5 | 5e-14 / 3e-14 / 2e-14 (0/24) | 4e-13 / 5e-13 / 4e-13 (0/20) | 5e-13 / 1e-12 / 6e-13 (0/16) | 8e-12 / 1e-12 / 4e-12 (0/12) | 1.9e-14 / 1.5e-14 / 1.3e-14 | 3.9e+02 |

Here cancellation is `P_Z sum_T (1 + w_T) / |P|` from the reference. Transmissivity and
displacement have little effect next to r and k. The native error counts above 1e-6 are
30 / 28 / 22 for eta = 0.5 / 0.9 / 1.0, and 49 / 31 for undisplaced / displaced.

### Where double precision fails

80 of the 360 random cases (native) and 72 (Python) have a relative error above 1e-6. All of
them have small squeezing and many clicks, so P is tiny next to the O(1) terms of the
alternating sum:

* r = 0.05: from k = 7 (8 modes) or k = 6 (16 modes) upwards.
* r = 0.2: from k = 7-10, depending on the mode count.
* r = 0.5: from k = 10-13 at 16-24 modes.
* r >= 1.0: never in this grid. The worst case is 1e-8.

The native error is 0.03-4 times `cancellation x eps` (median 0.03, 95th percentile 0.74).
No case with a cancellation below 1e11 failed, and every failure had a cancellation of at least
2.2e11. Past about 1e16 the double result carries no information (relative errors up to 3e16).

## Bound validity

**No violations.** Over all 382 cases on both backends, `|double - ref| <= B + ref uncertainty`
held every time. Four cases meet the bound only with the reference uncertainty included. In
each of them the double value is exactly 0 with `B = 0`: a vacuum mode clicks alone, and
`E = 0` exactly. The 40-digit reference returns about 1e-42 there, which is inside its own
uncertainty of about 1e-38. PhotoGraphiQ's mpmath path returns exactly 0. These are not
defects, so **no bound fix was needed**.

How loose the bound is (`B / actual` over the random cases):

| backend | min | 5th percentile | median | 95th percentile | max |
|---|---|---|---|---|---|
| native | 5.2 | 19 | 620 | 1.1e5 | 8.7e6 |
| Python | 12 | 30 | 1100 | 9.8e4 | 1.3e7 |

The bound is tightest at small squeezing and few clicks (median 78-91 at r <= 0.2). It is
loosest at large squeezing (median 4.2e3 at r = 1.0 and 3.5e4 at r = 1.5), where the
conditioning term dominates and the actual error stays near eps.

Warnings:

* Every case with an actual error above 1e-6 raised `ThresholdPrecisionWarning`: 80/80
  native and 72/72 Python. **No warning was missed.**
* 37 native and 45 Python warnings are false alarms. In these cases the actual error is
  between 3.5e-11 and 8.2e-7, while `B/|P|` is between 1.1e-6 and 6.4e-4. These false alarms
  are the cost of the bound's 5-1000x margin.
* All 22 zero-probability cases warn. 20 warn "consistent with zero". 2 warn "negative
  beyond": eta = 0.9 makes the vacuum block `0.9 + 0.1` round below 1, so the stored state
  is unphysical by -5.5e-17 and -1.8e-17, and the reference agrees.
* The largest absolute error in a zero case is 5.2e-15, against a bound of at most 7.0e-13.

## Piquasso comparison

Piquasso 8.0.1, `get_threshold_detection_probability`, was compared with the reference for
its own state. Piquasso's state matches PhotoGraphiQ's to 1.1e-14, and its reference P
differs from PhotoGraphiQ's by up to 8e-9 relative.

* Piquasso has an error above 1e-6 in 82 of 360 cases, against 80 for native and 72 for
  Python. All three fail in the same regime.
* In the 121 cases where the native error is above 1e-9, Piquasso's error is on average
  (geometric mean) 0.25x native and 0.67x Python. Piquasso is more accurate than native in
  66 of these cases and more accurate than Python in 43.
* At moderate squeezing Piquasso is ahead. At r = 0.5 and k = 13-16, the maximum error is
  6e-4 for Piquasso, against 7e-3 native and 9e-3 Python. This matches the earlier
  32-mode / 16-click result: 3.0e-4 Piquasso against 7.5e-3 native and 2.9e-3 Python.
* At small squeezing Piquasso is behind. At r = 0.05 the median error is 3.3e-5 for
  Piquasso, against 2.9e-6 native and 1.5e-6 Python. At r = 0.05 and k = 5-8 the maximum is
  0.5, against 3e-2.
* Piquasso returns exactly 0.0 for 14 nonzero probabilities (P from 2e-31 to 3e-16) and
  reports no error estimate or warning. PhotoGraphiQ warns in every one of those cases.
* In the zero cases Piquasso's absolute error is at most 1.2e-15.

## Adaptive precision (`precision="auto"`)

```python
from photographiq.sampling import threshold_probability
from photographiq.sampling.gbs import ThresholdInfo

p = threshold_probability(state, clicks, precision="auto")          # rtol = 1e-6
p, info = threshold_probability(state, clicks, precision="auto", rtol=1e-8, return_info=True)
info.path, info.digits, info.error, info.double_error  # "double" | "mpmath", ...
```

`precision="auto"` first evaluates in double precision. It recomputes in mpmath only when
`B > rtol |P|`, with `ceil(log10(X / max(|P|, B)) + log10(1/rtol)) + 10` digits (20..400),
where `X = P_Z (2^k + sum_T |w_T|)` is the term scale of the mpmath sum. If mpmath's own
estimate still exceeds `rtol`, it runs once more with more digits. In `auto` mode `rtol` is
also the warning threshold. `return_info=True` returns `(value, ThresholdInfo)` and cannot be
combined with `return_error`. Existing calls and return types are unchanged. A missing mpmath
raises `ImportError` naming `photographiq[precision]` on every `auto` call, not only on calls
that need it, so the behaviour does not depend on the data. The mpmath path itself now
extends Cholesky factors incrementally, `O(k^2)` per subset: 12 clicks went from 104 s to
1.3 s.

Results on the matrix:

* 133 of 382 cases went to mpmath (25-51 digits) and 249 stayed in double.
* The maximum relative error against the reference is 3.1e-8 on the double path and 7.6e-10 on
  the mpmath path, counting only the 109 mpmath cases whose reference uncertainty is below
  1e-10. This is below `rtol = 1e-6` in every case.
* No random case warned.
* The actual error never exceeded the mpmath estimate.
* Zero cases with `B = 0` stay in double and return the exact 0. The other zero cases return
  mpmath values of 1e-50 or less, or the genuinely negative value of the unphysical input,
  and warn.

Cost on one thread (median / max for the `auto` row; medians over all cases for double and
Piquasso):

| k | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| auto, mpmath cases (s) | 0.91 / 1.4 | 1.5 / 1.9 | 2.9 / 3.3 | 6.4 / 7.4 | 13 / 21 | 28 / 59 | 56 / 120 | 110 / 230 |
| native double (s) | 0.002 | 0.003 | 0.005 | 0.010 | 0.022 | 0.053 | 0.12 | 0.25 |
| Piquasso (s) | 0.001 | 0.001 | 0.002 | 0.004 | 0.009 | 0.015 | 0.038 | 0.053 |

## Recommendations

* Double precision, on either backend, is reliable when `B/|P| <= 1e-6`. When it does not
  warn, the error was always at least 5x below the reported bound. Treat warned results as
  unreliable or use `precision="auto"`.
* Use `precision="auto"` for small squeezing (r <= 0.5) with more than about 6-10 clicks.
  Prefer it to a fixed `precision=<digits>`, because it chooses the digits from the measured
  cancellation and costs nothing when the double result is good enough.
* Prefer the Python backend when accuracy matters more than speed in the warned regime. Its
  failures are slightly fewer (72 against 80). Its error is typically lower than native at
  large k, and its bound is about 2x looser.
* Do not rely on Piquasso for small probabilities. It is a little more accurate at moderate
  squeezing, but it returns 0.0 or wrong values without any indication.

## Remaining limits

* The sum part of `B` is first order and calibrated (factor `max(1, 16/sqrt(2^k-1))`), not
  proven. It held here and in the earlier 6140 random states, with a margin of at least 5.2x.
* The bound is very loose at large squeezing (up to 1e7x), so it raises false alarms near
  the 1e-6 threshold.
* The mpmath estimate `P_Z mp.eps sum|terms|` ignores the error growth of the Cholesky
  factorization. The `+10`-digit margin covers this in practice: the error never exceeded the
  estimate.
* mpmath is pure Python and single threaded. 16 clicks take 1-4 min per probability (median 110 s), and
  `threshold_distribution` / `sample_threshold` do not take `precision`.
* The grid stops at 24 modes and 16 clicks. The double kernel's cost grows as 2^k, and the
  cancellation grows with k at fixed squeezing.
* Zero-probability patterns can only be confirmed as consistent with zero. A result like
  `|P| <= B` cannot reach a relative tolerance.
