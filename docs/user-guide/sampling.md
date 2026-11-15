# Boson sampling and Gaussian boson sampling

`photographiq.sampling` computes exact output probabilities and draws exact samples for
two models:

- **Boson sampling (BS):** Fock states (single photons or several photons per mode) sent
  through a passive interferometer `U` and counted at the outputs. Probabilities are
  permanents.
- **Gaussian boson sampling (GBS):** a Gaussian state (squeezed, possibly lossy and
  displaced) measured with photon-number-resolving (PNR) detectors, where probabilities
  are hafnians or loop hafnians, or with threshold (click) detectors, where they are
  torontonian-type inclusion-exclusion sums.

The module stands on its own. It does not run through `pg.simulate`, and
`photographiq.sampling.sample` is unrelated to `pg.sample(pattern, shots)`, which runs
shots of an MBQC pattern. All routines run on the CPU, in NumPy or in the compiled C++
kernels. There is no GPU path.

## Conventions

- hbar = 2, interleaved quadratures `(q0, p0, q1, p1, ...)`, vacuum covariance `I`,
  `a_j = (q_j + i p_j)/2`. These are the conventions of
  [Gaussian states](gaussian.md) and `photographiq.states.GaussianState`.
- An interferometer acts as `a_k -> sum_j U_kj a_j` (Heisenberg picture), that is
  `a_j^dag -> sum_k U_kj a_k^dag` on states. `Rotate(phi)` is `U = exp(i phi)`, and
  `BeamSplitter(theta)` is `[[cos, -sin], [sin, cos]]`.
- `Squeeze(r)` is `diag(e^{-r}, e^{r})`, so `r > 0` squeezes q.
- Torontonians use the block ordering `(a_1..a_n, a_1^dag..a_n^dag)`. Piquasso pairs
  `(2i, 2i+1)` instead.
- Patterns are tuples of occupations ordered as `state.nodes` (GBS) or as the rows of
  `U` (BS).

### Interferometers

```python
import numpy as np
from photographiq import sampling as S

U = S.haar_unitary(3, rng=5)                  # Haar-random (Mezzadri QR with phase fix)
d = S.clements_decomposition(U)               # m(m-1)/2 Mach-Zehnder elements
print(len(d.elements), np.abs(d.unitary() - U).max() < 1e-13)
print(d.operations()[0], d.operations()[1][0])
s = S.passive_symplectic(U)                   # interleaved-quadrature symplectic
print(s.shape, np.allclose(s @ s.T, np.eye(6)))
# 3 True
# ('rotate', 0, 1.5122989091447172) beamsplitter
# (6, 6) True
```

`d.operations()` returns `("rotate", mode, angle)` and `("beamsplitter", u, v, theta)`
tuples in application order. The tests check that the product of the Gaussian-backend
gate symplectics equals `passive_symplectic(U)` to `1e-13`. `check_unitary` raises
`ValueError` for a non-unitary matrix (tolerance `1e-10`).

## Exact boson-sampling probabilities

For input occupation `s` and output occupation `t`,

`<t| W(U) |s> = Perm(U_{t,s}) / sqrt(prod s_i! prod t_j!)`,

where `U_{t,s}` repeats row `k` of `U` `t_k` times and column `j` `s_j` times. The
permanent is evaluated from the multiplicities directly, by Ryser's formula over
`prod_j (c_j + 1)` terms; the repeated matrix is never built. Photon number is conserved,
so the amplitude is zero unless `sum t == sum s`.

```python
import numpy as np
from photographiq import sampling as S

bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)          # 50:50 beam splitter
print(S.output_distribution(bs, (1, 1)))                # Hong-Ou-Mandel
U = S.haar_unitary(4, rng=1)
print(S.transition_amplitude(bs, (1, 1), (2, 0)))
print(S.output_probability(U, [1, 1, 0, 0], [0, 1, 1, 0]))
d = S.output_distribution(U, [1, 1, 1, 0])
print(len(d), S.collision_probabilities(d))
print(len(S.output_patterns(4, 3)), S.is_collision_free((1, 0, 2)))
# {(0, 2): 0.4999999999999998, (1, 1): 0.0, (2, 0): 0.4999999999999998}
# (-0.7071067811865474+0j)
# 0.06908404014531996
# 20 {'collision_free': 0.47610419972468104, 'collision': 0.5238958002753178}
# 20 False
```

`output_distribution` enumerates all `C(n+m-1, n)` patterns. It raises `ValueError` if
the probabilities do not sum to 1 within `1e-9`. Amplitudes are checked against an
independent creation-operator expansion and against Piquasso's `PureFockSimulator`
(`tests/sampling/test_sampling_boson.py`).

## Boson samplers

`sample(unitary, inputs, shots, *, rng=None, method="auto", backend="auto")` returns an
integer array of shape `(shots, modes)`. Both methods sample the exact ideal distribution:

| `method` | Algorithm | Inputs |
|---|---|---|
| `"clifford"` | Clifford & Clifford (2018), algorithm B. Each `(k-1)`-minor permanent is evaluated separately. | at most one photon per mode; otherwise `ValueError` |
| `"enumerate"` | sample from the full `output_distribution` | any input, small systems |
| `"auto"` (default) | `"clifford"` for collision-free inputs, `"enumerate"` otherwise | any |

```python
from photographiq import sampling as S

U = S.haar_unitary(4, rng=1)
print(S.sample(U, [1, 1, 1, 0], 3, rng=7).tolist())                    # Clifford & Clifford
print(S.sample(U, [2, 1, 0, 0], 3, rng=7, method="enumerate").tolist())  # full enumeration
# [[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0]]
# [[1, 0, 2, 0], [1, 1, 1, 0], [1, 0, 2, 0]]
```

The tests compare 8000-shot histograms with the exact distribution by a chi-square test.
A negative control checks that samples from distinguishable particles fail the same test.

## Gaussian boson sampling

### Input states and GBS matrices

`squeezed_interferometer_state(squeezing, unitary, transmissivity=1.0, displacement=None)`
applies single-mode squeezers to vacuum, then the interferometer, then per-mode pure loss,
then the coherent displacement `alpha_i`, which adds `(2 Re alpha_i, 2 Im alpha_i)` to the
mean. The result is an ordinary `GaussianState`, and you can pass any `GaussianState`,
for example from `pg.simulate`, to the GBS functions. The tests check covariance and mean
against Piquasso's `GaussianSimulator` (hbar = 2) to `1e-14`.

`gbs_matrices(state)` precomputes `sigma_Q`, the GBS matrix `A = X (I - sigma_Q^{-1})^*`,
`beta`, `gamma = sigma_Q^{-1} beta` and the vacuum probability `p_0`. Pass the result in
place of the state when you evaluate many patterns. `A` is the complex conjugate of
thewalrus' `A`. Both conventions give the same probabilities, but do not combine `gamma`
from one convention with `A` from the other.

```python
import math
import numpy as np
from photographiq import sampling as S

r = 1.0
st = S.squeezed_interferometer_state([r], np.eye(1))     # single-mode squeezed vacuum
print(S.pnr_probability(st, [2]), math.tanh(r) ** 2 / (2 * math.cosh(r)))

U = S.haar_unitary(3, rng=1)
lossy = S.squeezed_interferometer_state([0.4, 0.3, 0.2], U, transmissivity=0.9,
                                        displacement=[0.1, 0, 0])
g = S.gbs_matrices(lossy)
print(g.modes, g.pure, g.displaced, round(g.vacuum_probability, 12))
print(S.pnr_probability(g, [1, 1, 0]))
pure = S.squeezed_interferometer_state([0.4, 0.3, 0.2], U)
print(S.pnr_probability(pure, [1, 1, 0], method="pure"),
      S.pnr_probability(pure, [1, 1, 0], method="mixed"))
try:
    S.pnr_probability(lossy, [1, 1, 0], method="pure")
except ValueError as e:
    print("ValueError:", e)
# 0.18794405337586959 0.1879440533758696
# 3 False True 0.85905730507
# 0.02065294291793659
# 0.02604195986063082 0.026041959860630817
# ValueError: State is not pure; use method='mixed'
```

### PNR probabilities

`pnr_probability(state, pattern, *, method="auto", backend="auto")` evaluates

- `"mixed"`: `P(n) = p_0 lhaf(A_{n+n}; gamma_{n+n}) / prod n_i!`. This form is exact for
  every state, costs `2^N` power-trace terms for `N` photons, and reduces to a hafnian
  when the mean is zero.
- `"pure"`: `P(n) = p_0 abs(lhaf(B_n; gamma_n))^2 / prod n_i!` with `A = B (+) B^*`, at a
  cost of `2^{N/2}` terms. It raises `ValueError` for a mixed state.
- `"auto"`: uses `"pure"` only when the off-diagonal block of `A` is zero to
  `100 eps` relative (`GBSMatrices.pure`), so nearly pure states take the exact mixed
  formula.

The tests validate the PNR probabilities, with both kernels, against several independent
references: closed forms (single-mode and two-mode squeezed vacuum, coherent states), a
dense Fock-space evolution at per-mode cutoff 36 with binomial-thinning loss, and
Piquasso's Gaussian and pure-Fock simulators (`tests/sampling/test_sampling_gbs.py`,
absolute tolerances `1e-14` to `1e-13`).

### Truncated PNR distributions and samples

A PNR distribution has infinitely many patterns, so `pnr_distribution(state, cutoff)`
keeps every pattern with total photon number `sum n < cutoff` and **reports** the omitted
probability. It does not renormalise. `sample_pnr` keeps the omitted mass as an explicit
outcome: a shot that falls there is returned as a row of `-1`, and `overflow` counts such
rows.

```python
import numpy as np
from photographiq import sampling as S

state = S.squeezed_interferometer_state([0.4, 0.3, 0.2], S.haar_unitary(3, rng=1))
d = S.pnr_distribution(state, cutoff=4)
print(len(d.probabilities), d.cutoff, round(d.captured_mass, 12), round(d.truncated_mass, 12))

state = S.squeezed_interferometer_state([0.9, 0.7], S.haar_unitary(2, rng=2), [0.9, 0.9])
res = S.sample_pnr(state, 1000, cutoff=6, rng=4)
print(res.samples.shape, res.overflow, round(res.truncated_mass, 6))
print(S.sample_threshold(state, 4, rng=0).tolist())
# 20 4 0.98380143068 0.01619856932
# (1000, 2) 75 0.067295
# [[0, 1], [0, 0], [0, 0], [0, 0]]
```

Check `truncated_mass` before you treat a truncated distribution as complete. The tests
check it against the Markov bound `<n>/cutoff` and check that it decreases as the cutoff
grows. Both GBS samplers enumerate the distribution before sampling: `sample_pnr`
enumerates `C(cutoff-1+m, m)` patterns and `sample_threshold` enumerates `2^m`. They are
exact but practical only for small systems. No chain-rule or approximate GBS sampler is
implemented.

Known limitation: `sample_pnr` clips a negative truncated mass to 0 but does not clip
negative in-cutoff probabilities. When the exact distribution has zeros and the state
takes the mixed formula, rounding can make some of them slightly negative, and NumPy
then rejects the vector: `sample_pnr` raises
`ValueError("Probabilities are not non-negative")`. One observed case is the two-mode
squeezed vacuum with `r = 0.8`, `transmissivity=1 - 1e-13` and `cutoff=8`, where
`P(2, 5) = -2.3e-18`. `sample_threshold` clips such values and is not affected.

## Threshold detection

`threshold_probability(state, clicks, *, backend="auto", precision=None, return_error=False)`
gives the probability of a 0/1 click pattern. With `S` the clicked modes and `Z` the dark
modes, the code conditions on the dark modes first and then evaluates the
inclusion-exclusion sum over vacuum projections:

`P(S) = P_Z ([S empty] + sum_{T subset S} (-1)^{abs(T)} w_T)`,
`w_T = expm1(-sum log1p(eig E_T) / 2 - e_T^T C_T^{-1} e_T / 2)`.

Each vacuum-projection term `1 + w_T` enters through its deviation `w_T` from 1, which
`expm1` computes to relative accuracy. The sum equals the loop torontonian
`p_0 ltor(O_S, gamma'_S)` with `O = I - sigma_Q^{-1}`. `threshold_probability` does not
call `torontonian` itself.

```python
import math
import numpy as np
from photographiq import sampling as S

bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
tmsv = S.squeezed_interferometer_state([0.8, -0.8], bs)   # two-mode squeezed vacuum
print(S.threshold_probability(tmsv, [0, 0]), 1 / math.cosh(0.8) ** 2)
print(S.threshold_probability(tmsv, [1, 1], return_error=True))
print(S.threshold_distribution(tmsv))
# 0.5590551677322436 0.5590551677322438
# (0.4409448322677561, 6.661338147750938e-16)
# {(0, 0): 0.5590551677322436, (0, 1): 3.103379596260079e-17, (1, 0): 3.103379596260079e-17, (1, 1): 0.4409448322677561}
```

The exact values of `(0, 1)` and `(1, 0)` are 0; the computed `3e-17` is rounding.
`sample_threshold` clips negative rounding-level probabilities to 0 and renormalises
before it samples.

### Precision, error estimate and `ThresholdPrecisionWarning`

The alternating sum cancels more and more as the number of clicks grows and the click
probability shrinks. `return_error=True` also returns the estimate
`eps sum_T abs(w_T) / abs(P)` of the relative rounding error. This estimate is a
heuristic, not a rigorous bound. When it exceeds `1e-6`, the function issues a
`ThresholdPrecisionWarning` (a `RuntimeWarning` subclass). In that case, pass
`precision=<digits>` (an integer `>= 16`) to evaluate the same sum in mpmath. This needs
the `precision` extra: `pip install "photographiq[precision]"`.

```python
import warnings
import numpy as np
from photographiq import sampling as S

state = S.squeezed_interferometer_state(np.full(16, 0.2), S.haar_unitary(16, rng=3), 0.9)
clicks = [1] * 8 + [0] * 8
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    p, err = S.threshold_probability(state, clicks, return_error=True)
print(p, err, caught[0].category.__name__)
ref = S.threshold_probability(state, clicks, precision=30)
print(ref, abs(p - ref) / ref <= err)
# 1.13297519146997e-09 2.580299906299249e-06 ThresholdPrecisionWarning
# 1.1329737713467033e-09 True
```

Notes:

- The mpmath path is pure Python and slow: it computes `2^clicks` multiprecision
  determinants, and it ignores `backend=`.
- It takes the covariance and mean exactly as stored in double precision. The result is
  therefore exact for the stored state, not for an idealised one.
- The tests check the mpmath path against closed forms to `1e-15` relative, and check
  that in the 16-mode, 8-click case above the double-precision result lies within its own
  error estimate (`tests/sampling/test_sampling_gbs.py`). This is the only configuration
  in which the test suite checks the estimate itself.
- `threshold_distribution` and `sample_threshold` can emit the warning once per pattern.
- The estimate ignores rounding in the dark-mode conditioning step. A pattern whose exact
  probability is 0 can therefore come out as a rounding-level value without a warning.
  A result of exactly 0 has estimate `inf` and always triggers the warning, although
  `precision=` cannot help there. Both happen for a lossy squeezed mode next to a vacuum
  mode (`apply_loss(squeezed_interferometer_state([0.5, 0.0], np.eye(2)), 0.9)`): the
  pattern `(0, 1)` gives `-4.9e-17` with no warning, and `(1, 1)` gives exactly `0.0`
  with a warning.

## Loss

`apply_loss(state, transmissivity)` is the per-mode pure-loss channel:
`V -> eta V + (1 - eta) I` on each 2x2 block and `d -> sqrt(eta) d`. The transmissivity
is the intensity transmission, either one scalar or one value per mode, in `[0, 1]`. The
environment is vacuum; this function has no thermal photons. In
`squeezed_interferometer_state`, the code applies loss after the interferometer and
before the displacement. To apply loss after a displacement, call `apply_loss` on the
displaced state, as the Fock-reference test does.

```python
from photographiq import sampling as S

tmsv = S.squeezed_interferometer_state([0.8, -0.8], [[2**-0.5, -2**-0.5], [2**-0.5, 2**-0.5]])
lossy = S.apply_loss(tmsv, [0.9, 0.5])
print(S.threshold_probability(tmsv, [1, 0]) < 1e-15, S.threshold_probability(lossy, [1, 0]))
try:
    S.apply_loss(tmsv, 1.2)
except ValueError as e:
    print("ValueError:", e)
# True 0.14551309357464642
# ValueError: Transmissivities must lie in [0, 1]
```

The tests check loss against binomial thinning of the lossless photon-number
distribution and against Piquasso's `Attenuator`.

## Matrix functions

| Function | Algorithm | Notes |
|---|---|---|
| `permanent(matrix, rows=None, columns=None)` | Ryser with multiplicities | the matrix may be rectangular when multiplicities are given |
| `hafnian(matrix)` | Glynn-type power traces, `O(n^4 2^n)` | symmetric input; diagonal ignored; odd dimension gives 0 |
| `loop_hafnian(matrix, diagonal=None)` | power traces with loop correction | loop weights from `diagonal` or the matrix diagonal |
| `torontonian(matrix)` | subset sum of `1/sqrt(det(I - O_Z))` | `2n x 2n`, block `(a, a^dag)` order |
| `loop_torontonian(matrix, gamma)` | subset sum with the displacement exponent | `gamma` of length `2n` |
| `permanent_by_definition`, `hafnian_by_definition`, `loop_hafnian_by_definition` | defining sums | reference values for small sizes only |

```python
import numpy as np
from photographiq import sampling as S

J = np.ones((6, 6))
print(S.hafnian(J), S.hafnian_by_definition(J))            # (2n-1)!! = 15
print(S.loop_hafnian(J), S.loop_hafnian_by_definition(J))  # telephone number T_6 = 76
print(S.permanent([[1, 2], [3, 4]]))
# (15+0j) (15+0j)
# (76+0j) (76+0j)
# (10+0j)
```

The tests check the permanent, hafnian and loop hafnian against their defining sums
(permanent `n <= 7`, hafnians `d <= 10`, relative `1e-12`) and closed-form identities,
the torontonian and loop torontonian against closed-form products for direct-sum
matrices, and all five against Piquasso's `piquasso._math` routines
(`tests/sampling/test_sampling_linalg.py`). There is no by-definition torontonian.
`repeat_indices(matrix, repetitions)`, which builds the explicit repeated-index matrix,
is not tested.

## Native vs Python kernels

Every matrix function and probability API takes a `backend` argument:

- `"native"`: the compiled C++ kernels in `photographiq._native.sampling` (Eigen, with
  OpenMP when available). Raises `ImportError` if the `sampling` component was not built.
- `"python"`: NumPy implementations of the same algorithms.
- `"auto"` (default): native when compiled, NumPy otherwise. The fallback is silent.

Any other value raises `ValueError`. The native kernels release the GIL. They split the
work into a fixed number of chunks and sum them in a fixed order, so results are bitwise
identical for any thread count (the test checks this for `permanent`, `hafnian` and
`torontonian`). The thread count is a process-wide setting. Native errors
(`std::invalid_argument`, including `PhysicsError`) are raised as `ValueError`.

```python
import numpy as np
from photographiq import native, sampling as S

print(native.components())
kernels = native.require("sampling")                     # photographiq._native.sampling
print(kernels.max_threads() >= 1)
a = np.random.default_rng(0).normal(size=(8, 8)); a = a + a.T
print(abs(S.hafnian(a, backend="native") - S.hafnian(a, backend="python")) < 1e-10)
try:
    S.permanent(np.eye(2), backend="gpu")
except ValueError as e:
    print("ValueError:", e)
# ('gaussian', 'fock', 'sampling')
# True
# True
# ValueError: Unknown backend 'gpu'; use 'auto', 'native' or 'python'
```

`tests/sampling/test_sampling_native.py` checks that both kernels agree at larger sizes
and that results do not depend on the thread count. To build the extension, see the
[native build notes](../development/native-build.md).

To compare speed with Piquasso, run `benchmarks/boson_sampling/bench_boson_sampling.py`
and `benchmarks/gaussian_boson_sampling/bench_gbs.py` (`--smoke` for a quick check). The
method and caveats are in `benchmarks/README.md`. The notebook
`examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb` runs the same code with
validation plots.

## Limits

- Exponential cost: permanents scale as `prod (c_j + 1)` terms, hafnians as `2^n`
  power-trace terms, and threshold probabilities as `2^clicks` subsets. The native
  hafnian and torontonian kernels reject more than 40 pairs or modes.
- Every sampler is exact; none is approximate. The `"enumerate"` boson sampler and
  both GBS samplers enumerate the distribution before they sample; Clifford & Clifford
  does not.
- PNR distributions are truncated by total photon number, and the omitted mass is
  reported.
- Threshold probabilities with many clicks and small `P` lose digits. Use
  `return_error=True` and `precision=` for these cases.
- The module runs on the CPU only.
