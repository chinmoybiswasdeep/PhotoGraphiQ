# Boson sampling and Gaussian boson sampling

`photographiq.sampling` computes exact output probabilities and draws exact samples for Fock
inputs through passive interferometers (boson sampling, BS) and for Gaussian states measured
by photon-number-resolving (PNR) or threshold detectors (GBS). The matrix functions run in the
compiled `photographiq._native.sampling` kernels when available and fall back to NumPy
implementations of the same algorithms. Conventions: $\hbar=2$, interleaved quadratures,
vacuum $V=I$. The module docstrings of `sampling/boson_sampling.py` and `sampling/gbs.py` hold
the full derivations.

## Boson sampling: permanents

With $a_j^\dagger\mapsto\sum_kU_{kj}a_k^\dagger$, an input occupation $s$ reaches output $t$ with
amplitude $\mathrm{Perm}(U_{t,s})/\sqrt{\prod_is_i!\prod_jt_j!}$, where $U_{t,s}$ repeats row
$k$ $t_k$ times and column $j$ $s_j$ times. Photon number is conserved. Permanents use
Ryser's inclusion–exclusion formula with row/column multiplicities ($\prod_j(c_j+1)$ terms
instead of $2^{N}$; the native kernel walks them in Gray-code order). Hafnians and loop
hafnians use the Glynn-type power-trace formula (Björklund, Gupt, Quesada 2019),
$O(n^{4}2^{n})$.

Samplers: `method="clifford"` is Clifford & Clifford (2018) algorithm B, exact for inputs with at
most one photon per mode; `method="enumerate"` samples from the fully enumerated distribution
(tiny systems, any input).

## Gaussian boson sampling: hafnians and torontonians

With ladder variables $\xi=(a,a^\dagger)$, a state $(d,V)$ maps to $\sigma_Q=W V W^\dagger+I/2$
and the GBS matrix $A=X(I-\sigma_Q^{-1})^{*}$, $X=\begin{pmatrix}0&I\\I&0\end{pmatrix}$, loop
vector $\gamma=\sigma_Q^{-1}\beta$ and vacuum probability
$p_0=2^{m}e^{-d^{T}(V+I)^{-1}d/2}/\sqrt{\det(V+I)}$ (Hamilton et al. 2017; Kruse et al. 2019).

- **PNR detectors:** $P(n)=p_0\,\mathrm{lhaf}(A_{n\oplus n};\gamma_{n\oplus n})/\prod_in_i!$, a
  hafnian when $d=0$. Pure states use the $N\times N$ form
  $P(n)=p_0|\mathrm{lhaf}(B_n;\gamma_n)|^{2}/\prod n_i!$.
- **Threshold detectors** (click = at least one photon): $P(S)=p_0\,\mathrm{ltor}(O_S,\gamma'_S)$
  with $O=I-\sigma_Q^{-1}$ (Quesada et al. 2018), the torontonian when $d=0$.
  `threshold_probability` evaluates the equivalent inclusion–exclusion sum after conditioning
  on the dark modes, with each term entered through `expm1` so it is computed to relative
  accuracy.

**Precision.** The alternating threshold sum still cancels as the number of clicks grows.
`threshold_probability(..., return_error=True)` returns a first-order rounding bound (rigorous
for the conditioning step, calibrated for the alternating sum on a 382-case mpmath reference
matrix, with zero bound violations), warns with `ThresholdPrecisionWarning` when the bound is
large, and `precision="auto"` or an integer number of digits re-evaluates in mpmath
([accuracy study](../validation/gbs-threshold-accuracy.md)).

```python
import math
import numpy as np
from photographiq import sampling

# Boson sampling: P(t|s) = |Perm(U_{t,s})|^2 for collision-free s and t.
U = sampling.haar_unitary(3, rng=2)
s, t = [1, 1, 0], [0, 1, 1]
sub = U[np.ix_([1, 2], [0, 1])]                                   # rows of t, columns of s
print(np.isclose(sampling.output_probability(U, s, t), abs(sampling.permanent(sub)) ** 2))

# GBS: single-mode squeezed vacuum, P(2n) = tanh(r)^(2n) (2n)! / (4^n (n!)^2 cosh r).
r = 0.6
state = sampling.squeezed_interferometer_state([r], np.eye(1))
for n in range(3):
    exact = (math.tanh(r) ** (2 * n) * math.factorial(2 * n)
             / (4**n * math.factorial(n) ** 2 * math.cosh(r)))
    print(2 * n, np.isclose(sampling.pnr_probability(state, [2 * n]), exact))

# Threshold detection, returning the probability and its rounding-error bound.
U4 = sampling.haar_unitary(4, rng=1)
lossy = sampling.squeezed_interferometer_state([0.5, 0.5, 0, 0], U4, transmissivity=0.9)
print(sampling.threshold_probability(lossy, [1, 0, 1, 0], return_error=True))
```

## Relation to CV-MBQC

BS and GBS are not MBQC protocols; they share the Gaussian-state conventions, the native
kernels and the validation infrastructure. A Gaussian state produced by a PhotoGraphiQ
pattern (`result.state`) can be passed directly to the GBS functions.

## Scope and performance

Kernel speed is comparable to, or slower than, Piquasso on single matrices; the exact sampler
is faster and the full output distribution slower (Python loop over patterns). GPU versions
were measured and not shipped ([GPU](../gpu.md),
[Phase 4 benchmarks](../validation/PHASE4_MANUSCRIPT_REPRODUCIBILITY.md)).

## References

H. J. Ryser (1963); P. Clifford and R. Clifford, SODA
(2018); C. S. Hamilton et al., PRL 119, 170501 (2017); R. Kruse et al., PRA 100, 032326
(2019); N. Quesada, J. M. Arrazola, N. Killoran, PRA 98, 062322 (2018); A. Björklund,
B. Gupt, N. Quesada, ACM J. Exp. Algorithmics 24 (2019). User guide:
[sampling](../user-guide/sampling.md).
