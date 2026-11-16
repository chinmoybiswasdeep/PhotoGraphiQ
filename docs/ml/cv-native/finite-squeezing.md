# Finite squeezing

Physical execution (`mode="physical"`) uses resource modes with finite squeezing $r$
($10\log_{10}e^{2r} = 8.686\,r$ dB). This page states what finite squeezing changes, what
it does not change, and how to train with it.

## What does not change: unconditional means

For flow-corrected CV-MBQC, both the compiled `GaussianProgram` and the `CVCluster` give the
unconditional output mean $X d + t$ with the same $X$ and $t$ as the ideal map, at any
finite $r$. The test `test_penalties_and_finite_squeezing_semantics` checks
`moments(mode="ideal")` against `moments(mode="physical")` to 1e-12.

Consequence: a mean-readout model evaluated with exact moments predicts identically at
every $r$. In R70 the compiled (r = 1) and ideal layered regressors reach the same test MSE
to a relative 4.5e-12 for every seed.

## What changes: output covariance

The covariance is $X V X^T + Y(r)$, with $Y \propto e^{-2r}$ for compiled programs
([compilation](compilation.md)) and $Y = e^{-2r} L_p L_p^T$ for clusters. Finite squeezing
therefore matters through:

1. **Shot noise.** An $S$-shot mean prediction has expected squared error
   $(\mu - y)^2 + w^T C_{qq}(r)\,w / S$.
2. **Covariance readouts.** `readout="moments"` reads q variances, which contain $Y$.
3. **The physical homodyne head.** $\Phi((\mu_q - b)/\sigma_q)$ depends on
   $\sigma_q(r)$.
4. **Kernels.** Outputs become mixed, which lowers purity and changes the kernel
   ([kernels](kernels.md)).

Conditional (single-trajectory) means still depend on the outcomes at finite $r$. Only the
ensemble mean is $r$-independent.

## Training with finite resources (R72)

R72: linear-regression split, 2 x 4 cluster, q readout of both outputs, affine head.
Expected test MSE is computed exactly from the moments, and it agrees with 200 sampled
shot evaluations (largest |z| = 1.36).

| r | A ideal-trained, S = 10 / 100 / 1000 | B shot-trained (S = 100), at S = 100 | C noise-penalised (lambda = 0.05), at S = 100 | Added noise A / C |
| --- | --- | --- | --- | --- |
| 0.25 | 0.242 / 0.0246 / 0.00284 | 0.0109 | 0.0177 | 1.68 / 1.31 |
| 0.5 | 0.192 / 0.0195 / 0.00234 | 0.0106 | 0.0153 | 1.02 / 0.83 |
| 1.0 | 0.143 / 0.0147 / 0.00185 | 0.0087 | 0.0128 | 0.375 / 0.330 |
| 2.0 | 0.118 / 0.0122 / 0.00161 | 0.0075 | 0.0119 | 0.051 / 0.050 |

Established (asserted): A falls with $r$ for each $S$ and with $S$ for each $r$, and C's
added noise is at most A's at every $r$. Reported, not asserted: shot training (B) trades
bias (bias MSE 9.2e-4 to 1.4e-3, against 4.3e-4 for A) for lower output variance, and lowers the
expected MSE at S = 100.

The R72 homodyne-head classifier on blobs has single-shot accuracy $E[P(\text{correct})]$
between 0.6765 and 0.6770 at every $r$ from 0.25 to 2, both trained once in the ideal
limit and retrained at each $r$. The fraction of test points with
$P(\text{correct}) > 0.5$ is 0.975 throughout. The bound comes from the coherent-state
encoding: a Gaussian unitary amplifies signal and input vacuum noise together. At
r = 0.25 the encoding variance was 118 and the resource variance 0.61. Retraining can
suppress only the resource part.

## API summary

- `mode="physical"`, plus `squeezing=r` (required for `GaussianProgram`; for `CVCluster` it
  overrides the cluster's own `squeezing`).
- `shots=S` for shot-limited training and `predict(..., shots=S, seed=s)` for
  evaluation.
- `penalty="noise", penalty_weight=lambda` penalises $\operatorname{tr}Y/2n$.
- `resource_report(X)["added_noise_per_quadrature"]` reports it after training.

## Limitations

- Finite squeezing is the only imperfection modelled. There is no optical loss,
  detector inefficiency or electronic noise.
- Every resource mode has the same $r$.
- Exact moments do not see finite-squeezing noise in mean readouts. Use shots, the noise
  penalty or a covariance-dependent loss to make training aware of it.
