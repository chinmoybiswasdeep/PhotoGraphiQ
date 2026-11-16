# Gradients

## Options

`gradient=` on `CVRegressor` / `CVClassifier`:

| Value | Method | Used for |
| --- | --- | --- |
| `"auto"` (default) | `"complex-step"` for Gaussian programs (`GaussianProgram`, `CVCluster`); `"finite-difference"` for non-Gaussian (Fock-engine) programs and for clusters with trainable rung weights | |
| `"complex-step"` | $\partial_j f = \operatorname{Im} f(x + i h e_j)/h$ with $h = 10^{-20}$ (`complex_step_gradient`) | Gaussian moments, exact to rounding (no subtractive cancellation). Rejected for non-Gaussian programs, whose Fock execution is real-valued |
| `"finite-difference"` | Central differences with step `fd_step` (default 1e-6) (`central_difference`) | Non-Gaussian programs; trainable `CVCluster` rung weights |
| callable `g(vector) -> array` | User supplied; recorded as `"user"` in `history.settings` | Any |

The standalone functions `complex_step_gradient(objective, x, step=1e-20)` and
`central_difference(objective, x, step=1e-6)` are exported.

## Why complex steps are exact here

Every operation in the Gaussian moment code is real-analytic in the parameters. Gate
matrices are written with plain arithmetic, branch decisions use real parts, and the shot
estimator uses an unpivoted Cholesky written the same way. A complex perturbation
therefore propagates through `channel`, `moments`, `features` and the losses, including
the physical homodyne-head loss, which depends on the output covariance and hence on $r$.
The shot estimator with fixed normals is a deterministic function of the parameters, so
its complex-step gradient is the exact pathwise gradient of that estimator. It is not a
derivative of sampled outcomes.

Exception: `CVCluster` rung weights enter the CV-flow certificate, which is solved for real
weights, so complex-step derivatives cannot see them. With `trainable_weights=True`,
`gradient="auto"` therefore uses central differences, and `gradient="complex-step"` raises
`ValueError` ([tutorial 3](examples.md#3-trainable-homodyne-angles)).

## Verification

From `tests/ml/cv_native/test_cv_learning.py`:

- Closed form: $\partial_\theta \langle q\rangle = -d\sin\theta$ for a rotation on a
  q-displaced input, to 1e-14.
- Complex step agrees with central differences on physical homodyne losses of compiled
  programs and clusters at r = 0.3 and 1.5 (rtol and atol 1e-6).
- The gradient is finite near the homodyne-angle margin and consistent with central
  differences there.

From R73 (`R73_gradients.json`):

| Check | Result |
| --- | --- |
| Closed form, chain of L = 1 to 8 rotations | max error 2.2e-16 |
| Complex step vs central differences (h = 1e-6), 180 cases: compiled layered programs (L = 1 to 6) and 2 x (L+1) clusters at r = 0.25, 1, 2 | max relative difference 8.8e-9 |
| Step sweep, h = 1e-1 to 1e-12 | central-difference error is smallest (2.8e-11) at h = 1e-5 and grows to 2.2e-4 at h = 1e-12; complex-step error falls to 3.2e-16 and stays there |
| 1 x 3 cluster, $\varphi$ = 0 to 5 (down to 0.050 rad from the singular angle) | finite; relative difference from central differences at most 4.2e-7 |
| Shot gradients (S = 100, 40 repeats, 7 components): CRN pathwise vs central differences with fresh shot noise | CRN standard deviation 0.0099 to 0.15; independent-noise standard deviation 1.6 to 2.0 |

R73 also records the variance of the first gradient component over 40 random
initialisations against depth. It grows with depth in this range: from 0.74 (L = 1) to
802 (L = 6) for compiled programs, and from 445 to 8.1e11 for clusters. This is
descriptive and unnormalised, and it is not a barren-plateau statement.

## Cost

Complex step costs one complex evaluation per parameter. Central differences cost two real
evaluations per parameter. On Fock programs each evaluation runs the engine on every
sample.

## Limitations

- No adjoint or backpropagation. Cost is linear in the number of parameters.
- No parameter-shift rule is assumed for any CV gate.
- Non-Gaussian gradients carry central-difference truncation and Fock-cutoff error.
  Choose `fd_step` against the cutoff (R71 used 1e-5).
