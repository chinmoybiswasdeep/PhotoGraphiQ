# Non-Gaussian programs (`NonGaussianProgram`, experimental)

`NonGaussianProgram` extends `GaussianProgram` with two gates:

| Method | Gate |
| --- | --- |
| `kerr(mode, kappa)` | $e^{i\kappa n^2}$ |
| `cubic_phase(mode, gamma)` | $e^{i\gamma q^3/6}$, $q = a + a^\dagger$ |

All Gaussian gates (including bounded squeezing and displacement) are still available.

## Execution

Non-Gaussian programs run on PhotoGraphiQ's native finite-Fock engine (`native-fock`). No
Gaussian approximation is substituted.

- `run(params, x, feature_map, cutoff, *, backend="native-fock", **kw)` encodes $x$ as a
  product `GaussianInput`, executes the gate-level pattern and returns the PhotoGraphiQ
  `Result`.
- `fock_moments(params, x, feature_map, cutoff, modes)` returns the q and p means and
  variances on `modes`, and the minimum retained norm. The engine renormalises after
  every operation, so $1 - $ norm bounds the probability truncated per step.
- `cutoff` is an exclusive total-photon cutoff. The Fock engine raises `ValueError`
  ("increase cutoff") when |1 - retained norm| exceeds its `norm_tolerance` (default
  1e-3). Smaller truncation lets the run continue, and the norm is reported. `run` silences the
  engine's `RuntimeWarning` about small truncation.
- `channel()` works only while the program contains no Kerr or cubic gate (`is_gaussian`).
  Otherwise it raises `NotImplementedError`, as does `circuit()`.
- `pattern(params, x=(), *, cubic="gate", injection_squeezing=0.5)` returns the gate-level
  `Pattern`. Gaussian engines reject its Kerr and cubic commands.

## Learning

Pass the program to `CVRegressor` or `CVClassifier` with a `cutoff`. Constraints enforced
by the learner:

- `cutoff` is required.
- `mode="ideal"` (gate level) only.
- No training shots, because sample means of non-Gaussian outputs are not Gaussian.
- Gradients default to central differences (`gradient="auto"`). `"complex-step"` is
  rejected.
- After `moments` or `predict`, `model.retained_norms` holds the per-sample retained norms.

Off-diagonal covariances are not computed; only the q and p variances are.

```python
program = NonGaussianProgram(1).displace(0, "b", 0.0, bound=2.0).kerr(0, "k").rotate(0, "r")
model = CVRegressor(program, CVFeatureMap(1), cutoff=30, optimizer="lbfgs", epochs=15)
```

**Parity.** Kerr, rotation and squeezing commute with photon-number parity, so with
displacement encoding $\langle q\rangle$ is odd in the input amplitude. To fit an even
target (such as $x^2$) the program needs a trainable displacement before the Kerr gate.
Bound it (`bound=2.0` above) so the line search cannot push the state past the cutoff. R71
needed this after L-BFGS drove the displacement beyond cutoffs 16 and 28.

## Cubic gates as resource injection

`pattern(..., cubic="injection", injection_squeezing=s)` lowers each cubic gate to
`photographiq.non_gaussian.cubic_injection`: a cubic-phase ancilla, an inverse SUM, an
ancilla q homodyne, and quadratic-phase and displacement corrections. The ancilla outcome
keys are in `pattern.ancilla_keys`. This lowering is a conditional finite-energy filter,
not the exact gate. `injection_channel(params, x, feature_map, cutoff, *,
injection_squeezing=0.5, outcomes=np.linspace(-8, 8, 161))` integrates the outcome record
(trapezoid rule over postselected runs) for a program with exactly one cubic gate. It
returns `density_matrix`, `basis`, `outcome_probability` (ideally 1; a range and
truncation check) and `retained_norms`.

## Validation

- `tests/validation/cv_fock_reference.py` is an independent dense matrix-exponential
  reference that shares no code with PhotoGraphiQ. The engine matches it for
  rotation+squeezing, Kerr+rotation and cubic+squeezing+Kerr at cutoffs 40 and 60 (means
  2e-6, variances 2e-5, retained norm 1 to 1e-6).
- Cutoff convergence: errors decrease monotonically over cutoffs 16, 20 and 32 against 48.
  Truncation at cutoff 16 is visible in the retained norm.
- A Gaussian-only `NonGaussianProgram` gives the same means on the Fock engine as from the
  moment channel (1e-8).
- A Kerr regressor fits $x^2$ with less than half the MSE of the Gaussian model.
- The injection pattern has one measurement and a quadratic-phase correction. Its
  integrated outcome probability is within 0.02 of 1, and its density matrix is PSD.
- R75 covers cutoff convergence against the reference, Wigner negativity, nonlinear
  regression with Kerr and cubic programs, and the injection lowering against the cubic
  gate. See R75 in the final report for its numbers.

## Limitations

- Experimental. Cost grows with the Fock dimension $\binom{\text{cutoff} + n - 1}{n}$, and
  each objective evaluation runs the engine on every sample. In R71 the two-mode Kerr
  classifier at cutoff 24 took 642 s.
- Cutoff convergence is the user's responsibility. Check it with a sweep. The retained
  norm is a per-step indicator, not a certificate.
- No kernels, no physical (compiled) mode, no shot estimator and no complex-step
  gradients for non-Gaussian programs.
- The injection lowering is approximate at finite `injection_squeezing`.
