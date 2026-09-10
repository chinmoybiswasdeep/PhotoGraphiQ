# Automatic differentiation

Install the `autodiff` extra. The dedicated JAX execution path preserves scalar
expression trees and uses dense finite-Fock generators with fixed topology.
It does not wrap NumPy `simulate()` in a gradient decorator.

```python
import jax
jax.config.update("jax_enable_x64", True)
import photographiq as pg
from photographiq import autodiff

pattern = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("theta")))
initial = pg.FockInput((2**-0.5, 2**-0.5))
def objective(theta):
    return autodiff.expectation(pattern, {"theta": theta}, cutoff=8,
                                inputs={0: initial}, observable="q")
print(jax.grad(objective)(0.3))  # approximately -0.2955202067
```

Supported observables are photon number, parity, q and p. `fock_state` exposes
the differentiable density matrix, basis, nodes and branch log likelihood for
custom objectives. Input descriptions are constants; use parameterized preparation
squeezing or later gates for trainable state preparation. Gaussian gates, cubic,
Kerr and vacuum loss are supported in this Fock representation.

Every measurement requires an explicit fixed outcome. These derivatives describe
the selected branch and include its normalization; branch likelihood is separately
available. Zero-probability branches have undefined conditional derivatives.
Eager execution raises `ValueError` for zero or invalid normalization; under
`jax.jit`/`jax.grad`, invalid states explicitly contain NaNs. A zero branch has
log likelihood `-inf`; its derivative is undefined. Do not replace those values
with zeros in an optimization objective.

`retained_norms` records preparation projection masses before normalization;
`boundary_population` reports the final top two total-photon shells. Tensor
preparation with lost mass above 1e-3 is rejected eagerly (NaNs under tracing).
Photon addition cannot silently discard support above the cutoff. Projected
unitaries conserve trace even when their infinite-space approximation is poor.
Thermal loss, noisy homodyne and arbitrary Python callable expressions are not
supported in this path. Loss derivatives at exact endpoints can be singular in
the Kraus parametrization; evaluate interior values for smooth optimization.

`gaussian_sample(mean,covariance,standard_normal)` supplies the reparameterization
primitive for Gaussian sampling. `score_function_surrogate(values,log_probabilities,
baseline=...)` supplies likelihood-ratio terms for discrete samples. Samples must
come from the parameter-dependent distribution, and a baseline must be independent
of each sample's outcome. This estimator can have high variance. A conditional
postselected objective does not automatically include success probability.

Compilation choices and command topology are fixed during differentiation.
Projected-generator evolution differs numerically from Piquasso's native truncated
gates. Check gradients against analytical derivatives or refined finite differences,
and study cutoff convergence of both the observable and its derivative.
