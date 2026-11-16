# Gaussian → Fock bridge

`gaussian_to_fock(mean, covariance, dims, *, modes=None, pad=None)` prepares any physical
n-mode Gaussian state (pure or mixed) on per-mode dimensions. Conventions: ħ = 2, vacuum
covariance I, interleaved (q, p).

## Construction

1. **Williamson decomposition.** V = S diag(ν₁, ν₁, …, νₙ, νₙ) Sᵀ. It is computed directly:
   - K = V^{−1/2} Ω V^{−1/2} is real antisymmetric;
   - its real Schur form QᵀKQ has blocks [[0, 1/ν], [−1/ν, 0]] once each block's orientation is fixed;
   - this gives S = V^{1/2} Q D^{−1/2}.

   Reconstruction and symplecticity are checked, and the decomposition raises on failure.
2. **Thermal core.** Each mode starts in a thermal state with n̄ = (ν − 1)/2, so the state is a pure vector when every ν = 1.
3. **Symplectic unitary.** The polar decomposition S = P Q splits S into a symmetric positive factor P and an orthogonal (passive) factor Q, each with a real logarithm L. The generator is G = ½ xᵀ A x with A = −½ Ω L. Because [x_j, x_k] = 2iΩ_jk, U = exp(−iG) satisfies U†xU = exp(2ΩA)x = e^L x. The generator is built from the adapter's `position` and `momentum` operators. U_Q is applied first, then U_P.
4. **Displacement.** D(d) uses the adapter's `displacement` generators.
5. **Projection.** Steps 2–4 run on `dims + pad` levels per mode (default pad = max(dims)). The result is projected onto `dims`, and the kept probability is stored as `retained_weight`. There is no silent renormalisation.

```python
import numpy as np
from photographiq.hybrid.states import gaussian_to_fock

rep = gaussian_to_fock([0.4, -0.3], [[1.8, 0.3], [0.3, 1.1]], (20,))
rep.representation, 1 - rep.retained_weight      # ('mixed-fock', ~3e-11)
mean, cov = rep.moments()                         # equal to the input to ~5e-9
```

## Accuracy (HYB-02)

Moments are reproduced with errors that shrink geometrically with the local dimension. At
the largest dimension of each sweep:

| State | dims | max mean error | max cov error | 1 − retained weight |
|---|---|---|---|---|
| 1-mode squeezed coherent (pure, r = 0.5) | 28 | 6.0e-9 | 6.7e-8 | 4.6e-10 |
| 1-mode displaced squeezed thermal (mixed) | 28 | 5.2e-14 | 7.7e-13 | 3.2e-15 |
| 2-mode squeezed + CZ (pure) | 24 | 6.6e-8 | 9.8e-7 | 5.9e-9 |
| 2-mode squeezed + CZ + thermal noise (mixed) | 24 | 1.5e-6 | 2.3e-5 | 1.6e-7 |

All mixed outputs are positive semidefinite. Data: `paper_experiments/results/json/HYB02_gaussian_to_fock_bridge.json`.
