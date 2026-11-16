# Finite squeezing and Gaussian channels

Diagrams never assume infinite squeezing. A resource `Prepare.squeezed(mode, r)` has
covariance diag(e^{2r}, e^{−2r}). Every teleportation step through it is a Gaussian
**channel**, not a unitary.

## One teleportation step

The step of the [introduction](introduction.md) at squeezing r has

    X = T(k) = [[−k, −1], [1, 0]],   Y = diag(0, e^{−2r}),   d = 0,

with k = cot θ for a homodyne at angle θ. At θ = π/2 (k = 0) and r = 0.5, `evaluate` gives
Y = diag(0, 0.367879), which is e^{−1}. The ideal map X does not depend on r.

## Compiled patterns

A pattern from `compile_circuit` is a product of such steps. Because every step adds
diag(0, e^{−2r}) and that noise is then propagated linearly:

- X and d do not depend on r (flow-corrected patterns implement the ideal map);
- Y(r) = e^{−2r} Y₁, where Y₁ is the noise at e^{−2r} = 1.

A logical rewrite (teleport-chain resynthesis) changes Y₁ but keeps the e^{−2r} scaling.
The noise **ratio** optimised/original therefore does not depend on r.
[Tutorial 7](examples.md#7-finite-squeezing-channels) shows this for a one-mode R, S, R, S
circuit:

```text
r=0.5: modes 18->3, added noise 4.06878->0.78408 (ratio 0.192706), squeezing energy 4.888->0.815
r=1.0: modes 18->3, added noise 1.49682->0.28845 (ratio 0.192706), squeezing energy 24.860->4.143
r=1.5: modes 18->3, added noise 0.55065->0.10611 (ratio 0.192706), squeezing energy 81.609->13.601
r=2.0: modes 18->3, added noise 0.20257->0.03904 (ratio 0.192706), squeezing energy 236.774->39.462
max spread of Y e^(2r) over r: 2.2e-12; of X over r: 0.0e+00
```

R65 confirms the same scaling on 24 random programs (ratio spread ≤ 7.6e-13); see
[Resource optimisation](resource-optimization.md#recorded-results). Fewer modes is not always
less noise. In R65, one program in 24 had a fewest-mode variant that was not the quietest.

## Loss and general channels

`GaussianLoss(mode, eta, thermal)` and `GaussianChannelBox(labels, X, Y, d)` are explicit
channels. `channel-fusion` composes two wire-adjacent one-mode channels into one
`GaussianChannelBox` (class `exact-channel`):

```python
import numpy as np
from photographiq.calculus.cv_diagrams import (
    CVDiagram, GaussianLoss, Squeezer, apply_rewrite, compare, evaluate, is_completely_positive,
)

lossy = CVDiagram((0,), (GaussianLoss(0, 0.8, 0.1), Squeezer(0, 0.4)), (0,))
fused = apply_rewrite(lossy, "channel-fusion")
X, Y, d = evaluate(fused).channel()
print(np.round(Y, 5).tolist(), is_completely_positive(X, Y), compare(lossy, fused).physical_equivalent)
# [[0.10784, 0.0], [0.0, 0.53413]] True True
```

`GaussianChannelBox` checks complete positivity, Y + iΩ − iXΩXᵀ ≥ 0 with ħ = 2, on
construction. For example, `GaussianChannelBox((0,), 2 * np.eye(2), np.zeros((2, 2)))`
(amplification without its required noise) raises `DiagramTypeError`.

## Homodyne detector noise

`Homodyne(mode, theta, key, efficiency, noise)` adds (1 − η)/η + `noise` to the record
variance. Only ideal homodyne (η = 1, `noise = 0`) matches the
`cz-q-measurement-to-feedforward` rule, and `teleport_steps` detects only ideal homodynes.
`prepare-measure-to-source` accepts any efficiency.

## Reading the noise

`resource_cost(d)["added_noise"]` is tr Y / dim, the mean added noise per output quadrature
in vacuum units. For the full matrix use `evaluate(d).channel()[1]`. For a comparison between
two diagrams use `compare(a, b).channel_Y_residual`.
