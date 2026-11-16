# Direct MBQC: trainable homodyne angles (`CVCluster`)

`CVCluster` is a directly trainable weighted grid cluster. Its trainable parameters are
the homodyne measurement angles, and optionally the CZ rung weights. Like the logical
MuTA, it trains measurement angles on a cluster. It is not a MuTA representation, does not
change `MuTA`, and is not the CVMuTA proposal on the roadmap, which remains planned.

```python
CVCluster(wires, columns, *, squeezing=1.0, rungs=None, rung_weight=1.0,
          trainable_weights=False, angles="bounded", margin=0.05)
```

## Layout

- Nodes are grid positions `(w, c)` for `w < wires` and `c < columns`. Column 0 holds the
  encoded input modes and column `columns - 1` the outputs. Every other node is a
  p-squeezed resource mode with resource squeezing `squeezing`.
- Horizontal CZ edges of weight 1 run along every wire. Vertical rungs of weight `g`
  connect neighbouring wires in the columns listed in `rungs` (default: all interior
  columns; none for one wire).
- Measurement order is column-major over all non-output nodes. That gives
  `wires * (columns - 1)` homodyne measurements and the same number of resource modes.

## Trainable parameters

| Name | Meaning |
| --- | --- |
| `phi.w{w}.c{c}` (`angles="bounded"`) | $\theta = \pi/2 + (\pi/2 - \text{margin})\tanh\varphi$. This covers one period except a margin around the singular angle |
| `theta.w{w}.c{c}` (`angles="direct"`) | $\theta$ itself. Values with $\lvert\sin\theta\rvert < 10^{-6}$ raise `ValueError` |
| `g.w{w}.c{c}` (`trainable_weights=True`) | Rung weight between `(w, c)` and `(w+1, c)` |

Node $i$ is read out by the homodyne $q\cos\theta_i + p\sin\theta_i$. It implements the
teleportation shear $k_i = \cot\theta_i$, so $\theta$ is $\pi$-periodic and
$\theta = 0 \bmod \pi$ (a q measurement, which deletes the node) is singular.
`angles(params)` returns the physical angles. `angle_parameters(thetas)` inverts the
parameterisation after reducing mod $\pi$, and raises `ValueError` inside the excluded
margin. At the defaults ($\varphi = 0$ or $\theta = \pi/2$) every step is a Fourier
transform.

## Feedforward: CV flow

The corrections are the CV-flow corrections of Booth and Markham
(`photographiq.cvflow.certify_cv_flow`). After outcome $m_i$, every unmeasured node gets
the displacement $-(c\,q,\ A c\,p)\,m_i/\sin\theta_i$. The certificate depends only on the
edge weights. `certificate(params)` caches one certificate per weight tuple and computes a
new one when trainable weights change. A layout or weight tuple without a certificate
raises `ValueError`, at construction for the initial weights.

## Evaluation

| Method | Returns |
| --- | --- |
| `channel(params, x=(), *, mode="ideal", squeezing=None)` | `(X, t, Y, measurements)` by forward substitution ([mathematical model](mathematical-model.md#path-b-weighted-cluster-channel)). `mode="ideal"` sets $Y = 0$. `squeezing` overrides the resource squeezing |
| `pattern(params, *, squeezing=None)` | Executable finite-squeezing `photographiq.Pattern` from `flow_pattern`, with shears $\cot\theta_i$ |
| `graph(params)` | `photographiq.CVGraph` of the weighted cluster |
| `config()` | Constructor arguments |

Properties checked by `tests/ml/cv_native/test_cv_cluster.py`:

- $X$ and $Y$ equal `gaussian_channel(model.pattern(params))` to within 1e-11.
- The ideal map is symplectic.
- `measurements == wires * (columns - 1)`.
- A single wire gives the product of teleportation matrices $T(\cot\theta_i)$.
- Angles are $\pi$-periodic.
- Recertified weights reproduce the pattern channel.
- Trajectory runs of the pattern reproduce the ensemble mean and covariance within 5 SE.
- Conditional covariances do not depend on the outcomes, while conditional means still do
  at finite squeezing.

## Backends

Training uses the in-package forward-substitution channel. `pattern()` runs on
PhotoGraphiQ's Gaussian backend (`photographiq.run_shots` for trajectories).

## Limitations

- Grid topology only (wires x columns, rungs between neighbouring wires).
- The output displacement $t$ is zero. Data enters only through the input state.
- Angles near the margin give large shears and large output covariance at finite
  squeezing. Gradients stay finite up to the margin (R73).
- The flow certificate is solved for real rung weights, so complex-step derivatives
  cannot see trainable weights. With `trainable_weights=True`, `gradient="auto"` selects
  central differences, and an explicit `gradient="complex-step"` raises `ValueError`. `fit`
  starts each trainable rung weight at `rung_weight` (plus `normal(0, init_scale)`) and each
  angle parameter at its default. The certificate cache is cleared after 256 weight tuples.
  See [gradients](gradients.md).
- Gaussian only. There are no non-Gaussian resource nodes.
