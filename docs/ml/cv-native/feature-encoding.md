# Feature encoding (`CVFeatureMap`, `Feature`)

## `CVFeatureMap`

```python
CVFeatureMap(modes, encoding="displacement", *, scale=1.0, quadrature="q", offset=1.0,
             max_squeezing=1.5, input_squeezing=0.0)
```

The map turns a feature vector $x \in \mathbb{R}^{\text{modes}}$ into a product Gaussian
state, one feature per mode (formulas in [mathematical model](mathematical-model.md#encoding)).

| Encoding | Physical preparation | Output dependence on x through a Gaussian program |
| --- | --- | --- |
| `displacement` | Coherent state $D(\text{scale}\,x_j)\lvert 0\rangle$; the q (or p) mean equals `scale * x_j` | Means affine in x; covariance independent of x |
| `squeezing` | $S(\text{scale}\,x_j)\lvert 0\rangle$, with `abs(scale * x_j) <= max_squeezing` enforced | Means zero (without displacements); covariance depends on $e^{\pm 2\,\text{scale}\,x_j}$ |
| `rotation` | $R(\text{scale}\,x_j)$ applied to a coherent reference with q mean `offset` | Means depend on $(\cos, \sin)(\text{scale}\,x_j)$ |

`input_squeezing` squeezes every mode before the data enters. It is part of state
preparation, not a trainable parameter, and requires `abs(input_squeezing) <= max_squeezing`.

Methods:

- `check(X)`: returns a float array of shape `(N, modes)`. It raises `ValueError` for a
  wrong shape, non-finite values, or encoded squeezing above `max_squeezing`.
- `moments(x)`: `(mean, covariance)` of the encoded state.
- `mean_photons(x)`: $(\operatorname{tr}V + |d|^2)/4 - n/2$.
- `config()`.

Constructor errors (`ValueError`): `modes` not a positive integer, an unknown encoding or
quadrature, non-finite numeric arguments, `max_squeezing <= 0`, or
`abs(input_squeezing) > max_squeezing`.

## Data inside the program: `Feature`

`Feature(index, scale=1.0)` uses `scale * x[index]` as a gate argument. Gates that take
features are input-dependent Gaussian gates. They make the processing itself depend on x.
For example, `rotate(1, Feature(0))` makes the output mean nonlinear in $x_0$.
`layered_program(..., reupload=True)` re-uploads $x$ as q displacements at the start of
every layer after the first. With displacement encoding those displacements keep the mean
affine in x, so re-uploading changes nonlinearity only when combined with other
data-dependent gates.

A program with a `Feature` is `data_dependent`. Its channel is evaluated per sample, not
once per batch.

## Where the nonlinearity comes from

R71 (circles task, radius 0.62, 2 x 3 CVCluster at r = 1, L-BFGS, seed 0) attributes it
by source. Test accuracies are taken from `R71_nonlinear_classification.json`.

| Model | Source | Test accuracy |
| --- | --- | --- |
| Linear logistic (classical) | baseline | 0.629 |
| Gaussian, displacement, linear readout | none (affine) | 0.629 |
| (1) classical preprocessing $2x^2 - 1$ | classical encoding | 1.000 |
| (2) rotation encoding (scale 2, offset 1.5), `qp` readout | input-dependent Gaussian gates | 0.986 |
| (3) squeezing encoding (scale 0.8), `moments` readout | covariance observables | 1.000 |
| (4) quadratic classical head on the affine features | classical head | 0.929 |
| (5) Kerr program, cutoff 24 | non-Gaussian | 0.829 |
| Quadratic logistic (classical) | baseline | 0.929 |

These numbers attribute the nonlinearity to its source. They do not show an advantage:
one dataset, one seed.

## Limitations

- One feature per mode. There is no amplitude or multi-feature-per-mode encoding.
- Squeezing encoding needs `scale * |x|` inside the squeezing budget. Rescale the features.
- Encoding energy is reported (`mean_input_photons` in the resource report), not
  constrained.
