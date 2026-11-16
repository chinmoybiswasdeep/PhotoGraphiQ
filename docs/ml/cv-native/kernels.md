# Kernels (`CVKernel`)

```python
CVKernel(feature_map=None, program=None, params=None, *, kind="overlap", mode="ideal",
         squeezing=None, modes=None)
```

`CVKernel` turns encoded (and optionally processed) Gaussian states into a Gram matrix for
classical kernel methods such as an SVM or kernel ridge regression. The states are
quantum. The kernel method that uses the Gram matrix is classical.

| `kind` | Value | Properties |
| --- | --- | --- |
| `"overlap"` | $\operatorname{Tr}[\rho(x)\rho(x')]$ (Hilbert–Schmidt) | A Gram matrix of vectors in operator space, so symmetric PSD for any states; $K(x,x)$ is the purity |
| `"normalized"` | $\operatorname{Tr}[\rho\rho'] / \sqrt{\operatorname{Tr}\rho^2\,\operatorname{Tr}\rho'^2}$ | Unit diagonal, PSD |
| `"fidelity"` | Squared Uhlmann fidelity (Banchi–Braunstein–Pirandola formula for mixed Gaussian states) | Unit diagonal. Equals the overlap only when one state is pure. PSD is not guaranteed for mixed states |

Methods:

- `states(X)`: a list of `photographiq.GaussianState`.
- `kernel(X1, X2)`: a cross-kernel matrix.
- `gram_matrix(X)`: a symmetric Gram matrix.
- `diagnostics(X)`: `symmetry_error`, `max_diagonal`, `min_diagonal` and
  `minimum_eigenvalue`. Raises `ValueError` for an empty `X`.

If `program` is given, each encoded state is passed through
`program.channel(params, x, mode=mode, squeezing=squeezing)` (once, if the program is not
data dependent). `program` may be a `GaussianProgram` or a `CVCluster`.

## Established properties

From `tests/ml/cv_native/test_cv_kernels_nongaussian.py`:

- Coherent displacement encoding gives $K = \exp(-\text{scale}^2|x - x'|^2/4)$, a
  Gaussian RBF kernel (to 1e-13).
- A Gaussian unitary program leaves the kernel unchanged:
  $\operatorname{Tr}[U\rho U^\dagger U\sigma U^\dagger] = \operatorname{Tr}[\rho\sigma]$.
- In physical mode (finite squeezing), outputs are mixed. The diagonal (purity) is below 1,
  the Gram matrix stays symmetric and PSD, and it approaches the ideal kernel as $r$ grows.
- For pure states, fidelity equals overlap. For mixed cluster outputs, fidelity and
  normalized kernels have unit diagonal and differ from the overlap.

## Interpretation

An ideal Gaussian unitary cannot change the kernel. A Gaussian-unitary model therefore
adds nothing to the encoding's kernel, which for coherent encoding is a classical RBF
kernel. Only a noisy channel (`mode="physical"`) or the encoding itself changes the
kernel. In [tutorial 6](examples.md#6-gaussian-kernel-learning), kernel ridge on the
circles task reaches 0.983 test accuracy with the ideal RBF kernel. With a 2 x 3 cluster
at r = 0.25, 1 and 2 (purity 0.31, 0.70, 0.95) it reaches 0.983, 1.000 and 0.983. One
dataset; no advantage is claimed.

## Limitations

- Gaussian states only. A `NonGaussianProgram` with Kerr or cubic gates raises
  `NotImplementedError` from `channel`.
- `mode` is passed to `program.channel` and is ignored when `program` is `None`.
- No trainable kernel alignment. The kernel parameters are fixed when the kernel is built.
- The cost is $O(N^2)$ closed-form overlaps in Python.
