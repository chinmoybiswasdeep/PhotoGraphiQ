# Training (`CVRegressor`, `CVClassifier`)

Both learners share one quantum pipeline (encoding, program, readout) and differ in the
classical head and the loss.

```python
CVRegressor(program, feature_map=None, *, readout="q", readout_modes=None, mode="ideal",
            squeezing=None, shots=None, penalty=None, penalty_weight=0.0, optimizer="adam",
            epochs=100, learning_rate=0.05, gradient="auto", init_scale=0.1, seed=0,
            cutoff=None, fd_step=1e-6)
CVClassifier(program, feature_map=None, *, head="logistic", classes=2, **same_keywords)
```

`program` is a `GaussianProgram` (including `NonGaussianProgram`) or a `CVCluster`.
`feature_map` defaults to `CVFeatureMap(program.modes)` (displacement, scale 1).

## Quantum part

| Argument | Meaning |
| --- | --- |
| `mode` | `"ideal"` or `"physical"`. Physical execution of a `GaussianProgram` needs `squeezing`. For a `CVCluster`, `squeezing=None` uses the cluster's own squeezing |
| `readout`, `readout_modes` | `q`, `p`, `qp` or `moments` (q means plus q variances) on a nonempty subset of modes (default all) |
| `shots` | `None` (exact moments) or an integer >= 2. Features become sample means of `shots` homodyne outcomes. Not allowed with `readout="moments"` |
| `cutoff` | Fock cutoff, required for non-Gaussian programs |

`moments(params, X, *, mode=None)` returns the readout-mode output means `(N, 2k)` and
covariances `(N, 2k, 2k)`. `features(params, X, *, mode=None, shots=None, noise=None)`
returns the readout features.

### Shot estimator

With `shots = S`, each mean feature is drawn exactly as $\mu + \operatorname{chol}(C/S)\,z$
from the output moments. The standard normals $z$ are drawn once per `fit` from `seed`
and then held fixed (common random numbers). The training objective is therefore a
deterministic, differentiable function of the parameters (a pathwise estimator). The
sampled outcomes themselves are never differentiated. `qp` splits the shots,
$\lfloor S/2\rfloor$ for q and the same for p, because one homodyne cannot read both.
`predict(X, shots=S, seed=s)` draws fresh normals from `seed`.

## Classical heads and losses

| Learner / head | Head parameters | Loss |
| --- | --- | --- |
| `CVRegressor` | `head.w{j}`, `head.b` (affine) | MSE (divided by Var y when a penalty is set) |
| `CVClassifier(head="logistic")` | `head.w{j}`, `head.b` | Binary cross-entropy on the logits |
| `CVClassifier(head="softmax", classes=K)` | `head.W{k}.{j}`, `head.b{k}` | Softmax cross-entropy; required exactly when `classes > 2` |
| `CVClassifier(head="homodyne")` | `head.threshold` (b) | Cross-entropy of $P(y{=}1) = \Phi((\mu_q - b)/\sigma_q)$ for the first readout mode |

The homodyne head is the only head whose probability is a physical Born probability: the
probability that one q homodyne outcome exceeds $b$. It needs `readout="q"` and no training
shots. `predict_proba(X, shots=S, seed=s)` then returns the observed fraction of $S$ binned
outcomes (binomial sampling). All other heads are classical functions of the features.

Initialisation: quantum parameters are drawn from `normal(0, init_scale)` with `seed`. The
regression head starts at $w = 1/n_f$, $b = 0$; classification heads start at zero.

## Resource penalties

| `penalty` | Term added to the loss (times `penalty_weight`) | Units |
| --- | --- | --- |
| `"noise"` | Mean added noise per quadrature, $\operatorname{tr}Y/2n$, of the physical channel (first sample if data enters gates). Requires `mode="physical"` | Vacuum units |
| `"squeezing"` | $\sum \sinh^2 r$ over the program's squeeze gates (after bounding). This is the photon number those gates add to vacuum. It is zero for `CVCluster`, which has no squeeze gates | Photons |

With any penalty the regression MSE is divided by $\operatorname{Var}(y)$, so both terms
are dimensionless. Classification losses are not rescaled. Validation losses exclude the
penalty.

## Optimisation

`optimizer` is `"adam"`, `"sgd"` or `"lbfgs"` (the existing `photographiqml.Trainer`;
`epochs` is the L-BFGS `maxiter`). Gradients are covered in [gradients](gradients.md).

`fit(X, y, X_val=None, y_val=None)` returns the learner. Afterwards:

- `params` (quantum parameters dict), `head` (array), `vector`.
- `history`: a `CVTrainingHistory` with `losses`, `validation_losses`, `gradient_norms`,
  `parameters`, `seconds`, `evaluations` and `settings` (mode, squeezing, shots, gradient,
  complex and finite-difference step sizes, cutoff, optimiser, epochs, learning rate, seed,
  penalty, parameter names). `as_dict()` is
  JSON-serialisable. Losses include the initial point.
- `predict`, `score` (regressor: `mse`, `mae`; classifier: `accuracy`, `log_loss`) and
  `predict_proba` (classifier). Each accepts `mode=` to evaluate a trained model in the
  other execution mode.
- `get_params()`, `set_params(params, head)` to move parameters between models (for
  example, an ideal-trained model evaluated by a physical one).
- `resource_report(X)` ([resource analysis](resource-analysis.md)).

## Errors

- `TypeError`: the program is not a `GaussianProgram` or `CVCluster`.
- `ValueError`: the feature map and program have different mode counts; an unknown
  readout, mode, penalty or head; physical `GaussianProgram` without `squeezing`; `shots`
  below 2 or with `moments`; `"noise"` penalty without physical mode; a softmax head with
  two classes or another head with more than two; a homodyne head without `readout="q"`
  or with shots; a non-Gaussian program without `cutoff`, with shots, in physical mode or
  with complex-step gradients.
- `RuntimeError`: `predict`, `score` or `resource_report` before `fit` (or `set_params`).

## What the model can represent

With displacement encoding, no `Feature` gates and a linear mean readout, the output is
affine in $x$. The test `test_gaussian_mean_readout_is_affine_in_displacement_features`
checks this to 1e-10. In R71 such a model's circles accuracy equals linear logistic
regression (0.629), and on blobs every Gaussian head equals the linear baseline (0.975).
Nonlinearity must come from an identified source ([feature encoding](feature-encoding.md#where-the-nonlinearity-comes-from)).

On affine targets (R70; 5 seeds, Adam, 120 epochs), the median test MSE is 3.95e-4 for
the layered program in both ideal and compiled (r = 1) execution, and 3.90e-4 for a 2 x 3
cluster. Ordinary least squares on the same split gives 3.95e-4. On the three-feature task
a 3 x 3 cluster reaches 4.36e-4 against 4.36e-4 for OLS. An affine classical model is
optimal here by construction.

## Limitations

- Shot readout is supported for mean features only. Shot-estimated variances are not
  implemented.
- Inputs must fit in the feature map. For a data-dependent program the channel is
  rebuilt per sample, so it is slower.
- Non-Gaussian learners use central differences and the Fock engine, which is orders of
  magnitude slower: in R71 the Kerr classifier took 642 s, against under 4 s for each
  Gaussian model.
