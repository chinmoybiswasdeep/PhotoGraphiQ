# Examples

Eight runnable tutorials live in `examples/ml/cv_native/`. Run one with
`.venv/Scripts/python examples/ml/cv_native/01_first_regressor.py`. Each prints its
results and ends with `assert` checks of its key claim. The outputs below were
recorded on 2026-11-16 (Windows, Python 3.12.14). Each script took under 10 s of wall
time, including imports. Small numerical differences across platforms are expected.

## 1. First regressor

Gaussian layered program, displacement encoding, affine head on an affine target. Checks: test MSE within 2 x OLS + 1e-4; compiled (r = 1) predictions equal ideal ones.

??? example "`01_first_regressor.py`"
    ```python
    --8<-- "examples/ml/cv_native/01_first_regressor.py"
    ```

Output:

```text
quantum parameters: 11, head parameters: 3
train loss: 0.4049 -> 0.000340
test MSE: CV model 0.000515, OLS 0.000459
max |ideal - compiled (r=1)| prediction difference: 4.44e-16
runtime: 1.6 s
```

## 2. First classifier

Circles task on a 2 x 3 CVCluster (r = 1). Displacement encoding with a mean readout is affine and matches linear logistic regression, which here only reaches the majority-class fraction. Squeezing encoding with q-variance readout supplies the nonlinearity. Checks: affine model within 0.05 of linear logistic; squeezing model at least 0.9.

??? example "`02_first_classifier.py`"
    ```python
    --8<-- "examples/ml/cv_native/02_first_classifier.py"
    ```

Output:

```text
majority class fraction (test): 0.783
classical linear logistic:                  0.783
A displacement encoding, mean readout:      0.783
B squeezing encoding, means + q variances:  0.933
runtime: 2.9 s
```

## 3. Trainable homodyne angles

CVCluster angle parameterisation, channel vs flow pattern, pi-periodicity, singular-angle rejection, and training of angles and rung weights. Rung weights are trained with central differences, which `gradient="auto"` selects automatically when `trainable_weights=True`, because complex-step derivatives cannot see them ([gradients](gradients.md)). The weights start at `rung_weight`.

??? example "`03_trainable_homodyne_angles.py`"
    ```python
    --8<-- "examples/ml/cv_native/03_trainable_homodyne_angles.py"
    ```

Output:

```text
parameters: ('phi.w0.c0', 'phi.w1.c0', 'phi.w0.c1', 'phi.w1.c1', 'phi.w0.c2', 'phi.w1.c2', 'g.w0.c1', 'g.w0.c2')
measurements: 6; |X - X_pattern| = 0.0e+00, |Y - Y_pattern| = 8.3e-31
theta = pi rejected: theta.w0.c0 = 0 mod pi is a q measurement: singular teleportation
pi-periodicity error: 2.0e-15
trained angles (rad, mod pi): [1.294 1.288 1.687 1.5   1.516 2.084]
rung weights: initial [1.13  1.095] -> trained [1.129 0.881]
certificates cached: 47
train MSE: 5.01e-12
runtime: 0.5 s
```

## 4. Circuit to MBQC equivalence

Ideal circuit vs compiled channel vs `gaussian_channel` of the compiled Pattern, exp(-2r) scaling, gate-level execution and trajectory averaging. Checks: all errors below 1e-9; trajectory mean within 4 SE.

??? example "`04_circuit_to_mbqc_equivalence.py`"
    ```python
    --8<-- "examples/ml/cv_native/04_circuit_to_mbqc_equivalence.py"
    ```

Output:

```text
teleportation steps: 142, pattern measurements: 142
         |X - S|: 6.3e-14
 |X - X_pattern|: 3.1e-15
 |t - t_pattern|: 8.3e-17
 |Y - Y_pattern|: 3.6e-15
Y exp(2r) constant over r in (0.5, 1, 2): max deviation 5.0e-14
added noise per quadrature tr(Y)/4 at r = 1: 5.549 (vacuum units)
gate-level Pattern vs ideal mean: 2.8e-16
trajectory ensemble mean vs channel (300 shots): max |z| = 1.72
runtime: 3.7 s
```

## 5. Noise-aware training

Exact, shot-trained (S = 100) and noise-penalised 2 x 4 clusters at r = 0.5. Expected shot-limited test MSE from the moments, checked against 200 sampled evaluations. Checks: sampled vs exact within 4 SE; ideal = physical means; penalty lowers added noise.

??? example "`05_noise_aware_training.py`"
    ```python
    --8<-- "examples/ml/cv_native/05_noise_aware_training.py"
    ```

Output:

```text
r = 0.5 (4.34 dB), S = 100 shots per prediction
model       added noise   bias MSE  E[MSE] S=100
A exact          0.9452    0.00062       0.02441
B shots          0.8579    0.00120       0.01052
C penalty        0.8508    0.00063       0.01830
sampled mean MSE over 200 draws: 0.02350 (z = -2.06 vs exact)
max |ideal - physical| output mean: 0.0e+00
runtime: 4.2 s
```

## 6. Gaussian kernel learning

Overlap kernel = RBF, unitary invariance, finite-squeezing dependence through a CVCluster channel, and classical kernel ridge on circles. Checks: RBF and invariance identities, PSD Gram matrices, purity rising with r, unit fidelity diagonal.

??? example "`06_gaussian_kernel_learning.py`"
    ```python
    --8<-- "examples/ml/cv_native/06_gaussian_kernel_learning.py"
    ```

Output:

```text
overlap kernel vs exp(-s^2 |x-x'|^2 / 4): max error 3.3e-16
after a Gaussian unitary program: max change 5.6e-16
ideal kernel ridge test accuracy: 0.983
    r   purity    min eig  test acc
 0.25   0.3076   1.21e-13     0.983
  1.0   0.6952   5.13e-11     1.000
  2.0   0.9459   4.88e-10     0.983
fidelity kernel diagonal at r = 0.25 (mixed states): [1. 1. 1. 1. 1.]
runtime: 0.6 s
```

## 7. Non-Gaussian learning (experimental)

One-mode Kerr program, cutoff 30, L-BFGS for 15 iterations, central-difference gradients, fit to x^2 against an affine Gaussian model. Checks: Kerr MSE below half the Gaussian MSE; retained norm; parity; engine vs independent expm reference.

??? example "`07_non_gaussian_learning.py`"
    ```python
    --8<-- "examples/ml/cv_native/07_non_gaussian_learning.py"
    ```

Output:

```text
constant-predictor MSE (Var y): 0.1481
Gaussian (affine) MSE:          0.1481
Kerr program MSE:               0.0060  gradient: finite-difference
trained b = 2 tanh(theta_b) = 1.606, kappa = 0.849
1 - minimum retained Fock norm: 2.2e-16
no displacement: <q>(+0.8) = 0.359555, <q>(-0.8) = -0.359555
engine vs expm reference <q>: |difference| = 3.3e-16
runtime: 5.4 s
```

## 8. Resource-quality trade-offs

Resource reports of a compiled program and a cluster, cluster size x squeezing vs shot-limited expected MSE, and the squeezing penalty. Checks: structural counts; error decreasing in r for each size; penalty lowers sum sinh^2 r.

??? example "`08_resource_quality_tradeoffs.py`"
    ```python
    --8<-- "examples/ml/cv_native/08_resource_quality_tradeoffs.py"
    ```

Output:

```text
                                compiled layer  2x4 cluster
trainable_quantum_parameters                11            6
resource_modes                             130            6
entangling_edges                           136            8
measurements                               130            6
measurement_depth                           73            5
feedforward_corrections                    130           17
peak_live_modes                              3            8
resource_squeezing_db                    8.686        8.686
added_noise_per_quadrature               5.071        0.347
mean_input_photons                      0.1829       0.1829

expected test MSE with S = 100 shots (ideal-trained clusters):
 columns  measurements r=0.5    r=1.0    r=1.5   
       3             4 0.00974  0.00893  0.00863 
       4             6 0.02452  0.01767  0.01515 
       5             8 0.01688  0.01383  0.01271 

penalty weight 0.0: gate r = [0.054 0.521], sum sinh^2 r = 0.3003, test MSE = 0.000636

penalty weight 0.5: gate r = [-0.004 -0.02 ], sum sinh^2 r = 0.0004, test MSE = 0.000606
runtime: 7.1 s
```

## Where the claims are validated

Each tutorial is a small instance of a test or experiment listed in [validation](validation.md).
The tutorials are illustrations. Their datasets and seeds differ from R68 to R75.
