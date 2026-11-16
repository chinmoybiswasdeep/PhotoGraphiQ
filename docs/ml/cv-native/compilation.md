# Compilation: direct circuit versus compiled CV-MBQC channel

A trainable `GaussianProgram` has two separate propagations:

| `mode` | Object | Physical meaning |
| --- | --- | --- |
| `"ideal"` | $S$, $t$ ($Y = 0$) | The intended direct optical circuit. `gate_pattern(params, x)` is its gate-level `Pattern` |
| `"physical"` | $X$, $t$, $Y(r)$ | The unconditional channel of the finite-squeezing CV-MBQC `Pattern` built by `photographiq.compile_circuit` (`program.circuit(params, x).compile(squeezing=r)`) |

```python
from photographiq.analysis import gaussian_channel
from photographiqml.cv import layered_program

prog = layered_program(2, 1)
params = dict.fromkeys(prog.parameters, 0.3)
S, t0, _, _ = prog.channel(params)
X, t, Y, steps = prog.channel(params, mode="physical", squeezing=1.0)
ref = gaussian_channel(prog.circuit(params).compile(squeezing=1.0))
# X == S; (X, t, Y) == (ref.matrix, ref.displacement, ref.noise)
```

## How the mirror works

`channel(mode="physical")` repeats the compiler's lowering step by step. It does not call
`photographiq.analysis`, so the comparison with `gaussian_channel` is an independent check.

- Each teleportation step $T(k)$ adds $\operatorname{diag}(0, e^{-2r})$ to $Y$ on its mode.
- A rotation, squeezer or shear takes four shears (five on the compiler's
  $|S_{10}| < 10^{-8}$ branch).
- A CZ takes eight Fourier steps (four on each mode).
- A displacement takes four Fourier steps plus a shift applied as a correction.
- A beam splitter takes two half rotations. Each is three SUM gates, and each SUM is
  $R(\pi/2)\,CZ\,R(-\pi/2)$.

Teleportation steps observed in R69 and R74 for `layered_program(modes, layers)`:

| Modes | 1 layer | 2 layers | 3 layers | 4 layers |
| --- | --- | --- | --- | --- |
| 1 | 17 | 34 | 51 | 68 |
| 2 | 130 | 260 | 390 | 520 |
| 3 | 243 | 486 | 729 | 972 |
| 4 | 356 | 712 | 1068 | 1424 |

Each step consumes one resource mode and one homodyne measurement.

## What is established

From `tests/ml/cv_native/test_cv_gaussian_programs.py` and R69:

- $X = S$. The flow-corrected logical map is exact. The R69 maximum $|X - S|$ over 36
  configurations (1 to 3 modes, 1 to 4 layers, 3 seeds) is 7.6e-14.
- $X$, $t$ and $Y$ equal `gaussian_channel` of the compiled `Pattern`. R69 maxima are
  2.4e-14 for the matrix and 7.8e-15 for the noise, relative to 1 + max |Y|. The tests
  use r = 0.4 and r = 1.2.
- $Y$ scales exactly as $e^{-2r}$. In R69 (r from 0.25 to 2, 1 to 3 layers) the largest
  deviation of $Y e^{2r}$ from constant is 2.8e-15.
- `steps` equals the number of `Measure` commands in the pattern.
- Trajectory execution of the compiled pattern averages to $X d + t$ and
  $X V X^T + Y$. Conditional covariances do not depend on the outcomes. In R69 this was
  checked with 1000 shots, on a compiled one-mode layer and a 2 x 3 CVCluster, with a
  largest |z| of 1.44.
- The gate-level pattern reproduces the ideal moments on the `gaussian` and
  `native-gaussian` backends (atol 1e-12).

## Quantum and classical parts

The compiled pattern is entirely physical: resource preparation, CZ entangling, homodyne
measurement and feedforward displacements. The feedforward is classical arithmetic on the
outcomes. Training uses the unconditional channel; it never samples the pattern.

## Limitations

- The compiled pattern is much longer than the circuit (130 steps for one two-mode
  layer), and the added noise $\operatorname{tr}Y/2n$ grows with its length (R74, r = 1,
  random parameters: 4.9 for one two-mode layer, 39.8 for four four-mode layers).
- Only Gaussian gates are lowered here. For cubic resource injection see
  [non-Gaussian](non-gaussian.md).
- Finite squeezing is the only modelled imperfection. Loss and detector noise are not
  included.
