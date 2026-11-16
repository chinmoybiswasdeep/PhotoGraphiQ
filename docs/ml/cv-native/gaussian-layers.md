# Gaussian layers (`GaussianProgram`, `layered_program`)

A `GaussianProgram` is an ordered list of Gaussian gates. Each gate argument is a number
(fixed), a string (a trainable parameter name) or a `Feature` (data, never trainable).

```python
from photographiqml.cv import Feature, GaussianProgram

prog = (GaussianProgram(2)
        .rotate(0, "a")
        .squeeze(1, "r", bound=0.5)       # r_phys = 0.5 tanh(r)
        .beamsplitter(0, 1, "theta")
        .cz(0, 1, "g")
        .shear(1, 0.2)                     # fixed quadratic phase, p -> p + 0.2 q
        .displace(0, "dq", Feature(1, 0.5)))  # p shift = 0.5 x[1]
prog.parameters       # ('a', 'r', 'theta', 'g', 'dq'), first-appearance order
prog.data_dependent   # True (a Feature appears in a gate)
```

## Gates

| Method | Arguments | Notes |
| --- | --- | --- |
| `rotate(mode, angle)` | angle | $R(\phi)$ |
| `squeeze(mode, r, *, bound=None)` | r | With `bound`, a trainable r becomes `bound * tanh(r)` |
| `shear(mode, s)` | s | Quadratic phase $e^{i s q^2/4}$ |
| `displace(mode, q=0.0, p=0.0, *, bound=None)` | q, p | `bound` applies to each trainable argument |
| `beamsplitter(u, v, theta)` | theta | Two-mode |
| `cz(u, v, weight)` | weight | Weighted CZ |

Validation rules: gate modes must be distinct integers in `range(modes)`. Numeric
arguments must be finite. `bound` must be positive and finite. A `Feature` index must be
a nonnegative integer and its scale finite. An index beyond the input dimension raises
`ValueError` at evaluation.

## Evaluation

| Method | Returns |
| --- | --- |
| `channel(params, x=(), *, mode="ideal", squeezing=None)` | `(X, t, Y, steps)`. In ideal mode, $X = S$ and $Y = 0$. Physical mode needs a finite, nonnegative `squeezing` and returns the compiled channel and the number of teleportation steps |
| `squeezing_values(params, x=())` | Physical r of every squeeze gate after bounding |
| `circuit(params, x=())` | The equivalent `photographiq.Circuit` (compile with `.compile(squeezing=r)`) |
| `gate_pattern(params, x=())` | The direct circuit as a gate-level `Pattern`, with no measurements or resources |
| `config()` / `GaussianProgram.from_config(config)` | JSON-serialisable round trip |

Arithmetic is real-analytic in the parameters, so complex-step derivatives pass through
`channel` ([gradients](gradients.md)).

## Standard ansatz: `layered_program`

`layered_program(modes, layers, *, reupload=False, squeezing_bound=1.0, prefix="L")`
builds, per layer $\ell$:

1. If `reupload` and $\ell > 0$: `displace(j, Feature(j), 0.0)` on every mode.
2. `rotate(j, "L{l}.r{j}")`, `squeeze(j, "L{l}.s{j}", bound=squeezing_bound)` on every mode.
3. A beam-splitter chain `beamsplitter(j, j+1, "L{l}.bs{j}")`.
4. `rotate(j, "L{l}.r'{j}")`, `displace(j, "L{l}.dq{j}", "L{l}.dp{j}")` on every mode.

That gives $6n - 1$ trainable parameters per layer for $n$ modes (11 for two modes).

## Quantum and classical parts

- Quantum: every gate. In physical mode, also the finite-squeezing teleportation steps of
  the compiled pattern.
- Trainable: the named arguments. Bounded squeezing and displacement keep optimisation
  inside a squeezing or energy budget without clipping.
- Classical: nothing inside the program. The head lives in the learner.

## Limitations

- Gaussian only. Kerr and cubic gates are in `NonGaussianProgram`.
- With displacement encoding and no `Feature` gates, the output mean is affine in $x$.
- The compiled pattern is long. A two-mode layer compiles to 130 teleportation steps
  (R69, R74), mostly from the beam splitter.
