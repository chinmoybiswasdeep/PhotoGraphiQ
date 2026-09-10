# Input states

Input labels must belong to `pattern.inputs`. Omitted inputs default to vacuum.
Choose either an `inputs={label: description}` mapping or one correlated
`initial_state`; combining the two is an error. Resource modes created later use
`Prepare` or `PrepareResource` and must not duplicate input labels.

## GaussianInput: vacuum, coherent, squeezed and thermal

`GaussianInput(mean=(0,0), covariance=((1,0),(0,1)))` describes one mode. Its
resulting `.state(label)` is a validated `GaussianState`. Means are quadrature
coordinates and covariance is statistical V, with $V+i\Omega\geq0$.

```python
import photographiq as pg
vacuum = pg.GaussianInput()
coherent = pg.GaussianInput.coherent(0.3 + 0.2j)
squeezed = pg.GaussianInput.squeezed(r=0.4, angle=0.2)
thermal = pg.GaussianInput(covariance=((1.2, 0), (0, 1.2)))
print(coherent.mean)  # (0.6, 0.4)
```

Vacuum has mean zero and V=I. A coherent state is $D(\alpha)|0\rangle$, retaining
V=I with $\langle q\rangle=2\Re\alpha$ and $\langle p\rangle=2\Im\alpha$.
The squeezed constructor rotates $\mathrm{diag}(e^{-2r},e^{2r})$ through `angle`
radians. Thermal covariance is $(2\bar n+1)I$; the example has mean photon number
0.1. Gaussian backends accept physical pure or mixed covariance states. Pure Fock
execution accepts only pure Gaussian inputs; mixed Fock accepts mixed ones too.
Nonfinite moments, asymmetric covariance and uncertainty violations fail.

## GaussianState: correlated inputs

```python
import numpy as np
state = pg.GaussianState(np.zeros(4), np.eye(4), nodes=("a", "b"))
pattern = pg.Pattern(inputs=("a", "b"))
result = pg.simulate(pattern, initial_state=state, backend="gaussian")
```

Mean has shape `(2*modes,)`, covariance `(2*modes,2*modes)`, and `nodes` must
match ordered pattern inputs. Off-diagonal covariance blocks describe correlations.
This direct correlated-Gaussian injection path is supported by Gaussian backends.
For external Piquasso Gaussian states, use `PiquassoBackend.import_state(native,
nodes, source_hbar=...)`, then supply its `.get_state()` snapshot. Declare source
hbar explicitly; native covariance cannot simply be copied into V.

## FockInput: one-mode amplitudes

```python
one_photon = pg.FockInput.number(1)
superposition = pg.FockInput((2**-0.5, 1j * 2**-0.5))
cat = pg.FockInput.cat(alpha=0.7, cutoff=16, parity=1)
```

`number(n)` is $|n\rangle$ for a nonnegative integer n. `FockInput(amplitudes)`
stores $\sum_n c_n|n\rangle$ in ascending occupation order. Coefficients must be
finite and normalized; the constructor does not silently normalize arbitrary
vectors. `cat` returns a normalized finite projection of
$|\alpha\rangle+\mathrm{parity}|-\alpha\rangle$, with parity +1 or -1. An odd cat
at alpha=0 has zero norm and is rejected. These inputs require a Fock backend.

An explicit vector retains the same truncation when the execution cutoff grows.
For convergence, rebuild it at each cutoff or use `CatResource(alpha, parity)`,
which regenerates the projected resource and records its retained weight.

## Correlated Fock resources and external states

```python
resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
pattern = pg.Pattern(inputs=("a", "b"))
result = pg.simulate(pattern, initial_state=resource,
                     backend="piquasso-fock", cutoff=8)
```

Occupation tuples follow preparation-label order. `FockSuperposition.number((2,1))`
creates a multimode number state. `from_piquasso(native_pure_state)` imports public
native amplitudes into an independent normalized description. Mixed native states
require a density-matrix description rather than this pure-state importer.
`PrepareResource(("a","b"), resource)` introduces the same state as an ancilla.

The cutoff c retains total occupation $\sum_i n_i<c$, not c levels independently
per mode. Resource support exceeding that limit fails before preparation.

## Mixed density matrices

```python
mixed = pg.FockDensityMatrix([[0.3, 0], [0, 0.7]], ((0,), (1,)))
thermal_fock = pg.FockDensityMatrix.thermal(mean_photons=0.1, cutoff=12)
result = pg.simulate(pg.Pattern(inputs=(0,)), initial_state=mixed,
                     backend="piquasso-mixed-fock", cutoff=12)
```

`matrix[i,j]` multiplies $|\mathrm{basis}_i\rangle\langle\mathrm{basis}_j|$.
The basis is explicit, unique and ordered; the matrix must be finite, Hermitian,
positive semidefinite and trace one. Thermal construction normalizes a finite
geometric distribution; rebuild at each cutoff. For external mixed data, pass
the native density matrix together with its matching occupation basis.

## Cubic and GKP resources

`CubicPhaseResource(gamma, squeezing=r)` prepares
$\exp(i\gamma q^3/6)S(-r)|0\rangle$. `GKPResource(logical=0, peak_width=0.4,
envelope=0.4)` describes a finite Gaussian comb, projected at execution cutoff.
Both require Fock execution. GKP peak count, integration grid and Fock cutoff are
independent numerical controls. See [GKP](gkp.md) and
[non-Gaussian numerics](non-gaussian.md) before interpreting their diagnostics.
