# Mixed Fock evolution

Select `backend="piquasso-mixed-fock"` and an explicit cutoff. This separate
backend evolves density matrices with Piquasso's mixed simulator and implements
destructive conditional counting/homodyne locally. It supports pure and mixed
inputs, Gaussian and nonlinear gates, and thermal attenuation.

```python
import photographiq as pg
pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, transmissivity=0.6))
result = pg.simulate(pattern, inputs={0: pg.FockInput.number(2)},
                     backend="piquasso-mixed-fock", cutoff=5)
print(result.state.probabilities)
```

The nonzero probabilities are 0.16 for vacuum, 0.48 for one photon, and 0.36 for
two photons: the binomial loss distribution. `state_vector` is unavailable for a
mixed state; inspect `density_matrix`, moments, Wigner functions or reductions.

Loss transmissivity is intensity transmission eta. Thermal photons specify the
environment occupation. Homodyne detector efficiency acts as loss on the detected
mode; the reported outcome is rescaled to incident-quadrature units, matching the
Gaussian backend. Added electronics noise is a classical convolution of the
conditional operator, not a random displacement of the surviving state.

The matrix has D² complex entries, where D=comb(modes+cutoff-1,modes). A configurable
`max_matrix_bytes` guard bounds one matrix, not all native workspaces. Start with
one or two modes. Trace drift, boundary weight and cutoff sensitivity all matter.
Fock heterodyne/general-dyne remain unsupported.

Hardening tests cover n=1 through 4 attenuation, coherent attenuation, thermal
environment dilation, independent unitaries, and correlated noisy-homodyne
Schur complements. These regimes do not establish dense-state scalability or
uniform accuracy for every input. Inspect pre-normalization retained trace;
normalizing a truncated thermal tail does not turn it into physical photon loss.
