# Gaussian states, channels and measurements

This page fixes the phase-space conventions that every PhotoGraphiQ backend uses, then
derives the Gaussian operations the simulators implement. The code is in
`src/photographiq/gaussian.py` (matrices), `src/photographiq/states.py` (`GaussianState`) and
`src/photographiq/backends/gaussian.py` (the NumPy reference engine; `native-gaussian`
subclasses it and uses the same formulas in C++).

## Quadratures and canonical commutation relations

Each optical mode has a bosonic annihilation operator $a$ with $[a,a^\dagger]=1$. With
$\hbar=2$ the quadratures are

$$q=a+a^\dagger,\qquad p=-i(a-a^\dagger),\qquad [q,p]=2i .$$

For $n$ modes the quadrature vector is ordered **interleaved**,
$R=(q_0,p_0,q_1,p_1,\dots)$, and

$$[R_i,R_j]=2i\,\Omega_{ij},\qquad \Omega=\bigoplus_{k=1}^{n}\begin{pmatrix}0&1\\-1&0\end{pmatrix}.$$

## Gaussian states and the covariance matrix

A Gaussian state is fixed by its mean $\mu=\langle R\rangle$ and its **statistical
covariance**

$$V_{ij}=\tfrac12\langle\{\Delta R_i,\Delta R_j\}\rangle,\qquad \Delta R=R-\mu .$$

Vacuum has $V=I$, so a quadrature variance below 1 means squeezing. A real symmetric $V$ is a
physical state if and only if the uncertainty relation $V+i\Omega\ge0$ holds;
`GaussianState.validate` checks shape, finiteness, symmetry and this condition.
Piquasso's anticommutator covariance is $2V$ at the same $\hbar$; the conversion happens only
inside the `piquasso` backend adapter.

| Input (`GaussianInput`) | Mean | Covariance |
|---|---|---|
| vacuum (the default for a missing input) | $0$ | $I$ |
| `coherent(alpha)` | $(2\,\mathrm{Re}\,\alpha,\ 2\,\mathrm{Im}\,\alpha)$ | $I$ |
| `squeezed(r)`, $r>0$ | $0$ | $\mathrm{diag}(e^{-2r},e^{2r})$ ($q$ squeezed, then rotated by `angle`) |
| resource preparation, squeezing $r\ge0$ | $0$ | $\mathrm{diag}(e^{2r},e^{-2r})$ ($p$ squeezed) |

Useful scalar functions (all implemented on `GaussianState`): the purity is
$\det V^{-1/2}$; the mean photon number of mode $i$ is
$(\mathrm{Tr}\,V_i+\mu_i^{T}\mu_i-2)/4$; the parity is
$\exp(-\mu^{T}V^{-1}\mu/2)/\sqrt{\det V}$; the overlap of two $n$-mode states is
$2^{n}\exp(-\Delta^{T}(V_1+V_2)^{-1}\Delta/2)/\sqrt{\det(V_1+V_2)}$, which equals the
fidelity only when at least one state is pure (otherwise it is $\mathrm{Tr}\,\rho_1\rho_2$, and
`fidelity` uses the general Uhlmann formula instead).

## Symplectic maps

A Gaussian unitary acts on the quadratures linearly, $R\mapsto SR+d$, with $S$ symplectic
($S\Omega S^{T}=\Omega$). On states,

$$\mu\mapsto S\mu+d,\qquad V\mapsto SVS^{T}.$$

| Gate (command) | Generator / unitary | Symplectic block |
|---|---|---|
| `Rotate(phi)` | phase rotation | $\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$ |
| `Squeeze(r)` | single-mode squeezing | $\mathrm{diag}(e^{-r},e^{r})$ (positive $r$ squeezes $q$) |
| `QuadraticPhase(s)` | $e^{isq^{2}/4}$ | $\begin{pmatrix}1&0\\ s&1\end{pmatrix}$ ($p\mapsto p+sq$) |
| `Entangle(u, v, g)` (weighted CZ) | $e^{ig\,q_uq_v/2}$ | $p_u\mapsto p_u+g\,q_v,\ p_v\mapsto p_v+g\,q_u$ |
| `BeamSplitter(theta)` | passive two-mode | $q_u\mapsto\cos\theta\,q_u-\sin\theta\,q_v$, $q_v\mapsto\sin\theta\,q_u+\cos\theta\,q_v$ (same for $p$) |
| `Displace(q=x, p=z)` | $X(x)Z(z)$ | $d=(x,z)$; $X(s)=e^{-isp/2}$, $Z(t)=e^{itq/2}$ |

$X$ and $Z$ commute up to a global phase, which no stored state observes. The native engine
updates only the $k$ affected modes ($O(nk)$ work), the NumPy reference builds the full
matrix; both are checked against SciPy `expm` of the generators
([independent validation matrix](../validation/independent-validation-matrix.md)).

## Gaussian channels

A Gaussian channel acts as $\mu\mapsto X\mu$, $V\mapsto XVX^{T}+Y$ with
$Y+i\Omega-iX\Omega X^{T}\ge0$. The implemented channel is the thermal-loss channel
`Loss(node, transmissivity=eta, thermal_photons=n_th)`:

$$X=\sqrt\eta\,I,\qquad Y=(1-\eta)(2n_{\rm th}+1)\,I .$$

Pure loss is $n_{\rm th}=0$. The same $(X,Y)$ form describes the *unconditional* effect of a
finite-squeezing teleportation step; see [CV cluster states](cluster-states.md). PhotoGraphiQ
reports the intended linear map and the added noise **separately**
(`gaussian_channel(pattern).matrix` and `.noise`): a finite-squeezing step is never an ideal
unitary.

## Gaussian measurements: homodyne, heterodyne, general-dyne

Split the covariance and mean into the measured mode $(B,b)$ and the rest $(A,a)$, with
cross-covariance $C$:

$$V=\begin{pmatrix}B&C^{T}\\ C&A\end{pmatrix},\qquad \mu=(b,a).$$

**Homodyne at angle $\theta$** measures $h^{T}R$ with $h=(\cos\theta,\sin\theta)$, i.e.
$q\cos\theta+p\sin\theta$. With efficiency $\eta$ and additional calibrated readout noise
$\nu_{\rm add}$ the record is rescaled to the incident quadrature, so

$$v=h^{T}Bh+\frac{1-\eta}{\eta}+\nu_{\rm add},\qquad y\sim\mathcal N(h^{T}b,\ v),$$

and the surviving modes are conditioned by a Schur complement:

$$a'=a+Ch\,\frac{y-h^{T}b}{v},\qquad A'=A-\frac{(Ch)(Ch)^{T}}{v}.$$

The conditional covariance does not depend on $y$; the conditional mean does. The measured
mode is removed from the state and from the label map.

**General-dyne** with detector covariance $M$ ($M+i\Omega\ge0$) returns a phase-space pair
$y\sim\mathcal N(b,B+M)$ and updates $a'=a+C(B+M)^{-1}(y-b)$, $A'=A-C(B+M)^{-1}C^{T}$.
**Heterodyne** is $M=I$; it returns $(q,p)$, not a complex amplitude, so the vacuum record has
covariance $2I$.

A homodyne record has a probability **density**, not an event probability: `simulate`
records it as `kind="density"`. Fixing the outcome (`measurement_outcomes=` or
`outcome=`) applies the same update and is postselection, not a deterministic operation.

The following check reproduces the conditioning formula against the backend:

```python
import numpy as np
from photographiq import Homodyne
from photographiq.backends.gaussian import GaussianBackend

# Two resource modes (r = 0.6), a CZ of weight 0.8, a displacement on mode 1.
backend = GaussianBackend()
backend.prepare(0, 0.6)
backend.prepare(1, 0.6)
backend.entangle(0, 1, 0.8)
backend.displace(1, q=0.5, p=-0.2)
s = backend.get_state()
V, mu = s.covariance, s.mean

# Inefficient homodyne on mode 0 at theta = 0.4, conditioned on the record y = 0.3.
theta, eta, y = 0.4, 0.8, 0.3
h = np.array([np.cos(theta), np.sin(theta)])
B, C, A = V[:2, :2], V[2:, :2], V[2:, 2:]
v = h @ B @ h + (1 - eta) / eta                     # calibrated record variance
mean_ref = mu[2:] + C @ h * (y - h @ mu[:2]) / v
cov_ref = A - np.outer(C @ h, C @ h) / v

backend.measure(0, Homodyne(efficiency=eta), theta, outcome=y)
out = backend.get_state()
print(np.allclose(out.mean, mean_ref), np.allclose(out.covariance, cov_ref))   # True True
print(backend.last_measurement["kind"])             # 'density'
```

and the thermal-loss channel:

```python
import numpy as np
from photographiq import GaussianInput
from photographiq.backends.gaussian import GaussianBackend

eta, nth = 0.7, 0.2
backend = GaussianBackend()
backend.prepare(0, state=GaussianInput.squeezed(0.5))     # V = diag(e^-1, e^1)
V_in = backend.get_state().covariance
backend.loss(0, eta, thermal_photons=nth)
V_out = backend.get_state().covariance
print(np.allclose(V_out, eta * V_in + (1 - eta) * (2 * nth + 1) * np.eye(2)))   # True
```

## Displacement frame

With `simulate(..., frame=True)` displacements are deferred: a pending displacement $f$ is
transformed as $f\mapsto Sf$ by Gaussian gates and $f\mapsto\sqrt\eta f$ by loss. A homodyne
outcome is shifted by $h^{T}f$ (general-dyne by $f$), and the frame is materialized at outputs
and before non-Gaussian operations. This is a trajectory-dependent classical frame, not a
symbolic qubit Pauli frame, and it is available only on Gaussian backends.

## Further reading

- [CV cluster states and teleportation](cluster-states.md)
- [Gaussian compilation](compilation.md)
- [Conventions](../getting-started/conventions.md) and the [Gaussian user guide](../user-guide/gaussian.md)
