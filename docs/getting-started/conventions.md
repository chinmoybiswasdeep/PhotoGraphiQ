# Conventions

We use $[q,p]=2i$, $q=a+a^\dagger$, $p=-i(a-a^\dagger)$, and interleaved mode order
$(q_0,p_0,q_1,p_1,\ldots)$. Vacuum statistical covariance is $V=I$.
Piquasso's exported anticommutator covariance is $2V$ at the same hbar.

| Quantity | Convention |
|---|---|
| Angles | radians |
| Coherent amplitude | $\langle q\rangle=2\Re\alpha$, $\langle p\rangle=2\Im\alpha$ |
| Displace(q=x,p=z) | quadrature translations x,z |
| CZ(g) | $p_i\mapsto p_i+gq_j$, $p_j\mapsto p_j+gq_i$ |
| BeamSplitter(theta) | $q_i\mapsto q_i\cos\theta-q_j\sin\theta$, $q_j\mapsto q_i\sin\theta+q_j\cos\theta$ (same for $p$) |
| Squeeze(r)>0 | $S=\mathrm{diag}(e^{-r},e^{r})$ ($q$ squeezed) |
| Fock cutoff c | keeps $\sum_i n_i<c$ (total photon number, exclusive) |
| CubicPhase(gamma) | $\exp(i\gamma q^3/6)$ |
| Kerr(kappa) | $\exp(i\kappa n^2)$ |
| QuadraticPhase(s) | $\exp(isq^2/4)$ |
| Resource squeezing r>0 | momentum squeezed, $V=\mathrm{diag}(e^{2r},e^{-2r})$ |
| GaussianInput.squeezed(r)>0 | q squeezed before rotation, $V=\mathrm{diag}(e^{-2r},e^{2r})$ |

Homodyne angle theta measures $q\cos\theta+p\sin\theta$. Outcomes with inefficient
detection are calibrated to the incident quadrature: vacuum variance is
$1/\eta+\nu$, where eta is efficiency and nu is added calibrated noise.

Ideal infinitely squeezed states are not normalizable numerical inputs. Finite
squeezing is physical; Fock cutoff is a numerical approximation. Keep their error
budgets separate. Read the [theory pages](../theory/index.md) for the derivations, starting with
[Gaussian states and channels](../theory/gaussian.md).
