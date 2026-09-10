# Mixed-state conditioning

Represent the density matrix on explicit occupation tuples. A gate acts as
$\rho\mapsto U\rho U^\dagger$. A channel uses a Kraus sum
$\rho\mapsto\sum_k K_k\rho K_k^\dagger$; tracing an environment can destroy purity.

For vacuum attenuation of intensity transmission eta,

$$K_\ell|n\rangle=\sqrt{\binom n\ell}
(1-\eta)^{\ell/2}\eta^{(n-\ell)/2}|n-\ell\rangle.$$

This gives the independent binomial number-state loss reference. Thermal
attenuation also has $\langle n_{out}\rangle=\eta\langle n_{in}\rangle+
(1-\eta)\bar n_{env}$, with thermal tails requiring cutoff refinement.

A selected PNR result uses $K=\langle k|$ on the measured mode. Homodyne uses
$K_x=\langle x_\theta|$, evaluated through normalized Hermite wavefunctions at
hbar=2. The unnormalized survivor is $K_x\rho K_x^\dagger$; its trace is a density,
and division by that trace gives the conditional state.

Inefficient detection first attenuates the measured mode. Because the public
outcome x is calibrated to the incident quadrature, the detector coordinate is
$y=\sqrt\eta x$ and the reported density includes the Jacobian $\sqrt\eta$.
Independent electronics noise convolves the *unnormalized* conditional operator
with a Gaussian density in x. Convolving normalized states would incorrectly
discard their different likelihood weights.

Density matrices scale as D², where D is the total-photon basis dimension.
Check trace, positivity, basis ordering and cutoff convergence before interpreting
a mixed-state observable. See [mixed Fock usage](../user-guide/mixed-fock.md).
