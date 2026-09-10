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

## Thermal channel reference

The implementation factors thermal attenuation into vacuum loss with transmission
eta/G and a quantum-limited amplifier with $G=1+(1-\eta)\bar n_{env}$.
The amplifier's Kraus coefficient for $|n\rangle\to|n+\ell\rangle$ is
$\sqrt{\binom{n+\ell}{\ell}(G-1)^\ell/G^{n+\ell+1}}$.
An independent reference mixes the system and a thermal environment on a
number-conserving beamsplitter, then explicitly traces the environment. Tests
include input coherences, not just a thermal output mean.

Physical loss preserves trace. Truncated amplification can lose represented mass.
Retained-norm diagnostics record that trace before renormalization; it is not an
absorption probability. Loss beyond the configured tolerance raises an error.
