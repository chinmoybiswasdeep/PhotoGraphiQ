# Gradient contracts

For a fixed measurement branch m with unnormalized output sigma(theta), the
conditional expectation is

$$f_m(\theta)=\frac{\mathrm{Tr}(O\sigma_m(\theta))}
{\mathrm{Tr}(\sigma_m(\theta))}.$$

Differentiating includes both numerator and normalization derivatives. The branch
likelihood is separately $p_m=\mathrm{Tr}\sigma_m$ (a density for homodyne).
Conditional derivatives are undefined when this mass is zero. Small masses can
make the derivative ill-conditioned even if the normalized state looks ordinary.

For the normalized input (|00>+|11>)/sqrt(2), attenuate the first mode with
transmission eta and select zero photons there. Then p=1-eta/2, the conditional
mean occupation of the second mode is (1-eta)/(2-eta), and p times that mean is
(1-eta)/2. Their derivatives are respectively -1/2,
-1/(2-eta)^2, and -1/2; the log-likelihood derivative is -1/(2-eta).
These identities independently test the three distinct gradient objectives.

Invalid normalization raises eagerly. Traced execution returns a NaN state for
zero/invalid mass or unresolved tensor/addition truncation; zero likelihood has
log likelihood -infinity. There is no gradient assigned to an impossible branch.

For an unconditional discrete objective,

$$\nabla E_m[f_m]=E_m[\nabla f_m+(f_m-b)\nabla\log p_m],$$

where the baseline b must be independent of the sampled outcome. A fixed list of
postselected records is not an unbiased sample from p. The provided surrogate
supplies the score term; callers remain responsible for the sampling contract and
variance estimates. Finite differences serve as validation, not as the claimed
automatic-differentiation implementation.

For a Gaussian draw, $x=\mu+L\epsilon$, with $LL^T=V$ and fixed standard-normal
epsilon, gives a pathwise derivative when the covariance is positive definite.
Discrete photon outcomes do not admit this same construction.

The JAX Fock path exponentiates projected finite-dimensional generators, so a
derivative is initially a derivative of that truncated realization. Compare both
forward values and gradients over cutoffs and against independent analytical
derivatives. Compilation topology and decomposition branches remain fixed.

See [automatic differentiation](../user-guide/automatic-differentiation.md).
