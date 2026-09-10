# Gradient contracts

For a fixed measurement branch m with unnormalized output sigma(theta), the
conditional expectation is

$$f_m(\theta)=\frac{\mathrm{Tr}(O\sigma_m(\theta))}
{\mathrm{Tr}(\sigma_m(\theta))}.$$

Differentiating includes both numerator and normalization derivatives. The branch
likelihood is separately $p_m=\mathrm{Tr}\sigma_m$ (a density for homodyne).
Conditional derivatives are undefined when this mass is zero. Small masses can
make the derivative ill-conditioned even if the normalized state looks ordinary.

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
