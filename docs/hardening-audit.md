# Pre-change physics and release audit

Baseline: `8705e6934cc5f83b257d20c3c4bf8aa6da411a1e` (Included non-Gaussianity).
The requested production modules and analytical, Fock, non-Gaussian, cutoff,
Piquasso and Graphix test suites were inspected before changing production code.

## Remote evidence

[Run 34454699908](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/runs/34454699908)
failed on Ubuntu 24.04 / Python 3.13.15. Job 102798272710 collected 87 tests,
passed 84 and failed the three reported injection assertions. Jobs for 3.11 and
3.12 were **cancelled by fail-fast**, not passed. Ruff, formatting, mypy and build
were skipped after pytest. The fetched log identifies Piquasso 8.0.1, NumPy 2.5.3,
SciPy 1.18.1 and Graphix 0.4. The stored Windows report cannot certify Linux CI.

## Conventions and derivation before modifications

The IR, Gaussian symplectic gates, Fock adapter and independent operators agree
on [q,p]=2i, vacuum V=I, q=a+a†, p=-i(a-a†), and native hbar=2. Piquasso Gaussian
covariance is 2V. CP(gamma)=exp(i gamma q³/6) gives p→p+gamma q²;
K(kappa)=exp(i kappa n²); Q(s)=exp(i s q²/4) gives p→p+s q.
No sign or scale change to these operations is proposed.

Chronological R_a(pi/2), CZ_ua(-1), R_a(-pi/2) gives q_a→q_a-q_u,
p_u→p_u+p_a. Measuring q_a=m yields psi(q) phi(q+m). For the cubic resource,
the extra phase is gamma(3mq²+3m²q+m³)/6. Q(-2gamma m) contributes
-gamma m q²/2 and Z(-gamma m²) contributes -gamma m²q/2, leaving the cubic
phase and global phase gamma m³/6. The envelope exp(-(q+m)²/(4exp(2r))) stays.
The resource/filter is finite-energy; the protocol is not an exact unitary.

## Numerical diagnosis

The failing density is evaluated at a **fixed outcome**. Neither homodyne CDF
integration nor inverse sampling runs on that path. The expected density is an
analytic Gaussian convolution. Thus the discrepancy is not a SciPy quadrature
tolerance boundary. It is dominated by finite Fock preparation/gate projection
and normalization; native Gaussian decompositions change that error by platform.

The untouched Windows baseline gives density errors 1.80e-5, 3.69e-6, 8.27e-7,
4.94e-8 at c=36,48,64,96. The failing Linux c=36 error is 2.02e-5. Independent
wavefunction errors also decrease. Raw SUM versus Fourier-CZ infidelity on
Windows drops from 3.96e-7 at c=24 to 1.80e-8 at c=48; Linux gives 5.95e-6
and 1.94e-7. This is finite-decomposition sensitivity, not a global-phase error:
the comparison already uses squared overlap. Low-cutoff density error need not
be monotonic (c=12→18 is a counterexample). Tests must check asymptotic trends
and independently specified high-cutoff accuracy, not arbitrary monotonicity.

## Gaps requiring targeted hardening

- Resource cutoff/mode checks currently occur after earlier preparations can
  allocate state. Mixed/invalid Gaussian covariance validation on the Fock path
  uses only a determinant and can miss nonphysical matrices or small mixedness.
- Native reduction converts a pure state into a full density matrix before
  tracing; state metrics/norm also unnecessarily allocate pure density matrices.
- High-order moments are absent. They must include intermediate ladder paths
  above stored support before returning, not introduce a false boundary in q^k.
- Homodyne checks total interval mass but ignores the error estimate in each
  partial-CDF integration. Tail adequacy needs explicit finite-support evidence.
- Fidelity is correctly squared; mixed/mixed and label/order tests are missing.
- Wigner and cat checks need independent analytic cat interference and grids.
- Graphix isomorphism tests miss semantic label errors. Domain and order tests
  must be expanded without comparing CV and qubit physics.
- CV-flow is a numerical real-linear certificate for a supplied total order.
  It is neither an arbitrary causal DAG nor Graphix binary gflow/Pauli flow.

The plan is to add failing guard regressions, high-order observables, numerical
error-budget tests and evidence artifacts; preserve physical gate definitions;
disable CI fail-fast; record environments and all checks for Python 3.11–3.14;
and report remote CI status separately from local matrix results. Release
readiness remains pending until a new remote workflow succeeds.
