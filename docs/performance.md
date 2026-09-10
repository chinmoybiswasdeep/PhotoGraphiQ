# Performance and allocation

For m modes and exclusive total-photon cutoff c, the Fock dimension is
$D=\binom{m+c-1}{m}$. A complex128 vector needs 16D bytes; a density matrix needs
16D² bytes before gate workspaces. This is not c raised to the number of modes.

For example, two modes at cutoff 20 give D=210, while four modes give D=8855.
Pure-Fock default allocation guards limit vector dimension; mixed Fock also
checks a configured density-matrix byte budget. Differentiable dense evolution
has its own conservative dimension limit. Observable calculations and native
partial traces can allocate additional dense matrices.

Gaussian covariance storage is quadratic in modes; dense factorizations and
physicality checks can be cubic. Fock matrix exponentials and mixed-state metrics
are far more expensive. Product-formula Kerr compilation can create many Gaussian
teleportation steps and nonlinear injections even for one logical gate.

Start with one/two-mode examples, use Gaussian execution when sufficient, and
increase cutoffs only after inspecting energy and boundary weight. Record elapsed
time and environment for your experiment; this project makes no cross-package
benchmark claims from a single machine's timings.
