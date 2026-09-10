# Mathematical reference

## Canonical coordinates

Let R=(q0,p0,...), [R_i,R_j]=2i Omega_ij and
Omega=direct_sum([[0,1],[-1,0]]). Means are <R>; covariance is
V_ij=<{Delta R_i,Delta R_j}>/2. Vacuum has V=I and physical Gaussian states obey
V+i Omega >= 0. A symplectic map S sends (mu,V) to (S mu, S V S^T).
The numerical state validator checks shape, finiteness, symmetry and uncertainty.
Piquasso stores sigma=2V; conversion happens only at its backend boundary.

Rotation R(phi)=[[cos(phi),-sin(phi)],[sin(phi),cos(phi)]]. A squeezing gate has
S(r)=diag(exp(-r),exp(r)); a resource preparation uses S(-r), with r>=0.
X(s)=exp(-i s p/2), Z(t)=exp(i t q/2) shift q by s and p by t.
Their ordering differs only by a global phase, irrelevant for stored density states.
The complex displacement amplitude in Piquasso is (s+i t)/2.

## Canonical weighted graph states

For real symmetric zero-diagonal adjacency A, prepare independent modes with
Dq=diag(exp(2r_i)), Dp=diag(exp(-2r_i)), then apply
CZ_ij(g)=exp(i g q_i q_j/2). This leaves q fixed and sends p to p+Aq.
In grouped coordinate order, the resulting covariance is

```
V = [[Dq,       Dq A],
     [A Dq, Dp + A Dq A]].
```

The commuting nullifiers delta=p-Aq have covariance Dp. Their variance tends to
zero as r increases, while the full state does not converge to a normalizable
infinite-squeezing vector. `CVGraph(..., ideal=True)` retains nullifier mathematics
but cannot create a numerical Pattern. Input nodes replace their resource
preparation with supplied input states; the nullifier formula for a uniform graph
vacuum resource must not be applied blindly to arbitrary inputs.

## Destructive Gaussian measurements

Partition the measured mode and survivors as V=[[B,C^T],[C,A]], mu=(b,a).
For h=(cos(theta),sin(theta)), exact homodyne measures y=h R_measured:

```
v = h B h^T + nu
y ~ Normal(h b, v)
mu' = a + C h^T (y-h b)/v
V' = A - C h^T h C^T/v
```

Here nu=(1-eta)/eta + noise for an efficiency eta, with outcomes calibrated to
the incoming quadrature. This is detector inefficiency modeled by mixing with
vacuum and rescaling; `noise` is additional nonnegative readout variance.
The measured mode is removed from both the state and label map. For general-dyne
detector covariance M satisfying M+i Omega>=0, sample y~Normal(b,B+M), then
mu'=a+C(B+M)^-1(y-b) and V'=A-C(B+M)^-1 C^T. Heterodyne uses M=I and returns
phase-space coordinates (q,p), not complex amplitude alpha. Thus its vacuum
outcome covariance is 2I. Conditional moments are checked by Schur complements;
sample statistics are checked separately.

## One-step teleportation and finite-resource noise

Entangle input mode 0 and p-squeezed ancilla 1 with unit CZ. Measure
m=p0+k q0 and apply X1(-m). Immediately before measurement,
p0'=p0+q1 and p1'=p1+q0. Therefore after correction

```
q_out = -k q_in - p_in
p_out = q_in + p_ancilla
T(k) = [[-k,-1],[1,0]].
```

PhotoGraphiQ measures the normalized quadrature at theta=atan2(1,k) and uses
m=y/sin(theta) in feed-forward. Neglecting this rescaling would implement the
wrong gate whenever k!=0. The unconditional Gaussian channel is
V_out=T V_in T^T+diag(0,exp(-2r)). This relation refers to averaging corrected
trajectories; finite-resource conditional output means and covariances depend
on the outcome. Four k=0 steps implement identity with noise 2 exp(-2r) I.
An L-step wire recursively accumulates N_j=T_j N_(j-1) T_j^T+N_resource,j.
High squeezing reduces noise but increases source energy and covariance condition
numbers. No numeric ideal limit is hidden behind a finite-r simulation.

## Gaussian compiler

For S=[[A,B],[C,D]], det S=1 and C!=0, the chronological shear list
[0,(1-D)/C,C,(1-A)/C] gives S=T(k4) T(k3) T(k2) T(k1).
For C near zero, prepend T(0) and decompose S T(0)^-1, giving five steps.
The compiler checks its reconstructed matrix. For extreme near-singular numerical
requests it raises an error rather than returning an unstable decomposition.
Angles of circuit rotations and squeezing are numeric at compile time; symbolic
angles, squeezing and couplings remain supported in hand-built patterns and wires.

Logical CZ uses a resource edge between the current wires followed by four-step
identity transport on each wire. All operations are preparation, entanglement,
measurement and displacement correction; no direct rotation/squeezing gate is
inserted as a substitute for a compiled MBQC operation. Logical SUM is a CZ
conjugated by Fourier transforms on the target. For q-coordinate rotation,
U(-tan(phi/2)) L(sin(phi)) U(-tan(phi/2)) implements a beam splitter, where
U and L are the two SUM directions. Two half-angle decompositions avoid the
singularity at phi=pi. This implementation is deliberately resource expensive.

`gaussian_channel` propagates affine Wigner variables for fixed-angle homodyne
patterns, independently of conditional simulation. It returns S,N,d with
mu_out=S mu_in+d and V_out=S V_in S^T+N. General-dyne, nonlinear feed-forward,
outcome-dependent gates and Fock resources are rejected by this analyzer.
Trajectories still support general Gaussian measurements and nonlinear adaptation.

## CV-flow versus causal scheduling

`flow.dependency_graph` combines classical producer/consumer edges and consecutive
touches of each quantum mode. Topological scheduling preserves these requirements.
It does not prove determinism. `cvflow.certify_cv_flow` implements a separate
real-linear condition for a supplied total order, following Booth and Markham.
For each measured j, let P(j) be the measured prefix including j; form
A[P(j), V minus (P(j) union I)]. Solve A_cut c_j=e_j over the reals.
A certificate reports the correction coefficients and numerical residuals.
Failure for one order does not imply the graph has no CV-flow in another order.
`flow_pattern` applies X(-c_j m) and the corresponding Z(-A c_j m) on surviving
nodes. The guarantee is about the ideal-limit protocol, not a finite-r unitary.

## Displacement frame and safe rewriting

A pending displacement f is transformed by f->S f through Gaussian gates and
by f->sqrt(eta) f through loss. For homodyne, the frame shifts the reported
outcome by h f, and for general-dyne by f. The frame on surviving modes persists.
Corrections are materialized at outputs and before non-Gaussian operations.
The implementation is a trajectory-dependent classical frame, not a symbolic
qubit Pauli frame. `Pattern.standardize` only swaps preparations/entanglers past
disjoint commands whose classical values are already available, combines adjacent
displacements and removes numeric zero shifts. It makes no full normal-form claim.

## Fock truncation and observables

The experimental backend uses the total-photon subspace sum(n_i)<cutoff. Dimension
is binomial(d+cutoff-1,d), not cutoff^d. Pure state vectors are evolved by
Piquasso. Photon counting samples exact finite-space probabilities and projects
the remaining amplitudes. Each active gate/tensor preparation checks retained norm;
deviation above the configured tolerance raises an error, otherwise the norm is
recorded and the vector normalized. Norm loss alone is not a rigorous truncation
error bound: observable convergence versus increasing cutoff is still required.

Number, cat and arbitrary normalized Fock vectors are available. Photon addition
and subtraction construct normalized offline resources; their normalization is
not a detector heralding probability. A cubic-phase gate can create a truncated
non-Gaussian resource. Fock homodyne conditioning, mixed-input injection and GKP
states are not implemented. No universal non-Gaussian MBQC compilation is claimed.

Gaussian mean photon number is (Tr V_i+mu_i^T mu_i-2)/4.
Parity is exp(-mu^T V^-1 mu/2)/sqrt(det V). Gaussian overlap is
2^n exp(-Delta^T(V1+V2)^-1 Delta/2)/sqrt(det(V1+V2)); it equals fidelity when at
least one state is pure, and is otherwise only Tr(rho1 rho2).
