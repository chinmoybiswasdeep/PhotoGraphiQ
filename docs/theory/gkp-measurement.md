# Encoded measurement derivation

We retain [q,p]=2i, vacuum covariance I, L=√(2π),
X_L=exp(-iLp/2) and Z_L=exp(iLq/2). Ideal zero and one have
q=(2s+μ)L. Therefore q-homodyne distinguishes μ through cell parity.
Fourier transformation with kernel exp(-ipq/2)/√(4π) maps logical plus/minus
to even/odd p lattice cells. Thus X readout uses **p**, not q homodyne.
The code and Gaussian logical-gate setting originate in
[Gottesman, Kitaev and Preskill (2001)](https://arxiv.org/abs/quant-ph/0008040).

For interval C_m=[(m-1/2)L,(m+1/2)L), the physical effects are

    E_b = Σ_{m mod 2=b} ∫_{C_m} |x><x| dx.

In a finite Fock space, compress these operators with P_N. The resulting effects
remain positive and sum to I_N; they are generally not projectors. The implementation
integrates each full cell separately, checks the finite quadrature window against
I_N and never renormalizes overlaps with logical codewords.

For analog readout the conditional survivor is obtained with the quadrature bra
<x|, preserving all inter-mode correlations. If only parity is retained, the
conditional survivor is an integral over analog branches and is generally mixed.
A coarse-grained POVM and a trajectory retaining x are distinct instruments.

## Ideal reference and finite discrimination

`IdealLogicalXYMeasurement` acts on a normalized two-dimensional ideal qubit:
|±_α>=(|0>±exp(iα)|1>)/√2. It is deliberately not a Pattern measurement.
Finite oscillator codewords are nonorthogonal, so their naive superposition
projectors do not form this resolution of identity.

Let B contain the two normalized finite codewords, G=B†B, and D=BG^-1.
The reciprocal vectors satisfy D†B=I. With w=λ_min(G), define
E_μ=w|d_μ><d_μ|. Their sum has eigenvalues w/λ_i(G) on span(B), hence
it is at most the subspace projector P=BG^-1B†. Complete the POVM with
E_inc=P-E_0-E_1 and E_out=I-P. Correct-label probability on each basis
state is w, wrong-label probability is zero, and inconclusive probability is 1-w.
The square-root Kraus choice defines a mathematical Lüders instrument explicitly.
This algebra does not provide an optical implementation of those matrices.

## Logical gates and unsupported angles

The H realization is exp(iπn/2), with (q,p)→(-p,q). On the ideal square
lattice the Fourier transform exchanges X and Z up to irrelevant signs/stabilizers.
S is exp(iq²/4), giving p→p+q. At q=mL its phase is exp(iπm²/2),
equal to 1 for even m and i for odd m. Physical CZ is exp(iq₁q₂/2);
at lattice sites its phase is exp(iπmn)=(-1)^(mn). These identities are exact
only for the ideal lattice. Finite peak phases and displaced/sheared envelopes
produce real target infidelity and code-subspace leakage.

For a Y readout, ideal S† preprocessing followed by X has observable
S X S†=Y. This identifies a candidate physical shear/readout construction.
It is not enabled here because its complete finite-energy instrument and
convergence tests have not been established. In particular homodyne(π/4) is not Y.

For arbitrary XY, measuring X after Rz(-α) would give
Rz(α) X Rz(-α)=cos(α)X+sin(α)Y. A generic logical Rz is not supplied
by rotating an oscillator homodyne axis. Non-Clifford synthesis or injection,
including ancillas, all branches, correction rules and resource/cutoff validation,
remains missing. The π/4 case is unsupported too.
