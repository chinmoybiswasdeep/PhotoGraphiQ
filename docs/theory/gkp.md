# Finite Gaussian-comb GKP states

For hbar=2, let $L=\sqrt{2\pi}$. The implemented logical basis wavefunction is
proportional to

$$\psi_\mu(q)=\sum_{s=-S}^{S}
\exp[-(q-(2s+\mu)L)^2/(4\Delta^2)]
\exp[-\kappa^2((2s+\mu)L)^2/4],\quad \mu\in\{0,1\}.$$

Here Delta is `peak_width`, kappa is `envelope`, and S is `peaks`. Each isolated
peak has probability variance Delta². The finite envelope makes the state
normalizable. Numerical grid normalization precedes Fock projection
$c_n=\int dq\,\langle n|q\rangle\psi_\mu(q)$.

The projected norm is reported before the finite vector is normalized. This norm
does not certify the lattice-sum or grid approximations. Finite logical states
need not be exactly orthogonal, so superpositions use their actual overlap.

The two stabilizer translations are $e^{-iLp}$ and $e^{iLq}$. Their commutator
phase is unity because $2L^2=4\pi$. Logical shifts use half those translations.
The basic decoder reduces a measured displacement modulo L and reports the cell
parity. Finite ancillas retain likelihood-dependent filtering and can introduce
back-action; a one-round correction example does not establish fault tolerance.

The construction follows the oscillator-code principles of
[Gottesman, Kitaev and Preskill](https://arxiv.org/abs/quant-ph/0008040).
See [the resource guide](../user-guide/gkp.md) for numerical controls.

## Derivation and independent references

Position translation is $X(a)=e^{-iap/2}$ and momentum translation is
$Z(b)=e^{ibq/2}$. The Weyl relation is
$X(a)Z(b)=e^{-iab/2}Z(b)X(a)$. Choosing a=b=L gives the logical Pauli
anticommutator when $L^2=2\pi$. Squaring gives stabilizers $X(2L),Z(2L)$
with phase $e^{-2iL^2}=1$; each also commutes with the other logical operator.

For equal peak width Delta, Gaussian-peak overlap integrals are
$\sqrt{2\pi}\Delta\exp[-(c-d)^2/(8\Delta^2)]$, for centers c,d.
Summing with the envelope weights gives independent analytic normalization and
logical overlap without grid quadrature or Fock projection. Shifting centers
and integrating an added phase similarly validates displacement expectations.

The q-syndrome SUM yields the unnormalized wavefunction $\psi(q)\phi(m-q)$.
For decoded residual r, translation by -r gives
$\psi(q+r)\phi(m-q-r)$. Its norm before correction is the measurement density.
The p branch Fourier-conjugates the complete protocol, giving the same kernel in
p coordinates. Ancilla envelopes and fixed measurement records must be retained.

Decoder tests include exact floating-point half-cell boundaries and their adjacent
representable values. At sufficiently large shifts, floating-point spacing loses
subcell resolution, so parity/residual interpretation requires a resolved input.
