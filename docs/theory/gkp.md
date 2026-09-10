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
