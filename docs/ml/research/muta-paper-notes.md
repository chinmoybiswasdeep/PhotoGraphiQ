# MuTA paper analysis

Source: Mantilla Calderón et al., *Measurement-based quantum machine learning*,
[PRA 113, 042421 (2026)](https://doi.org/10.1103/2snk-m8c6).
The supplied 14-page PDF was read in full, including Appendices A–E.
Equations below use its logical convention; they are not CV gate identities.

## Definition and execution (II, III, Appendix B)

An open graph has inputs I, outputs O and |+> resources elsewhere. Apply CZ on
each edge, then measure M(alpha)=cos(alpha)X+sin(alpha)Y on V minus O.
Its positive eigenbra is (1, exp(-i alpha))/sqrt(2).
A wire step therefore implements H exp(i alpha Z/2), up to a global phase.
An MBQC neuron is a local parametrized measurement in this adaptive network;
there is no classical nonlinear activation acting on statevector amplitudes.

A paper layer (n,i) contains n five-site wires Q(w,k), k=0,...,4. For each
connected base wire j, connect Q(i,1) to Q(j,0) and Q(j,2). The named triangle
T(i,j,1) contains the tip and three base sites: it is a four-vertex cycle,
not a three-cycle. Its base center Q(j,1) controls the XX rotation. Inputs
are column 0; outputs column 4; every other measurement is XY. The original
paper permits all 4n measurement angles, including column 3.

Concatenation identifies an output with the next input, giving n(4d+1) nodes,
4nd wire edges and 2 times the number of triangles cross edges. Full
connectivity adds 2d(n-1) edges. There are 4nd unrestricted parameters.
For fixed column-3 X measurements there are 3nd parameters. This restriction
is a choice, not the full paper ansatz.

Flow is the next site on each wire. For outcome s_v=1, correct with
X on f(v) and Z on N(f(v)) minus {v}. The causal DAG includes all those
targets. Equivalently, accumulated Pauli frames change each angle to
(-1)^x alpha + pi z and determine final output corrections.

Appendix B supplies an efficient logical circuit translation: process nodes
in flow order, apply remaining junction CZs on the active wires, then
H exp(i alpha Z/2) and advance that wire. This costs statevector memory
O(2^n), rather than O(2^{n(4d+1)}). It is our ideal logical execution path.
An independent small graph-state contraction validates the adaptive branches.

Equation B2 can also be written (rightmost acts first):

    U = [product_k exp(i a[k,3] X_k/2) exp(i a[k,2] Z_k/2)]
        exp(i a[i,1] X_i/2)
        [product_{j != i} exp(i a[j,1] X_i X_j/2) exp(i a[j,0] Z_j/2)]
        exp(i a[i,0] Z_i/2).

For disconnected j replace XX by X_j. Table I is checked with a single
(2,0) layer: only a[1,1]=phi gives IsingXX(-phi). All zeros give identity.

## Claims and their limits

Properties 1–7 establish flow determinism, universality with sufficient
unrestricted layers, tunable XX entanglement, architectural bias, linear
parameters per paper layer, weak monotonic inclusion and bipartiteness.
For an isolated XX gate on |00>, reduced purity is (1+cos(phi)^2)/2.
An entangling gate need not entangle every input.

Bias engineering (Property 4) means topology/connectivity and allowed-angle
constraints. Freezing measurements explicitly implements part of that bias;
CV displacement offsets have a different physical meaning.

Adding an all-X layer is identity in the ideal qubit model, proving depth
inclusion. Adding triangles alone is not the stated theorem: Property 6(3)
also inserts a compensating layer. Finite squeezing adds noise even at
identity settings, so the ideal theorem does not establish physical-channel
inclusion. Appendix C concerns the infinite-depth dynamical Lie algebra;
su(2^n) has dimension 4^n-1, not the dimension of every finite-depth family.

## Experimental targets (IV, V, Appendices D, E)

| Target | Paper protocol | Reproduction requirement |
|---|---|---|
| Fig. 3 gates | Haar U on first qubit and IsingXX(pi/2), one (2,0) layer; ten Haar states, 7/3 split; Adam; 20 runs | Record independent datasets and initializations; Eq. 2 mean infidelity |
| Fig. 4 noisy data | Brownian and bit flips, five runs; N=20/100; equal splits; 60/200 steps | Noiseless test targets; do not confuse noise on labels with resource noise |
| Fig. 5 QFI classification | Disconnected two wires, tied angles; quadratic head in p00,p11; S1 and S2 families; 50 each, 80/20; margin 0.5 | Report excluded near-threshold states separately |
| Fig. 6 instrument | Three-stage teleportation, destructive Z nodes, controlled XY measurements; 35/15 states, ten runs | Compare both branch probabilities and conditional quantum outputs |
| Figs. 7–8 kernel | Eq. 5 feature map; circles, blobs, moons; 160/40 split | Fidelity kernel, SVM and classical baselines |
| Figs. 9–10 HEA | Angles 0,pi/4,pi/2; epsilon-greedy slices or DQN | Logical discrete-angle search is not a physical GKP simulation |
| Fig. 11 resource noise | Independently apply (1-p)rho + p(XrhoX+YrhoY+ZrhoZ)/3 to all nodes | Average all adaptive outcomes; cannot postselect only zero |

Equation 2 is 1-mean |<target|prediction>|^2. Equation 3 is a soft-margin
classification objective centered at QFI=2. Equation 5 is
Rz_1(-x1) Rz_0(-x0) exp(i cos(x0)cos(x1)XX/2)
Rz_1(-x1) Rz_0(-x0)|00>. Appendix E supplies search pseudocode.
The text normalizes h so ||h||=1/2, but later says h=Z: reproductions must
declare whether they use Z/2 (SQL=2) or rescale thresholds for Z.
MentPy's Brownian helper conjugates U by noise; IV B describes noisy output
labels V U. These are different experiments and must be labeled.

Section V discusses GKP-accessible logical Pauli bases and magic-state
injection for pi/4, not a replacement alpha -> homodyne angle on arbitrary
bosonic states. Arbitrary logical XY measurements need additional resources
or gate synthesis, an explicit protocol and convergence validation.
