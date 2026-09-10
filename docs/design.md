> Historical v0.1 design audit. The v0.2 extension is recorded in [non_gaussian_design.md](non_gaussian_design.md).

> Historical v0.2 reference. For the v0.3 additions and current backend support, see the [documentation home](index.md) and [feature matrix](validation/feature-matrix.md).

# Pre-implementation design record

This record was written before implementation, following inspection of the supplied
Piquasso and Graphix papers, documentation and upstream source on 2026-09-10.

## Sources and compatibility target

* Piquasso 8.0.1, source commit `bd47eebdf2a0dd857aae48050c9cecfea1864093`:
  https://github.com/Budapest-Quantum-Computing-Group/piquasso
* Graphix 0.4, inspected upstream commit `2ea820e2978cfc8be8471814c9d938c2ea92abee`:
  https://github.com/TeamGraphix/graphix
* https://piquasso.readthedocs.io/en/stable/index.html
* https://graphix.readthedocs.io/en/latest/index.html
* Gu et al., Phys. Rev. A 79, 062318 (2009), https://arxiv.org/abs/0903.3233
* Booth and Markham, Quantum 7, 1146 (2023),
  https://doi.org/10.22331/q-2023-10-19-1146

Inspected Piquasso modules: Gaussian state, simulator and simulation steps;
Fock pure/general instruction maps and homodyne implementation; instructions for
preparation, Gaussian transformations, measurements and channels; Config and Result.
Inspected Graphix modules: command, pattern, optimization, parameter, transpiler,
simulator, backend base, open graph, causal flow and generalized/Pauli flow.
The supplied papers explain the motivation and architecture; installed source is
the executable compatibility reference because the papers predate these releases.

## A. Capability matrix

| Graphix concept / mathematical meaning | CV analogue | Piquasso primitive | Strategy | Validation |
|---|---|---|---|---|
| N: prepare plus state | finite momentum-squeezed vacuum | Squeezing(-r) | labelled weighted resource | nullifier covariance |
| E: qubit CZ | exp(i g q_i q_j / hbar) | ControlledZ(s=g) | weighted entangle command | independent symplectic matrix |
| M: projective qubit measurement | quadrature POVM | Gaussian moments | exact Schur conditioning | analytic and raw state comparison |
| X/Z: binary Pauli corrections | real q/p translations | Displacement | expression-driven correction | means and frame equivalence |
| measurement domains | real-valued classical signals | orchestration above backend | safe expression tree and declared callable dependencies | causal rejection tests |
| Pattern / measurement calculus | preparation, CZ, measurement, correction sequence | sequential execution | backend-independent IR | structural Graphix examples |
| signal shifting | affine outcome/frame changes | no extra primitive | phase-space frame tracking | identical seeded trajectories |
| standardization | commute only proven independent operations | no extra primitive | conservative dependency-preserving rewrite | state equivalence |
| flow / gflow | CV-flow is a separate real-linear theory | none | DAG scheduling first; no false flow certificate | graph/order comparisons |
| transpiler | Gaussian teleportation gadgets | CZ, squeezing, homodyne, displacement | single-mode symplectic decomposition and two-wire CZ | transfer matrices and finite-noise channel |
| statevector/tensor network | Gaussian moments or truncated Fock state | Gaussian / PureFock / Fock simulators | separate backend capabilities | analytical and raw backend tests |
| noise model | loss, thermal channel, detector noise | Attenuator | explicit channels | covariance channel equation |
| serialization and visualization | labelled resources and dependencies | none | versioned JSON and matplotlib | round trips and smoke tests |

## B. Package architecture

Use a `src/photographiq` package. Small modules contain `graph`, `commands`,
`expressions`, `measurements`, `pattern`, `flow`, `compiler`, `protocols`,
`simulator`, `states`, `serialization`, `visualization`, and `gaussian`.
`backends/base.py` defines the execution contract; `backends/gaussian.py` is the
independent NumPy implementation; `backends/piquasso.py` is the main physical
adapter; `backends/fock.py` isolates cutoff-dependent functionality.
Tests are partitioned into unit, analytical, integration, piquasso_reference,
graphix_reference, protocols and regression. Examples, reproducible experiments,
documentation and the Quantum manuscript live outside the runtime package.

## C. Mathematical conventions

Internally hbar=2, [q,p]=2i, R=(q0,p0,q1,p1,...).
V_ij = <{Delta R_i,Delta R_j}>/2; vacuum V=I. Piquasso sigma=2V.
The uncertainty condition is V+i Omega >=0. Resource r>=0 means p squeezed:
V=diag(exp(2r),exp(-2r)), implemented by Piquasso Squeezing(r=-r).
Rotation is R(phi)=[[cos(phi),-sin(phi)],[sin(phi),cos(phi)]].
CZ(g) sends p_i to p_i+g q_j and p_j to p_j+g q_i.
X(s)=exp(-i s p/hbar), Z(t)=exp(i t q/hbar) shift q,p by s,t.
Piquasso displacement alpha=(s+i t)/2. Homodyne measures
q_theta=q cos(theta)+p sin(theta), angles in radians.

One unit-weight cluster step measures p+kq, reports m, and corrects X(-m).
Its ideal map is T(k)=[[-k,-1],[1,0]]. Finite resource squeezing gives an
unconditional Gaussian channel V -> T V T^T + diag(0,exp(-2r)).
Individual conditional trajectories do NOT obey that unconditional equation.
Homodyne records the normalized quadrature, so m=record/sin(theta),
theta=atan2(1,k). This scaling is mandatory for feed-forward.

## D. Prioritized roadmap

1. Conventions, graph validation, Gaussian states and independently tested gates.
2. Typed command IR, safe symbolic parameters, causal validation and execution.
3. Piquasso gate adapter, exact measurements, repeated shots and frame tracking.
4. Teleportation gadgets, Gaussian compiler, finite-noise channel validation.
5. Conservative rewriting, serialization, visualization and structural tests.
6. Explicit Fock capability layer, noise, examples and reproducible experiments.
7. Documentation, Quantum-style manuscript, complete checks and evidence record.

## E. Features not directly ported

Binary outcome parity, GF(2) gflow equations, Pauli measurement removal,
the finite local-Clifford group, Bloch-sphere planes, and qubit local
complementation optimizations are not imported as CV rules. A dependency DAG
certifies causality only, not determinism, unitarity or CV-flow. The literature
contains CV-flow; a complete CV-flow finder is a separate future extension.
Post-design implementation: `cvflow.py` now checks its real-linear equations for
a supplied total order and constructs the associated protocol. It does not search
orders. Couplings are frozen at certification so later parameter changes cannot
invalidate the generated corrections.

## F. Backend limitations and discrepancies

Piquasso Gaussian photon counting cannot return a non-Gaussian conditional state.
PureFock homodyne returns samples with no conditional state in the inspected
implementation; adaptive Fock homodyne must be rejected until implemented.
Global total-photon cutoff is not a per-mode cutoff. Truncation must be reported.
Autodifferentiation connectors exist upstream; Python stochastic orchestration
does not thereby become differentiable. Repeated trajectories are not vectorized.
Piquasso Graph is a Gaussian boson-sampling construction, not the canonical
momentum-squeezed CZ graph used here; prepare this resource explicitly.

The ControlledZ docstring has a negative exponent whereas its Bogoliubov blocks
give p_i -> p_i+s q_j: follow the tested transformation. Homodyne's docstring says
z -> infinity, but diag(z^2,z^-2) approaches q homodyne as z -> 0.
The inspected Gaussian sampler passes sigma+sigma_detector to NumPy without
dividing by two, despite sigma being the anticommutator covariance. PhotoGraphiQ
uses statistical V for its own exact conditioner. Tests must distinguish raw
gate/state agreement, conditional-state agreement and sampling statistics;
raw measurement samples are not silently treated as a physics oracle.

## G. Minimal API

```python
import photographiq as pg

graph = pg.CVGraph.line(3, squeezing=1.2, inputs=(0,))
pattern = pg.Pattern(graph)
pattern.measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
pattern.measure(1, pg.Homodyne(angle=pg.Parameter("theta")))
result = pg.simulate(pattern, parameters={"theta": 1.0}, seed=12)
print(result.outcomes, result.state.mean, result.state.covariance)
```

Non-Gaussian universality, fault tolerance, a hardware temporal-mode scheduler,
and PhotoGraphiQML are not claimed by v0.1. Unsupported operations fail explicitly.
