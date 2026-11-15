# MentPy audit

Reference: [MentPy](https://github.com/mentpy/mentpy), commit
`63c3d83e495696b4491c9d376dab7e3e6cf6c863`, package `0.1.0a15`.
Documentation: <https://docs.mentpy.com/en/latest/>.
Implementation is independent; no MentPy source is vendored into the package.
MentPy is Apache-2.0 reference software and an optional test dependency.

Inspected `mbqc/templates.py`, `mbqc/mbqcircuit.py`, `states/graphstate.py`,
`operators/ment.py`, `simulators/pattern_simulator.py`, numpy statevector
execution, optimizer Adam, generated datasets, noise base and Lie tools.
GraphState extends NetworkX; its equality uses isomorphism, insufficient for
semantic comparison. MBQCircuit stores measurements, inputs, outputs and flow.
Ment represents fixed Pauli or parametrized plane measurements.
PatternSimulator dispatches PennyLane and numpy density/statevector backends.
The numpy SV backend defaults to the zero-outcome branch and disallows random
outcomes; numerical agreement there alone does not validate feed-forward.

`templates.muta(n,L)` stacks n pivot blocks per layer unless `one_column=True`.
Paper depth d=L*n or d=L respectively. `hstack` identifies boundary vertices;
integer labels are compacted in insertion order. Semantic coordinates (wire,
global column) are the stable comparison keys. Flow permits multiple total
orders; compare causal validity rather than requiring identical tie breaking.

`restrict_trainable` removes column-3 nodes from a list but does not fix their
Ment angles. Constructing a new MBQCircuit during composition reconstructs
that list from the still-parametrized measurements, losing the restriction.
Adding cross edges also rebuilds attributes: even a single multi-wire block
loses the restriction. Only an unstacked one-wire block retains the list
restriction, with the omitted measurement still having angle=None.
Reference tests must report this discrepancy. Production models explicitly
fix them at zero and retain the restriction across layers.

Adam delegates gradients; SGD and coordinate descent are available. Losses
are user callables (there is no separate mature losses package). Haar and
Brownian utilities use global randomness; PhotoGraphiQML uses local seeded
generators. The noise module is a minimal base, not a complete physical noise
backend. Lie generators derive from correction operators; histogram-based
expressivity is a numerical diagnostic, not a proof of finite-depth inclusion.

Docs and source APIs are inspected as references; successful runtime tests
and exact reference version are recorded separately in the release report.
