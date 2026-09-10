# API and extension guide

| Module | Public concepts | Execution contract |
|---|---|---|
| graph | CVGraph, ClusterState | labels, resource edges, squeezing and I/O |
| states | GaussianState, GaussianInput, FockInput | hbar=2, validated moments or normalized amplitudes |
| measurements | Homodyne, Heterodyne, Generaldyne, PhotonNumber | destructive outcome and conditioning semantics |
| expressions | Expr, Parameter, Outcome, CallableExpression | finite real values with declared dependencies |
| commands | Prepare, Entangle, Measure, Displace, Rotate, Squeeze, BeamSplitter, Loss, CubicPhase, Signal, Output | backend-independent sequence |
| pattern | Pattern | construct, validate, inspect, copy, standardize, serialize, draw |
| flow | dependency_graph, topological_schedule | command-level causal analysis |
| cvflow | CVFlow, certify_cv_flow, flow_pattern | real-linear supplied-order certificate and protocol |
| compiler | Circuit, compile_circuit, decompose_symplectic | Gaussian gates to teleportation patterns |
| protocols | wire, identity, displacement, rotation, squeeze, gaussian, entangling, teleportation, adaptive | reusable computations |
| analysis | GaussianChannel, gaussian_channel | unconditional S,N,d for fixed-angle affine homodyne patterns |
| simulator | simulate, sample, run_shots, Result, ShotResult | independent conditional trajectories and ensembles |
| backends | BaseBackend, GaussianBackend, PiquassoBackend | physical execution, Gaussian state export |
| backends.fock | PiquassoFockBackend, FockResultState | experimental total-cutoff pure Fock execution |
| gaussian | omega, rotation, squeezing, cz, beamsplitter, wire_channel | independent matrices and wire noise |
| serialization | dumps, loads | version 1 allowlisted JSON |
| visualization | draw_graph, draw_pattern, draw_dependencies | matplotlib figures |

`Prepare(node, squeezing, state)` creates a fresh labelled mode. If `state` is
supplied it overrides the resource squeezing preparation. Inputs are prepared by
the execution engine and must not be prepared again in the command list.
`Measure(node, measurement, key=None)` removes its node and writes one key,
defaulting to the node label. `Output(nodes)` is optional; if supplied it must
be last and may trace out additional surviving modes. `Entangle` is physical CZ,
whereas `Circuit.cz` compiles a logical gate including transport. `Rotate` and
`Squeeze` commands are physical operations; the compiler implements their circuit
counterparts using homodyne gadgets.

`GaussianState.quadrature(node,angle)` returns (mean,variance).
`reduced(nodes)` respects requested label order; `photon_number`, `parity` and
`overlap` provide Gaussian observables. `FockResultState.probabilities` is an
occupation-tuple mapping; `density_matrix`, `photon_number`, `parity` and
`retained_norms` are available. All state getters return independent snapshots.

## Adding a backend

Implement BaseBackend's reset, prepare, entangle, displace, rotate, squeeze,
beamsplitter, measure and get_state methods. Add loss/cubic_phase where supported.
Pass the backend instance to `simulate`; it will be reset at each trajectory.
Maintain a node-label-to-current-mode map after every destructive measurement.
Use statistical V and quadrature translations at the interface even if the native
simulator uses other units. Unsupported operations must raise NotImplementedError.
Use the provided analytic tests as a model, then add direct native references.

## Future PhotoGraphiQML boundary

Patterns can be reused with new bindings and shot results expose features and
observables. Backend isolation permits future differentiation-aware engines.
The current engine converts scalar parameters to Python floats and performs
stochastic control flow; its execution is not end-to-end differentiable. Batching
means independent trajectories, not a vectorized accelerator implementation.
PhotoGraphiQ deliberately introduces no ML runtime dependencies.
