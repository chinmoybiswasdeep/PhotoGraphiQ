# Specification coverage for v0.1

This audit distinguishes implemented release behavior from the longer-term vision
in the supplied specification. See `design.md` for the pre-implementation choices,
`theory.md` for derivations and `validation.md` for evidence boundaries.

| Specification sections | Delivered implementation | Boundary |
|---|---|---|
| 1–3, 35–37, 41–43: architecture and study | Separate resources, IR, expressions, execution, backends; inspected upstream sources and supplied papers; capability matrix | GraphiX is structural, never the CV numerical oracle |
| 4: conventions | Central hbar=2, statistical covariance, interleaved order, gate and detector conventions | Native Piquasso discrepancies explicitly characterized |
| 5: graph states | NetworkX graphs, labels, weighted edges, adjacency, role tuples, graph families, node squeezing, ideal nullifiers | Temporal/dual-rail are topologies rather than hardware models |
| 6: measurements | q/p/rotated homodyne, heterodyne, physical general-dyne covariance, Fock photon counting | Fock homodyne conditioning and general non-Gaussian POVMs unavailable |
| 7, 9, 11: patterns and adaptation | Typed commands, records, outputs, causal validation, adaptive expressions and declared callables | Callables are not portable serialized objects |
| 8: corrections | Quadrature translations, physical and Gaussian virtual frames | Exact frame mode requires a Gaussian backend |
| 10: flow | Dependency DAG and stable topological scheduling; real-linear supplied-order CV-flow certificate and protocol | No optimal-order or general partial-order finder; no qubit Pauli-flow port |
| 12–13: backends and Gaussian simulation | BaseBackend, Piquasso native gates, exact conditioning, NumPy backend, affine channel analyzer | Dense moments and eigenvalue checks; measurement adapter is shared and independently tested |
| 14: non-Gaussian support | Explicit pure-Fock cutoff, number/cat/custom vectors, offline photon addition/subtraction, cubic-phase operations, conditional counting | GKP, mixed Fock input injection and adaptive Fock homodyne remain future work |
| 15: inputs | Coherent/squeezed/general Gaussian, correlated Gaussian states, native Gaussian conversion with explicit hbar, pure Fock input | Arbitrary native non-Gaussian state import is not a general interchange API |
| 16–18: protocols and compiler | Optical teleportation, identity and multi-step wires, displacement, rotation, squeezing, arbitrary single-mode symplectic gates, CZ and beam splitter compilation, adaptive wire | Gaussian compilation only; two-mode decompositions prioritize transparency over noise/resource efficiency |
| 17, 23: noise | Exact finite-squeezing channel accumulation, thermal attenuation, detector inefficiency and readout variance | Phase diffusion, mode mismatch and complete hardware noise models are not implemented |
| 19: optimization | Safe disjoint swaps, adjacent displacement fusion, numerical zero removal, execution frames | No full normal-form, graph minimization or general symbolic algebra optimizer |
| 20–22: parameters, shots, observables | Reusable real parameters; independent seeded shots; Gaussian moments, photon number, parity, overlap; Fock probabilities and parity | General circuit matrix decomposition is numeric; no differentiable stochastic runtime; overlap is not general mixed-state fidelity |
| 24–25: visualization and serialization | Resource/pattern/DAG matplotlib exports; versioned allowlisted JSON with label and expression preservation | No pickle/eval; runtime callables explicitly rejected |
| 26–30: API and validation | Public API, analytical tests, raw Piquasso tests, GraphiX structural/causal-flow tests, pytest suites and error cases | Tests establish the stated examples and invariants, not a proof for every possible computation |
| 31–34: documentation, engineering and ML boundary | Tutorials, API guide, derivations, modular src layout, Ruff, mypy, CI, pre-commit, build metadata | Python 3.12 is locally tested; CI configuration is not a report of remote runs; no ML runtime dependencies |
| 38–40: development and release | Design-first incremental tests, v0.1 API, source/wheel builds, reproducibility artifacts | Research extensions above remain explicit future work; no remote release or commit is implied |
| Additional request: paper | Complete 11-page Quantum-class manuscript, compiled PDF, equations, source audit, data, plots, bibliography and AI-use disclosure | Collective author metadata must be finalized by the actual authors; not peer-reviewed or submitted |
