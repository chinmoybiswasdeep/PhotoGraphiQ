# Hybrid Gaussian–Fock executor: architecture

!!! warning "Experimental API, CPU-validated only"
    `photographiq.hybrid` is experimental and is not a registered backend. Only its NumPy
    reference engine has been executed. The CUDA-Q engine is implemented but has **never
    run**: CUDA-Q is Linux-only and the development machine runs Windows without WSL (see
    [Validation](validation.md)).

The executor runs a PhotoGraphiQ `Pattern` (CV-MBQC with homodyne, photon counting and
feed-forward) on a truncated Fock state that uses **per-mode** local dimensions, which is
CUDA-Q Dynamics' representation. Gaussian preparations enter through an exact
Gaussian→Fock bridge. All gates are Hamiltonian evolutions built from the generators of
`photographiq.cudaq_dynamics`, and measurements are physical Fock instruments.

```text
Pattern ──plan()──► ExecutionPlan (dry run: segments, engines, dims, memory)
   │
   └─run()─► for each command, in order (never reordered):
              Prepare        → gaussian_to_fock (Williamson + symplectic generators)
                               | fock_input | CubicPhaseResource = bridge + CP Hamiltonian
              gates          → U = exp(-i H) of cudaq_dynamics generators
                               engine="numpy-reference" (local expm) | "cudaq" (cudaq.evolve)
              Loss           → Lindblad channel, L = sqrt(1+n) a, sqrt(n) a†, t = -ln(eta)
              Homodyne       → Fock instrument: density, conditional state, sampling, bins
              PhotonNumber   → projective count
              outcome-valued → parameters resolved on the actual records (feed-forward)
           ► HybridResult(state, records, densities, probabilities, retained_weight,
                          tail_mass, truncation, execution_record, plan, runtime, engine)
```

| Module | Role |
|---|---|
| `photographiq.hybrid.states` | `FockRep` contract, total-photon mapping, Williamson, `gaussian_to_fock`, `moment_matched_gaussian` (approximate, labelled) |
| `photographiq.hybrid.homodyne` | quadrature wavefunctions, densities, `measure` (ideal / binned, postselected / sampled) |
| `photographiq.hybrid.planner` | `classify`, `plan` (dry run) |
| `photographiq.hybrid.executor` | `run`, `run_shots`, `ensemble_statistics`, `HybridResult` |

## Four execution modes compared

| | Native Gaussian (`gaussian`, `native-gaussian`) | Native Fock (`native-fock`, `native-mixed-fock`) | CUDA-Q Hamiltonian (`cudaq_dynamics`) | Hybrid CUDA-Q CV-MBQC (`photographiq.hybrid`) |
|---|---|---|---|---|
| Input | `Pattern` | `Pattern` | a Hamiltonian, dims, initial state | `Pattern` |
| State | mean + covariance (exact) | total-photon truncated vector / ρ | per-mode truncated vector / ρ | per-mode truncated `FockRep` (contract, retained weight) |
| Non-Gaussian gates | no | yes (projected exact matrices) | yes (truncated generator) | yes (truncated generator) |
| Loss | exact Gaussian channel | Kraus / mixed engine | Lindblad master equation | Lindblad superoperator (NumPy) or `cudaq.evolve` |
| Homodyne, counting, feed-forward | yes | yes | **no** | yes (Fock instrument, causal records) |
| Registered backend | yes | yes | no | no (experimental API) |
| Executed here | yes | yes | NumPy reference only | NumPy engine only; CUDA-Q engine **never run** |

## Why a separate executor

CUDA-Q Dynamics evolves Hamiltonians and master equations. It has no measurement
instruments, classical records or feed-forward, so it cannot execute CV-MBQC by itself.
Hamiltonian evolution alone does not amount to CV-MBQC. The executor supplies the
missing pieces around the adapter: the instrument, causal classical control and the
bridge. It also uses the native Fock and Gaussian engines as independent references.

See also: [state representations](state-representations.md), [bridge](gaussian-to-fock.md),
[dynamics](dynamics-execution.md), [homodyne](homodyne-instruments.md),
[feed-forward](feedforward.md), [trajectories](trajectories.md), [noise](noise.md),
[performance](performance.md), [limitations](limitations.md).
