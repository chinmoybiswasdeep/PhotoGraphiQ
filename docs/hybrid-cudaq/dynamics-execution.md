# Hamiltonian segment execution

Each unitary command becomes the adapter generator H, and the executor applies
U = exp(−iH) on the command's modes:

| Command | Generator (`photographiq.cudaq_dynamics`) |
|---|---|
| `Rotate(angle)` | `rotation` = −angle n |
| `Squeeze(r)` | `squeezing` = i r (a² − a†²)/2 |
| `QuadraticPhase(s)` | `quadratic_phase` = −s q²/4 |
| `Displace(q, p)` | `displacement` = i(α a† − α* a), α = (q + ip)/2 |
| `BeamSplitter(theta)` | `beamsplitter` |
| `Entangle(weight)` (CZ) | `controlled_z` = −g q_u q_v / 2 |
| `Kerr(kappa)` | `kerr` = −κ n² |
| `CubicPhase(gamma)` | `cubic_phase` = −γ q³/6 |
| `Prepare(state=CubicPhaseResource(γ, r))` | bridge for S(−r)|0⟩, then `cubic_phase` |

The conventions are the ones verified against native-fock in `tests/cudaq_adapter`. The
native engines also implement cross-Kerr, but PhotoGraphiQ has no cross-Kerr pattern
command, so the executor does not use it.

## Semantics: truncated generator

H is truncated to the per-mode space before it is exponentiated (U = exp(−iH_d)). This is
the semantics of CUDA-Q's per-mode truncation, so both engines compute the same object.
Two consequences follow:

- U is exactly unitary on the truncated space. The state's norm therefore says nothing about truncation error, which is why the result carries [boundary diagnostics](limitations.md#truncation-diagnostics).
- Agreement with native-fock, which projects exact Gaussian matrix elements onto a total-photon cutoff, is a statement about the converged limit of two different truncations (see [validation](validation.md)).

## Engines

| `engine=` | What runs | Record |
|---|---|---|
| `"numpy-reference"` (default) | dense `scipy.linalg.expm` of the local generator, applied with a tensor contraction to the gate's modes (pure: U ψ; mixed: U ρ U†) | `{"engine": "numpy-reference", "device": "cpu"}` |
| `"cudaq"` | `cudaq_dynamics.evolve(H_global, dims, [0, 1], state)`, with H relabelled to global positions and the full state evolved | `EvolutionResult.record`: CUDA-Q version, target, simulator, platform, `verified: False` |

Any other value raises `ValueError`. `"cudaq"` checks `cudaq` up front and raises
`ImportError` if it is missing; it never falls back.

Each entry of `HybridResult.execution_record` holds the operation, modes, seconds, the
boundary population and the engine record. `HybridResult` itself holds the total
`runtime`, the `engine` and the `plan`.

## Dry run

```python
from photographiq.hybrid.planner import plan
dry = plan(pattern, dims=20)          # never evolves a state
[s.kind for s in dry.segments]        # gaussian-preparation, non-gaussian-preparation, ...
dry.peak_dimension, dry.peak_bytes    # 8000, 128000 for three live 20-level modes (pure)
dry.unsupported                       # [(index, "CommandName"), ...]; run() refuses if non-empty
```

`plan` classifies each command and builds maximal runs of consecutive commands of the same
class, never reordering them. Every measurement and feed-forward command starts a new
segment. The classes are:

`gaussian-preparation`, `fock-preparation`, `non-gaussian-preparation`, `gaussian-unitary`,
`non-gaussian-unitary`, `open-system`, `homodyne`, `photon-counting`, `feedforward`,
`output` and `unsupported`.

Each segment records its engine, live modes, the classical records it reads and the
records it writes. The memory estimate is 16·D bytes for a pure state and 16·D² for a mixed
state, at the peak live dimension. In HYB-12 it was always below the measured peak
allocation.
