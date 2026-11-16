# State representations

`photographiq.hybrid.states.FockRep` is the single state type in the executor. It is
frozen and validated on construction.

| Field / property | Meaning |
|---|---|
| `data` | vector, shape (D,), for a pure state, or density matrix, shape (D, D), for a mixed state; D = ∏ dims |
| `modes` | mode labels, in tensor order (first mode = most significant index, as `cudaq_dynamics.to_matrix`) |
| `dims` | per-mode local dimensions (levels 0..d−1) |
| `retained_weight` | product of every truncation or projection probability applied so far |
| `physical_parameters` | provenance (source, mean, symplectic eigenvalues, padding, ...) |
| `precision` | `complex128` |
| `representation` | `"pure-fock"` or `"mixed-fock"` |
| `normalization` | ‖ψ‖² or Tr ρ of `data` (1 after an explicit renormalisation) |

`rep.contract()` returns the full contract:
`mode_order, hilbert_space_dimensions, total_photon_cutoff (None: per-mode), representation_type, normalization, retained_weight, physical_parameters, numerical_precision`.

Further methods:

- `tensor` combines two states (pure ⊗ pure stays pure; anything mixed gives a mixed state);
- `reduced(modes)` takes a partial trace (pure states never form the full density matrix);
- `moments()` returns the ħ = 2 mean and covariance with interleaved (q, p) ordering;
- `boundary_population()` gives the per-mode population within two levels of the truncation (see [limitations](limitations.md)).

## No silent renormalisation

Every projection, whether the bridge onto `dims`, a truncated `fock_input`, or a
total-photon projection, renormalises the state explicitly. The kept probability goes
into `retained_weight`, so the physical probability of the discarded part stays visible:

```python
from photographiq.hybrid.states import fock_input
rep = fock_input([1, 1, 1], "a", 2)     # |0> + |1> + |2>, two levels kept
rep.retained_weight                      # 0.666...: one third was truncated
```

## Per-mode vs total-photon truncation

The native Fock engines keep states with Σn < cutoff. CUDA-Q and this executor use
n_k < d_k for each mode. The conversions reuse the adapter's
`per_mode_to_total_photon` and `total_photon_to_per_mode`:

```python
from photographiq.hybrid.states import to_total_photon, from_total_photon
data, weight = to_total_photon(rep, cutoff)        # not renormalised; weight = kept probability
back = from_total_photon(data, rep.modes, rep.dims, cutoff)
```

## Fock → Gaussian

There is only one conversion in this direction, and it is approximate and explicitly
requested: `moment_matched_gaussian(rep)` returns `{"mean", "covariance",
"approximate": True, "note": ...}`. The executor never applies it implicitly.
