# CV quantum teleportation

## Goal

Compare the optical teleportation channel with its analytical added noise. Install the [required extras](../getting-started/installation.md) first.

## Theory

The Braunstein–Kimble protocol uses an entangled two-mode resource and two quadrature measurements. Its unconditional covariance adds 2 exp(-2r) I. A selected trajectory differs from this average channel. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/02-teleportation.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/02-teleportation.py"
```

## Visualization

The script creates a headless matplotlib view and closes it after verification.
To inspect it interactively, remove the final close call and use `plt.show()`.
To export, call `plt.gcf().savefig("experiment.svg")` before closing the figure.
Graph connectivity and the temporal command DAG describe different aspects of
the experiment; a graph picture alone is not a gate-equivalence certificate.

## Output interpretation

The following output was captured by the executable documentation check:

```text
--8<-- "docs/tutorials/outputs/02-teleportation.txt"
```

The channel preserves the coherent displacement and adds noise. The sampled outcomes are real measurement readings, not probabilities or target amplitudes.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Sweep r and compute overlap with a pure coherent target relabelled to the output. Compare channel moments with a large shot ensemble.
