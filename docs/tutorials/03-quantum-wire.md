# One-dimensional quantum wire

## Goal

Distinguish one Fourier teleportation step from four-step identity transport. Install the [required extras](../getting-started/installation.md) first.

## Theory

Each p measurement in a canonical CZ wire implements a Fourier step. Four such steps compose to identity on first moments, with accumulated finite-resource noise. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/03-quantum-wire.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/03-quantum-wire.py"
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
--8<-- "docs/tutorials/outputs/03-quantum-wire.txt"
```

Identity refers to the ideal linear transformation. It does not mean the finite-squeezing output covariance equals the input covariance.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Compare lengths 4, 8 and 12. Keep squeezing fixed to isolate accumulated noise.
