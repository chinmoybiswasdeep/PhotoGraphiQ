# Wigner-function analysis

## Goal

Compute finite-grid Wigner mass and negativity for a single photon. Install the [required extras](../getting-started/installation.md) first.

## Theory

The Wigner function is a quasiprobability. Negative values cannot be interpreted as event probabilities. Its integral should approach one as the window and resolution converge. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/16-wigner-analysis.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/16-wigner-analysis.py"
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
--8<-- "docs/tutorials/outputs/16-wigner-analysis.txt"
```

A negative region near the origin is expected for an odd photon-number state. Finite-grid negativity is a numerical estimate dependent on window and spacing.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Repeat with 51 and 151 points, then shrink the window. Identify integration error versus lost tails.
