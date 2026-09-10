# Two-mode entangling computation

## Goal

Compile a logical CZ gate and inspect cross-mode correlations. Install the [required extras](../getting-started/installation.md) first.

## Theory

A logical CZ transforms both momenta using the other mode position. Compilation includes transport steps after entanglement, unlike a single physical Entangle command. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/08-two-mode-entangling.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/08-two-mode-entangling.py"
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
--8<-- "docs/tutorials/outputs/08-two-mode-entangling.txt"
```

Nonzero cross blocks record correlations. Their presence alone is not a complete entanglement witness for arbitrary mixed states.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Change CZ weight sign and inspect which correlations change sign. Compare physical and logical CZ resource counts.
