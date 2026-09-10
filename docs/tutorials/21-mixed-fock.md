# Mixed Fock evolution

## Goal

Track a lossy nonlinear trajectory as a density matrix. Install the [required extras](../getting-started/installation.md) first.

## Theory

Loss destroys purity by entangling the field with an unobserved environment. A density matrix retains the unconditional mixture while allowing later conditional measurement. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/21-mixed-fock.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/21-mixed-fock.py"
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
--8<-- "docs/tutorials/outputs/21-mixed-fock.txt"
```

The density matrix is positive and trace one, but purity is below one. Kerr has no observable phase effect on the chosen number-state input; the mixture here comes from loss.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Replace the input with a superposition to make Kerr phase observable. Compare thermal and vacuum environments.
