# Circuit to MBQC compilation and visualization

## Goal

Follow each source gate into resources, measurements and corrections. Install the [required extras](../getting-started/installation.md) first.

## Theory

A correspondence diagram needs compiler provenance. Graph layout alone cannot reliably reconstruct which teleportation gadget came from a source gate. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/20-compilation-visualization.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/20-compilation-visualization.py"
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
--8<-- "docs/tutorials/outputs/20-compilation-visualization.txt"
```

Color identifies generated resource ownership. Frontier nodes may be shared between consecutive gates; correction indices refer to the full temporal command list.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Save SVG and PDF. Modify the pattern after compiling and verify that the correspondence view rejects the stale trace.
