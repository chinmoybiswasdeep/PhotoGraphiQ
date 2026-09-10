# Heralded photon subtraction

## Goal

Compare a physical one-photon tap probability with its analytical value. Install the [required extras](../getting-started/installation.md) first.

## Theory

A vacuum beamsplitter tap followed by one detected photon implements an attenuated subtraction Kraus map. It approaches ideal subtraction only in the weak-tap limit. All examples use [hbar=2 conventions](../getting-started/conventions.md).

## Circuit and resource

The executable code below constructs the resource and command order explicitly.
Input labels identify injected states; preparation commands introduce ancillas.
Inspect those boundaries before changing a measurement or correction.

## Code and simulation

Run `python examples/tutorials/13-photon-subtraction.py` from the repository root. The script
uses a fixed seed or deterministic postselection where measurement is involved.

```python
--8<-- "examples/tutorials/13-photon-subtraction.py"
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
--8<-- "docs/tutorials/outputs/13-photon-subtraction.txt"
```

One photon survives after one is detected from an initial two-photon state. The herald probability falls toward zero with the tap angle.

## Validation

The assertions in the script check the stated example property. They complement
the independent [validation suites](../validation/index.md); they do not certify
arbitrary parameter regimes. For Fock experiments, repeat at larger cutoff before
using a numerical result in a publication. Keep selected measurement branches
fixed when comparing conditional states.

## Things to try

Sweep the tap angle and compare success probability with fidelity to normalized ideal subtraction for a cat input.
