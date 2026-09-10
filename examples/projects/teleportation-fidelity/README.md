# Teleportation fidelity versus squeezing

Sweep the exact unconditional teleportation channel and compare with a pure coherent target.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/teleportation-fidelity/main.py --output outputs/teleportation-fidelity
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

Compare numerical values with 1/(1+exp(-2r)). This is an unconditional channel result, not a single selected trajectory.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.
