# Graph topology comparison

Compare total photon occupation of equal-squeezing resource families.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/topology-comparison/main.py --output outputs/topology-comparison
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

Connectivity changes energy and correlations. This is a resource comparison, not a claim that one graph computes better.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.
