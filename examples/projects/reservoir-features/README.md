# Introductory reservoir-style feature extraction

Map coherent input amplitudes to a small observable feature table without ML dependencies.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/reservoir-features/main.py --output outputs/reservoir-features
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

This demonstrates observable extraction only. There is no trained model, prediction benchmark or PhotoGraphiQML dependency.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.
