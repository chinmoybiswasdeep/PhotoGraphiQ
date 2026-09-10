# Small graph-state experiment

Vary a CZ edge weight and inspect resource correlations.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/graph-experiment/main.py --output outputs/graph-experiment
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

Cross-covariance changes with coupling strength. A complete entanglement claim requires an appropriate witness.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.


```python
--8<-- "examples/projects/graph-experiment/main.py"
```
