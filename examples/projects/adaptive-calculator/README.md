# Adaptive MBQC calculator

Bind an adaptive parameter and extract repeated-trajectory means.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/adaptive-calculator/main.py --output outputs/adaptive-calculator
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

Thirty-two shots are a demonstration, not a precision estimate. Increase shots and report uncertainty before drawing conclusions.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.
