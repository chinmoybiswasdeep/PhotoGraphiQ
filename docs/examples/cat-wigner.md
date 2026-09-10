# Cat states and Wigner negativity

Compare finite-grid negativity over cat amplitudes.

## Run

From a source checkout with the visualization extra installed:

```sh
python examples/projects/cat-wigner/main.py --output outputs/cat-wigner
```

The script writes `data.csv` and `figure.svg` to the chosen directory. Inputs and
sweep values are visible in `main.py`; edit those values to reproduce a modified
experiment. Deterministic channels or explicitly seeded/selected trajectories are
used so results are reproducible within numerical tolerances.

## Interpretation

Negativity depends on both physical amplitude and numerical grid. Refine the window and spacing before comparing small differences.

## Validation and extension

Check the corresponding analytical identity or independent validation suite before
using the result in a paper. For Fock examples, increase cutoff and compare
observables; for sampling, increase shots and estimate uncertainty. Record package
versions and all numerical settings with exported data. This project can be run
independently of other demos and requires no downloaded datasets.


```python
--8<-- "examples/projects/cat-wigner/main.py"
```
