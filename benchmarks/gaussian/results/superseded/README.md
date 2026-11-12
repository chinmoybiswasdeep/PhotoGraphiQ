# Superseded native Gaussian results

`native_gaussian_20261110-174011.json` was produced by the previous version of
`bench_native_gaussian.py` (row keys `operation` / `samples`, no `kind` labels). That
version compared the native batched kernels only with a Python loop over the per-state NumPy
path and had no vectorized-NumPy baseline, so its `batch_*` ratios overstate the C++ gain.
The script that produced it no longer exists in this form, so the file cannot be reproduced.
It is kept for provenance only; do not cite its batched speedups. Use the results of the
current script (`native_gaussian_<config>_<timestamp>.json` in the parent directory).
