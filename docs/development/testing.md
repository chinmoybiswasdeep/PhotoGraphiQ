# Running tests

Run `python -m pytest` for the installed core test suite. Categories include tests/analytical, tests/piquasso_reference, tests/graphix_reference, tests/fock, tests/non_gaussian, tests/mixed_fock, tests/compilation, tests/gkp and tests/autodiff. The last category requires JAX. `python experiments/validate_release.py --output .validation` preserves gate logs. Add a regression for a demonstrated bug and keep independent numerical error budgets explicit.

See [numerical policy](../validation/numerical-policy.md), [API](../api/index.md) and the [feature matrix](../validation/feature-matrix.md).
