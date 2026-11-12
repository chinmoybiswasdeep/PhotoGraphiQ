"""Report tests that need an uninstalled optional extra as skipped, naming the extra.

PhotoGraphiQ raises ``ImportError`` with an ``pip install 'photographiq[<extra>]'`` hint
when a feature needs an optional dependency (Piquasso backends, ...). Only that exact
error becomes a skip; with all extras installed (CI, development) it never occurs.
"""

import pytest

_OPTIONAL_HINT = "pip install 'photographiq["


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    error = call.excinfo
    if (
        error is not None
        and error.errisinstance(ImportError)
        and _OPTIONAL_HINT in str(error.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {error.value}")
    return report
