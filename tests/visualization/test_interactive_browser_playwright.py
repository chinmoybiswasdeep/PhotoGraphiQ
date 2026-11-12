"""Real mouse/keyboard checks of the interactive pages (scripts/check_interactive_browser.py).

Skips when Playwright is not importable, no installed Edge/Chrome can be launched, or
Cytoscape.js cannot be loaded (offline; set PHOTOGRAPHIQ_CYTOSCAPE_JS to a local copy).
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_interactive_browser.py"


def load_script():
    spec = importlib.util.spec_from_file_location("check_interactive_browser", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pages_with_real_input_events(tmp_path):
    from playwright.sync_api import Error

    script = load_script()
    for channel in ("msedge", "chrome"):
        try:
            results = script.run_checks(tmp_path, channel)
            break
        except Error as error:
            if "Executable doesn't exist" not in str(error) and "not found" not in str(error):
                raise
    else:
        pytest.skip("No installed Edge or Chrome for Playwright")
    if any("Cytoscape.js did not load" in r.get("error", "") for r in results):
        pytest.skip("Cytoscape.js could not be loaded (offline); set PHOTOGRAPHIQ_CYTOSCAPE_JS")
    failed = {r["check"]: r["error"] for r in results if not r["ok"]}
    assert not failed, failed
    assert len(results) == 9 and (tmp_path / "report.json").exists()
