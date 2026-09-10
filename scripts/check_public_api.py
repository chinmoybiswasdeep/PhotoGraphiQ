"""Check convenience exports and generated API coverage without plotting imports."""

import importlib
from pathlib import Path

import photographiq as pg


def main():
    if len(pg.__all__) != len(set(pg.__all__)):
        raise AssertionError("Duplicate public exports")
    for name in pg.__all__:
        if not hasattr(pg, name):
            raise AssertionError(f"Missing public export {name}")
    pages = Path(__file__).resolve().parents[1] / "docs/api"
    for page in pages.glob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("::: "):
                importlib.import_module(line[4:])
    print(f"Checked {len(pg.__all__)} convenience exports and API module imports")


if __name__ == "__main__":
    main()
