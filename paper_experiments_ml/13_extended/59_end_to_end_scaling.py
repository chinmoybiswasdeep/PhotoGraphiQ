"""R59: end-to-end runtime and peak-memory scaling over measured ranges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_studies import run

if __name__ == "__main__":
    run("R59")
