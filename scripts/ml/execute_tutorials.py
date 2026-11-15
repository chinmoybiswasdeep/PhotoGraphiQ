"""Execute every tutorial and curated notebook with the current interpreter."""

import json
import runpy
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

# Repository-root anchored, so running from any cwd only touches the PhotoGraphiQML trees.
ROOT = Path(__file__).resolve().parents[2]


def main():
    for path in sorted((ROOT / "examples" / "ml" / "tutorials").glob("*.py")):
        print(path, flush=True)
        runpy.run_path(str(path), run_name="__main__")
    with tempfile.TemporaryDirectory() as directory:
        spec = Path(directory) / "pgml"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "PhotoGraphiQML checks",
                    "language": "python",
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[directory])
        for path in sorted((ROOT / "notebooks" / "ml").glob("*.ipynb")):
            print(path, flush=True)
            notebook = nbformat.read(path, as_version=4)
            kernel = KernelManager(kernel_name="pgml", kernel_spec_manager=manager)
            try:
                NotebookClient(notebook, km=kernel, timeout=120).execute()
                # Validation must not mutate checked-in notebook outputs.
            finally:
                if kernel.has_kernel:
                    kernel.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
