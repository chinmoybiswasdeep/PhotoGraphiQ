"""Execute documentation sources, projects and optional notebooks in isolation."""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--notebooks", action="store_true")
    parser.add_argument("--projects", action="store_true")
    parser.add_argument(
        "--only", default="", help="Execute tutorial filenames containing this text"
    )
    parser.add_argument("--skip-autodiff", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    for source in sorted((root / "examples/tutorials").glob("*.py")):
        if (args.only and args.only not in source.name) or (
            args.skip_autodiff and source.name.startswith("24-")
        ):
            continue
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, str(source)],
                cwd=temp,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
            )
        if result.returncode:
            raise RuntimeError(f"{source.name}\n{result.stdout}\n{result.stderr}")
        if args.write_outputs:
            (root / "docs/tutorials/outputs" / f"{source.stem}.txt").write_text(
                result.stdout, encoding="utf-8"
            )
        print(f"PASS {source.name}", flush=True)
    if args.projects:
        for source in sorted((root / "examples/projects").glob("*/main.py")):
            with tempfile.TemporaryDirectory() as temp:
                subprocess.run(
                    [sys.executable, str(source), "--output", temp],
                    cwd=temp,
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                if (
                    not (Path(temp) / "data.csv").is_file()
                    or not (Path(temp) / "figure.svg").is_file()
                ):
                    raise RuntimeError(f"Missing project artifacts: {source}")
                with (Path(temp) / "data.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                if len(rows) < 2 or any(len(row) != 2 for row in rows):
                    raise RuntimeError(f"Expected a two-column CSV with header: {source}")
            print(f"PASS project {source.parent.name}", flush=True)
    if args.notebooks:
        import nbformat
        from nbclient import NotebookClient

        for path in sorted((root / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            with tempfile.TemporaryDirectory() as temp:
                client = NotebookClient(
                    notebook,
                    timeout=180,
                    kernel_name="python3",
                    resources={"metadata": {"path": temp}},
                )
                client.create_kernel_manager()
                client.km.kernel_spec.argv[0] = sys.executable
                client.execute()
            if args.write_outputs:
                nbformat.write(notebook, path)
            print(f"PASS notebook {path.name}", flush=True)


if __name__ == "__main__":
    main()
