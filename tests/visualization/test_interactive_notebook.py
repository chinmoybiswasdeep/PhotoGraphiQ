"""The interactive visualization example notebook executes without errors."""

from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "notebooks"
    / "interactive_graph_visualization.ipynb"
)


def test_notebook_executes():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOK.parent}},
    ).execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    frames = [o for o in outputs if "<iframe srcdoc=" in o.get("data", {}).get("text/html", "")]
    assert len(frames) == 7  # four graph stages and three plots
