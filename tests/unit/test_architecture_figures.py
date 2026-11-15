"""The architecture figures may only depict implemented code (docs/architecture/figures.json)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "generate_architecture_figures", ROOT / "scripts" / "generate_architecture_figures.py"
)
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)


def test_every_reference_exists():
    spec = json.loads(figures.SPEC.read_text(encoding="utf-8"))
    assert [f["id"] for f in spec["figures"]] == list("ABCDEFGH")
    assert figures.validate(spec) == []


def test_validation_rejects_missing_code():
    box = {"id": "x", "label": "X", "kind": "api", "col": 0, "row": 0}
    spec = {
        "figures": [
            {
                "id": "Z",
                "slug": "z",
                "title": "t",
                "caption": "c",
                "boxes": [
                    {**box, "refs": ["photographiq.ir:NoSuchClass"]},
                    {**box, "id": "y", "col": 1, "refs": ["path:cpp/CMakeLists.txt#no_such"]},
                    {**box, "id": "w", "col": 2, "refs": []},
                ],
                "edges": [{"from": "x", "to": "missing"}],
            }
        ]
    }
    errors = figures.validate(spec)
    assert any("NoSuchClass" in e for e in errors)
    assert any("no_such" in e for e in errors)
    assert any("no refs" in e for e in errors)
    assert any("unknown box" in e for e in errors)


def test_edge_crossing_detection():
    # Boxes as (centre x, centre y, width, height); a straight edge a -> c passes through b.
    rects = {"a": (0, 0, 1, 1), "b": (2, 0, 1, 1), "c": (4, 0, 1, 1), "d": (2, 2, 1, 1)}
    assert figures._crossed_boxes((0.5, 0), (3.5, 0), 0.0, rects, ("a", "c")) == ["b"]
    # Bent upwards (control point y = 4) through d; a straight a -> d edge crosses nothing.
    assert "d" in figures._crossed_boxes((0.5, 0), (3.5, 0), -1.33, rects, ("a", "c"))
    assert figures._crossed_boxes((0.5, 0.5), (1.5, 2.0), 0.0, rects, ("a", "d")) == []


def test_print_readability_check():
    # 14 in wide: placed at 7 in, so every native size is halved.
    texts = [("title", "a", 8.4), ("detail", "a", 6.6), ("edge", "a->b", 13.0)]
    scale, smallest, problems = figures.print_sizes("Z", 14.0, texts)
    assert scale == 0.5 and smallest == {"title": 4.2, "detail": 3.3, "edge": 6.5}
    assert any("title" in p for p in problems) and any("detail" in p for p in problems)
    assert not any("edge" in p for p in problems)
    # Narrower than the print width: placed at native size, never enlarged.
    assert figures.print_sizes("Z", 5.0, [("title", "a", 7.0)])[:2] == (1.0, {"title": 7.0})
    assert figures.print_sizes("Z", 5.0, [("title", "a", 7.0)])[2] == []


def test_wide_figure_with_tiny_text_fails_and_lanes_render():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box = {"kind": "api", "refs": ["photographiq.ir:OpenGraph"], "detail": "detail text"}
    figure = {
        "id": "Z",
        "slug": "z",
        "title": "t",
        "caption": "c",
        "cell": [12.0, 1.3],  # 24 in native width -> text shrinks to ~0.3x when placed
        "boxes": [
            {**box, "id": "a", "label": "A", "col": 0, "row": 0},
            {**box, "id": "b", "label": "B", "col": 1, "row": 1},
        ],
        "edges": [{"from": "a", "to": "b", "label": "edge"}],
        "lanes": [{"label": "Top", "rows": [0, 0]}, {"label": "Bottom", "rows": [1, 1]}],
    }
    spec = {"figures": [figure]}
    assert figures.validate(spec) == []
    fig, problems, report = figures.render(figure)
    plt.close(fig)
    assert problems == []
    assert set(report["smallest"]) == set(figures.MIN_PT)
    assert {p.split()[2] for p in report["problems"]} == set(figures.MIN_PT)
    # Lanes must be inside the grid and must not overlap.
    figure["lanes"] = [{"label": "X", "rows": [0, 1]}, {"label": "Y", "rows": [1, 2]}]
    errors = figures.validate(spec)
    assert any("'Y' needs" in e for e in errors)
    figure["lanes"] = [{"label": "X", "rows": [0, 1]}, {"label": "Y", "rows": [1, 1]}]
    assert any("overlaps another lane" in e for e in figures.validate(spec))
