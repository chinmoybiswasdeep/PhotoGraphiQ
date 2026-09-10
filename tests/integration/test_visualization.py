import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg
from photographiq.visualization import draw_dependencies, draw_pattern


def test_export_diagrams(tmp_path):
    p = pg.protocols.wire([0.0, 0.3], squeezing=0.5)
    for name, draw in [("pattern", draw_pattern), ("dependencies", draw_dependencies)]:
        ax = draw(p)
        path = tmp_path / f"{name}.pdf"
        ax.figure.savefig(path, bbox_inches="tight")
        assert path.stat().st_size > 1000
        plt.close(ax.figure)


def test_circuit_trace_and_exports(tmp_path):
    circuit = pg.Circuit(1).rotate(0, 0.2).cubic_phase(0, 0.01)
    pattern, trace = circuit.compile(squeezing=0.2, return_trace=True)
    before = pattern.to_json()
    for suffix in ("svg", "png", "pdf"):
        ax = circuit.draw(output=tmp_path / f"circuit.{suffix}")
        plt.close(ax.figure)
        fig = pg.visualize_compilation(
            circuit, pattern, trace=trace, output=tmp_path / f"mapping.{suffix}"
        )
        assert len(fig.axes) == 2
        plt.close(fig)
    assert before == pattern.to_json()
    import pytest

    circuit.rotate(0, 0.1)
    with pytest.raises(ValueError, match="unchanged"):
        pg.visualize_compilation(circuit, pattern)


def test_symbolic_labels_and_pnr(tmp_path):
    pattern = pg.Pattern(inputs=(("in", 1),))
    pattern.append(pg.Prepare("ancilla"))
    pattern.append(pg.Entangle(("in", 1), "ancilla", pg.Parameter("g")))
    pattern.measure(("in", 1), pg.Homodyne(pg.Parameter("theta")), key="q")
    pattern.measure("ancilla", pg.PhotonNumber())
    original = pattern.to_json()
    for layout in ("spring", "grid", "circular"):
        ax = pattern.draw(layout=layout, output=tmp_path / f"{layout}.svg")
        plt.close(ax.figure)
    ax = pattern.draw(positions={("in", 1): (0, 0), "ancilla": (1, 0)})
    plt.close(ax.figure)
    assert original == pattern.to_json()


def test_physical_command_glyphs(tmp_path):
    from photographiq.visualization import draw_commands

    pattern = pg.Pattern(inputs=("signal",)).extend(
        [
            pg.Prepare("tap"),
            pg.CubicPhase("signal", 0.01),
            pg.Kerr("signal", 0.2),
            pg.BeamSplitter("signal", "tap", 0.1),
            pg.Loss("signal", 0.9),
            pg.Measure("tap", pg.PhotonNumber()),
        ]
    )
    ax = draw_commands(pattern, output=tmp_path / "physical.svg")
    assert any("CP" in text.get_text() for text in ax.texts)
    assert any("PNR" in text.get_text() for text in ax.texts)
    plt.close(ax.figure)
