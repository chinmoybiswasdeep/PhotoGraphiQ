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
