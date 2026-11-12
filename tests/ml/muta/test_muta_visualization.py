import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photographiqml import MuTA, TriangleNeuron


def test_draw_logical_graphs(tmp_path):
    for index, model in enumerate([TriangleNeuron(), MuTA(2)]):
        ax = model.draw()
        target = tmp_path / f"graph{index}.svg"
        ax.figure.savefig(target)
        assert target.stat().st_size > 1000
        plt.close(ax.figure)
