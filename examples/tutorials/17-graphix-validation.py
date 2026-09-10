"""Graphix structural validation. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import graphix
import matplotlib.pyplot as plt
from graphix import command

import photographiq as pg

dv = graphix.Pattern(input_nodes=[0])
dv.add(command.N(1))
dv.add(command.E((0, 1)))
dv.add(command.M(0))
cv = pg.Pattern(pg.CVGraph.line(2, inputs=(0,))).measure(0, pg.Homodyne.p())


def edges(graph):
    return {frozenset(e) for e in graph.edges()}


assert edges(dv.to_opengraph().graph) == edges(cv.graph.network)
assert tuple(dv.output_nodes) == cv.outputs
print("Matching edges:", sorted(tuple(sorted(e)) for e in edges(cv.graph.network)))
print("Matching outputs:", cv.outputs)
cv.draw()
plt.close("all")
