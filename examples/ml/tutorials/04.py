import photographiqml as pqml

cell = pqml.TriangleNeuron(2)
assert cell.graph.has_edge((0, 1), (1, 0))
assert cell.graph.has_edge((0, 1), (1, 2))
print(sorted(cell.graph.edges))
