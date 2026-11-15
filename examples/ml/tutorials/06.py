import photographiqml as pqml

m = pqml.MuTA(2, 2, one_column=True)
assert len(m.graph) == 18
assert m.output_nodes == ((0, 8), (1, 8))
print(m.output_nodes)
