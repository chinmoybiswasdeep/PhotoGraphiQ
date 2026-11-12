import photographiqml as pqml

m = pqml.MuTA(2, one_column=True, connections=())
assert m.graph.number_of_edges() == 8
m.freeze("alpha.w0.c1", 0)
assert m.n_parameters == 7
m.unfreeze("alpha.w0.c1")
assert m.n_parameters == 8
print("Topology and angle constraints are explicit")
