import photographiqml as pqml

m = pqml.MuTA(3, 1)
assert m.paper_depth == 3
assert len(m.graph) == 39
print(m.summary())
