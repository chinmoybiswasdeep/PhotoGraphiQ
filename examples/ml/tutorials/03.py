import photographiqml as pqml

m = pqml.MuTA(2, one_column=True)
assert m.n_parameters == 8
print(m.summary())
