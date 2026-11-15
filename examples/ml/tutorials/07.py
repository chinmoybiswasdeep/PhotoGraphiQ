import photographiqml as pqml

m = pqml.MuTA(1)
m.freeze("alpha.w0.c3", 0)
assert m.n_parameters == 3
print(list(m.trainable_parameters()))
m.unfreeze("alpha.w0.c3")
assert m.n_parameters == 4
