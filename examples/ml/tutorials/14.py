import photographiqml as pqml

k = pqml.MuTAKernel()
X = [[0, 0], [1, 0], [0, 1]]
d = k.diagnostics(X)
assert d["minimum_eigenvalue"] > -1e-10
print(k.gram_matrix(X).round(4))
