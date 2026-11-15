import photographiqml as pqml

X = [[0], [0.1], [3.0], [3.14]]
y = [1, 1, 0, 0]
c = pqml.MuTAClassifier(pqml.MuTA(1), trainer=pqml.Trainer(epochs=60))
c.fit(X, y)
assert c.score(X, y) == 1
print(c.predict([[0.05], [3.1]]))
