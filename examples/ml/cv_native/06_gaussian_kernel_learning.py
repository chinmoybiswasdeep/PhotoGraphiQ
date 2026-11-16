"""Tutorial 6: Gaussian-state kernels and classical kernel ridge classification.

Quantum: the encoded states rho(x) (and optionally a Gaussian program or a finite-squeezing
CVCluster channel applied to them); the kernel is the closed-form overlap Tr[rho(x) rho(x')].
Classical: kernel ridge regression on +-1 labels, decision by sign.

For coherent displacement encoding the overlap kernel is exp(-scale^2 |x - x'|^2 / 4), a
Gaussian RBF kernel; a Gaussian unitary applied to every state leaves it unchanged; only a
noisy channel (finite squeezing) changes it.
"""

from time import perf_counter

import numpy as np

from photographiqml.cv import CVCluster, CVFeatureMap, CVKernel, layered_program

start = perf_counter()
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (140, 2))
y = np.where(np.hypot(*X.T) < 0.62, 1.0, -1.0)
X_train, y_train, X_test, y_test = X[:80], y[:80], X[80:], y[80:]
fmap = CVFeatureMap(2, scale=3.0)


def kernel_ridge_accuracy(kernel, lam=1e-3):
    alpha = np.linalg.solve(kernel.gram_matrix(X_train) + lam * np.eye(len(X_train)), y_train)
    return float(np.mean(np.sign(kernel(X_test, X_train) @ alpha) == y_test))


base = CVKernel(fmap)
gram = base.gram_matrix(X_train)
distance = ((X_train[:, None] - X_train[None]) ** 2).sum(-1)
rbf_error = np.abs(gram - np.exp(-(3.0**2) * distance / 4)).max()
print(f"overlap kernel vs exp(-s^2 |x-x'|^2 / 4): max error {rbf_error:.1e}")

program = layered_program(2, 1)
params = {n: 0.4 for n in program.parameters}
unitary_error = np.abs(CVKernel(fmap, program, params).gram_matrix(X_train) - gram).max()
print(f"after a Gaussian unitary program: max change {unitary_error:.1e}")
print(f"ideal kernel ridge test accuracy: {kernel_ridge_accuracy(base):.3f}")

cluster = CVCluster(2, 3)
cparams = dict.fromkeys(cluster.parameters, 0.2)
print(f"{'r':>5} {'purity':>8} {'min eig':>10} {'test acc':>9}")
rows = []
for r in (0.25, 1.0, 2.0):
    kernel = CVKernel(fmap, cluster, cparams, mode="physical", squeezing=r)
    diag = kernel.diagnostics(X_train)
    acc = kernel_ridge_accuracy(kernel)
    rows.append((r, diag["max_diagonal"], diag["minimum_eigenvalue"], acc))
    print(f"{r:>5} {diag['max_diagonal']:>8.4f} {diag['minimum_eigenvalue']:>10.2e} {acc:>9.3f}")
fid = CVKernel(fmap, cluster, cparams, kind="fidelity", mode="physical", squeezing=0.25)
fid_diag = np.diag(fid.gram_matrix(X_train[:5]))
print(f"fidelity kernel diagonal at r = 0.25 (mixed states): {np.round(fid_diag, 6)}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert rbf_error < 1e-12 and unitary_error < 1e-10
assert kernel_ridge_accuracy(base) >= 0.9
assert all(row[1] < 1 and row[2] > -1e-10 for row in rows)  # mixed outputs, PSD Gram
assert rows[0][1] < rows[1][1] < rows[2][1]  # purity rises with squeezing
assert np.allclose(fid_diag, 1, atol=1e-8)
