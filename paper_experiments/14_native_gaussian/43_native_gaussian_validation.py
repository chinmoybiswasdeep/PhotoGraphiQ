"""R43: Native Gaussian backend validation against analytic phase-space formulas.

Scientific question: does the compiled ``native-gaussian`` backend (in-place O(nk) C++
kernels) reproduce the exact Gaussian phase-space evolution (states, gates, loss,
homodyne/heterodyne conditioning, adaptive feed-forward) to floating-point accuracy, at
every mode count and circuit depth we use, and does it agree trajectory-by-trajectory with
the NumPy ``gaussian`` reference backend?

Model and conventions: hbar = 2, interleaved (q1, p1, ...), vacuum covariance I.

References (all independent of the code under test):

* Single-mode states: closed forms. Coherent mean (2 Re a, 2 Im a); squeezed covariance
  R(t) diag(e^{-2r}, e^{2r}) R(t)^T; thermal (2 nbar + 1) I; mean photon number
  (tr V - 2)/4 + |d|^2/4.
* Gates: Heisenberg map S = expm(2 Omega M) of each gate's quadratic generator
  U = exp(-i x^T M x / 2), embedded densely in 2n x 2n (``tests/validation/test_val_gaussian.py``
  helpers ``from_generator``/``gen_*``/``embed``, which are themselves checked against the
  Fock-space Heisenberg action in that test-suite). Loss: X = sqrt(eta) I,
  Y = (1-eta)(2 nth + 1) I.
* Conditioning: physical dilation (vacuum-ancilla beam splitter for detector inefficiency,
  50:50 splitter + two homodynes for heterodyne) and the PRECISION-matrix form of Gaussian
  conditioning (``_homodyne_reference``/``_heterodyne_reference`` of the same test module),
  not the Schur-complement formula the backends implement.
* Adaptive feed-forward: two-step teleportation wire with outcome-dependent measurement
  angle, propagated by hand (``_wire_step``).

Quantities: mean-vector error, covariance error (relative to max |V_ref|), symplectic
residual ||S Omega S^T - Omega||_max / ||S||_2^2 of the composed native map (S extracted
column by column from the native mean update), physicality min eig(V + i Omega), and
conditional-state / density agreement for the same outcomes.

Assumptions/limitations: double precision only; deep random circuits use bounded gate
parameters (|r| ~ 0.15) so that covariances stay O(10^2) and the relative-error criterion is
meaningful; the forward-error model of the acceptance bound is heuristic, not proven.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "validation"))
import test_val_gaussian as ref  # noqa: E402  (first-principles reference helpers)

import photographiq as pg  # noqa: E402
from photographiq import registry  # noqa: E402
from photographiq.states import GaussianState  # noqa: E402

SEED = 20261143
EPS = np.finfo(float).eps

# ----- acceptance criteria (declared before any evaluation) ---------------------------------
TOL_STATE = 1e-13  # closed-form single-mode moments: a handful of flops, O(1) entries
# Deep circuits: heuristic forward-error model, relative error <= C * depth * eps with
# C = 64 (each local k-mode update V -> S V S^T costs ~4k^2 flops per entry, k <= 2, times a
# 4x margin for amplification by later gates with ||S|| ~ e^{0.3}).
C_ROUND = 64
TOL_PHYSICAL_REL = 1e-12  # min eig(V + i Omega) >= -TOL * ||V||_2 (rounding of an O(n^3) eig)
TOL_CONDITIONAL = 1e-10  # conditional moments relative to max(1, ||V||_max) (precision vs Schur)
TOL_DENSITY_REL = 1e-10  # outcome density, relative
TOL_TRAJECTORY = 1e-11  # native vs numpy, same seed: outcomes and final moments
TOL_FEEDFORWARD = 1e-11  # adaptive wire vs hand propagation (as in test_val_gaussian)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="publication sweep (64/128 modes, depth 2560)")
    return parser.parse_args()


def omega(n):
    return ref.omega(n)


def random_state(rng, n, nus=None):
    """Random mixed Gaussian state V = S diag(nu) S^T (as ``test_val_gaussian.random_state``) with
    the generator scaled by 1/sqrt(4n) so that ||S|| stays ~e^0.5 at every mode count."""
    h = rng.normal(size=(2 * n, 2 * n))
    s = ref.from_generator(0.25 * (h + h.T) / np.sqrt(4 * n))
    nus = rng.uniform(1.0, 2.0, size=n) if nus is None else np.asarray(nus, dtype=float)
    cov = s @ np.diag(np.repeat(nus, 2)) @ s.T
    return rng.normal(scale=0.7, size=2 * n), (cov + cov.T) / 2, nus


def physical_margin(cov):
    n = len(cov) // 2
    lam = np.linalg.eigvalsh(cov + 1j * omega(n)).min()
    return float(lam / np.linalg.norm(cov, 2))


# ----- random circuits ------------------------------------------------------------------------
GATES_UNITARY = ("rotate", "squeeze", "beamsplitter", "cz", "quadratic", "displace")
GATES_NOISY = GATES_UNITARY + ("loss",)


def random_circuit(rng, n, depth, kinds):
    # Active-gate scale shrinks as sqrt(40 n / depth) for deep circuits so the accumulated
    # (random-walk) squeezing per mode stays O(1) at every depth: the study measures rounding
    # growth with depth, not the conditioning of ever more squeezed states.
    k = min(1.0, np.sqrt(40 * n / depth))
    ops = []
    for _ in range(depth):
        kind = kinds[rng.integers(len(kinds))] if n > 1 else kinds[rng.integers(len(kinds))]
        if n == 1 and kind in ("beamsplitter", "cz"):
            kind = "rotate"
        u, v = (int(x) for x in rng.choice(n, 2, replace=False)) if n > 1 else (0, 0)
        if kind == "rotate":
            ops.append((kind, (u,), float(rng.uniform(-np.pi, np.pi))))
        elif kind == "squeeze":
            ops.append((kind, (u,), float(rng.normal(scale=0.15 * k))))
        elif kind == "beamsplitter":
            ops.append((kind, (u, v), float(rng.uniform(-np.pi, np.pi))))
        elif kind == "cz":
            ops.append((kind, (u, v), float(rng.normal(scale=0.3 * k))))
        elif kind == "quadratic":
            ops.append((kind, (u,), float(rng.normal(scale=0.3 * k))))
        elif kind == "displace":
            ops.append((kind, (u,), (float(rng.normal()), float(rng.normal()))))
        else:
            ops.append((kind, (u,), (float(rng.uniform(0.6, 1.0)), float(rng.uniform(0, 0.2)))))
    return ops


def apply_backend(engine, ops):
    for kind, modes, value in ops:
        if kind == "rotate":
            engine.rotate(modes[0], value)
        elif kind == "squeeze":
            engine.squeeze(modes[0], value)
        elif kind == "beamsplitter":
            engine.beamsplitter(*modes, value)
        elif kind == "cz":
            engine.entangle(*modes, value)
        elif kind == "quadratic":
            engine.quadratic_phase(modes[0], value)
        elif kind == "displace":
            engine.displace(modes[0], *value)
        else:
            engine.loss(modes[0], *value)


def reference_evolution(mean, cov, ops, n):
    """Dense first-principles propagation; returns mean, cov and the composed symplectic."""
    s_total = np.eye(2 * n)
    for kind, modes, value in ops:
        if kind == "displace":
            mean = mean.copy()
            mean[2 * modes[0] : 2 * modes[0] + 2] += value
            continue
        if kind == "loss":
            x, y = ref.loss_xy(*value)
            xf = ref.embed(x, modes, n)
            yf = np.zeros((2 * n, 2 * n))
            yf[2 * modes[0] : 2 * modes[0] + 2, 2 * modes[0] : 2 * modes[0] + 2] = y
            mean, cov = xf @ mean, xf @ cov @ xf.T + yf
            s_total = None
            continue
        generator = {
            "rotate": ref.gen_rotation,
            "squeeze": ref.gen_squeeze,
            "beamsplitter": ref.gen_beamsplitter,
            "cz": ref.gen_cz,
            "quadratic": ref.gen_quadratic,
        }[kind](value)
        s = ref.embed(ref.from_generator(generator), modes, n)
        mean, cov = s @ mean, s @ cov @ s.T
        if s_total is not None:
            s_total = s @ s_total
    return mean, (cov + cov.T) / 2, s_total


def fresh(name, mean, cov, n, seed=0):
    engine = registry.create_backend(name)
    engine.reset(seed)
    engine.set_state(GaussianState(mean, cov, tuple(range(n))))
    return engine


def extract_symplectic(name, ops, n):
    """Columns S e_j from the backend's mean update (unitary, displacement-free circuits)."""
    columns = []
    for j in range(2 * n):
        e = np.zeros(2 * n)
        e[j] = 1.0
        engine = fresh(name, e, np.eye(2 * n), n)
        apply_backend(engine, ops)
        columns.append(engine.get_state().mean)
    return np.column_stack(columns)


def symplectic_residual(s):
    n = len(s) // 2
    return float(np.abs(s @ omega(n) @ s.T - omega(n)).max() / np.linalg.norm(s, 2) ** 2)


def circuit_rows(modes_list, depths, rng):
    rows = []
    for n in modes_list:
        for depth in depths:
            for family, kinds in (("unitary", GATES_UNITARY), ("noisy", GATES_NOISY)):
                mean0, cov0, _ = random_state(rng, n, nus=np.ones(n) if family == "unitary" else None)
                ops = random_circuit(rng, n, depth, kinds)
                m_ref, v_ref, s_ref = reference_evolution(mean0.copy(), cov0.copy(), ops, n)
                scale = np.abs(v_ref).max()
                row = {"family": family, "modes": n, "depth": depth, "cov_scale": scale,
                       "cov_condition": float(np.linalg.cond(v_ref))}
                for name in ("native-gaussian", "gaussian"):
                    engine = fresh(name, mean0, cov0, n)
                    apply_backend(engine, ops)
                    out = engine.get_state()
                    tag = "native" if name == "native-gaussian" else "numpy"
                    row[f"{tag}_mean_error"] = float(np.abs(out.mean - m_ref).max() / max(1.0, np.abs(m_ref).max()))
                    row[f"{tag}_cov_rel_error"] = float(np.abs(out.covariance - v_ref).max() / scale)
                    row[f"{tag}_physical_margin"] = physical_margin(out.covariance)
                    row[f"{tag}_min_symplectic_eigenvalue"] = float(np.min(out.symplectic_eigenvalues()))
                    if tag == "native":
                        native_state = out
                    else:
                        numpy_state = out
                row["native_vs_numpy_cov_rel"] = float(
                    np.abs(native_state.covariance - numpy_state.covariance).max() / scale
                )
                row["reference_physical_margin"] = physical_margin(v_ref)
                if family == "unitary":
                    s_nat = extract_symplectic("native-gaussian", [o for o in ops if o[0] != "displace"], n)
                    s_clean = reference_evolution(np.zeros(2 * n), np.eye(2 * n),
                                                  [o for o in ops if o[0] != "displace"], n)[2]
                    row["native_symplectic_residual"] = symplectic_residual(s_nat)
                    row["reference_symplectic_residual"] = symplectic_residual(s_clean)
                    row["native_S_rel_error"] = float(np.abs(s_nat - s_clean).max() / np.abs(s_clean).max())
                    row["purity_native"] = float(1 / np.sqrt(np.linalg.det(native_state.covariance)))
                rows.append(row)
                print(f"  {family:7s} n={n:3d} depth={depth:5d}: native cov rel {row['native_cov_rel_error']:.2e}"
                      f" numpy {row['numpy_cov_rel_error']:.2e} margin {row['native_physical_margin']:.1e}",
                      flush=True)
    return rows


# ----- single-mode states ---------------------------------------------------------------------
def state_rows():
    cases = []
    for alpha in (0.0, 0.7 - 0.3j, -1.5 + 2.0j):
        cases.append(("coherent", {"alpha": alpha}, pg.GaussianInput.coherent(alpha),
                      np.array([2 * alpha.real, 2 * alpha.imag]), np.eye(2)))
    for r, t in ((0.3, 0.0), (1.0, 0.7), (2.0, -1.2)):
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        cov = rot @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rot.T
        cases.append(("squeezed", {"r": r, "theta": t}, pg.GaussianInput.squeezed(r, t), np.zeros(2), cov))
    for nbar in (0.1, 1.0, 5.0):
        cov = (2 * nbar + 1) * np.eye(2)
        cases.append(("thermal", {"nbar": nbar}, pg.GaussianInput(covariance=tuple(map(tuple, cov))),
                      np.zeros(2), cov))
    r, t, nbar, alpha = 0.6, 0.4, 0.5, 0.8 + 0.2j
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    cov = (2 * nbar + 1) * rot @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rot.T
    mean = np.array([2 * alpha.real, 2 * alpha.imag])
    cases.append(("displaced squeezed thermal", {"r": r, "theta": t, "nbar": nbar, "alpha": alpha},
                  pg.GaussianInput(tuple(mean), tuple(map(tuple, cov))), mean, cov))
    rows = []
    for kind, params, prep, mean, cov in cases:
        expected_n = (np.trace(cov) - 2) / 4 + mean @ mean / 4
        row = {"state": kind, **{k: (str(v) if isinstance(v, complex) else v) for k, v in params.items()}}
        for name in ("native-gaussian", "gaussian"):
            engine = registry.create_backend(name)
            engine.reset(0)
            engine.prepare("m", state=prep)
            out = engine.get_state()
            tag = "native" if name == "native-gaussian" else "numpy"
            row[f"{tag}_mean_error"] = float(np.abs(out.mean - mean).max())
            row[f"{tag}_cov_error"] = float(np.abs(out.covariance - cov).max() / np.abs(cov).max())
            row[f"{tag}_photon_number_error"] = abs(out.photon_number("m") - expected_n)
            row[f"{tag}_physical_margin"] = physical_margin(out.covariance)
        rows.append(row)
    return rows


# ----- conditional measurements ---------------------------------------------------------------
def conditional_rows(rng):
    rows = []
    for n in (2, 4, 8):
        for theta, eta, noise, outcome in ((0.0, 1.0, 0.0, 0.8), (1.1, 1.0, 0.0, -1.3), (2.3, 0.7, 0.15, 0.4)):
            mean, cov, _ = random_state(rng, n)
            mode = int(rng.integers(n))
            r_mean, r_cov, r_density = ref._homodyne_reference(mean, cov, mode, theta, eta, noise, outcome)
            row = {"measurement": "homodyne" if (eta, noise) == (1.0, 0.0) else "inefficient homodyne",
                   "modes": n, "theta": theta, "efficiency": eta, "noise": noise, "outcome": outcome}
            states = {}
            for name in ("native-gaussian", "gaussian"):
                engine = fresh(name, mean, cov, n)
                engine.measure(mode, pg.Homodyne(theta, eta, noise), theta, outcome=outcome)
                out = engine.get_state()
                tag = "native" if name == "native-gaussian" else "numpy"
                scale = max(1.0, np.abs(r_cov).max())
                row[f"{tag}_conditional_error"] = float(
                    max(np.abs(out.mean - r_mean).max(), np.abs(out.covariance - r_cov).max()) / scale)
                row[f"{tag}_density_rel_error"] = abs(engine.last_measurement["value"] / r_density - 1)
                states[tag] = out
            row["native_vs_numpy"] = float(max(np.abs(states["native"].mean - states["numpy"].mean).max(),
                                               np.abs(states["native"].covariance - states["numpy"].covariance).max()))
            rows.append(row)
        mean, cov, _ = random_state(rng, n)
        mode, outcome = int(rng.integers(n)), (0.7, -1.1)
        r_mean, r_cov, r_density = ref._heterodyne_reference(mean, cov, mode, outcome)
        row = {"measurement": "heterodyne", "modes": n, "theta": np.nan, "efficiency": 1.0, "noise": 0.0,
               "outcome": str(outcome)}
        states = {}
        for name in ("native-gaussian", "gaussian"):
            engine = fresh(name, mean, cov, n)
            engine.measure(mode, pg.Heterodyne(), outcome=outcome)
            out = engine.get_state()
            tag = "native" if name == "native-gaussian" else "numpy"
            scale = max(1.0, np.abs(r_cov).max())
            row[f"{tag}_conditional_error"] = float(
                max(np.abs(out.mean - r_mean).max(), np.abs(out.covariance - r_cov).max()) / scale)
            row[f"{tag}_density_rel_error"] = abs(engine.last_measurement["value"] / r_density - 1)
            states[tag] = out
        row["native_vs_numpy"] = float(max(np.abs(states["native"].mean - states["numpy"].mean).max(),
                                           np.abs(states["native"].covariance - states["numpy"].covariance).max()))
        rows.append(row)
    return rows


# ----- adaptive feed-forward ------------------------------------------------------------------
def feedforward_rows(n_seeds, wire_steps):
    rows = []
    r, alpha = 0.6, 0.3 - 0.2j
    mean0 = np.array([2 * alpha.real, 2 * alpha.imag])
    inputs = {"L": pg.GaussianInput.coherent(alpha)}
    pattern = ref._wire_pattern(r)
    for seed in range(n_seeds):
        res = {}
        for name in ("native-gaussian", "gaussian"):
            res[name] = pg.simulate(pattern, backend=name, inputs=inputs, seed=SEED + seed)
        a, b = res["native-gaussian"].outcomes["a"], res["native-gaussian"].outcomes["b"]
        m1, c1, d1 = ref._wire_step(mean0, np.eye(2), r, np.arctan2(1, 0.3), a)
        m2, c2, d2 = ref._wire_step(m1, c1, r, np.arctan2(1, 0.2 + 0.5 * a), b)
        out = res["native-gaussian"].state
        # Same outcomes replayed on the NumPy backend (postselection) -> same conditional state.
        replay = pg.simulate(pattern, backend="gaussian", inputs=inputs, measurement_outcomes={"a": a, "b": b})
        rows.append({
            "pattern": "2-step adaptive wire (angle depends on outcome a)",
            "seed": SEED + seed,
            "outcome_a": a,
            "outcome_b": b,
            "native_vs_hand_error": float(max(np.abs(out.mean - m2).max(), np.abs(out.covariance - c2).max())),
            "native_loglik_error": abs(res["native-gaussian"].log_likelihood - np.log(d1 * d2)),
            "same_seed_outcome_diff": float(max(abs(a - res["gaussian"].outcomes["a"]),
                                                abs(b - res["gaussian"].outcomes["b"]))),
            "same_seed_state_diff": float(max(np.abs(out.mean - res["gaussian"].state.mean).max(),
                                              np.abs(out.covariance - res["gaussian"].state.covariance).max())),
            "replayed_outcomes_state_diff": float(max(np.abs(out.mean - replay.state.mean).max(),
                                                      np.abs(out.covariance - replay.state.covariance).max())),
        })
    shears = list(np.linspace(-0.6, 0.6, wire_steps))
    long_wire = pg.protocols.wire(shears, squeezing=1.0)
    long_inputs = {long_wire.inputs[0]: pg.GaussianInput.coherent(0.4 + 0.1j)}
    for seed in range(n_seeds):
        nat = pg.simulate(long_wire, backend="native-gaussian", inputs=long_inputs, seed=SEED + 1000 + seed)
        num = pg.simulate(long_wire, backend="gaussian", inputs=long_inputs, seed=SEED + 1000 + seed)
        keys = sorted(nat.outcomes, key=repr)
        replay = pg.simulate(long_wire, backend="gaussian", inputs=long_inputs,
                             measurement_outcomes=dict(nat.outcomes))
        rows.append({
            "pattern": f"{wire_steps}-step Fourier wire with shears (corrections by feed-forward)",
            "seed": SEED + 1000 + seed,
            "outcome_a": float(nat.outcomes[keys[0]]),
            "outcome_b": float(nat.outcomes[keys[-1]]),
            "native_vs_hand_error": np.nan,
            "native_loglik_error": abs(nat.log_likelihood - num.log_likelihood),
            "same_seed_outcome_diff": float(max(abs(nat.outcomes[k] - num.outcomes[k]) for k in keys)),
            "same_seed_state_diff": float(max(np.abs(nat.state.mean - num.state.mean).max(),
                                              np.abs(nat.state.covariance - num.state.covariance).max())),
            "replayed_outcomes_state_diff": float(max(np.abs(nat.state.mean - replay.state.mean).max(),
                                                      np.abs(nat.state.covariance - replay.state.covariance).max())),
        })
    return rows


def main():
    args = parse_args()
    plt = common.setup_style()
    rng = common.rng(SEED)
    modes_list = [1, 2, 4, 8, 16, 32] + ([64, 128] if args.full else [])
    depths = [10, 40, 160, 640] + ([2560] if args.full else [])
    n_seeds, wire_steps = (20, 24) if args.full else (8, 12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        states = state_rows()
        circuits = circuit_rows(modes_list, depths, rng)
        conditional = conditional_rows(rng)
        feedforward = feedforward_rows(n_seeds, wire_steps)

    failures = []
    for r in states:
        for tag in ("native", "numpy"):
            if max(r[f"{tag}_mean_error"], r[f"{tag}_cov_error"], r[f"{tag}_photon_number_error"]) > TOL_STATE \
                    or r[f"{tag}_physical_margin"] < -TOL_PHYSICAL_REL:
                failures.append({"check": f"state:{r['state']}:{tag}", **r})
    for r in circuits:
        tol = C_ROUND * r["depth"] * EPS
        for tag in ("native", "numpy"):
            if r[f"{tag}_cov_rel_error"] > tol or r[f"{tag}_mean_error"] > tol \
                    or r[f"{tag}_physical_margin"] < -TOL_PHYSICAL_REL:
                failures.append({"check": f"circuit:{tag}", "tolerance": tol, **r})
        if r["family"] == "unitary" and (r["native_symplectic_residual"] > tol or r["native_S_rel_error"] > tol):
            failures.append({"check": "circuit:symplectic", "tolerance": tol, **r})
    for r in conditional:
        for tag in ("native", "numpy"):
            if r[f"{tag}_conditional_error"] > TOL_CONDITIONAL or r[f"{tag}_density_rel_error"] > TOL_DENSITY_REL:
                failures.append({"check": f"conditional:{r['measurement']}:{tag}", **r})
    for r in feedforward:
        worst = max(r["same_seed_outcome_diff"], r["same_seed_state_diff"], r["replayed_outcomes_state_diff"])
        hand = 0.0 if np.isnan(r["native_vs_hand_error"]) else r["native_vs_hand_error"]
        if worst > TOL_TRAJECTORY or hand > TOL_FEEDFORWARD or r["native_loglik_error"] > TOL_FEEDFORWARD:
            failures.append({"check": "feedforward", **r})

    config = {
        "backend": "native-gaussian (compiled photographiq._native.gaussian) vs NumPy 'gaussian'",
        "reference": "first-principles expm(2 Omega M) symplectics, closed-form states, "
                     "precision-matrix dilation conditioning (tests/validation/test_val_gaussian.py)",
        "precision": "float64",
        "cutoff": None,
        "seed": SEED,
        "full": args.full,
        "parameters": {"modes": modes_list, "depths": depths, "feedforward_seeds": n_seeds,
                       "long_wire_steps": wire_steps, "squeeze_scale": "0.15 * min(1, sqrt(40 n / depth))", "cz_and_shear_scale": "0.3 * min(1, sqrt(40 n / depth))",
                       "loss_eta_range": [0.6, 1.0], "loss_nth_range": [0.0, 0.2]},
        "acceptance": {"TOL_STATE": TOL_STATE, "circuit_relative_bound": f"{C_ROUND} * depth * eps",
                       "TOL_PHYSICAL_REL": TOL_PHYSICAL_REL, "TOL_CONDITIONAL": TOL_CONDITIONAL,
                       "TOL_DENSITY_REL": TOL_DENSITY_REL, "TOL_TRAJECTORY": TOL_TRAJECTORY,
                       "TOL_FEEDFORWARD": TOL_FEEDFORWARD},
    }
    common.save_result(states, "R43_native_gaussian_states", extra=config)
    common.save_result(conditional, "R43_native_gaussian_conditional", extra=config)
    common.save_result(feedforward, "R43_native_gaussian_feedforward", extra=config)
    common.save_result(circuits, "R43_native_gaussian_validation", extra=config)
    if failures:
        common.save_csv(failures, "R43_native_gaussian_validation_FAILURES")

    # ----- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(modes_list)))
    for n, color in zip(modes_list, colors):
        sub = [r for r in circuits if r["modes"] == n and r["family"] == "noisy"]
        axes[0].plot([r["depth"] for r in sub], [max(r["native_cov_rel_error"], 1e-17) for r in sub],
                     "o-", color=color, label=f"n={n}")
        sub = [r for r in circuits if r["modes"] == n and r["family"] == "unitary"]
        axes[1].plot([r["depth"] for r in sub], [max(r["native_symplectic_residual"], 1e-17) for r in sub],
                     "s-", color=color, label=f"n={n}")
    d = np.array(depths, dtype=float)
    for ax in axes[:2]:
        ax.plot(d, C_ROUND * d * EPS, "k--", lw=1, label=r"bound $64\,D\,\epsilon$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Circuit depth D (gates)")
    axes[0].set_ylabel(r"$\max|V_{\rm native}-V_{\rm ref}|/\max|V_{\rm ref}|$")
    axes[0].set_title("(a) covariance error, gates + loss")
    axes[1].set_ylabel(r"$\|S\Omega S^T-\Omega\|_{\max}/\|S\|_2^2$")
    axes[1].set_title("(b) symplectic residual, native S")
    axes[0].legend(fontsize=7, ncol=2)
    short = {"homodyne": "hom", "inefficient homodyne": "hom(eta=0.7)", "heterodyne": "het"}
    labels = [f"{short[r['measurement']]} n={r['modes']}" for r in conditional]
    x = np.arange(len(conditional))
    axes[2].semilogy(x, [max(r["native_conditional_error"], 1e-17) for r in conditional], "o",
                     label="native: conditional moments")
    axes[2].semilogy(x, [max(r["native_density_rel_error"], 1e-17) for r in conditional], "^",
                     label="native: outcome density (rel.)")
    axes[2].semilogy(x, [max(r["native_vs_numpy"], 1e-17) for r in conditional], "x",
                     label="native vs NumPy (same outcome)")
    axes[2].axhline(TOL_CONDITIONAL, color="k", ls="--", lw=1, label="acceptance 1e-10")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    axes[2].set_title("(c) conditioning vs dilation reference")
    axes[2].set_ylim(1e-17, 1e-8)
    axes[2].legend(fontsize=6.5, loc="upper left")
    common.save_figure(fig, "R43_native_gaussian_validation")
    plt.close(fig)

    noisy = [r for r in circuits if r["family"] == "noisy"]
    unitary = [r for r in circuits if r["family"] == "unitary"]
    worst_cov = max(r["native_cov_rel_error"] for r in circuits)
    worst_cov_noisy = max(r["native_cov_rel_error"] for r in noisy)
    worst_sym = max(r["native_symplectic_residual"] for r in unitary)
    worst_margin = min(r["native_physical_margin"] for r in circuits)
    worst_cond = max(r["native_conditional_error"] for r in conditional)
    worst_density = max(r["native_density_rel_error"] for r in conditional)
    worst_traj = max(max(r["same_seed_outcome_diff"], r["same_seed_state_diff"]) for r in feedforward)
    worst_hand = max(r["native_vs_hand_error"] for r in feedforward if not np.isnan(r["native_vs_hand_error"]))
    common.save_caption(
        "R43_native_gaussian_validation",
        f"""
**R43 - native Gaussian backend vs analytic phase-space evolution.** (a) Relative covariance
error of `native-gaussian` against a dense first-principles reference (gate symplectics
S = expm(2 Omega M) from each quadratic generator; loss X = sqrt(eta) I, Y = (1-eta)(2 n_th+1) I)
for seeded random circuits of rotations, squeezers (|r| ~ 0.15 min(1, sqrt(40 n/D)), keeping the accumulated squeezing O(1)), beam splitters, CZ, shears,
displacements and thermal loss (eta in [0.6, 1], n_th <= 0.2) on n = {modes_list[0]}..{modes_list[-1]}
modes and depth D = {depths[0]}..{depths[-1]} (worst shown {worst_cov_noisy:.1e}; worst over the
unplotted unitary family, dominated by the expm reference itself at n = 1, {worst_cov:.1e}; worst physicality margin
min eig(V + i Omega)/||V|| = {worst_margin:.1e}). (b) Symplectic residual of the composed native
map S, extracted column-by-column from the native mean update (worst {worst_sym:.1e}). Dashed:
acceptance bound 64 D eps (heuristic forward-error model, not a proof). (c) Homodyne (ideal and
eta = 0.7, noise 0.15), and heterodyne conditioning on random mixed states (n = 2, 4, 8) against
a physical-dilation, precision-matrix reference: worst conditional-moment error {worst_cond:.1e},
worst relative outcome-density error {worst_density:.1e} (exact zeros drawn at 1e-17). Not shown: closed-form coherent/squeezed/
thermal states (worst {max(max(r['native_mean_error'], r['native_cov_error']) for r in states):.1e}),
and adaptive feed-forward: the two-step wire with outcome-dependent angle agrees with hand propagation
to {worst_hand:.1e}; native and NumPy trajectories with the same seed agree to {worst_traj:.1e} in
outcomes and final moments. hbar = 2, float64, seed {SEED}. Source: R43 (14_native_gaussian/43_native_gaussian_validation.py).
""",
    )

    common.print_summary(
        "R43 native Gaussian validation",
        circuits=len(circuits),
        worst_cov_rel=f"{worst_cov:.2e}",
        worst_symplectic_residual=f"{worst_sym:.2e}",
        worst_physical_margin=f"{worst_margin:.2e}",
        worst_state_error=f"{max(max(r['native_mean_error'], r['native_cov_error']) for r in states):.2e}",
        worst_conditional=f"{worst_cond:.2e}",
        worst_density_rel=f"{worst_density:.2e}",
        worst_feedforward_hand=f"{worst_hand:.2e}",
        worst_trajectory_diff=f"{worst_traj:.2e}",
        noisy_cases=len(noisy),
        failures=len(failures),
    )
    if failures:
        raise AssertionError(f"{len(failures)} R43 checks failed; see R43_native_gaussian_validation_FAILURES.csv")


if __name__ == "__main__":
    main()
