"""Experimental non-Gaussian CV programs executed on PhotoGraphiQ's Fock engines.

Adds Kerr exp(i kappa n^2) and cubic phase exp(i gamma q^3 / 6) (hbar = 2, q = a + a^dagger)
to the Gaussian gate list. Execution is the native finite-Fock engine with an exclusive
total-photon ``cutoff``; every prediction reports the retained norm, and no Gaussian
approximation is substituted. Convergence in the cutoff must be checked (see the tests and
the cutoff sweeps of the experiments); the engine conventions are validated against an
independent dense matrix-exponential reference (tests/validation/cv_fock_reference.py).

Cubic phase gates can alternatively be lowered to PhotoGraphiQ's finite-energy resource
injection (``photographiq.non_gaussian.cubic_injection``: cubic-phase ancilla, inverse SUM,
ancilla q homodyne, quadratic-phase and displacement corrections). That lowering is a
conditional finite-energy filter, not an exact gate; ``injection_channel`` integrates the
outcome record numerically to give its unconditional output and compares it with the gate.
"""

from __future__ import annotations

import warnings

import numpy as np

import photographiq as pg
from photographiq.fock_analysis import quadrature_moment
from photographiq.non_gaussian import cubic_injection

from .gaussian import GaussianProgram


class NonGaussianProgram(GaussianProgram):
    non_gaussian = True

    def kerr(self, mode, kappa):
        return self._add("kerr", (mode,), kappa)

    def cubic_phase(self, mode, gamma):
        return self._add("cubic", (mode,), gamma)

    @property
    def is_gaussian(self):
        return not any(k in ("kerr", "cubic") for k, *_ in self.gates)

    def channel(self, params, x=(), *, mode="ideal", squeezing=None):
        if not self.is_gaussian:
            raise NotImplementedError(
                "Non-Gaussian programs have no Gaussian channel; use fock_moments"
            )
        return super().channel(params, x, mode=mode, squeezing=squeezing)

    def circuit(self, params, x=()):
        raise NotImplementedError("Non-Gaussian programs execute on a Fock engine (pattern())")

    def pattern(self, params, x=(), *, cubic="gate", injection_squeezing=0.5):
        """Gate-level pattern; ``cubic="injection"`` lowers cubic gates to resource injection."""
        if cubic not in ("gate", "injection"):
            raise ValueError("cubic must be 'gate' or 'injection'")
        gaussian = GaussianProgram(self.modes)
        commands: list = []
        ancillas: list = []
        for i, (kind, modes, v) in enumerate(self._resolved(params, x)):
            v = [float(a) for a in v]
            if kind == "kerr":
                commands.append(pg.Kerr(modes[0], v[0]))
            elif kind == "cubic" and cubic == "gate":
                commands.append(pg.CubicPhase(modes[0], v[0]))
            elif kind == "cubic":
                gadget = cubic_injection(
                    v[0], injection_squeezing, input_node=modes[0], ancilla=f"anc{i}", key=f"m{i}"
                )
                commands += [c for c in gadget.commands if not isinstance(c, pg.Output)]
                ancillas.append(f"m{i}")
            else:
                gaussian.gates = [(kind, modes, tuple(v), None)]
                commands += [
                    c
                    for c in gaussian.gate_pattern({}, ()).commands
                    if not isinstance(c, pg.Output)
                ]
        pattern = pg.Pattern(inputs=tuple(range(self.modes))).extend(commands)
        pattern.append(pg.Output(tuple(range(self.modes))))
        pattern.ancilla_keys = tuple(ancillas)
        return pattern.validate()

    def run(self, params, x, feature_map, cutoff, *, backend="native-fock", **kw):
        """Execute one data point; returns the PhotoGraphiQ Result (Fock state)."""
        mean, cov = feature_map.moments(x)
        inputs = {
            j: pg.GaussianInput(
                tuple(mean[2 * j : 2 * j + 2]),
                tuple(map(tuple, cov[2 * j : 2 * j + 2, 2 * j : 2 * j + 2])),
            )
            for j in range(self.modes)
        }
        pattern_kw = {k: kw.pop(k) for k in ("cubic", "injection_squeezing") if k in kw}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return pg.simulate(
                self.pattern(params, x, **pattern_kw),
                backend=backend,
                cutoff=cutoff,
                inputs=inputs,
                **kw,
            )

    def fock_moments(self, params, x, feature_map, cutoff, modes, **kw):
        """(means, variances, minimum retained norm) of q, p on ``modes`` for one data point.

        The engine renormalises after every operation; the retained norm is the smallest
        pre-renormalisation norm, so 1 - norm bounds the truncated probability per step.
        """
        state = self.run(params, x, feature_map, cutoff, **kw).state
        means, variances = [], []
        for m in modes:
            for angle in (0.0, np.pi / 2):
                first = quadrature_moment(state, m, 1, angle)
                means.append(first)
                variances.append(quadrature_moment(state, m, 2, angle) - first**2)
        return np.array(means), np.array(variances), float(min(state.retained_norms))

    def resources(self, params, x, cutoff):
        kinds = [k for k, *_ in self.gates]
        return {
            "fock_cutoff": cutoff,
            "kerr_gates": kinds.count("kerr"),
            "cubic_gates": kinds.count("cubic"),
            "fock_dimension": int(
                np.prod([cutoff + j for j in range(self.modes)])
                // int(np.prod(range(1, self.modes + 1)))
            ),
        }

    def injection_channel(
        self,
        params,
        x,
        feature_map,
        cutoff,
        *,
        injection_squeezing=0.5,
        outcomes=np.linspace(-8, 8, 161),
    ):
        """Unconditional output of the injection lowering (one cubic gate) by outcome quadrature.

        Integrates rho_out = int p(m) rho(m) dm with the trapezoid rule over ``outcomes`` from
        postselected executions; returns the averaged density matrix, the integrated outcome
        probability (truncation and range check, ideally 1) and the per-outcome retained norms.
        """
        keys = None
        weights, states, norms = [], [], []
        for m in outcomes:
            if keys is None:
                keys = self.pattern(
                    params, x, cubic="injection", injection_squeezing=injection_squeezing
                ).ancilla_keys
                if len(keys) != 1:
                    raise NotImplementedError("injection_channel integrates exactly one cubic gate")
            result = self.run(
                params,
                x,
                feature_map,
                cutoff,
                cubic="injection",
                injection_squeezing=injection_squeezing,
                measurement_outcomes={keys[0]: float(m)},
            )
            density = result.measurement_statistics[keys[0]]["value"]
            rho = np.asarray(result.state.density_matrix)
            weights.append(density)
            states.append(rho / np.trace(rho).real)
            norms.append(float(min(result.state.retained_norms)))
        density = np.asarray(weights)
        dx = np.gradient(np.asarray(outcomes))
        total = float(np.sum(density * dx))
        rho = sum(w * d * s for w, d, s in zip(density, dx, states)) / total
        return {
            "density_matrix": rho,
            "basis": result.state.basis,
            "outcome_probability": total,
            "retained_norms": norms,
        }
