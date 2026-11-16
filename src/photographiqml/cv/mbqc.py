"""Directly trainable weighted CV cluster states (CVCluster): homodyne angles as parameters.

A ``wires x columns`` grid cluster: horizontal CZ edges of weight 1 along each wire and
vertical CZ rungs (weight g, optionally trainable) between neighbouring wires on chosen
columns. Column 0 holds the externally encoded input modes, the last column the outputs, and
every other node is a finite-squeezing p-squeezed resource mode (variances e^{2r}, e^{-2r}).
Measured node i is read out by the physical homodyne q cos(theta_i) + p sin(theta_i).

Feedforward is the CV-flow correction of Booth and Markham (``photographiq.cvflow``): after
outcome m_i every unmeasured node gets the certified displacement -(c q, A c p) m_i / sin(theta_i).
The certificate depends on the edge weights only, so it is recomputed whenever trainable
weights change and an uncertifiable configuration is rejected. Measurement angles do not
enter the certificate; each sets the teleportation shear k_i = cot(theta_i), so theta is
pi-periodic and theta = 0 (mod pi) is singular (a q measurement deletes the node). With the
default ``angles="bounded"`` the trainable parameter phi_i sets
theta_i = pi/2 + (pi/2 - margin) tanh(phi_i), which covers one full period except a margin
around the singular angle; ``angles="direct"`` binds theta_i itself and rejects singular values.

``channel`` evaluates the exact unconditional Gaussian channel by forward substitution of the
linear measurement records (independent of ``photographiq.analysis``); ``pattern`` returns the
executable ``photographiq.Pattern`` built by ``flow_pattern``. Tests compare the two.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

import photographiq as pg
from photographiq.cvflow import certify_cv_flow, flow_pattern

ANGLE_TOLERANCE = 1e-6


class CVCluster:
    """Trainable homodyne-angle CV cluster program on a weighted grid.

    Not the CVMuTA proposal of the PhotoGraphiQML roadmap (MuTA's triangle cells with CV
    gates, which remains planned): this is a generic grid cluster whose measurement angles,
    and optionally rung weights, are the trainable parameters.

    Args:
        wires: number of optical wires (rows of the grid), also the input/output mode count.
        columns: grid columns (>= 2); columns - 1 measurement layers per wire.
        squeezing: finite resource squeezing r of every non-input node.
        rungs: columns carrying vertical CZ rungs between neighbouring wires.
        rung_weight: initial (or fixed) rung weight g.
        trainable_weights: expose rung weights as trainable parameters.
    """

    def __init__(
        self,
        wires,
        columns,
        *,
        squeezing=1.0,
        rungs=None,
        rung_weight=1.0,
        trainable_weights=False,
        angles="bounded",
        margin=0.05,
    ):
        for name, v in (("wires", wires), ("columns", columns)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if columns < 2:
            raise ValueError("A cluster program needs at least two columns")
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Resource squeezing must be finite and nonnegative")
        rungs = tuple(range(1, columns - 1)) if rungs is None else tuple(rungs)
        if any(not 0 <= c < columns for c in rungs):
            raise ValueError("Rung columns must lie inside the grid")
        rungs = rungs if wires > 1 else ()
        if not np.isfinite(rung_weight) or rung_weight == 0:
            raise ValueError("Rung weight must be finite and nonzero")
        if angles not in ("bounded", "direct") or not 0 < margin < np.pi / 2:
            raise ValueError("angles must be 'bounded' or 'direct', with 0 < margin < pi/2")
        self.angle_mode, self.margin = angles, float(margin)
        self.wires, self.columns, self.squeezing = wires, columns, float(squeezing)
        self.rungs, self.trainable_weights = rungs, bool(trainable_weights)
        self.nodes = tuple((w, c) for c in range(columns) for w in range(wires))
        self.inputs = tuple((w, 0) for w in range(wires))
        self.outputs = tuple((w, columns - 1) for w in range(wires))
        self.order = tuple(n for n in self.nodes if n not in self.outputs)  # column-major
        self.edges = [((w, c), (w, c + 1), 1.0) for w in range(wires) for c in range(columns - 1)]
        self.rung_edges = [((w, c), (w + 1, c)) for c in rungs for w in range(wires - 1)]
        prefix = "phi" if angles == "bounded" else "theta"
        self.angle_names = tuple(f"{prefix}.w{w}.c{c}" for (w, c) in self.order)
        self.weight_names = tuple(f"g.w{w}.c{c}" for (w, c), _ in self.rung_edges)
        # theta = pi/2 everywhere: every step is a Fourier transform (identity after four)
        self.defaults = dict.fromkeys(self.angle_names, 0.0 if angles == "bounded" else np.pi / 2)
        self.defaults.update(dict.fromkeys(self.weight_names, float(rung_weight)))
        self._flow_cache: dict = {}
        self.certificate(self.defaults)  # reject an uncertifiable layout at construction

    @property
    def modes(self):
        return self.wires

    @property
    def parameters(self):
        return self.angle_names + (self.weight_names if self.trainable_weights else ())

    def _weights(self, params):
        return tuple(
            float(np.real(params.get(n, self.defaults[n])))
            if self.trainable_weights
            else self.defaults[n]
            for n in self.weight_names
        )

    def graph(self, params):
        network = nx.Graph()
        network.add_nodes_from(self.nodes)
        network.add_weighted_edges_from(self.edges)
        network.add_weighted_edges_from(
            (u, v, w) for (u, v), w in zip(self.rung_edges, self._weights(params))
        )
        return pg.CVGraph(
            network, squeezing=self.squeezing, inputs=self.inputs, outputs=self.outputs
        )

    def certificate(self, params):
        """CV-flow certificate for the current weights (cached per weight tuple)."""
        key = self._weights(params)
        if key not in self._flow_cache:
            flow = certify_cv_flow(self.graph(params), self.order)
            if flow is None:
                raise ValueError(f"No CV-flow certificate for rung weights {key}")
            if len(self._flow_cache) >= 256:  # weights change every step while training
                self._flow_cache.clear()
            self._flow_cache[key] = flow
        return self._flow_cache[key]

    def angles(self, params):
        """Physical homodyne angles theta_i in measurement order."""
        values = [params.get(n, self.defaults[n]) for n in self.angle_names]
        if self.angle_mode == "bounded":
            return [np.pi / 2 + (np.pi / 2 - self.margin) * np.tanh(v) for v in values]
        return values

    def angle_parameters(self, thetas):
        """Parameter values that realise the angles ``thetas`` (reduced mod pi)."""
        thetas = np.mod(np.asarray(thetas, dtype=float), np.pi)
        if self.angle_mode == "direct":
            return dict(zip(self.angle_names, thetas))
        span = np.pi / 2 - self.margin
        if np.any(np.abs(thetas - np.pi / 2) >= span):
            raise ValueError("Angle lies inside the excluded margin around 0 mod pi")
        return dict(zip(self.angle_names, np.arctanh((thetas - np.pi / 2) / span)))

    def _angles(self, params):
        angles = self.angles(params)
        for name, a in zip(self.angle_names, angles):
            if abs(np.sin(np.real(a))) < ANGLE_TOLERANCE:
                raise ValueError(f"{name} = 0 mod pi is a q measurement: singular teleportation")
        return angles

    def channel(self, params, x=(), *, mode="ideal", squeezing=None):
        """(X, t, Y, measurements) of the unconditional output channel on the input modes."""
        if mode not in ("ideal", "physical"):
            raise ValueError("mode must be 'ideal' or 'physical'")
        r = self.squeezing if squeezing is None else float(squeezing)
        angles = self._angles(params)
        flow = self.certificate(params)
        index = {n: i for i, n in enumerate(self.nodes)}
        size = 2 * len(self.nodes)
        dtype = complex if any(np.iscomplexobj(a) for a in angles) else float
        rows = {n: np.eye(size, dtype=dtype)[2 * i : 2 * i + 2].copy() for n, i in index.items()}
        adjacency = np.zeros((len(self.nodes), len(self.nodes)))
        for u, v, w in self.edges + [
            (u, v, w) for (u, v), w in zip(self.rung_edges, self._weights(params))
        ]:
            adjacency[index[u], index[v]] = adjacency[index[v], index[u]] = w
        q_rows = {n: rows[n][0].copy() for n in self.nodes}
        for n in self.nodes:  # CZ network: p_i += sum_j A_ij q_j
            rows[n][1] += sum(adjacency[index[n], index[m]] * q_rows[m] for m in self.nodes)
        measured = set()
        for node, theta in zip(self.order, angles):
            scaled = (np.cos(theta) * rows[node][0] + np.sin(theta) * rows[node][1]) / np.sin(theta)
            measured.add(node)
            correction = flow.corrections[node]
            for target in self.nodes:
                if target in measured:
                    continue
                q = correction.get(target, 0.0)
                p = sum(adjacency[index[target], index[m]] * v for m, v in correction.items())
                rows[target][0] -= q * scaled
                rows[target][1] -= p * scaled
        L = np.vstack([rows[n] for n in self.outputs])
        cols_in = [j for n in self.inputs for j in (2 * index[n], 2 * index[n] + 1)]
        res = [n for n in self.nodes if n not in self.inputs]
        X = L[:, cols_in]
        anti = np.array([2 * index[n] for n in res])
        leak = float(np.abs(L[:, anti]).max()) if len(anti) else 0.0
        if leak > 1e-9:  # flow must cancel the anti-squeezed resource quadratures exactly
            raise RuntimeError(f"Anti-squeezed resource quadratures reach the output ({leak:.2e})")
        Y = np.zeros((2 * self.wires, 2 * self.wires), dtype=dtype)
        if mode == "physical":
            squeezed = L[:, anti + 1]
            Y = np.exp(-2 * r) * squeezed @ squeezed.T
        return X, np.zeros(2 * self.wires, dtype=dtype), Y, len(self.order)

    def pattern(self, params, *, squeezing=None):
        """Executable finite-squeezing pattern (``flow_pattern`` with shears cot(theta))."""
        r = self.squeezing if squeezing is None else float(squeezing)
        graph = self.graph(params)
        for n in graph.nodes:
            graph.network.nodes[n]["squeezing"] = r
        angles = self._angles(params)
        shears = {n: float(np.cos(a) / np.sin(a)) for n, a in zip(self.order, angles)}
        return flow_pattern(graph, self.order, shears=shears)

    def config(self):
        return {
            "wires": self.wires,
            "columns": self.columns,
            "squeezing": self.squeezing,
            "rungs": list(self.rungs),
            "rung_weight": next(iter(self.defaults[n] for n in self.weight_names), 1.0),
            "trainable_weights": self.trainable_weights,
            "angles": self.angle_mode,
            "margin": self.margin,
        }
