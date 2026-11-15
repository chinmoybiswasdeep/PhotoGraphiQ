"""R46: Fock-space optimisations that PhotoGraphiQ actually implements, with ablations.

Scientific question: what do the implemented Fock-space optimisations buy in runtime and
memory, at identical physics and identical numerical output?

Implemented optimisations studied (inspected in cpp/src/fock/{basis,kernels,gates}.cpp,
src/photographiq/backends/native_fock.py and src/photographiq/gpu/fock.py):

1. Total-photon truncation basis, n_1 + ... + n_m < c, dimension C(m + c - 1, m), instead of a
   per-mode product basis of c^m states. This is a MODEL choice (the two truncations are different
   approximations), so the product basis is reported only as a dimension/memory count, never as
   an output-compared variant.
2. Fibre-local gate application (``apply_pure``): a k-mode gate acts on each fibre of fixed
   occupations of the other m - k modes through the leading block of its local matrix (P U P);
   no D x D operator is formed; cost sum_f dim_f^2. Compared, on the SAME P U P model, with
   (a) the global sparse D x D gate G assembled by ``photographiq.gpu.fock`` and applied with
   scipy.sparse (CPU) and cuSPARSE (GPU, CuPy), and (b) the dense D x D matrix of G (small D).
3. Photon-number conservation of passive gates: a passive Gaussian unitary (beam splitter,
   rotation) conserves the total photon number, so its local matrix is block diagonal in the
   sectors N = 0..c-1; ``gaussian_unitary`` skips the off-sector blocks (gates.cpp: "passive"
   branch). With total-photon truncation a passive gate is then exactly unitary on the truncated
   space (retained norm 1). The skip cannot be switched off from Python, so the block structure is
   verified (off-sector entries exactly 0) and its cost compared with a non-passive two-mode gate
   (CZ) of the same local dimension: an observation, not a controlled ablation.
4. Diagonal gates (rotation, Kerr, cross-Kerr) as phase multiplication (``phase_pure``) instead of
   a matrix application of the diagonal operator: same output, compared directly.
5. Caching: basis tables and the cubic-phase matrix (``functools.lru_cache`` in native_fock.py),
   and the sparse structure of the GPU/CPU gate (``_maps``/``_template`` in gpu/fock.py): first
   call vs cached call.

Accuracy: every variant's output is compared with the native fibre kernel (max |difference|,
states normalised). Memory: exact byte counts of the representation each variant needs
(``.nbytes`` of the arrays it holds: local matrix, CSR data+indices+indptr, dense matrix, state).
Timing: ``common.benchmark`` (warm-up excluded, median and IQR); final only with ``PGQ_IDLE_MACHINE=1``, otherwise provisional.
GPU timings include host<->device transfers (results returned to NumPy) and are taken only when
CuPy reports a usable device; ``Execution.device`` is recorded so a CPU fallback cannot be
mistaken for GPU execution (the GPU kernels never fall back).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from photographiq import native  # noqa: E402
from photographiq.backends import native_fock as nf  # noqa: E402
from photographiq.gpu import fock as gfock  # noqa: E402

SEED = 20261146
ENGINE = native.require("fock")
# ----- acceptance criteria (declared before evaluation) ---------------------------------
TOL_VARIANT = 1e-12  # same P U P model, different summation order: float64 rounding of <= D terms
TOL_PHASE = 1e-14  # elementwise phases vs diagonal matrix application
TOL_PASSIVE_NORM = 1e-13  # passive gate on the truncated space is unitary (norm 1)
OFF_SECTOR_EXACT = 0.0  # off-sector entries of a passive local matrix are skipped: exactly zero
DENSE_LIMIT = 4000  # dense D x D variant only when 16 D^2 <= 256 MB


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="larger (modes, cutoff) grid")
    parser.add_argument("--repeats", type=int, default=None)
    return parser.parse_args()


def gpu_available():
    try:
        from photographiq.gpu.device import require

        cp = require()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - any failure means "no usable GPU", recorded below
        return False


def gate_ops(cutoff):
    c, s = np.cos(0.7), np.sin(0.7)
    bs = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = 0.4
    sq = np.diag([np.exp(-0.3), np.exp(0.3)])
    return {
        "beamsplitter (passive, 2 modes)": ((0, 1), ENGINE.gaussian_unitary(bs, np.zeros(4), cutoff)),
        "CZ g=0.4 (active, 2 modes)": ((0, 1), ENGINE.gaussian_unitary(cz, np.zeros(4), cutoff)),
        "squeezer r=0.3 (active, 1 mode)": ((0,), ENGINE.gaussian_unitary(sq, np.zeros(2), cutoff)),
    }


def sparse_bytes(m):
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def gate_rows(grid, repeats, use_gpu, rng):
    rows = []
    for modes, cutoff in grid:
        dim = ENGINE.dimension(modes, cutoff)
        assert dim == comb(modes + cutoff - 1, modes)
        assert ENGINE.basis(modes, cutoff).shape == (dim, modes)
        for batch in (1, 32):
            psi = rng.normal(size=(batch, dim)) + 1j * rng.normal(size=(batch, dim))
            psi /= np.linalg.norm(psi, axis=1, keepdims=True)
            for gate, (local, op) in gate_ops(cutoff).items():
                def native_run(psi=psi, local=local, op=op):
                    out = psi.copy()
                    for b in range(len(out)):
                        ENGINE.apply_pure(out[b], modes, cutoff, list(local), op)
                    return out

                t_nat = common.benchmark(native_run, warmup=1, repeats=repeats)
                ref = t_nat["result"]
                variants = {"native fibre kernel (apply_pure)": (t_nat, int(op.nbytes), "cpu")}
                t_sp = common.benchmark(lambda psi=psi, local=local, op=op: gfock.apply_operator(
                    psi, modes, cutoff, local, op, device="cpu"), warmup=1, repeats=repeats)
                gate_tpl = gfock._template((modes, cutoff, tuple(local)), "complex128", None)[0]
                variants["global sparse G, scipy (CPU)"] = (t_sp, sparse_bytes(gate_tpl) + int(op.nbytes),
                                                            t_sp["result"].execution.device)
                t_sp["result"] = t_sp["result"].states
                if use_gpu:
                    t_gpu = common.benchmark(lambda psi=psi, local=local, op=op: gfock.apply_operator(
                        psi, modes, cutoff, local, op, device="gpu"), warmup=2, repeats=repeats)
                    device = t_gpu["result"].execution.device
                    t_gpu["result"] = t_gpu["result"].states
                    variants["global sparse G, cuSPARSE (GPU, incl. transfers)"] = (
                        t_gpu, sparse_bytes(gate_tpl) + int(op.nbytes), device)
                if dim <= DENSE_LIMIT:
                    dense = gfock.apply_operator(np.eye(dim), modes, cutoff, local, op, device="cpu").states.T
                    t_dn = common.benchmark(lambda psi=psi, dense=dense: psi @ dense.T, warmup=1, repeats=repeats)
                    variants["dense global D x D matrix"] = (t_dn, int(dense.nbytes), "cpu")
                for variant, (timing, op_bytes, device) in variants.items():
                    out = np.asarray(timing["result"])
                    rows.append({
                        "modes": modes, "cutoff": cutoff, "dimension": dim, "product_basis_dimension": cutoff**modes,
                        "batch": batch, "gate": gate, "variant": variant, "device": device,
                        "median_seconds": timing["median_seconds"], "iqr_seconds": timing["iqr_seconds"],
                        "operator_bytes": op_bytes, "state_bytes": int(psi.nbytes),
                        "max_abs_diff_vs_native": float(np.abs(out - ref).max()),
                        "output_norm_min": float(np.linalg.norm(out, axis=1).min()),
                        "output_norm_max": float(np.linalg.norm(out, axis=1).max()),
                    })
            print(f"  m={modes} c={cutoff} D={dim} B={batch}: done", flush=True)
    return rows


def phase_rows(grid, repeats, rng):
    rows = []
    for modes, cutoff in grid:
        dim = ENGINE.dimension(modes, cutoff)
        psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi /= np.linalg.norm(psi)
        n = np.arange(cutoff)
        phases = 0.35 * n**2  # Kerr exp(i kappa n^2)
        diag = np.diag(np.exp(1j * phases))

        def by_phase():
            out = psi.copy()
            ENGINE.phase_pure(out, modes, cutoff, [0], phases)
            return out

        def by_matrix():
            out = psi.copy()
            ENGINE.apply_pure(out, modes, cutoff, [0], diag)
            return out

        a = common.benchmark(by_phase, warmup=1, repeats=repeats)
        b = common.benchmark(by_matrix, warmup=1, repeats=repeats)
        rows.append({"modes": modes, "cutoff": cutoff, "dimension": dim,
                     "phase_median_seconds": a["median_seconds"], "matrix_median_seconds": b["median_seconds"],
                     "phase_iqr": a["iqr_seconds"], "matrix_iqr": b["iqr_seconds"],
                     "max_abs_diff": float(np.abs(a["result"] - b["result"]).max())})
    return rows


def sector_rows(cutoffs, repeats):
    rows = []
    c, s = np.cos(0.7), np.sin(0.7)
    bs = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = 0.4
    for cutoff in cutoffs:
        basis = ENGINE.basis(2, cutoff)
        sector = basis.sum(axis=1)
        off = sector[:, None] != sector[None, :]
        t_bs = common.benchmark(lambda: ENGINE.gaussian_unitary(bs, np.zeros(4), cutoff), warmup=1, repeats=repeats)
        t_cz = common.benchmark(lambda: ENGINE.gaussian_unitary(cz, np.zeros(4), cutoff), warmup=1, repeats=repeats)
        u_bs, u_cz = t_bs["result"], t_cz["result"]
        block_bytes = sum(16 * (k + 1) ** 2 for k in range(cutoff))  # sector N has N+1 states
        rows.append({"cutoff": cutoff, "local_dimension": len(basis),
                     "bs_off_sector_max_abs": float(np.abs(u_bs[off]).max()),
                     "bs_zero_fraction": float(np.mean(u_bs == 0)),
                     "cz_off_sector_max_abs": float(np.abs(u_cz[off]).max()),
                     "bs_unitarity_error": float(np.abs(u_bs.conj().T @ u_bs - np.eye(len(basis))).max()),
                     "bs_median_seconds": t_bs["median_seconds"], "cz_median_seconds": t_cz["median_seconds"],
                     "dense_local_bytes": int(u_bs.nbytes), "sector_block_bytes": block_bytes})
    return rows


def cache_rows():
    rows = []

    def first_and_cached(label, call, clear):
        clear()
        t0 = time.perf_counter()
        call()
        first = time.perf_counter() - t0
        cached = common.benchmark(call, warmup=0, repeats=7)["median_seconds"]
        rows.append({"cache": label, "first_call_seconds": first, "cached_call_seconds": cached,
                     "speedup": first / cached if cached > 0 else np.inf})

    first_and_cached("basis table (6 modes, cutoff 12) - nf._occupations",
                     lambda: nf._occupations(6, 12), nf._occupations.cache_clear)
    first_and_cached("cubic phase matrix (projected, gamma 0.5, cutoff 60) - nf._cubic_phase_matrix",
                     lambda: nf._cubic_phase_matrix(0.5, 60, "projected", 1e-12), nf._cubic_phase_matrix.cache_clear)
    psi = np.ones((1, ENGINE.dimension(6, 10)), complex)
    op = ENGINE.gaussian_unitary(np.eye(4), np.zeros(4), 10)

    def clear_gpu():
        gfock._maps.cache_clear()
        gfock._template.cache_clear()
        gfock._rank.cache_clear()

    first_and_cached("sparse gate structure (6 modes, cutoff 10) - gpu.fock._maps/_template (CPU path)",
                     lambda: gfock.apply_operator(psi, 6, 10, (2, 4), op, device="cpu"), clear_gpu)
    return rows


def main():
    args = parse_args()
    repeats = args.repeats or (9 if args.full else 5)
    grid = [(2, 30), (3, 16), (4, 12), (5, 10), (6, 8), (8, 6)]
    if args.full:
        grid += [(4, 20), (6, 12), (8, 9), (10, 7)]
    use_gpu = gpu_available()
    rng = common.rng(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gates = gate_rows(grid, repeats, use_gpu, rng)
        phases = phase_rows(grid, repeats, rng)
        sectors = sector_rows([6, 10, 16, 24, 40] + ([64] if args.full else []), repeats)
        caches = cache_rows()
    dims = [{"modes": m, "cutoff": c, "total_photon_dimension": comb(m + c - 1, m), "product_dimension": c**m,
             "ratio": c**m / comb(m + c - 1, m), "total_vector_bytes": 16 * comb(m + c - 1, m),
             "product_vector_bytes": 16 * c**m, "total_density_bytes": 16 * comb(m + c - 1, m) ** 2}
            for m in range(1, 11) for c in (4, 6, 8, 12)]

    failures = [{"check": "variant disagreement", **r} for r in gates if r["max_abs_diff_vs_native"] > TOL_VARIANT]
    failures += [{"check": "passive gate not norm preserving", **r} for r in gates
                 if r["gate"].startswith("beamsplitter") and max(abs(r["output_norm_min"] - 1), abs(r["output_norm_max"] - 1)) > TOL_PASSIVE_NORM]
    failures += [{"check": "phase vs matrix", **r} for r in phases if r["max_abs_diff"] > TOL_PHASE]
    failures += [{"check": "passive off-sector entries nonzero", **r} for r in sectors
                 if r["bs_off_sector_max_abs"] > OFF_SECTOR_EXACT]
    failures += [{"check": "GPU variant did not run on a CUDA device", **r} for r in gates
                 if "GPU" in r["variant"] and not str(r["device"]).startswith("cuda")]

    extra = {"backend": "photographiq._native.fock (apply_pure, phase_pure, gaussian_unitary) and "
                        "photographiq.gpu.fock (scipy.sparse CPU / cuSPARSE GPU)",
             "precision": "complex128", "cutoff": "total-photon cutoff, see rows", "seed": SEED, "full": args.full,
             "gpu_used": use_gpu, "timing_status": common.timing_label(),
             "parameters": {"grid": grid, "batches": [1, 32], "repeats": repeats, "beamsplitter_theta": 0.7,
                            "cz": 0.4, "squeeze": 0.3, "kerr": 0.35},
             "native_build_info": native.build_info(),
             "acceptance": {"TOL_VARIANT": TOL_VARIANT, "TOL_PHASE": TOL_PHASE, "TOL_PASSIVE_NORM": TOL_PASSIVE_NORM,
                            "OFF_SECTOR_EXACT": OFF_SECTOR_EXACT}}
    common.save_result(gates, "R46_fock_space_optimization", extra=extra)
    common.save_result(phases, "R46_fock_phase_gates", extra=extra)
    common.save_result(sectors, "R46_fock_passive_sectors", extra=extra)
    common.save_result(caches, "R46_fock_caches", extra=extra)
    common.save_result(dims, "R46_fock_basis_dimensions", extra=extra)
    if failures:
        common.save_csv(failures, "R46_fock_space_optimization_FAILURES")

    plt = common.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for c, color in zip((4, 6, 8, 12), plt.cm.viridis(np.linspace(0, 0.85, 4))):
        sub = [d for d in dims if d["cutoff"] == c]
        axes[0, 0].semilogy([d["modes"] for d in sub], [d["product_dimension"] for d in sub], "--", color=color)
        axes[0, 0].semilogy([d["modes"] for d in sub], [d["total_photon_dimension"] for d in sub], "o-", color=color,
                            label=f"c={c}")
    axes[0, 0].set(xlabel="modes m", ylabel="basis dimension",
                   title="(a) total-photon C(m+c-1,m) (solid) vs product c^m (dashed)")
    axes[0, 0].legend(fontsize=7)
    var_colors = {"native fibre kernel (apply_pure)": "#24677b", "global sparse G, scipy (CPU)": "#c66d27",
                  "global sparse G, cuSPARSE (GPU, incl. transfers)": "#8a3b8f", "dense global D x D matrix": "#627a36"}
    gate = "beamsplitter (passive, 2 modes)"
    order = sorted(grid, key=lambda mc: ENGINE.dimension(*mc))
    xlab = [f"m={m},c={c}" + chr(10) + f"D={ENGINE.dimension(m, c)}" for m, c in order]
    xpos = {mc: i for i, mc in enumerate(order)}
    for variant, color in var_colors.items():
        for batch, ls in ((1, "o-"), (32, "s--")):
            sub = sorted((r for r in gates if r["variant"] == variant and r["gate"] == gate and r["batch"] == batch),
                         key=lambda r: xpos[(r["modes"], r["cutoff"])])
            if sub:
                axes[0, 1].semilogy([xpos[(r["modes"], r["cutoff"])] for r in sub], [r["median_seconds"] for r in sub],
                                    ls, color=color, label=f"{variant}, B={batch}", ms=4)
    mem = {"native: local matrix only": ("native fibre kernel (apply_pure)", "#24677b", "o-"),
           "global sparse G (CSR, scipy or cuSPARSE)": ("global sparse G, scipy (CPU)", "#c66d27", "s-"),
           "dense global D x D matrix": ("dense global D x D matrix", "#627a36", "^-")}
    for label, (variant, color, ls) in mem.items():
        sub = sorted((r for r in gates if r["variant"] == variant and r["gate"] == gate and r["batch"] == 1),
                     key=lambda r: xpos[(r["modes"], r["cutoff"])])
        axes[1, 0].semilogy([xpos[(r["modes"], r["cutoff"])] for r in sub], [r["operator_bytes"] for r in sub], ls,
                            color=color, label=label, ms=4)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlab, fontsize=6, rotation=40, ha="right")
    axes[0, 1].set(xlabel="(modes, cutoff), ordered by Fock dimension D", ylabel="median runtime per call (s)",
                   title="(b) one beam splitter on B states: identical P U P output")
    axes[0, 1].legend(fontsize=5.8, ncol=2, loc="upper left")
    axes[0, 1].set_ylim(top=axes[0, 1].get_ylim()[1] * 30)
    axes[1, 0].set(xlabel="(modes, cutoff), ordered by D", ylabel="operator representation (bytes)",
                   title="(c) memory of the beam-splitter representation")
    axes[1, 0].set_ylim(top=3e9)
    axes[1, 0].legend(fontsize=6.5, loc="upper left")
    axes[1, 1].semilogy([r["cutoff"] for r in sectors], [r["bs_median_seconds"] for r in sectors], "o-",
                        color="#24677b", label="beam splitter (passive: sector blocks only)")
    axes[1, 1].semilogy([r["cutoff"] for r in sectors], [r["cz_median_seconds"] for r in sectors], "s-",
                        color="#c66d27", label="CZ (active: full local matrix)")
    axes[1, 1].set(xlabel="cutoff (two-mode local matrix of dimension C(c+1, 2))",
                   ylabel="median runtime of gaussian_unitary (s)", title="(d) photon-number conservation of passive gates")
    axes[1, 1].legend(fontsize=7)
    fig.suptitle("Implemented Fock-space optimisations" + ("" if common.timing_final() else " (PROVISIONAL timings)"), fontsize=11)
    common.save_figure(fig, "R46_fock_space_optimization")
    plt.close(fig)

    largest = max(grid, key=lambda mc: ENGINE.dimension(*mc))
    big = [r for r in gates if (r["modes"], r["cutoff"]) == largest and r["gate"] == gate and r["batch"] == 1]
    nat = next(r for r in big if r["variant"].startswith("native"))
    spr = next(r for r in big if r["variant"].startswith("global sparse G, scipy"))
    worst_diff = max(r["max_abs_diff_vs_native"] for r in gates)
    top_sector = sectors[-1]
    phase_speed = np.median([r["matrix_median_seconds"] / r["phase_median_seconds"] for r in phases])
    common.save_caption(
        "R46_fock_space_optimization",
        f"""
**R46 - implemented Fock-space optimisations.** Timings: {common.timing_label()}.
(a) Basis size of the total-photon truncation n_1 + ... + n_m < c used by every Fock backend,
C(m + c - 1, m), vs a per-mode product basis c^m (a different truncation, shown for size only).
(b) Runtime of one beam splitter (theta = 0.7) on B = 1 and 32 random states for implementations of
the SAME truncated operator P U P: the native fibre kernel (no global operator), the global sparse
gate G applied with scipy.sparse and with cuSPARSE{' (CUDA, incl. host-device transfers)' if use_gpu else ' (not run: no usable GPU)'},
and the dense D x D matrix (D <= {DENSE_LIMIT}); all outputs agree to {worst_diff:.1e} (acceptance
{TOL_VARIANT:g}). At the largest grid point (m = {largest[0]}, c = {largest[1]}, D = {nat['dimension']}) the native kernel takes
{nat['median_seconds']:.2e} s vs {spr['median_seconds']:.2e} s for scipy.sparse, B = 1. (c) Bytes of each gate representation
(local matrix vs CSR vs dense). (d) Photon-number conservation: the passive beam splitter's local
matrix is block diagonal in total-photon sectors (off-sector entries exactly
{top_sector['bs_off_sector_max_abs']:.0f}, {100 * top_sector['bs_zero_fraction']:.1f}% zeros at cutoff {top_sector['cutoff']}), so `gaussian_unitary` builds only
the sector blocks; compared with the active CZ of the same local size (an observation: the skip cannot
be disabled). Diagonal gates by phase multiplication are a median {phase_speed:.1f}x faster than applying the
diagonal matrix, with identical output. Cached basis/cubic/sparse-structure speedups are in
R46_fock_caches.csv. complex128, seed {SEED}. Source: R46 (15_native_fock/46_fock_space_optimization.py).
""",
    )
    common.print_summary(f"R46 Fock-space optimisation ({common.timing_label()})", rows=len(gates), gpu=use_gpu,
                         worst_variant_diff=f"{worst_diff:.2e}",
                         native_vs_scipy_largest=f"{nat['median_seconds']:.2e}s vs {spr['median_seconds']:.2e}s",
                         phase_speedup=f"{phase_speed:.1f}", caches=[(c['cache'][:30], round(c['speedup'], 1)) for c in caches],
                         failures=len(failures))
    if failures:
        raise AssertionError(f"{len(failures)} R46 checks failed; see R46_fock_space_optimization_FAILURES.csv")


if __name__ == "__main__":
    main()
