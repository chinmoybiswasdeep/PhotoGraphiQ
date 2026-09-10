# PhotoGraphiQ v0.2 physics-validation and release-hardening report

Audited baseline: `8705e6934cc5f83b257d20c3c4bf8aa6da411a1e`.
The [pre-change audit](hardening-audit.md) records inspected files, derivations
and original CI evidence. This pass preserves the architecture and physical
gate conventions. It adds tests, diagnostics and targeted guards.

## A. Bugs actually found

1. **Small mixedness was silently discarded on Fock input conversion.**
   Default `np.isclose(det(V),1)` admitted V=1.0000001 I as pure. The conversion
   then prepared a pure squeezed state instead of rejecting the thermal input.
   Physical covariance validation plus absolute purity tolerance 1e-10 and
   zero relative tolerance now rejects it.
2. **Resource validation was too late.** Insufficient explicit cutoff support,
   mismatched resource mode counts and invalid Gaussian covariance could fail
   after earlier preparations allocated a native Fock state. Allocation-spy
   regressions failed before the fix and pass with whole-pattern preparation
   preflight. Dynamic outcome-dependent support growth remains checked at execution.
3. **Partial homodyne CDF errors were ignored.** The total-interval integral
   was checked, but each inverse-CDF partial integral discarded its error
   estimate. It now raises on unresolved or nonfinite partial integration.
4. **Boolean dimensions were accepted as integers.** `dimension(True)` and
   boolean allocation limits now fail explicitly.
5. **Avoidable quadratic allocation in pure analysis.** Pure norms and pure/pure
   fidelity/trace distance formed full density matrices. They now use native
   norms and aligned vectors, with stable phase alignment near identical states.

No sign/factor defect was found in cubic phase, Kerr, ladder factors, the
homodyne Hermite recurrence, CZ Bogoliubov blocks or cubic feedforward.
The Wigner and mixed-state fidelity audits found missing evidence rather than
a demonstrated wrong formula. Mixed native reduction still uses dense matrices.

## B. Numerical-validation issues found

The [original GitHub run](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/runs/34454699908)
failed three tests on Linux/Python 3.13.15. Python 3.11 and 3.12 were cancelled,
and lint/type/build steps were skipped. The original local 87-pass report did
not establish remote or multi-version success.

At fixed outcome m=.4, gamma=.3 and r=.2, Linux's c=36 density error was
2.023493614e-5 against 2e-5. This path performs **no numerical quadrature**.
Its expected density is analytical. Finite Fock preparation, native Gaussian
projection and renormalization dominate the discrepancy. Independent reference
integration is tested separately by tolerance and interval refinement.

Direct SUM versus Fourier-CZ infidelity at c=24/48 was approximately
5.95e-6/1.94e-7 on Linux, compared with 3.96e-7/1.80e-8 on Windows.
The same infinite-space map can have decomposition-dependent finite-space
projection error, including platform-dependent numerical eigenspace choices.
Neither a phase correction nor an arbitrary relaxed fixed threshold is the fix.
Low-cutoff density error is not universally monotonic (c=12→18 is a counterexample).

## C. Tests changed and why

The original cubic/cat wavefunction cases and the raw c=24/48 cases remain.
Identical native sequences still compare vectors directly. Different Gaussian
realizations now have an explicit c=12,18,24,32,48 convergence regression tracking
infidelity, both retained norms and both boundary populations. A tenfold
24→48 improvement is required; original measurements on both platforms exceed
twentyfold. This empirical criterion applies to the tested regime, not all inputs.

The density regression retains c=36 and requires improvement at c=96 with a
**stricter 1e-6 density target**. Four (gamma,r,m) cases compare c=36,64,96 to an
independent wavefunction oracle with 1e-7 final infidelity target. Refining the
oracle's tolerance and integration window must change its vector by less than
1e-9, independently of the Hilbert truncation budget.

Additional tests cover cubic gamma=.05/.2/.5/.8 and Kerr kappa=.1/.3/.7/1.1
on vacuum, coherent, complex-superposition and cat inputs; optional q²/q³/q⁴,
p²/p³/p⁴/n²; analytical homodyne densities and CDFs; complex cats; Wigner grids;
mixed/mixed fidelity and reordered reductions; physical tap Kraus probabilities;
weak-tap approach to ideal subtraction; guard failures and memory dimensions.
See [validation.md](validation.md) for the feature matrix and error classes.

## D. Physics conventions re-verified

[q,p]=2i, V_vac=I, q=a+a†, p=-i(a-a†), interleaved mode order and Piquasso
hbar=2 remain unchanged. CP(gamma)=exp(i gamma q³/6), Kerr=exp(i kappa n²),
Q(s)=exp(i s q²/4), and displacement Z(t)=exp(i tq/2) are consistent across code,
independent matrices, native instructions and manuscript derivations.

Inverse SUM followed by resource q=m gives psi(q) phi(q+m). Q(-2gamma m)
and Z(-gamma m²) cancel its outcome-dependent polynomial phases, leaving the
finite Gaussian envelope and exp(i gamma q³/6), up to exp(i gamma m³/6).
That envelope is retained in fidelity comparisons. Finite injection is not an
exact unitary or an experimentally free preparation of a nonlinear ancilla.

High-order moments include intermediate ladder excursions outside stored
support. Cubic vacuum tests use <p²>=1+3gamma², <p³>=gamma+15gamma³ and
<p⁴>=3+10gamma²+105gamma⁴, including noncommutativity. A c=160 check budgets
0.02% relative high-moment error for gamma up to .8, separately from fidelity.

## E. Graphix validation expanded

Line, ring, star, 2×3 cluster and nonconsecutive labelled graphs now compare
exact canonical edge sets, I/O labels and measured order. Arbitrary heterogeneous
CV labels use an explicit bijection to Graphix integers; a counterexample shows
that plain isomorphism can hide a wrong semantic mapping. Multiple s/t domains,
correction targets, transitive classical signal support, valid/invalid causal
orders, safe standardization and line-flow correction support are checked.

No Graphix amplitudes, quadrature distributions, Fock probabilities or nonlinear
gate matrices are used. A dependency DAG, a valid causal schedule, a numerical
supplied-total-order real CV-flow certificate, literature CV-flow and Graphix
qubit flow/gflow/Pauli flow are distinct notions. The implementation certifies
the specified real linear equation and order; it does not find general orders
or prove arbitrary non-Gaussian patterns deterministic.

## F. Python matrix and CI status

The final matrix status is recorded below after execution. The workflow runs
pytest, Ruff lint, Ruff formatting, mypy, package builds and numerical evidence
on Python 3.11–3.14. Fail-fast is disabled and logs/versions are uploaded even on
failure. `validation/hardening/original-ci.json` retains the original failed evidence.

Local Windows validation: **216 tests passed on each of Python 3.11.16,
3.12.14, 3.13.15 and 3.14.4**. Ruff lint, Ruff formatting, mypy, package
builds and physics-evidence generation also passed on every version. Exact
packages, logs and numerical results are in `validation/hardening/local-python-*`.

The [artifact-preserving Linux matrix](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/runs/34459232823)
passed at commit `201c9b0`. All four artifact archives were downloaded and their
JUnit counts, exact environments, numerical evidence and six exit codes inspected.

| Python | Windows pytest | Linux pytest | Lint / format / mypy / build / evidence |
|---|---:|---:|---|
| 3.11.16 | 216 passed | 216 passed | All passed on both |
| 3.12.14 | 216 passed | 216 passed | All passed on both |
| 3.13.15 | 216 passed | 216 passed | All passed on both |
| 3.14 (Windows .4; Linux .7) | 216 passed | 216 passed | All passed on both |

There were no failures or skips. The expected truncation/boundary warnings remain.
Piquasso 8.0.1 and Graphix 0.4 were used throughout. Python 3.11 used NumPy 2.4.6
and SciPy 1.17.1; 3.12-3.14 used NumPy 2.5.3 and SciPy 1.18.1. Full dependency
lists and logs are in `validation/hardening/{local,linux}-python-*`. Linux source
hash: `3daf6f2ee13b755c95f9846ed058fd5fd6dd70caa24a20e7fd12a737543be401`.
The local hash differs because the workflow subsequently gained explicit hidden
artifact inclusion and missing-artifact failure; production and test code match.

The first hardened run also passed but artifact upload silently excluded the
hidden output directory. This was fixed and verified by the linked second run.
An isolated installed-wheel smoke check also verified the number-state fourth
moment, and the revised 16-page manuscript compiled with resolved references.

Linux reproduced the original c=36 density discrepancy on 3.12/3.13 exactly,
then reduced it to 5.447e-8 at c=96. Raw decomposition infidelity fell from
5.954e-6 at c=24 to 1.940e-7 at c=48. Thus the original difficult regime was
retained, with independently measured convergence rather than a looser boundary.

`experiments/validate_release.py` writes exact versions, normalized source/test
hash, head commit, separate logs and exit codes. Local Windows results are not
substituted for Ubuntu GitHub Actions results. No remote release is created.

## G. Remaining unsupported features

- Mixed-state Fock input/evolution, Fock loss/thermal channels and noisy Fock detection.
- Fock heterodyne/general-dyne, generic non-Gaussian POVMs and automatic representation switching.
- GKP preparation/decoding, fault tolerance, universal non-Gaussian compilation and hardware execution.
- Differentiable stochastic execution, gradient estimators and accelerator batching for PhotoGraphiQML.
- Physical heralded photon addition; ideal a† remains an explicitly mathematical normalized operation.

Mixed **output reductions and their analysis** are supported; this is different
from mixed-state trajectory execution. Existing Gaussian noisy simulation remains.

## H. Remaining numerical limitations

Finite-Fock error depends on input energy, gate strength and native decomposition.
Norm and boundary population are diagnostics, not rigorous observable-error bounds.
High moments converge more slowly than low moments or fidelity. State convergence
does not remove the physical finite-energy envelope in injection.

The homodyne interval is validated for the represented finite support using its
Hermite tail envelope and actual mass checks; this does not certify a badly
truncated infinite-state preparation or arbitrarily huge cutoffs. Integration
errors raise explicitly. Exact real postselection reports a density, not an
event probability. Conditioning near extremely unlikely outcomes remains sensitive.

Wigner negativity requires both window and grid convergence. Mixed metrics and
native partial trace remain dense; the allocation guard bounds vector dimension,
not peak memory of every requested observable. Metadata compatibility alone is
not evidence for an untested Python/platform/dependency combination.

## I. v0.2 release decision

**Ready for an experimental v0.2 research release** within the documented
parameter regimes and capabilities: both Windows and actual Linux CI passed
all required gates, including package builds. This is not a claim of universal
CV-MBQC, unrestricted numerical accuracy or broad production stability. No
release tag, package publication or default-branch merge was performed.

## J. PhotoGraphiQML readiness

The resource/IR/backend separation, explicit classical dependencies, reproducible
seeds, state diagnostics and capability failures provide a defensible basis for
starting feature extraction and model prototypes now that the matrix passed.
They do not supply autodifferentiation or a validated gradient estimator. A future
ML layer must state its cutoff, postselection, sampling and differentiation
contracts; it must not treat normalized heralded trajectories as unbiased
unconditional samples or pure-state feature analysis as mixed-channel support.

## Exact file changes and reasons

The accompanying [file-changes.json](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/blob/validation/v0.2-physics-hardening/validation/hardening/file-changes.json) lists every changed
file against the audited baseline with its concrete role. Generated numerical
evidence and the revised manuscript are distinguished from production fixes.
