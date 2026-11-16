# Examples

Nine runnable tutorials live in `examples/cv_diagrams/`. Run one with
`.venv/Scripts/python examples/cv_diagrams/01_build_gaussian_diagram.py`. Each prints its
results and ends with `assert` checks of its key claim.

The outputs below were recorded on 2026-11-16 (Windows, project `.venv`). Wall times,
including imports, ranged from 1.8 s (tutorial 7) to 6.6 s (tutorial 6). The `runtime` line is
the time after imports. Small numerical differences across platforms are expected; the
trajectory ensembles in tutorial 6 are seeded.

## 1. Build a Gaussian diagram

A corrected teleportation step built by hand. The tutorial shows:

- the averaged channel against `analysis.gaussian_channel` of the executable pattern;
- the conditional instrument against a postselected Gaussian-backend trajectory;
- a unitary diagram's symplectic matrix;
- typing errors, serialisation, the fingerprint and a drawing.

??? example "`01_build_gaussian_diagram.py`"
    ```python
    --8<-- "examples/cv_diagrams/01_build_gaussian_diagram.py"
    ```

Output:

```text
CVDiagram(inputs=(0,), boxes=4, outputs=('a',)) records: ('m',) gaussian: True
X = [[-0.0, -1.0], [1.0, 0.0]]
Y = [[0.0, 0.0], [0.0, 0.367879]]  e^{-2r} = 0.367879
max |diagram - gaussian_channel(Pattern)| = 0.0e+00
conditional mean: [-0.023863, 0.449747]  record density: 0.18752
max |diagram - postselected trajectory| = 8.3e-17
R(0.3) then S(0.2): S = [[0.782163, -0.241951], [0.360949, 1.166851]]
DiagramTypeError: Box 1 (Rotation) uses bosonic wire 'm' which is not alive (it is a classical record)
DiagramTypeError: Box 1 (Rotation) uses bosonic wire 0 which is not alive
DiagramTypeError: Box 0 (Feedforward) reads record 'k' before it is produced
fingerprint: 21f7f25f6977e224 round trip equal: True
drawing saved to the temp directory as cv_diagram_teleport_step.png
runtime: 0.4 s
```

## 2. Sound rewrite

The tutorial applies these rules and prints the rewrite trace:

- fusion, which requires wire adjacency;
- displacement fusion, which records the Weyl phase;
- ancilla elimination (CZ + q-homodyne → feed-forward, then preparation + homodyne →
  classical source);
- feed-forward merge;
- channel fusion.

??? example "`02_sound_rewrite.py`"
    ```python
    --8<-- "examples/cv_diagrams/02_sound_rewrite.py"
    ```

Output:

```text
rules: rotation-fusion [exact-instrument], squeezing-fusion [exact-instrument], shear-fusion [exact-instrument], displacement-fusion [exact-instrument], identity-removal [exact-instrument], symplectic-fusion [exact-instrument], channel-fusion [exact-channel], feedforward-merge [exact-instrument], cz-q-measurement-to-feedforward [exact-instrument], prepare-measure-to-source [exact-instrument], commute [exact-instrument], phase-polynomial-fusion [exact-instrument], displacement-through-phase-polynomial [exact-instrument], teleport-chain-resynthesis [logical]
blocked: Rule rotation-fusion does not match
rotation-fusion: boxes (0, 2) -> Rotation, scalar None
displacement-fusion: boxes (2, 3) -> Displacement, scalar global phase exp(i*0.022500000000000006)
boxes 5 -> 3; physical True, conditional True, X residual 1.1e-16
after elimination: ['ClassicalSource', 'Feedforward', 'Feedforward']
  cz-q-measurement-to-feedforward: conditional True (instrument residual 0.0e+00)
  prepare-measure-to-source: conditional True (instrument residual 0.0e+00)
merged feed-forward: (('m', (0.3, 0.8)),) | branch-dependent global phase (density operators unchanged)
channel fusion [exact-channel]: physical True, Y residual 0.0e+00
trace rules: ['cz-q-measurement-to-feedforward', 'prepare-measure-to-source', 'feedforward-merge']
runtime: 0.0 s
```

## 3. Verify a rewrite

The tutorial runs four checks:

- every exact Gaussian rule at every match on a random diagram, using `compare` and
  executed patterns;
- the ancilla-eliminated diagram against a trajectory of the original pattern;
- the Weyl phase against dense matrices;
- a false rule, which is caught, and a logical rewrite, which is reported as not physical.

??? example "`03_verify_rewrite.py`"
    ```python
    --8<-- "examples/cv_diagrams/03_verify_rewrite.py"
    ```

Output:

```text
17 exact rewrites verified (12 also executed); worst residual 4.4e-16
ancilla eliminated; |rewritten conditional - original trajectory| = 4.4e-16
recorded Weyl phase 0.042500; dense-matrix residual (cutoff 100) 5.0e-16
false swap: logical False, X residual 0.529
resynthesis: 8 -> 3 steps; logical True, physical False (Y residual 0.458); records differ: 8 -> 3; conditional equivalence not defined
runtime: 0.2 s
```

## 4. Pattern to diagram

The tutorial covers:

- a compiled two-mode circuit as a diagram, with its costs and its teleportation steps;
- channel agreement, including the round trip, and the ideal circuit;
- rejection of an adaptive pattern;
- the `GaussianGraph` cluster and the q-measurement rule as diagram conditioning.

??? example "`04_pattern_to_diagram.py`"
    ```python
    --8<-- "examples/cv_diagrams/04_pattern_to_diagram.py"
    ```

Output:

```text
CVDiagram(inputs=(0, 1), boxes=86, outputs=(22, 18))
{'resource_modes': 21, 'measurement_count': 21, 'measurement_depth': 13, 'feedforward_commands': 21, 'entangling_edges': 22}
teleportation steps found: 21
max |diagram - gaussian_channel| incl. round trip: 1.8e-15
ideal circuit vs compiled X: 6.1e-16 (flow-corrected: X is ideal)
adaptive pattern rejected: Outcome-dependent parameter (adaptive / nonlinear feed-forward)
Z of built cluster: [[0.2466j, (1+0j), 0j], [(1+0j), 0.2466j, (0.5+0j)], [0j, (0.5+0j), 0.2466j]]
|Z(diagram) - GaussianGraph.cluster| = 0.0e+00; round trip 0.0e+00
q-measurement: |diagram conditioning - GaussianGraph.measure_q| = 0.0e+00
averaged state: Fix every record with outcomes=...; averaged states are mixed
runtime: 0.0 s
```

## 5. Optimise a CV-MBQC pattern

The tutorial runs the search in each mode:

- physical, logical (greedy and beam) and approximate (ε = 1e-6);
- a depth objective;
- CV-flow certification before and after;
- weighted costs and a Pareto front over (modes, noise).

??? example "`05_optimize_mbqc.py`"
    ```python
    --8<-- "examples/cv_diagrams/05_optimize_mbqc.py"
    ```

Output:

```text
original: {'resource_modes': 29, 'measurement_depth': 16, 'entangling_edges': 30, 'added_noise': 1.0634}
                physical: improved False explored    0 rejected   0 | resource_modes 29, measurement_depth 16, entangling_edges 30, added_noise 1.063 | logical True physical True Y residual 0
          logical greedy: improved True  explored    3 rejected   0 | resource_modes 22, measurement_depth 11, entangling_edges 23, added_noise 0.9854 | logical True physical False Y residual 0.335
            logical beam: improved True  explored    4 rejected   0 | resource_modes 22, measurement_depth 11, entangling_edges 23, added_noise 0.9854 | logical True physical False Y residual 0.335
         approx eps=1e-6: improved False explored    2 rejected   2 | resource_modes 29, measurement_depth 16, entangling_edges 30, added_noise 1.063 | logical True physical True Y residual 0
logical, depth objective: improved True  explored    3 rejected   0 | resource_modes 22, measurement_depth 11, entangling_edges 23, added_noise 0.9854 | logical True physical False Y residual 0.335
rules applied (logical greedy): ['teleport-chain-resynthesis', 'teleport-chain-resynthesis']
CV flow: [('certified', 16, 31), ('certified', 11, 24)]
weighted costs (default weights): [58.905, 48.715, 44.238]
(original, one chain, all chains) Pareto front over (modes, noise): [2]
runtime: 2.2 s
```

Here, resynthesising both chains lowers the modes **and** the noise, so the front has one
point. That is not true in general (see R65 in
[Resource optimisation](resource-optimization.md#recorded-results)).

## 6. Recompile and execute

The tutorial takes a logically optimised diagram and:

- recompiles it with `to_pattern` and executes it;
- checks its own channel, and that X and d equal the original's;
- checks a postselected trajectory against the conditional instrument;
- runs a 400-shot ensemble of each pattern against its own channel;
- shows a `ClassicalSource` recompiled as one ancilla.

??? example "`06_recompile_and_execute.py`"
    ```python
    --8<-- "examples/cv_diagrams/06_recompile_and_execute.py"
    ```

Output:

```text
resource modes 13 -> 3; commands 53 -> 13
|gaussian_channel(recompiled) - own diagram channel| = 1.8e-15
|X - X_original| = 1.1e-16, |d - d_original| = 0.0e+00, noise tr Y/2: 1.0007 -> 0.2079
conditional: |trajectory - diagram| = 1.1e-15 (density 0.002376)
400 shots each: max|z| original 1.30, recompiled 0.88; ensemble means [0.268, -0.532] vs [0.289, -0.393] (both predict [0.281, -0.442])
ClassicalSource: diagram modes 0, recompiled preparations 1, channel residual 1.1e-16
runtime: 4.6 s
```

## 7. Finite-squeezing channels

The tutorial shows:

- Y(r) e^{2r} and X are r-independent for the compiled and the optimised pattern, so the
  noise ratio is constant;
- channel fusion of loss and squeezing, with complete positivity;
- rejection of a non-CP channel box.

??? example "`07_finite_squeezing_channels.py`"
    ```python
    --8<-- "examples/cv_diagrams/07_finite_squeezing_channels.py"
    ```

Output:

```text
r=0.5: modes 18->3, added noise 4.06878->0.78408 (ratio 0.192706), squeezing energy 4.888->0.815
r=1.0: modes 18->3, added noise 1.49682->0.28845 (ratio 0.192706), squeezing energy 24.860->4.143
r=1.5: modes 18->3, added noise 0.55065->0.10611 (ratio 0.192706), squeezing energy 81.609->13.601
r=2.0: modes 18->3, added noise 0.20257->0.03904 (ratio 0.192706), squeezing energy 236.774->39.462
max spread of Y e^(2r) over r: 2.2e-12; of X over r: 0.0e+00
fused channel Y = [[0.10784, 0.0], [0.0, 0.53413]]; |fused - compose_channels| = 1.1e-16; CP: True; physical equivalent: True
added noise of the lossy diagram: 0.32098
rejected: Not a completely positive Gaussian channel
runtime: 0.1 s
```

## 8. Non-Gaussian diagram

The tutorial shows:

- phase-polynomial fusion and displacement propagation converging to a dense reference as
  the cutoff grows, while a false rule does not;
- the heralding probability of photon subtraction against μ e^{−μ};
- post-selection without its outcome being refused;
- Gaussian semantics being refused for Kerr.

??? example "`08_non_gaussian_diagram.py`"
    ```python
    --8<-- "examples/cv_diagrams/08_non_gaussian_diagram.py"
    ```

Output:

```text
fused coefficients: (0.0, 0.1, 0.030000000000000002, 0.04)
after propagation: [('PhasePolynomial', (0.04736000000000001, 0.14320000000000002, 0.078, 0.04)), ('Displacement', None)]
infidelity vs dense reference at cutoffs (20, 30, 45, 60)
      original: 3.78e-05  1.39e-06  2.18e-08  6.10e-10
         fused: 3.72e-05  1.43e-06  2.25e-08  6.20e-10
   fused+moved: 3.21e-05  1.22e-06  1.88e-08  5.12e-10
    false rule: 9.08e-02  9.08e-02  9.08e-02  9.08e-02
  retained norm: 0.9999987  1.0000000  1.0000000  1.0000000
  false-rule diagram vs its own true order at cutoff 60: 8.59e-13
herald probability: [(8, 0.052854194533), (16, 0.052854319838), (24, 0.052854319838)] closed form 0.052854319838
missing post-selection outcome: Postselect('n' = 1) needs outcomes['n'] = 1
Gaussian semantics refused: Non-Gaussian boxes ['Kerr'] have no Gaussian semantics; use calculus.cv_diagrams.fock
runtime: 0.4 s
```

## 9. GKP syndrome diagram

The tutorial shows:

- the finite-energy GKP Z readout as a diagram, integrated on decoding-cell-aligned
  midpoints and compared with `pg.multimode_readout`;
- the SUM gate built from rotations and CZ;
- Z-syndrome extraction onto a \|0~⟩ ancilla, with its finite-energy gap to the direct
  readout.

??? example "`09_gkp_syndrome_diagram.py`"
    ```python
    --8<-- "examples/cv_diagrams/09_gkp_syndrome_diagram.py"
    ```

Output:

```text
direct readout: P(bit=0) = 0.78692 (integrated density 1.000000); multimode_readout 0.78684; ideal cos^2(0.4) = 0.84835
SUM symplectic (d, a interleaved): [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
syndrome readout: P(bit=0) = 0.72341 (integrated density 1.000000); gap to direct readout +0.0635
diagram is Gaussian: False | live outputs: ('d',)
runtime: 2.6 s
```
