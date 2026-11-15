# Manuscript experiments: theory sections

One section per experiment, read by `scripts/build_colab_notebook.py` and placed above the experiment's code cell. Each section starts with a `<!-- suite:ID -->` marker and a `**Title:**` line.

<!-- core:R1 -->
**Title:** One-step CV wire and teleportation

**Why this experiment.** Every PhotoGraphiQ computation is built from one primitive: a single homodyne measurement on a cluster edge followed by displacement feed-forward, which teleports the input onto a fresh resource mode while applying a shear-Fourier map. If this primitive had a sign error, a wrong feed-forward coefficient, or a mis-modelled finite-squeezing noise term, every compiled gate, wire and multimode circuit downstream would inherit it. R1 checks the primitive at three levels at once: the analytic unconditional channel against an independently written closed form, the sampled trajectory ensemble against that channel, and three trajectory engines (NumPy, C++ and Piquasso-backed) against each other.

### Theory
Conventions: $\hbar = 2$, quadrature ordering $(q, p)$ per mode, vacuum covariance $V = I$. A coherent input $|\alpha\rangle$ has mean $\mu_{\rm in} = (2\,\mathrm{Re}\,\alpha,\ 2\,\mathrm{Im}\,\alpha)^T$ and covariance $V_{\rm in} = I$.

One teleportation step with shear $k$ (as built by the package compiler): prepare a resource mode momentum-squeezed with covariance $\mathrm{diag}(e^{2r}, e^{-2r})$, apply $CZ$ ($p_1 \to p_1 + q_2$, $p_2 \to p_2 + q_1$), measure the input in the rotated quadrature $x_\theta = q\cos\theta + p\sin\theta$ with $\cot\theta = k$, and displace the resource by $q_2 \to q_2 - m/\sin\theta$, where $m$ is the outcome. Substituting $m = q_1\cos\theta + (p_1 + q_2)\sin\theta$ cancels the anti-squeezed $q_2$ exactly and gives

$$
\begin{pmatrix} q_{\rm out}\\ p_{\rm out}\end{pmatrix}
= S(k)\begin{pmatrix} q_{\rm in}\\ p_{\rm in}\end{pmatrix} + \begin{pmatrix}0\\ p_2\end{pmatrix},
\qquad
S(k) = \begin{pmatrix} -k & -1\\ 1 & 0\end{pmatrix}.
$$

R1 uses $k = 0$ (a $p$-homodyne measurement), so $S(0) = F$ is the Fourier map $(q, p) \to (-p, q)$. The residual resource momentum enters as additive noise, so the unconditional (record-averaged) channel is the Gaussian affine map

$$
\mu_{\rm out} = S\mu_{\rm in} + d,\qquad V_{\rm out} = S V_{\rm in} S^T + N,\qquad N = \begin{pmatrix}0 & 0\\ 0 & e^{-2r}\end{pmatrix},\quad d = 0 .
$$

The noise is the physical price of finite squeezing and vanishes only as $r \to \infty$.

A sampled ensemble of $M$ trajectories has moment-matched mean and covariance (law of total covariance)

$$
\bar\mu = \frac1M\sum_j \mu_j,\qquad
V_{\rm ens} = \frac1M\sum_j V_j + \frac1M\sum_j (\mu_j - \bar\mu)(\mu_j - \bar\mu)^T .
$$

For a Gaussian pattern the conditional covariance $V_c$ is record independent, so only the spread $W = V_{\rm out} - V_c$ of the conditional means is estimated from samples. Its standard errors are

$$
\mathrm{SE}(\bar\mu_i) = \sqrt{W_{ii}/M},\qquad
\mathrm{SE}(V_{ij}) = \sqrt{(W_{ii}W_{jj} + W_{ij}^2)/M}.
$$

### What the code does
- System under test: `pg.protocols.wire([0.0], squeezing=1.0)` (one Fourier step, $r = 1$). The analytic channel $(S, N, d)$ comes from `pg.gaussian_channel`, which propagates Wigner source variables through the pattern; trajectories come from `pg.run_shots` on the `gaussian` (NumPy), `native-gaussian` (C++) and `piquasso` engines.
- Oracles: an independently written Fourier matrix $F$ and noise $\mathrm{diag}(0, e^{-2r})$ for the channel; the channel itself (a code path separate from the trajectory simulator) for the ensembles.
- Inputs: six coherent states $\alpha \in \{0,\ 0.3+0.2i,\ -0.5+0.1i,\ 1,\ 0.2-0.4i,\ -0.8-0.6i\}$, 4000 shots per input and per engine (base seed 20260913). One `simulate` call per input supplies $V_c$ for the standard errors.
- Noise sweep (channel only): $r \in \{0.25, 0.5, 1.0, 1.5, 2.0, 3.0\}$; map error vs $F$ and noise error vs $\mathrm{diag}(0, e^{-2r})$.
- Native contract: the C++ engine consumes the same random numbers as NumPy for equal seeds, so measurement records and output means are compared trajectory by trajectory.
- Metrics: Frobenius mean and covariance errors, max $z$-score over every first and second moment. Outputs: `R1_wire_teleportation` (CSV/JSON/metadata, figure, caption) and `R1_wire_teleportation_noise_sweep.csv`.

### Expected outcome
The channel identities are exact up to rounding, so map and noise errors should be about $10^{-16}$ at every $r$. Ensemble moments fluctuate statistically, with $|z|$ typically of order 1. NumPy and C++ trajectories should agree to rounding. Because all three engines share one random stream, their markers coincide: they form one statistical sample, not three. Acceptance (any violation raises and writes a FAILURES CSV):
- channel map and noise errors $\le 10^{-12}$ (at $r = 1$ and across the whole sweep);
- every ensemble moment within 5 standard errors of the analytic channel, for every engine;
- legacy check: Frobenius mean and covariance errors $\le 0.35$;
- max native-vs-NumPy record and output-mean difference $\le 10^{-10}$.

Figure `R1_wire_teleportation`: (left) $z$-scores of the ensemble $q$ and $p$ means per coherent input and engine, with a grey $\pm 5$ SE acceptance band; (centre) the same for the variances; (right) the added $p$ variance $N_{pp}$ from `gaussian_channel` against the dashed closed form $e^{-2r}$ on a log axis ($N_{qq} = 0$).

<!-- core:R2 -->
**Title:** Identity wire versus finite squeezing

**Why this experiment.** Transport over long cluster wires is the backbone of every compiled circuit, and the package separates the logical map, which must be exact, from the finite-squeezing noise, which is physical and must match theory. R2 checks both for the four-step identity wire: the logical map must be exactly the identity at every squeezing, and the added noise must follow the closed form $N = 2e^{-2r}I$. If noise were reported as a compilation error, or the recursion over steps were wrong, every resource estimate built on wire transport would be off by a constant or an exponential factor.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$, vacuum covariance $I$. One $k = 0$ teleportation step (see R1) acts as

$$
T = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix},\qquad T^4 = I ,
$$

so four steps implement the identity, $S = I$, $d = 0$. Each step adds the resource momentum noise $\mathrm{diag}(0, e^{-2r})$, so the noise obeys the recursion

$$
N_{j+1} = T N_j T^T + \begin{pmatrix}0 & 0\\ 0 & e^{-2r}\end{pmatrix},\qquad N_0 = 0 .
$$

Because $T\,\mathrm{diag}(a,b)\,T^T = \mathrm{diag}(b,a)$, the sequence is $\mathrm{diag}(0,\epsilon)$, $\mathrm{diag}(\epsilon,\epsilon)$, $\mathrm{diag}(\epsilon,2\epsilon)$, $\mathrm{diag}(2\epsilon,2\epsilon)$ with $\epsilon = e^{-2r}$, so

$$
N = 2e^{-2r} I,\qquad \mathrm{Tr}\,N = 4e^{-2r}.
$$

For the coherent input $\alpha = 0.4 + 0.3i$, $\mu_{\rm in} = (0.8, 0.6)^T$ and $V_{\rm in} = I$, so the unconditional output is $\mu_{\rm out} = \mu_{\rm in}$ and $V_{\rm out} = (1 + 2e^{-2r}) I$. On a log axis $\log_{10}\mathrm{Tr}\,N = \log_{10}4 - (2/\ln 10)\,r$, a straight line of slope $\approx -0.869$ per unit $r$. Increasing $r$ never changes the logical map; it only suppresses the noise exponentially.

The ensemble checks use the moment-matched estimator and standard errors of R1: with spread $W = V_{\rm out} - V_c$ (the conditional covariance $V_c$ is record independent), $\mathrm{SE}(\bar\mu_i) = \sqrt{W_{ii}/M}$ and $\mathrm{SE}(V_{ij}) = \sqrt{(W_{ii}W_{jj} + W_{ij}^2)/M}$.

### What the code does
- System under test: `pg.protocols.identity(squeezing=r)` (four Fourier steps), analysed by `pg.gaussian_channel` and executed by `pg.run_shots` on `gaussian` (NumPy) and `native-gaussian` (C++).
- Independent oracles: `photographiq.gaussian.wire_channel([0,0,0,0], r)`, a separate closed-form recursion that never sees the pattern, plus the hand-coded $I$ and $\mathrm{Tr}\,N = 4e^{-2r}$.
- Sweep: $r \in \{0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0\}$; coherent input $\alpha = 0.4+0.3i$; 1500 shots per engine per $r$ (seed 20261113).
- Metrics: $\|S - I\|_F$, $\|S - S_{\rm closed}\|_F$, $\|N - N_{\rm closed}\|_F$, $|\mathrm{Tr}\,N - 4e^{-2r}|$, output-mean error, single-trajectory $q$/$p$ variances, ensemble $z$-scores and the max native-vs-NumPy record difference.
- Output: `R2_identity_finite_squeezing` (CSV/JSON/metadata, figure, caption).

### Expected outcome
All channel identities hold to rounding (about $10^{-16}$). $\mathrm{Tr}\,N$ falls from about 2.68 at $r = 0.2$ to about 0.073 at $r = 2$. The sampled ensembles agree with the channel within a few standard errors, and the two engines agree trajectory by trajectory. Acceptance (any failure raises and writes a FAILURES CSV): $\|S-I\|_F$, both oracle errors and the trace error $\le 10^{-9}$; every ensemble moment within 5 SE for both engines; native-vs-NumPy record difference $\le 10^{-10}$.

Figure `R2_identity_finite_squeezing` (single panel, log $y$): $\mathrm{Tr}\,N$ from `gaussian_channel` (circles) on the dashed closed form $4e^{-2r}$, plus the sampled excess $\mathrm{Tr}\,V_{\rm ens} - 2$ for both engines (coincident markers, clipped at $10^{-4}$ where shot noise dominates at large $r$).

<!-- core:R3 -->
**Title:** Compiled phase-space rotation

**Why this experiment.** The compiler lowers every single-mode Gaussian gate to a short chain of one-step teleportations (R1). R3 checks that this lowering reproduces the ideal rotation for every angle and every resource squeezing, and shows where the finite-squeezing noise floor of a compiled gate comes from. A wrong decomposition, a sign error in the rotation convention, or noise leaking into the deterministic map would corrupt every circuit that uses `Circuit.rotate`, and the gauge-dependent noise amplification documented here would go unnoticed.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$, rotation $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$. `Circuit.rotate` passes $R(\theta)$ to `decompose_symplectic`, which writes a real symplectic $S = \begin{pmatrix}a & b\\ c & d\end{pmatrix}$ ($\det S = 1$) as a product of teleportation steps $T(k) = \begin{pmatrix}-k & -1\\ 1 & 0\end{pmatrix}$:

$$
S = T(k_4)\,T(k_3)\,T(k_2)\,T(k_1),\qquad k_1 = 0,\quad k_2 = \frac{1-d}{c},\quad k_3 = c,\quad k_4 = \frac{1-a}{c}.
$$

If $|c| < 10^{-8}$ a leading $k = 0$ step is prepended and $S$ is pre-multiplied by $T(0)^T$, giving five steps. For a rotation, $a = d = \cos\theta$ and $c = \sin\theta$, so the half-angle identity gives

$$
k_1 = 0,\qquad k_2 = k_4 = \tan(\theta/2),\qquad k_3 = \sin\theta .
$$

At $\theta = 0$ and $\theta = \pm\pi$ (where $\sin\theta = 0$) the five-step branch is used instead.

Each step consumes one resource of squeezing $r$, so the noise follows the recursion of R1 and R2,

$$
N_{j+1} = T(k_j)\,N_j\,T(k_j)^T + \mathrm{diag}(0, e^{-2r}),\qquad N_0 = 0,
$$

which is linear in $e^{-2r}$: $N(\theta, r) = e^{-2r}\,G(\theta)$ with a gauge factor $G$ fixed by the shears. The deterministic map is independent of $r$. The key physical point is that the noise floor of a compiled gate is $e^{-2r}$ amplified by the synthesis shears. Because $\tan(\theta/2)$ diverges as $\theta \to \pm\pi$, rotations close to a half-turn (but not exactly at it) need large shears and have a much larger noise floor at fixed $r$. This is a synthesis (gauge) effect, not a squeezing limitation and not a truncation artifact.

### What the code does
- System under test: `pg.Circuit(1).rotate(0, theta).compile(squeezing=r)`, analysed by `pg.gaussian_channel`.
- Oracle: an independently written NumPy rotation matrix, deliberately not `photographiq.gaussian.rotation`, which the compiler itself uses.
- Angle sweep: 25 angles `np.linspace(-pi, pi, 25)` (step $\pi/12$, including $0$ and $\pm\pi$) at $r = 1.2$. Squeezing sweep: $r \in \{0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5\}$ at $\theta = 0.7$.
- Metrics: map Frobenius error $\|S_{\rm pattern} - R(\theta)\|_F$, noise spectral norm $\|N\|_2$, $\mathrm{Tr}\,N$ and command count.
- Execution: for $\theta \in \{-2.0, 0.7, 2.5\}$ at $r = 1.2$, the compiled pattern is run with `pg.run_shots` on `gaussian` and `native-gaussian` (600 shots, seed 20261113, coherent input $\alpha = 0.3 - 0.2i$). Ensemble moments are compared with `channel.apply` using the shot-noise standard errors of R1, the two engines are compared record by record, and the planner's backend choice (`pg.select_backend`) is logged.
- Outputs: `R3_rotation_compile_theta_sweep`, `R3_rotation_compile_squeezing_sweep`, `R3_rotation_compile_execution` (CSV/JSON/metadata) and figure `R3_rotation_compile`.

### Expected outcome
The map error should be at machine precision (about $10^{-15}$) for every angle and every $r$. $\|N\|_2$ should be symmetric in $\theta$, smallest near small $|\theta|$ on the four-step branch, and rise sharply at the grid points next to $\pm\pi$, where $|\tan(\theta/2)| \approx 7.6$. At exactly $0$ and $\pm\pi$ the five-step branch gives a moderate, equal value. At fixed $\theta$, $\|N\|_2/e^{-2r}$ should be constant, so $\|N\|_2$ is a straight line on a log axis. Sampled ensembles should lie within a few standard errors of the channel.

Acceptance (any failure raises and writes `R3_rotation_compile_FAILURES.csv`): map error $\le 10^{-9}$ on both sweeps; for the executed patterns, every ensemble mean and covariance entry within 5 SE on both engines, and native-vs-NumPy record difference $\le 10^{-9}$.

Figure `R3_rotation_compile`: (left) $\|N\|_2$ vs $\theta$ at $r = 1.2$, log $y$, showing the half-turn peaks; (right) $\|N\|_2$ vs $r$ at $\theta = 0.7$, log $y$, showing pure $e^{-2r}$ decay.

<!-- core:R4 -->
**Title:** Compiled single-mode squeezing

**Why this experiment.** Squeezing is the second elementary single-mode Gaussian gate and the one whose numerics are most delicate: it stretches one quadrature and compresses the other by the same exponential factor. R4 checks that `Circuit.squeeze` reproduces the ideal squeezing map for positive and negative logical squeezing at fixed resource squeezing, stays finite, and runs correctly on the trajectory engines. It also exposes how the synthesis of a strong squeezer amplifies the finite-squeezing noise floor. Without it, sign conventions (which quadrature is squeezed) and the five-step decomposition branch would be untested.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$; positive $r_L$ squeezes $q$. The ideal logical squeezer is

$$
S(r_L) = \begin{pmatrix} e^{-r_L} & 0\\ 0 & e^{r_L}\end{pmatrix},\qquad
\kappa_2\big(S(r_L)\big) = e^{2|r_L|} .
$$

The 2-norm condition number is a property of the target gate itself, not of the compiler: strong squeezing is an intrinsically ill-conditioned linear map.

Shear synthesis (see R3): a pure squeezer has $c = S_{21} = 0$, so `decompose_symplectic` always takes the five-step branch. It pre-multiplies by $T(0)^T$, which gives $a = d = 0$ and $c = -e^{r_L}$, and the shears are

$$
k = \big(0,\ 0,\ -e^{-r_L},\ -e^{r_L},\ -e^{-r_L}\big),\qquad S(r_L) = T(k_5)\cdots T(k_1),\quad T(k) = \begin{pmatrix}-k & -1\\ 1 & 0\end{pmatrix}.
$$

One shear magnitude grows as $e^{\pm r_L}$, so at fixed resource squeezing $r$ the noise, built by the recursion

$$
N_{j+1} = T(k_j)\,N_j\,T(k_j)^T + \mathrm{diag}(0, e^{-2r}),\qquad N_0 = 0,
$$

is amplified by the synthesis itself. It grows with $|r_L|$ on the scale of $e^{2|r_L|}$, but not symmetrically in the sign of $r_L$, because the large shear sits at a different point of the step order relative to the quadrature it amplifies. The deterministic map stays exact for any $r$; only $N$ depends on $r$.

### What the code does
- System under test: `pg.Circuit(1).squeeze(0, r_L).compile(squeezing=1.2)`, analysed by `pg.gaussian_channel`.
- Oracle: an inline NumPy matrix $\mathrm{diag}(e^{-r_L}, e^{r_L})$, deliberately not `photographiq.gaussian.squeezing`, which the compiler uses.
- Sweep: $r_L \in \{-1.5, -1.0, -0.5, -0.2, 0, 0.2, 0.5, 1.0, 1.5\}$ at resource squeezing $r = 1.2$.
- Metrics: map Frobenius error, noise spectral norm $\|N\|_2$, $\mathrm{Tr}\,N$, condition number $\kappa_2$ of the compiled map, and finiteness flags of map and noise.
- Execution: for $r_L \in \{-1.0, 0.2, 1.5\}$ the compiled pattern runs on `gaussian` and `native-gaussian` via `pg.run_shots` (600 shots, seed 20261113, coherent input $\alpha = 0.2 + 0.3i$). Ensemble mean and covariance are compared with `channel.apply` in shot-noise standard errors (as in R1), the engines are compared record by record, and the planner's backend choice is logged.
- Outputs: `R4_squeezing_compile`, `R4_squeezing_compile_execution` (CSV/JSON/metadata) and figure `R4_squeezing_compile`.

### Expected outcome
The map error should be at machine precision (about $10^{-15}$) for all $r_L$, with every quantity finite. The compiled condition number should reproduce $e^{2|r_L|}$ (from 1 at $r_L = 0$ to about 20.1 at $|r_L| = 1.5$), symmetric in $r_L$. $\|N\|_2$ should grow with $|r_L|$ on both sides but asymmetrically, with its minimum near $r_L \approx 0$ to $0.2$ and its largest value at $r_L = -1.5$. Sampled ensembles should lie within a few standard errors of the channel.

Acceptance (any failure raises and writes `R4_squeezing_compile_FAILURES.csv`): map error $\le 10^{-9}$ and finite map and noise for every $r_L$; for the executed patterns, every ensemble moment within 5 SE on both engines and native-vs-NumPy record difference $\le 10^{-9}$.

Figure `R4_squeezing_compile`: (left) $\|N\|_2$ vs $r_L$, log $y$, showing synthesis amplification of the noise floor; (right) compiled-map condition number vs $r_L$, log $y$, a symmetric V-shape $e^{2|r_L|}$.

<!-- core:R5 -->
**Title:** Random single-mode symplectic compilation

**Why this experiment.** R3 and R4 test two special gate families. R5 is the broad statistical test of the single-mode compiler: arbitrary valid single-mode Gaussian unitaries, drawn at random, must compile to MBQC patterns whose unconditional channel reproduces the target exactly. The compiled patterns must also be structurally sound: graph metadata consistent with the command list, feed-forward causally ordered, and a lossless typed-IR round trip. A small sample of patterns is also executed on the trajectory engines. Without R5, a decomposition that works for rotations and squeezers but fails for generic matrices, or a compiler that emits inconsistent graph metadata, could pass every hand-picked test.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$. Every real $2\times 2$ symplectic matrix ($\det S = 1$) has an Euler (Bloch-Messiah) factorisation into rotation, squeezer and rotation. R5 samples

$$
S(\varphi_1, r, \varphi_2) = R(\varphi_1)\,\mathrm{diag}(e^{-r}, e^{r})\,R(\varphi_2),\qquad
\varphi_{1,2} \sim U(-\pi, \pi),\quad r \sim U(-1.6, 1.6),
$$

with $R(\varphi) = \begin{pmatrix}\cos\varphi & -\sin\varphi\\ \sin\varphi & \cos\varphi\end{pmatrix}$ built in plain NumPy, independently of `photographiq.gaussian`. `Circuit.gaussian` passes $S$ to `decompose_symplectic` (R3):

$$
S = T(k_4)T(k_3)T(k_2)T(k_1),\quad k_1 = 0,\ k_2 = \frac{1-S_{22}}{S_{21}},\ k_3 = S_{21},\ k_4 = \frac{1-S_{11}}{S_{21}},\qquad T(k) = \begin{pmatrix}-k & -1\\ 1 & 0\end{pmatrix},
$$

with a five-step branch when $|S_{21}| < 10^{-8}$ (probability essentially zero for continuous random draws). The decomposition re-multiplies its shears and raises `ValueError` if the product misses $S$ by more than $10^{-8}$. Such cases are recorded as decomposition failures, never skipped silently. Each step is one teleportation with resource squeezing $r_{\rm res}$; the finite-squeezing noise $N$ follows the recursion $N \to T N T^T + \mathrm{diag}(0, e^{-2r_{\rm res}})$ and is reported separately, never counted as map error.

Structural invariants of a valid compiled pattern: (i) its resource-graph edges and weights equal its `Entangle` commands; (ii) the squeezing of each non-input node equals its `Prepare` command; (iii) every non-output node is measured exactly once and no output is measured; (iv) every outcome-dependent command comes after the measurement it reads (all edges of `Pattern.dependencies()` point forward in node order); (v) `OpenGraph.from_cvgraph` survives a `to_dict`/`from_dict` round trip with inputs and outputs preserved.

### What the code does
- System under test: `pg.Circuit(1).gaussian(0, S).compile(squeezing=1.0, return_trace=True)`, analysed by `pg.gaussian_channel`.
- Oracle: the NumPy-built target $S$; the map error is $\epsilon = \|S_{\rm pattern} - S\|_F$.
- Sample: 400 random targets from a deterministic generator (seed 20260913), resource squeezing $r_{\rm res} = 1.0$.
- Per sample: $\det S$, $\epsilon$, $\log_{10}\epsilon$, $\mathrm{Tr}\,N$, resource-node and measurement counts from the compilation trace, command count, and the five graph-metadata checks above.
- Execution: the first 8 samples run on `gaussian` and `native-gaussian` with `pg.run_shots` (600 shots, seed 20261113, coherent input $\alpha = 0.25 + 0.15i$). Ensemble moments are compared with `channel.apply` in shot-noise standard errors (as in R1), engines are compared record by record, and the planner's backend is logged.
- Outputs: `R5_random_symplectic_compile` (CSV/JSON/metadata, figure, caption), plus `R5_random_symplectic_compile_decomposition_failures.csv` if any decomposition raised.

### Expected outcome
$\epsilon$ should cluster at machine precision (roughly $10^{-16}$ to $10^{-13}$) across the whole parameter space, with a slight tail at large $|r|$ where matrix entries reach $e^{1.6} \approx 5$. Decomposition failures should be zero or very rare. Every pattern should pass all metadata checks. Sampled ensembles should sit within a few standard errors of the channel.

Acceptance (any violation raises and writes `R5_random_symplectic_compile_FAILURES.csv`): $\epsilon \le 10^{-7}$ for every successful compilation; all graph-metadata checks true; for executed patterns, every ensemble moment within 5 SE on both engines and native-vs-NumPy record difference $\le 10^{-9}$.

Figure `R5_random_symplectic_compile` (single panel): histogram (30 bins) of $\log_{10}\epsilon$ over all successful compilations. The title reports the number of successes and of decomposition failures.

<!-- core:R6 -->
**Title:** Compiled two-mode CZ gate

**Why this experiment.** The controlled-Z interaction is the only native entangling gate of the compiler, and every multimode circuit (including the beamsplitter, R7) is built from it. R6 checks that `Circuit(2).cz` produces the exact CZ symplectic map for positive and negative weights, and that its finite-squeezing noise has the predicted, weight-independent block structure. The compiled pattern is also executed on the trajectory engines. A wrong sign or index convention here (which momentum is kicked by which position) would silently corrupt every entangling circuit.

### Theory
Conventions: $\hbar = 2$, two-mode ordering $(q_0, p_0, q_1, p_1)$, vacuum covariance $I$. The `Entangle(u, v, g)` command implements $e^{i g \hat q_u \hat q_v/2}$, i.e.

$$
p_0 \to p_0 + g\,q_1,\qquad p_1 \to p_1 + g\,q_0,\qquad
\mathrm{CZ}(g) = \begin{pmatrix}1 & 0 & 0 & 0\\ 0 & 1 & g & 0\\ 0 & 0 & 1 & 0\\ g & 0 & 0 & 1\end{pmatrix}.
$$

The compiler lowers a logical CZ as one instantaneous `Entangle` between the current frontier nodes, followed by a four-step $k = 0$ identity wire (R2) on each mode, so that every logical gate ends on fresh output nodes. The entangling step is an exact symplectic transformation with no resource state and adds no noise. All finite-squeezing noise comes from the two identity wires, so the noise is block diagonal and independent of $g$:

$$
N_{\rm CZ} = \begin{pmatrix} N_{\rm wire}(r) & 0\\ 0 & N_{\rm wire}(r)\end{pmatrix},\qquad N_{\rm wire}(r) = 2e^{-2r} I_2 .
$$

For a product coherent input with mean $\mu$ and covariance $I$, the exact output is $\mu_{\rm out} = \mathrm{CZ}(g)\,\mu$ and $V_{\rm out} = \mathrm{CZ}(g)\,\mathrm{CZ}(g)^T + N_{\rm CZ}$. Its entanglement signature is the conjugate cross-correlation $\mathrm{Cov}(p_0, q_1) = \mathrm{Cov}(q_0, p_1) = g$, while $\mathrm{Cov}(q_0, q_1) = 0$ because positions are never mixed. With $r = 1$, $\|N_{\rm CZ}\|_2 = 2e^{-2} \approx 0.271$ for every weight.

### What the code does
- System under test: `pg.Circuit(2).cz(0, 1, g).compile(squeezing=1.0)`, analysed by `pg.gaussian_channel`.
- Independent oracles: an inline CZ matrix (not `photographiq.gaussian.cz`), and a noise oracle assembled from `photographiq.gaussian.wire_channel([0,0,0,0], r)` as two diagonal blocks. The full output state for each input is predicted from these two oracles alone.
- Sweep: $g \in \{-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 1.5, 2\}$ at $r = 1$; coherent input pairs $(\alpha_0, \alpha_1) \in \{(0.3+0.1i, -0.2+0.4i),\ (0, 0.5)\}$.
- Metrics: map error $\|S - \mathrm{CZ}(g)\|_F$, noise error $\|N - N_{\rm CZ}\|_F$, max output mean and covariance errors over the inputs, $\|N\|_2$, $\mathrm{Tr}\,N$, and the recorded $q_0$-$q_1$ covariance.
- Execution: for $g \in \{-1, 0.5, 2\}$ the pattern runs on `gaussian` and `native-gaussian` via `pg.run_shots` (600 shots, seed 20261113, inputs $\alpha = (0.3+0.1i, -0.2+0.4i)$). Ensemble moments are compared with `channel.apply` in shot-noise standard errors (as in R1), engines are compared record by record, and the planner's backend is logged.
- Outputs: `R6_cz_compile`, `R6_cz_compile_execution` (CSV/JSON/metadata) and figure `R6_cz_compile`.

### Expected outcome
Map, noise and state errors should all be at machine precision (about $10^{-16}$ to $10^{-15}$) for every $g$. $\|N\|_2$ and $\mathrm{Tr}\,N = 8e^{-2r} \approx 1.083$ should be exactly flat in $g$, and the recorded $\mathrm{Cov}(q_0, q_1)$ should be zero. Sampled ensembles should lie within a few standard errors of the channel.

Acceptance (any violation raises and writes `R6_cz_compile_FAILURES.csv`): map, noise-oracle, output-mean and output-covariance errors all $\le 10^{-8}$; for executed patterns, every ensemble moment within 5 SE on both engines and native-vs-NumPy record difference $\le 10^{-9}$.

Figure `R6_cz_compile`: (left) compiled-map Frobenius error vs $g$ (linear axis, at rounding level); (right) $\|N\|_2$ vs $g$ on a log axis, a horizontal line showing that entangling is noiseless and only the appended wires contribute.

<!-- core:R7 -->
**Title:** Compiled beamsplitter via SUM gates

**Why this experiment.** The beamsplitter is the most resource-expensive two-mode primitive of the compiler. It is not native to the cluster: it is built from SUM gates, each a CZ conjugated by single-mode Fourier rotations, so it nests the teleportation machinery of R3 and R6 three layers deep. R7 checks that this construction reproduces the exact beamsplitter symplectic map over the whole angle range, records its fixed MBQC resource overhead, and executes a compiled instance on the trajectory engines. An error in the SUM sandwich, in the half-angle trick, or in the angle wrapping would break every interferometric circuit.

### Theory
Conventions: $\hbar = 2$, ordering $(q_0, p_0, q_1, p_1)$. The package beamsplitter, with $c = \cos\theta$ and $s = \sin\theta$, is

$$
\mathrm{BS}(\theta) = \begin{pmatrix} c & 0 & -s & 0\\ 0 & c & 0 & -s\\ s & 0 & c & 0\\ 0 & s & 0 & c\end{pmatrix}.
$$

A SUM gate is a CZ (R6) conjugated by $\pm\pi/2$ rotations of the target mode,

$$
\mathrm{SUM}_{u\to v}(g) = R_v(-\tfrac{\pi}{2})\,\mathrm{CZ}_{uv}(g)\,R_v(\tfrac{\pi}{2}),
$$

which realises a position shift $q_v \to q_v + g\,q_u$ (with the conjugate back-action on $p_u$). Three SUMs give a beamsplitter of angle $\varphi$,

$$
\mathrm{BS}(\varphi) = \mathrm{SUM}_{v\to u}\big(-\tan\tfrac{\varphi}{2}\big)\,\mathrm{SUM}_{u\to v}\big(\sin\varphi\big)\,\mathrm{SUM}_{v\to u}\big(-\tan\tfrac{\varphi}{2}\big),
$$

which is singular at $\varphi = \pm\pi$. The compiler first wraps $\theta$ into $[-\pi, \pi)$ and then applies the three-SUM sandwich twice with $\varphi = \theta/2$:

$$
\mathrm{BS}(\theta) = \Big[\mathrm{SUM}_{v\to u}\big(-\tan\tfrac{\theta}{4}\big)\,\mathrm{SUM}_{u\to v}\big(\sin\tfrac{\theta}{2}\big)\,\mathrm{SUM}_{v\to u}\big(-\tan\tfrac{\theta}{4}\big)\Big]^2 .
$$

This moves the pole to $\theta = \pm 2\pi$, outside the wrapped range. The sweep still stays $0.05$ rad away from $\pm\pi$ (the wrap boundary) as a numerical margin.

Resource count: each $\pm\pi/2$ rotation decomposes into 4 teleportation steps (R3), and each CZ into one `Entangle` plus a 4-step identity wire per mode (8 steps), so one SUM costs $4 + 8 + 4 = 16$ teleportations. Six SUMs give $96$ resource nodes and $96$ homodyne measurements per beamsplitter, independent of $\theta$. This is a deliberate design choice that favours a transparent compiler over efficiency. The finite-squeezing noise accumulates over all 96 steps and depends only weakly on $\theta$ through the SUM weights.

### What the code does
- System under test: `pg.Circuit(2).beamsplitter(0, 1, theta).compile(squeezing=1.2, return_trace=True)`, analysed by `pg.gaussian_channel`.
- Oracle: the inline matrix $\mathrm{BS}(\theta)$ above (package sign convention), written independently of `photographiq.gaussian.beamsplitter`.
- Sweep: 17 angles `np.linspace(-pi + 0.05, pi - 0.05, 17)` at resource squeezing $r = 1.2$.
- Metrics: map Frobenius error, $\|N\|_2$, $\mathrm{Tr}\,N$, and the resource nodes and measurements from the compilation trace, plus the total command count.
- Execution: $\theta = 0.3$ runs on `gaussian` and `native-gaussian` via `pg.run_shots` (300 shots, fewer than for single-mode gates because each shot has 96 homodynes; seed 20261113; coherent inputs $(0.3+0.1i, -0.2+0.2i)$). Ensemble moments are compared with `channel.apply` in shot-noise standard errors (as in R1), engines are compared record by record, and the planner's backend is logged.
- Outputs: `R7_beamsplitter_compile`, `R7_beamsplitter_compile_execution` (CSV/JSON/metadata) and figure `R7_beamsplitter_compile`.

### Expected outcome
The map error should stay near machine precision (roughly $10^{-15}$ to $10^{-13}$, slightly above R3-R6 because six nested syntheses compound rounding) with no growth toward $\pm\pi$. The resource counts should be exactly flat at 96 nodes and 96 measurements. $\|N\|_2$ should be of order a few units at $r = 1.2$, which shows the noise cost of the long synthesis chain. The executed ensemble should lie within a few standard errors of the channel.

Acceptance (any violation raises and writes `R7_beamsplitter_compile_FAILURES.csv`): map error $\le 10^{-6}$ at every angle; for the executed pattern, every ensemble moment within 5 SE on both engines and native-vs-NumPy record difference $\le 10^{-9}$.

Figure `R7_beamsplitter_compile`: (left) compiled-map Frobenius error vs $\theta$ on a log axis (flat at the rounding floor, no pole); (right) resource nodes and measurements vs $\theta$, two coincident flat lines at 96.

<!-- core:R8 -->
**Title:** Five-mode circuit-to-MBQC compilation showcase

**Why this experiment.** R3-R7 validate each gate family in isolation. R8 shows that the compiler composes them correctly in one realistic multimode circuit: every gate family at once, interleaved across five modes, with feed-forward between gates. It is both the main-text illustration of how a logical circuit becomes a measurement pattern and a quantitative check. The composed $10\times 10$ symplectic map and displacement must match an independent product of the eight gates, the graph metadata must be self-consistent, and the pattern must execute correctly on the trajectory engines. Composition bugs, such as frontier bookkeeping, mode embedding or the ordering of corrections, are invisible in single-gate tests.

### Theory
Conventions: $\hbar = 2$, interleaved ordering $(q_0, p_0, \ldots, q_4, p_4)$. The circuit is

$$
R_0(0.4),\ S_1(0.3),\ R_2(-0.6),\ \mathrm{CZ}_{01}(0.8),\ D_3(q{=}0.2,\,p{=}{-0.1}),\ \mathrm{BS}_{23}(0.3),\ R_4(0.5),\ \mathrm{CZ}_{34}(-0.5),
$$

with the conventions of R3 (rotation), R4 ($S(r) = \mathrm{diag}(e^{-r}, e^{r})$), R6 (CZ: $p_u \mathrel{+}= g q_v$, $p_v \mathrel{+}= g q_u$) and R7 (beamsplitter). A Gaussian circuit is an affine map $x \to S x + d$. Composing gate $j$ as $(S_j, d_j)$ gives

$$
S = S_8 \cdots S_1,\qquad d \leftarrow S_j d + d_j \ \text{(in gate order)} ,
$$

where each $S_j$ is a two- or four-dimensional block embedded into the identity on the acted-on modes, and only the displacement contributes $d_j = (0.2, -0.1)$ on mode 3. The compiled pattern's unconditional channel is $\mu \to S\mu + d$, $V \to S V S^T + N$, with finite-squeezing noise $N$ accumulated over all teleportation steps and reported separately.

Resource accounting follows R3-R7: rotations take 4 teleportation steps, the squeezer 5 (its $S_{21} = 0$ branch), each CZ 8 (identity wire on both modes), the displacement 4 (identity wire plus an output correction) and the beamsplitter 96 (six SUM gates of 16 steps). That gives $4+5+4+8+4+96+4+8 = 133$ homodyne measurements and $133 + 5 = 138$ graph nodes, below the `visualize_compilation` size guard ($\le 24$ gates, $\le 150$ nodes). The structural invariants checked are those of R5: edges and weights equal the `Entangle` commands, node squeezing equals the `Prepare` commands, every non-output node is measured exactly once and no output is measured, every dependency edge points forward in command order, and the `OpenGraph` IR survives a dict round trip.

### What the code does
- System under test: `pg.Circuit(5)` as above, compiled with `compile(squeezing=1.0, return_trace=True)` and analysed by `pg.gaussian_channel`.
- Independent oracle: a hand-written NumPy product of the eight embedded gate matrices and the displacement, sharing no code with the compiler.
- Metrics: map Frobenius error, displacement error, $\mathrm{Tr}\,N$, $\|N\|_2$, graph-metadata checks, and per-gate trace rows (resource nodes, measurements, corrections, commands).
- Execution: the pattern runs on `gaussian` and `native-gaussian` via `pg.run_shots` (200 shots, about 140 homodynes each; seed 20261113; coherent inputs $(0.2+0.1i, -0.1+0.2i, 0.3, -0.2i, 0.1+0.1i)$). Ensemble moments are compared with the analytic channel in shot-noise standard errors (as in R1), engines are compared record by record, and the planner's backend is logged.
- Outputs: `R8_multimode_showcase_summary.json`, `R8_multimode_showcase_steps.csv`, metadata, the serialized pattern `R8_multimode_showcase_pattern.json` (raw), the package's spring-layout view `R8_multimode_showcase_package_view`, and the main figure `R8_multimode_showcase` with caption.

### Expected outcome
The map and displacement should agree with the independent product at machine precision (about $10^{-15}$ to $10^{-14}$). The compiled pattern should have 138 nodes and 133 measurements, the beamsplitter accounting for 96 of them. All metadata checks should be true, and ensemble moments should lie within a few standard errors.

Acceptance (any violation raises and writes `R8_multimode_showcase_FAILURES.json`): map and displacement errors $\le 10^{-9}$; every graph-metadata check true; every ensemble moment within 5 SE on both engines; native-vs-NumPy record difference $\le 10^{-9}$.

Figures: `R8_multimode_showcase` is a measurement-order view. The x axis is the position of each node's homodyne in the command list (outputs at the right), the y axis is the logical mode carried by the node (nodes of two-mode gates go to the graph-closest input row), and colours mark the source gate (labelled above). Grey lines are CZ edges, red arrows are classical feed-forward from `Pattern.dependencies()`, squares are inputs and diamonds outputs. The long beamsplitter block on modes 2-3 dominates. `R8_multimode_showcase_package_view` is the package's own gate-to-resource spring-layout overview, kept for provenance.

<!-- core:R9 -->
**Title:** Compilation provenance and trace completeness

**Why this experiment.** The compiler can return a `CompilationTrace` that attributes every emitted command, resource node, measurement and correction to the logical gate that produced it. The per-gate resource accounting (R8, R10), the compilation visualisations, and any debugging of a compiled circuit all assume this record is complete, non-overlapping and faithful to the command list. R9 is a software-correctness check rather than a physics test: it verifies these properties exactly, with no tolerances, on the five-mode showcase circuit of R8.

### Theory
Let the compiled pattern have commands indexed $0, \ldots, M-1$, excluding the trailing `Output` command, and let $I_g$ be the indices that step $g$ of the trace attributes to logical gate $g$. Provenance is complete and non-overlapping exactly when the $I_g$ partition the index set,

$$
\bigcup_g I_g = \{0, 1, \ldots, M-1\},\qquad I_g \cap I_{g'} = \varnothing\quad (g \ne g').
$$

This is checked by comparing the sorted concatenation of all `command_indices` with `range(M)` element for element, which catches gaps and duplicates at once. Further exact invariants tie the trace to the command list:
- every index in `step.measurements` is a `Measure` command, and their union is exactly the set of all `Measure` indices;
- every index in `step.corrections` is a `Displace` command;
- every predecessor of a correction in the dependency DAG `Pattern.dependencies()` has a smaller command index (feed-forward is causal);
- the union of `step.resource_nodes` is exactly the set of nodes created by `Prepare` commands.

Expected per-gate structure (see R3-R8): one teleportation step consists of four commands (`Prepare`, `Entangle`, `Measure`, `Displace`). A rotation uses 4 steps (16 commands), the squeezer 5 steps (20), a CZ one `Entangle` plus 8 steps (33), the displacement 4 steps plus one output correction (17), and the beamsplitter 96 steps plus 6 SUM-internal `Entangle` commands (390). For the showcase circuit this totals $M = 541$ commands, plus `Output`.

### What the code does
- System under test: `pg.Circuit(5)` with the same eight gates as R8 ($R_0(0.4)$, $S_1(0.3)$, $R_2(-0.6)$, $\mathrm{CZ}_{01}(0.8)$, $D_3(0.2, -0.1)$, $\mathrm{BS}_{23}(0.3)$, $R_4(0.5)$, $\mathrm{CZ}_{34}(-0.5)$), compiled with `compile(squeezing=1.0, return_trace=True)`.
- Oracle: the pattern's own command list and dependency DAG, an independent view of the same compilation, compared with the trace record by record.
- Exported per gate: gate index, name, modes, parameters, input and output frontier nodes, resource nodes, command indices, edges, measurements, corrections and synthesis report, together with their counts.
- No backend is executed; R8 runs the same pattern on `gaussian` and `native-gaussian`.
- Outputs: `R9_compilation_trace.csv`, `R9_compilation_trace.json` (steps, pattern signature, trace checks), metadata, and the human-readable `R9_compilation_trace_summary.txt` (raw). No figure is produced.

### Expected outcome
All checks are combinatorial identities, so they hold exactly or fail. The trace should list eight steps with resource-node counts 4, 5, 4, 8, 4, 96, 4, 8 (133 in total) and command indices that tile $0..540$ without gaps. Acceptance: the partition identity must hold exactly (otherwise `AssertionError`), and all five trace-consistency checks must be true (otherwise the check dictionary is written to `R9_compilation_trace_FAILURES.json` and the script raises). The summary text gives one line per gate with its resource nodes, measurements, corrections, command count and input-to-output frontier.

<!-- core:R10 -->
**Title:** Compiler resource scaling

**Why this experiment.** A practical compiler must have predictable cost: the MBQC resources (nodes, measurements, entangling edges, commands) of a circuit should grow linearly with its gate count, with a fixed, known cost per gate family, and compile time should remain practical. R10 measures these growth laws for the four primitive families of R3, R4, R6 and R7. It also checks that the largest compiled patterns are still exactly correct and that the C++ engine reproduces the NumPy engine on them. Without it, hidden super-linear bookkeeping, or a loss of accuracy in long chains (for example 100 chained squeezers), would go unnoticed.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$ per mode. Each teleportation step consists of four commands (`Prepare`, `Entangle`, `Measure`, `Displace`) and adds one resource node, one measurement and one CZ edge. The per-gate costs follow from R3-R7:

| family | steps (nodes, measurements) | edges | commands |
|---|---|---|---|
| rotation | $4n$ | $4n$ | $16n + 1$ |
| squeeze | $5n$ | $5n$ | $20n + 1$ |
| CZ | $8n$ | $9n$ | $33n + 1$ |
| beamsplitter | $96n$ | $102n$ | $390n + 1$ |

The $+1$ is the trailing `Output` command. Rotations use the four-step branch because none of the angles $0.1 + 0.01i$ is near $0$ or $\pi$. The squeezer always uses the five-step branch (R4). A CZ is one `Entangle` plus two 4-step identity wires (R6). A beamsplitter is six SUM gates of 16 steps plus one `Entangle` each (R7). All counts are therefore exactly linear in $n$ (slope 1 on log-log axes), with family-dependent prefactors. Compile time is an empirical observation, not a physical prediction.

Correctness of the largest patterns uses the fact that gates of one family acting on the same modes commute, so their parameters add. With $n$ gates of parameter $x_0 + \delta\, i$, $i = 0, \ldots, n-1$,

$$
R\Big(\sum_i \theta_i\Big),\qquad \mathrm{diag}\big(e^{-\sum r_i}, e^{\sum r_i}\big),\qquad \mathrm{CZ}\Big(\sum_i g_i\Big),\qquad \mathrm{BS}\Big(\sum_i \theta_i\Big)
$$

are independent closed-form targets. The parameters are $\theta_i = 0.1 + 0.01 i$ (rotation), $r_i = 0.05 + 0.005 i$ (squeeze), $g_i = 0.3 + 0.01 i$ (CZ) and $\theta_i = 0.2 + 0.01 i$ (beamsplitter). For 100 squeezers the total is $r = 29.75$, so the map entries reach $e^{29.75} \approx 8\times 10^{12}$ and $\kappa_2 = e^{59.5}$. Absolute float64 errors are then meaningless, and the map is compared with the relative error $\|S - S_{\rm target}\|_F / \|S_{\rm target}\|_F$.

### What the code does
- System under test: `pg.Circuit(...).compile(squeezing=1.0, return_trace=True)` for $n$ repeated gates of each family (single mode for rotation and squeeze, two modes for CZ and beamsplitter).
- Sweep: $n \in \{1, 2, 5, 10, 20, 50, 100\}$ for rotation, squeeze and CZ; $n \in \{1, 2, 5, 10, 20\}$ for the much more expensive beamsplitter.
- Metrics: resource nodes (graph nodes minus inputs), measurements, `Entangle` edges, commands, and compile wall time (median of 3 repeats for $n \le 20$, single run above). Timings are labelled provisional (shared machine).
- Largest-pattern validation for (rotation, 100), (squeeze, 20), (squeeze, 100), (CZ, 100) and (beamsplitter, 20): relative and absolute map errors against the closed-form target from `pg.gaussian_channel`, target condition number, $\mathrm{Tr}\,N$, and one `pg.simulate` trajectory (seed 11) on `gaussian` and `native-gaussian`, comparing records and output means.
- Outputs: `R10_compiler_scaling` (CSV/JSON/metadata), `R10_compiler_scaling_largest_validation.csv` and figure `R10_compiler_scaling`.

### Expected outcome
The resource counts should follow the table above exactly: straight, parallel lines of slope 1 on log-log axes, with the beamsplitter about 24 times costlier than a rotation. Compile time should grow roughly linearly, from milliseconds up to the order of a second for the largest patterns. Relative map errors should be at the rounding level (about $10^{-15}$ to $10^{-12}$).

Acceptance (any violation raises and writes `R10_compiler_scaling_FAILURES.csv`): relative map error $\le 10^{-9}$ for every largest pattern; relative record difference $|m_{\rm native} - m_{\rm numpy}|/\max(1, |m_{\rm numpy}|) \le 10^{-9}$, asserted only when the target condition number is $\le 10^8$. That excludes the 100-squeezer chain ($\kappa \approx 7\times 10^{25}$), whose record difference is recorded but not asserted because differences of order $\epsilon_{\rm mach}\kappa$ are expected. The 20-squeezer chain ($r = 1.95$) carries the asserted trajectory contract for squeezing.

Figure `R10_compiler_scaling` (2x2, log-log, one line per family): resource nodes, measurements, commands and median compile time vs number of gates.

<!-- core:R11 -->
**Title:** Adaptive feed-forward of measurement angles

**Why this experiment.** Every gate compiled in R3-R10 uses fixed homodyne angles: only the displacement corrections depend on outcomes. Universal and non-Gaussian CV-MBQC also needs measurement bases chosen after earlier outcomes are known. R11 tests that PhotoGraphiQ represents and executes this genuine adaptivity correctly. The outcome-dependent angle must create a causal dependency, must be rejected by the fixed-channel analyzer, must be scheduled after the measurement it reads, and must resolve per trajectory to exactly the angle a hand-built non-adaptive pattern would use. If angle feed-forward were silently ignored, frozen at compile time, or resolved from the wrong record, adaptive protocols would give plausible but wrong results.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$, vacuum covariance $I$, homodyne quadrature $x_\theta = q\cos\theta + p\sin\theta = h_\theta^T x$ with $h_\theta = (\cos\theta, \sin\theta)^T$.

Pattern: a three-node line resource $0 - 1 - 2$ (`CVGraph.line(3, squeezing=1.0, inputs=(0,))`; nodes 1 and 2 are squeezed resource modes, node 0 is the input). The commands are

$$
m_0 = x_0(0),\quad D_1(q \mathrel{-}= m_0),\quad m_1 = x_{\theta_1}(1),\ \ \theta_1 = 0.05\,m_0 + \theta,\quad D_2(q \mathrel{-}= m_1),\quad \text{output } 2,
$$

with $\theta = 0.3$ supplied as a symbolic `Parameter`. Because $\theta_1$ is a function of a random outcome, the map applied to the remaining modes is a family of maps selected after $m_0$ is known. A fixed-angle pattern has one unconditional Gaussian channel $(S, N, d)$, with outcomes entering only through affine displacements. Here the measurement generator itself depends on the record, so no single $(S, N, d)$ describes all trajectories. `gaussian_channel` therefore raises `NotImplementedError`, and R11 uses this rejection as a structural test of adaptivity.

Conditional Gaussian update: measuring $x_\theta$ on mode $a$ of a state $(\mu, V)$, partitioned into $a$ and the rest $B$, gives the outcome density

$$
p(m) = \mathcal N\big(m;\ h^T\mu_a,\ h^T V_{aa} h\big),
$$

and the post-measurement moments

$$
\mu_B' = \mu_B + \frac{V_{Ba}h\,(m - h^T\mu_a)}{h^T V_{aa} h},\qquad
V_B' = V_{BB} - \frac{V_{Ba}h\,h^T V_{aB}}{h^T V_{aa} h}.
$$

Fixing the records $(m_0, m_1)$ by postselection (`measurement_outcomes=`) makes the conditional output a deterministic function of the records. It must therefore coincide exactly with a control pattern whose second angle is hard-coded to $0.05\,m_0 + 0.3$. Different $m_0$ must give different conditional outputs, both through the correction and through the angle. Postselected runs are conditional branches with an associated density $p(m_1 \mid m_0)$, not a deterministic execution.

### What the code does
- System under test: the adaptive pattern above, simulated with `pg.simulate` on `gaussian` (NumPy) and `native-gaussian` (C++), coherent input $\alpha = 0.3 + 0.1i$.
- Structural checks: `Pattern.dependencies()` contains the edge from $m_0$'s `Measure` to $m_1$'s; `pg.gaussian_channel` raises `NotImplementedError`; `Pattern.schedule()` places $m_0$ before $m_1$. Any failure raises immediately.
- Sampled trajectories: seeds 1-8 on both engines. The script records $m_0$, $m_1$, the realised angle $0.05\,m_0 + 0.3$, the output $q$ mean and the output quadrature mean at the realised angle, and the native-vs-NumPy difference (records and output mean).
- Postselected branches: $m_0 \in \{-1.5, -0.5, 0, 0.7, 2.0\}$, $m_1 \in \{-0.8, 0.4\}$ (10 branches) on both engines. Oracle: an independently built non-adaptive control pattern with the angle fixed by hand, compared on output mean, covariance and the density of $m_1$.
- Outputs: `R11_adaptive_feedforward` (CSV/JSON/metadata, including dependency edges and pattern JSON), `R11_adaptive_feedforward_branches.csv`, raw `R11_adaptive_feedforward_pattern.json`, and figures `R11_adaptive_feedforward_pattern` and `R11_adaptive_feedforward_dependencies`.

### Expected outcome
The structural checks hold exactly. The adaptive and control branches should agree to rounding (about $10^{-16}$), since they execute identical arithmetic once the angle is resolved, and the two engines should agree to rounding as well. All 10 branches should give distinct output means, proving the feed-forward is active.

Acceptance (any violation raises and writes `R11_adaptive_feedforward_FAILURES.csv`): adaptive-vs-control difference $\le 10^{-12}$ on both engines; native-vs-NumPy difference $\le 10^{-10}$ for every seed and every branch; the number of distinct conditional output means (rounded to 9 decimals) equals the number of branches.

Figures: `R11_adaptive_feedforward_pattern` is the package's drawing of the three-node pattern (graph, measurement angles, corrections). `R11_adaptive_feedforward_dependencies` is the command dependency DAG, including the $m_0 \to m_1$ measurement-angle edge.

<!-- core:R12 -->
**Title:** Dependency DAG and causal schedule

**Why this experiment.** Adaptive patterns such as R11 are well defined only if every classical outcome is known before any command that reads it executes, and if commands on the same mode keep their order. PhotoGraphiQ encodes both requirements in a command dependency DAG (`Pattern.dependencies()`) and derives an execution order from it (`Pattern.schedule()`). R12 checks that the schedule is a valid topological order of the DAG. It then checks the physical meaning of that DAG: any order consistent with it must produce the same conditional output on both Gaussian engines. A missing edge would let a correction run before its outcome exists; a spurious edge would only over-constrain the order, but a wrong order of same-mode commands would silently change the state.

### Theory
Conventions: $\hbar = 2$, ordering $(q, p)$. `Pattern.dependencies()` builds a directed graph $D = (V, E)$ over command indices with two edge types:
- a classical edge $u \to v$ when command $v$ reads `Outcome(key)` produced by measurement $u$;
- a quantum edge $u \to v$ when $u$ is the previous command touching a mode that $v$ also touches (mode lifetimes).

Cycles and missing or duplicate classical keys are rejected. `Pattern.schedule()` returns a lexicographic topological sort $\pi$ (smallest index first among ready commands). Correctness requires

$$
\forall (u, v) \in E:\qquad \pi(u) < \pi(v).
$$

Physical content: commands on disjoint modes act on different tensor factors and commute, and classical dependencies are honoured by construction. Hence every topological order of $D$ yields the same conditional output state for the same measurement records,

$$
\rho_{\rm out}\big(\pi \mid m_0, m_1, m_2\big) = \rho_{\rm out}\big(\pi' \mid m_0, m_1, m_2\big)\qquad \text{for all topological orders } \pi, \pi' .
$$

With random sampling, a different order may consume random numbers in a different sequence, so order invariance is tested on postselected branches, where records are fixed.

Test pattern: a four-node line resource $0 - 1 - 2 - 3$ (input 0, output 3, resource squeezing $r = 1$) with multiple and multi-source dependencies:

$$
m_0 = x_0(0),\ \ D_1(q \mathrel{-}= m_0),\ \ m_1 = x_{0.1 m_0}(1),\ \ D_2(q \mathrel{-}= m_1,\ p \mathrel{+}= 0.2\,m_0),\ \ m_2 = x_{\pi/2}(2),\ \ D_3(q \mathrel{-}= m_2,\ p \mathrel{+}= m_1 - m_0).
$$

The measurement angle of node 1 is adaptive (as in R11), and the corrections of nodes 2 and 3 each read two earlier outcomes.

### What the code does
- System under test: `Pattern.dependencies()`, `Pattern.schedule()` and `Pattern.validate()` on the pattern above, then `pg.simulate` on `gaussian` (NumPy) and `native-gaussian` (C++), with coherent input $\alpha = 0.2 - 0.3i$.
- Structural check: every DAG edge goes forward in the schedule (a discrete comparison, no tolerance); a per-command table lists type, schedule position and predecessors.
- Scheduled execution: the commands are re-assembled in `schedule()` order into a new pattern, which must validate, and are run against the original for seeds $\{3, 5, 8, 13, 21\}$ on both engines. Records, output mean and covariance are compared with the NumPy listed-order reference. For this pattern `schedule()` returns the listed order, so this part mainly checks the re-assembly and the native RNG contract.
- Non-trivial reordering: a genuinely different topological order (latest-index-first tie breaking, `networkx.lexicographical_topological_sort` with key $-i$) is assembled. It must differ from the listed order and is compared with the original on two postselected branches, $(m_0, m_1, m_2) = (0.4, -1.1, 0.3)$ and $(-2.0, 0.5, 1.7)$, on both engines.
- Outputs: `R12_dependency_dag` (CSV/JSON/metadata with edges, schedule, violations and pattern JSON), `R12_dependency_dag_execution.csv`, and figure `R12_dependency_dag`.

### Expected outcome
There should be zero backward violations. All executions, across orders and engines, should agree to rounding (about $10^{-16}$), because they perform the same floating-point operations on commuting factors with identical records. Acceptance: any backward edge raises and writes `R12_dependency_dag_VIOLATIONS.json`; a maximum difference (records, output mean, covariance) above $10^{-10}$ across orders and engines, or an alternative order that equals the listed order, raises and writes `R12_dependency_dag_FAILURES.csv`.

Figure `R12_dependency_dag`: the package's drawing (`draw_dependencies`) of the command DAG, showing preparation and entangling commands feeding the measurements, the classical edges from $m_0$ to the adaptive measurement of node 1 and to both downstream corrections, and from $m_1$ and $m_2$ to the corrections they feed.

<!-- core:R13 -->
**Title:** Rejection of invalid patterns and non-certifiable flows

**Why this experiment.** The paper claims that a PhotoGraphiQ `Pattern` is a causally checked program: a command cannot touch a mode that was never prepared or was already measured, a feed-forward expression cannot read a classical record that does not exist yet, and the command dependency graph is acyclic. Without this audit those guarantees would only be assertions in docstrings. A validator that silently accepts a malformed pattern would let a simulator produce numbers for a protocol that has no physical meaning. R13 is a software-safety experiment, not a physics one: every check is discrete (pass or fail), with no numerical tolerance.

### Theory
A pattern is an ordered command list $C_0, C_1, \dots$ acting on mode labels, together with a classical record set $\mathcal{R}$. Validity requires:

- **Mode lifetime.** Each mode is prepared at most once (inputs count as prepared), is used only while active, and becomes inactive after a destructive `Measure`.
- **Classical causality.** Each `Measure` or `Signal` defines a unique key $k$. A command whose parameters contain $\mathrm{Outcome}(k)$ is legal only if $k \in \mathcal{R}$ when it executes. In particular the key must exist and must already have been produced.
- **Acyclicity.** The dependency digraph (an edge from the producer of $k$ to every reader of $k$, plus an edge between successive commands on the same mode) must be a DAG (see R12).
- **Well-formed commands.** Inputs are distinct, an `Entangle` has two distinct endpoints, no command repeats a mode label, `Output` is the final command, and a measurement is a supported `MeasurementProtocol`.
- **Physical resource.** A `Pattern` cannot be built from an ideal ($r = \infty$) symbolic `CVGraph`.

The second part concerns determinism. For a CV open graph with homodyne measurements, the Booth-Markham real-linear CV flow exists only if the correction matrices admit exact solutions. The "singular bridge" (inputs $u, v$; outputs $a, b$; unit edges $u$-$a$, $u$-$b$, $v$-$a$, $v$-$b$) has correction block $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ of rank 1. No order can correct both measured nodes, so the flow finder must return the decision `impossible`, backed by exact rational witnesses. A qubit flow algorithm (gflow) asked about a CV graph must answer `unsupported`, never with a qubit certificate.

### What the code does
- **System under test:** `Pattern.validate()` on twelve hand-built invalid patterns, one per invariant: `duplicate_inputs`, `duplicate_resource_node`, `self_cz`, `output_not_final`, `duplicate_outputs`, `missing_or_measured_node`, `future_outcome_dependency`, `cyclic_dependency`, `undefined_classical_key`, `duplicate_classical_key`, `unsupported_measurement_description` and `ideal_graph_pattern`. The last case is rejected by the `Pattern` constructor itself.
- **Second pass:** the same twelve patterns go unvalidated to `pg.simulate(..., backend="native-gaussian")`. That call must raise the same exception type with the same message before it prepares any state, which shows that the C++ engine path validates first.
- **Flow rejections:** `photographiq.flows.analyze` on the singular bridge (squeezing 0.8 on every node), first with the default CV algorithm and then with `"gflow"`.
- **Oracle:** the expected outcome is fixed by construction (rejection), not computed. Accepted exceptions are `ValueError`, `NotImplementedError` and `TypeError`. Any other exception propagates and fails the run.
- **Outputs:** `R13_invalid_causality.csv/.json` (per-case exception type and message for both passes, with a `native_matches_validate` flag), `R13_invalid_causality_flows.csv`, and a metadata JSON. No figure is produced.

### Expected outcome
All twelve invalid patterns raise. Eleven raise `ValueError` with a specific diagnostic, for example "Self CZ is invalid", "Command 0 (Output) repeats a mode", "depends on a future outcome", "No producer for classical key", "Duplicate classical key", "Cyclic command dependencies" or "Ideal graph is symbolic". The unsupported measurement raises `NotImplementedError`. The validator checks self-loops and repeated modes before it builds the dependency graph, so `self_cz` and `duplicate_outputs` now get their specific messages instead of the generic cyclic-dependency message. The native-gaussian pass reproduces every exception type and message exactly. The singular bridge is reported as `not_certified` with decision `impossible` and non-empty witnesses, and gflow on the CV graph returns `unsupported`.

The acceptance criterion is strict. The run fails (`AssertionError`) if any case is not rejected, if any native message or type differs from `validate()`, or if either flow row is not `ok`.

<!-- core:R14 -->
**Title:** CV-flow certification, flow finding and principled declines

**Why this experiment.** Measurement-based protocols are useful only if the outcome-dependent byproducts can be corrected by feed-forward, which makes the computation deterministic. The paper claims two things. First, PhotoGraphiQ certifies CV determinism with the real-linear flow condition of Booth and Markham (Quantum 7, 1146, 2023). Second, its flow finder declines to certify when it cannot decide. R14 checks that a valid order is accepted and an invalid one rejected. It also checks that the finder reproduces the supplied-order certificate, that the corrections it reports actually work on two engines, that a wrong correction map is caught, and that `unknown`, `impossible` and `unsupported` are returned exactly when they should be. A certifier that never says "no" or "I don't know" would be worthless.

### Theory
Let $A$ be the weighted adjacency matrix of a resource graph with inputs $I$ and outputs $O$, and let $(v_1, \dots, v_m)$ be a total order of the measured nodes. At step $j$ define the past $P_j = (v_1, \dots, v_j)$ and the future $F_j = V \setminus (P_j \cup I)$. The order admits a real-linear CV flow if, for every $j$,
$$A[P_j, F_j]\, c_j = e_j$$
has an exact real solution, where $e_j$ is the unit vector on the row of $v_j$. The vector $c_j$ gives the q-corrections on future nodes, and $A c_j$ gives the induced p-corrections. `certify_cv_flow` solves each system by least squares and rejects the order if the residual $\lVert A c_j - e_j\rVert_\infty$ exceeds $10^{-10}$.

For the line $0$-$1$-$2$ (input 0, output 2) the order $(0,1)$ gives $A[\{0\},\{1,2\}] = (1, 0)$, so $c_0 = (1, 0)$, and $A[\{0,1\},\{2\}] = (0, 1)^T$, so $c_1 = 1$. Both residuals are zero. The order $(1,0)$ fails at step 2. Node 0's only neighbour (node 1) is already measured, so $A[\{1,0\},\{2\}] = (1, 0)^T$ cannot produce $(0,1)^T$. The minimum-norm least-squares solution is $c = 0$ with residual exactly 1.

Determinism is an infinite-squeezing statement. At finite squeezing $r$ the conditional output mean is $A\mu + G m$, where the residual gain $G \to 0$ as $r \to \infty$. The averaged channel $(S, N)$ also carries added noise $N$. Conventions: $\hbar = 2$ and $[q, p] = 2i$.

### What the code does
- **Supplied order:** on `CVGraph.line(3, squeezing=1.0, inputs=(0,), outputs=(2,))`, `certify_cv_flow` must certify $(0,1)$ and return `None` for $(1,0)$. A NumPy `lstsq` recomputation of the failing step reports the residual and the solution.
- **Finder:** `flows.analyze` on the same line as a `Model.CV_GAUSSIAN` `OpenGraph` must return a certificate with order $(0,1)$ and regimes {`ideal_limit`, `finite_squeezing`}. Its layer corrections must equal the supplied-order certificate exactly.
- **Execution:** `flows.cv.corrected_pattern` builds a pattern from the reported correction map, with coherent input $\alpha = 0.3 + 0.1i$. It runs `run_shots` with $M = 400$ shots and seed 20261113 on `gaussian` and `native-gaussian`. Ensemble mean and covariance are compared with `gaussian_channel` using standard errors derived from the conditional-mean spread $D = V_{\rm ideal} - V_{\rm cond}$: $\sigma_\mu = \sqrt{D_{ii}/M}$ and $\sigma_{V,ij} = \sqrt{(D_{ii}D_{jj} + D_{ij}^2)/M}$.
- **Wrong map:** every q-correction is scaled by 0.5, and `verify_cv_determinism` must report that the result is not ideal-limit deterministic.
- **Declines:** (i) singular bridge, which must give `impossible` with rational witnesses; (ii) near-singular bridge with determinant $10^{-10}$, which must give `unknown`/`numerically_uncertain` by default and an `ideal_limit`-only certificate with `exact=True`; (iii) a 5-node unit wire at $r = 4$, which must give `unknown` with the default check shift 3.0 (squeezing levels up to $r + 6 = 10$, beyond float64 range) and be certified in both regimes with shift 1.5; (iv) `"gflow"` on the CV graph; (v) a `CV_NONGAUSSIAN` model. Cases (iv) and (v) must return `unsupported`.
- **Outputs:** `R14_cvflow.json`, `R14_cvflow_declines.csv`, the raw pattern `R14_cvflow_pattern.json` and metadata. No figure is produced.

### Expected outcome
The valid order is certified with zero residuals. The invalid order is rejected, and the diagnostic residual equals 1, an $O(1)$ obstruction rather than a marginal failure. The finder agrees with the supplied certificate. Both engines' 400-shot ensembles lie within $5\sigma$ of the analytic channel (`SEM_FACTOR = 5`). Trajectories with the same seed give per-record differences of at most $10^{-10}$ between the NumPy and C++ engines. The halved correction map is rejected, and all six decline rows are `ok`. Any violation is written to `R14_cvflow_FAILURES.json` and raises an error.

<!-- core:R15 -->
**Title:** Gaussian engines against an independent raw Piquasso program

**Why this experiment.** PhotoGraphiQ runs one pattern on three exact Gaussian engines: the Piquasso adapter (`piquasso`), its own NumPy engine (`gaussian`) and the C++ engine (`native-gaussian`). All three share PhotoGraphiQ's gate conventions, so a convention error in the shared layer could pass every internal consistency test. Examples are a factor of 2 in the displacement amplitude, an opposite rotation sense, or a squeezing sign. R15 compares the engines against an external program written directly with the public Piquasso API, with no PhotoGraphiQ code in its numerical path. This fixes the package's conventions relative to an established simulator.

### Theory
Gaussian states are described by the mean vector $\mu$ and covariance $V$ in $(q_1, p_1, q_2, p_2, \dots)$ ordering, with $\hbar = 2$, so the vacuum has $V = I$. A Gaussian unitary acts as $\mu \mapsto S\mu + d$ and $V \mapsto S V S^T$. The gates under test are:

- displacement $D(q, p)$: $d = (q, p)$, equivalent to a Piquasso displacement of amplitude $(q + ip)/2$;
- rotation by $\phi$ (Piquasso `Phaseshifter`);
- squeezing $S(r)$: $q \mapsto e^{-r} q$, $p \mapsto e^{r} p$;
- beamsplitter $BS(\theta)$ with $\phi = 0$;
- weighted controlled-Z $CZ(g) = \exp(i g\, q_1 q_2 / 2)$, which maps $p_1 \mapsto p_1 + g q_2$ and $p_2 \mapsto p_2 + g q_1$ (Piquasso `ControlledZ(s=g)`).

A coherent input $|\alpha\rangle$ has $\mu = (2\,\mathrm{Re}\,\alpha, 2\,\mathrm{Im}\,\alpha)$ and $V = I$. Piquasso's `xpxp_covariance_matrix` at `hbar=2` equals $2V$, so the reference divides it by 2. As a third, scalar observable, the single-mode mean photon number
$$\langle \hat n\rangle = \frac{\mathrm{Tr}\,V + \mu^T\mu - 2}{4}$$
is evaluated with the same formula on both sides.

### What the code does
- **Reference:** a fresh `pq.Program` prepares vacuum, displaces it to the coherent input and applies the native Piquasso gate. It is executed with `pq.GaussianSimulator(d, config=pq.Config(hbar=2))`.
- **System under test:** the same gate as a PhotoGraphiQ `Pattern`, run with `pg.simulate` on `piquasso`, `gaussian` and `native-gaussian` with `GaussianInput.coherent(alpha)` inputs.
- **Sweep:** `N_RANDOM = 20` random draws (seed 20260913) for each of six families, 120 cases in total:
  - displacement: $q, p \in [-1, 1]$;
  - rotation: $\phi \in [-\pi, \pi]$;
  - squeezing: $r \in [-1.2, 1.2]$;
  - beamsplitter: $\theta \in [-\pi/2, \pi/2]$;
  - CZ: $g \in [-2, 2]$;
  - a compound circuit: squeeze ($r \in [-0.6, 0.6]$), rotate, CZ ($g \in [-1, 1]$) and displace on the second mode.

  Input amplitudes have real and imaginary parts uniform in $[-1, 1]$.
- **Metrics:** for each engine, $\lVert \mu_{\rm pg} - \mu_{\rm raw}\rVert$ and $\lVert V_{\rm pg} - V_{\rm raw}\rVert_F$. For the single-mode cases the photon-number difference of the `piquasso` engine is also recorded; it is reported but not part of the acceptance test.
- **Outputs:** `R15_raw_piquasso_gaussian.csv/.json` (all cases), `R15_raw_piquasso_gaussian_summary_table.csv` (worst case per family and engine), metadata, and the figure `R15_raw_piquasso_gaussian` (PDF/PNG/SVG).

### Expected outcome
All three engines solve the same linear-algebra problem exactly, so every mean and covariance error should sit at double-precision rounding level, roughly $10^{-16}$ to $10^{-14}$. Some errors may be exactly zero. The photon-number differences should be at the same level. The acceptance criterion is that every mean-vector and covariance error, for every engine and every case, is at most $10^{-9}$. Otherwise the failing cases go to `R15_raw_piquasso_gaussian_FAILURES.csv` and the run aborts.

The figure has one panel. It shows grouped bars of $\log_{10}$ of the worst covariance error per gate family (x axis, alphabetical), with one bar each for the `piquasso` backend, NumPy `gaussian` and C++ `native-gaussian`. An exact zero is drawn at $-300$. All bars should lie far below the $10^{-9}$ tolerance, and no family or engine should stand out.

<!-- core:R16 -->
**Title:** Independent NumPy analytic reference for Gaussian and Fock engines

**Why this experiment.** R15 compares PhotoGraphiQ with Piquasso, and R16 adds a ground truth that belongs to neither package. The symplectic matrices are typed out directly in NumPy, without importing `photographiq.gaussian`, and applied analytically. Every execution path, including raw Piquasso, is then measured against this target. Agreement among several independent implementations and a closed-form target rules out a convention error shared by any two of them. The Fock engines are included because the same Gaussian circuits must also come out right in a truncated number basis, which tests their gate matrices and Gaussian-input preparation.

### Theory
In $\hbar = 2$ units, with ordering $(q_1, p_1, q_2, p_2)$, the hand-written targets are:
$$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},\qquad S(r) = \mathrm{diag}(e^{-r}, e^{r}),$$
$$BS(\theta) = \begin{pmatrix} c & 0 & -s & 0\\ 0 & c & 0 & -s\\ s & 0 & c & 0\\ 0 & s & 0 & c\end{pmatrix},\qquad CZ(g) = I_4 + g\,(E_{p_1 q_2} + E_{p_2 q_1}),$$
with $c = \cos\theta$ and $s = \sin\theta$. A displacement adds $d = (q, p)$. The compound target is the explicit product $CZ(0.4)\,[R(0.3) \oplus I_2]\,[S(0.2) \oplus I_2]$. For coherent inputs ($\mu_{\rm in} = 2(\mathrm{Re}\,\alpha_k, \mathrm{Im}\,\alpha_k)$, $V_{\rm in} = I$) the analytic output is
$$\mu = S\mu_{\rm in} + d,\qquad V = S V_{\rm in} S^T,$$
with no added noise, because the gates act directly rather than through teleportation.

The Fock engines work at a total-photon cutoff $c$, which keeps states with at most $c - 1$ photons in total. They are compared on single-mode quadrature moments $\langle \hat x_\phi\rangle$ and $\mathrm{Var}(\hat x_\phi)$ for $\phi \in \{0, \pi/2\}$. Inputs have $|\alpha| \le 0.45$ and $r \le 0.5$. At $c = 30$ the population above 29 photons is far below $10^{-12}$, so these moments should match the Gaussian values almost exactly. The remaining difference comes from `piquasso-fock`, which truncates after each factor of its Euler-decomposed gates. The minimum retained norm is reported alongside so that truncation is never hidden by renormalisation.

### What the code does
- **Six fixed cases:**
  - rotation, $\theta = 0.6$, $\alpha = 0.4 + 0.2i$;
  - squeezing, $r = 0.5$, $\alpha = 0.3 - 0.1i$;
  - displacement, $(q, p) = (0.4, -0.3)$, $\alpha = 0.1 + 0.1i$;
  - beamsplitter, $\theta = 0.35$;
  - CZ, $g = 0.7$;
  - compound squeeze, rotate and CZ, $(r, \phi, g) = (0.2, 0.3, 0.4)$.

  The two-mode cases use small two-mode coherent inputs.
- **Exact paths:** `gaussian`, `piquasso`, `native-gaussian` and a raw Piquasso program with covariance $=$ `xpxp_covariance_matrix`$/2$. Each reports the Euclidean mean error and the Frobenius covariance error against the analytic target, which gives 48 exact errors.
- **Fock paths:** `native-fock`, `piquasso-fock` and `native-mixed-fock` at `FOCK_CUTOFF = 30`. Each reports the worst per-mode moment error and the minimum retained norm, which gives 18 moment errors.
- **Outputs:** `R16_numpy_analytic_reference.csv/.json` (with the cutoff and both tolerances), metadata, and the figure `R16_numpy_analytic_reference`.

### Expected outcome
The four exact paths should agree with the analytic target to rounding level, about $10^{-16}$, or exactly. The Fock moment errors should be small, set by truncation and decomposition, and orders of magnitude below an $O(0.1)$ convention error. Retained norms should be essentially 1.

Acceptance has two parts. Every exact error must be at most $10^{-9}$, and every Fock moment error must be at most `FOCK_TOLERANCE` $= 10^{-6}$. Otherwise the failures go to `R16_numpy_analytic_reference_FAILURES.csv` and the run aborts.

The figure is a single grouped-bar panel with the six cases on the x axis and seven series per case. Four series are the covariance errors of `gaussian`, `piquasso`, `native-gaussian` and raw Piquasso. Three are the moment errors of the Fock engines at $c = 30$. The y axis is $\log_{10}$ error from $-17.5$ to $0.5$, with exact zeros drawn at $-17$. The Gaussian bars should sit at the rounding floor and the Fock bars clearly below $-6$.

<!-- core:R17 -->
**Title:** Structural and causal-flow agreement with Graphix

**Why this experiment.** CV and qubit MBQC share a combinatorial layer: the resource graph, the input and output sets, the measurement order, which outcome each later command depends on, and, for qubits, causal flow. Graphix is an established, independent qubit-MBQC package, so it can serve as an external reference for that shared layer. A mismatch there would mean that PhotoGraphiQ's graph or dependency bookkeeping departs from standard MBQC conventions. The comparison is deliberately limited to structure. Graphix has no quadratures, amplitudes or Wigner functions, so R17 is never evidence for CV numerical correctness. It also checks that a qubit flow algorithm is never accepted as a CV determinism certificate.

### Theory
A pattern defines (i) the open graph $(G, I, O)$, (ii) a sequence of measured nodes, and (iii) dependency domains, which state that a measurement angle or correction on node $j$ depends on the outcome of node $i$. In Graphix these are the `s_domain` of `M` and the `domain` of `X`/`Z`. In PhotoGraphiQ, $\mathrm{Outcome}(i)$ appears in a later command, which creates an edge in `Pattern.dependencies()` (R12).

A qubit causal flow (Danos and Kashefi) is a map $f: O^c \to I^c$ with a strict partial order $\prec$ such that:

- $i \sim f(i)$;
- $i \prec f(i)$;
- $i \prec k$ for every neighbour $k \ne i$ of $f(i)$.

The X-correction support of a measured node $i$ is $\{f(i)\}$. Causal flows need not be unique. For example, a star centre may target any leaf. Two implementations can therefore return different but valid flows, and the correct cross-check is to verify one implementation's certificate with the other's verifier. A necessary condition for causal flow is $|I| \le |O|$, so the triangle $K_3$ with two inputs and one output has none.

### What the code does
- **Requirement:** if `graphix` is not installed the script prints `SKIPPED` and exits with status 0.
- **Structural cases:** line graphs with $n = 3, 5$ and star graphs with 3 and 5 leaves, all with input 0.
  - Line: the CV pattern measures node 0 with $\hat q$ and each inner node $i$ with angle $0.2 + 0.1i + \mathrm{Outcome}(i-1)$. It then displaces the output by $q = \mathrm{Outcome}(n-2)$, and by $p = \mathrm{Outcome}(0)$ when $n > 2$. The hand-matched Graphix pattern uses `M(i, s_domain={i-1})`, `X(n-1, domain={n-2})` and `Z(n-1, domain={0})`.
  - Star: the hub is measured and every leaf is displaced by $\mathrm{Outcome}(0)$, mirrored by `X(leaf, domain={0})`.
- **Structural checks:**
  - `networkx.is_isomorphic` and canonical edge-set equality between Graphix's `to_opengraph()` and `CVGraph.network`;
  - equal input and output tuples;
  - identical measurement order;
  - every Graphix `M.s_domain` and `X.domain` entry matches an edge in the PhotoGraphiQ dependency DAG, with at least one such check per case. `Z` domains are not tested separately.
- **Flow comparison:** five qubit open graphs (`Model.QUBIT`, XY measurements): path graphs with 3 and 5 nodes, stars with 3 and 5 leaves, and $K_3$ with inputs $(0, 1)$. For each, `flows.analyze(og, "causal")` is compared with Graphix's `to_causalflow_or_none()`:
  - existence must agree;
  - the X-correction supports must agree, or else Graphix's flow must pass PhotoGraphiQ's independent `verify_causal_flow`;
  - the same graph typed `Model.CV_GAUSSIAN` must give `unsupported` under `"causal"`.
- **Outputs:** `R17_graphix_structure.csv/.json` (with the flow table embedded), `R17_graphix_structure_flows.csv` and metadata. No figure is produced.

### Expected outcome
All four structural cases pass every check: isomorphic graphs, identical edges, inputs/outputs and measurement order, and all dependency domains present in the DAG. In the flow table, the lines and stars have a causal flow in both packages. Where the X supports differ (stars), Graphix's flow passes PhotoGraphiQ's verifier. $K_3$ with two inputs has no flow in either package. Every CV-typed graph is rejected as `unsupported`. Any failure is written to `R17_graphix_structure_FAILURES.csv` and raises `AssertionError`.

<!-- core:R18 -->
**Title:** Exact Gaussian engines and finite-cutoff Fock engines on random circuits

**Why this experiment.** The package offers several engines for the same physical program. Three are exact Gaussian engines (NumPy `gaussian`, `piquasso`, C++ `native-gaussian`) and four are truncated Fock engines (`native-fock`, `piquasso-fock`, `native-mixed-fock`, `piquasso-mixed-fock`). Users switch between them freely, so the paper must show that the exact engines agree exactly on arbitrary Gaussian circuits. It must also show that the Fock engines approximate them to a declared, truncation-limited accuracy. The two claims are kept separate. Comparing a truncated simulation against an exact tolerance would produce false failures, while a loose tolerance applied to exact engines would hide real bugs.

### Theory
A non-adaptive Gaussian circuit acts on $(\mu, V)$ ($\hbar = 2$) by the composed symplectic-affine map, and every exact engine must reproduce it to rounding. The mean photon number per mode is
$$\langle \hat n_k\rangle = \frac{V_{q_kq_k} + V_{p_kp_k} + \mu_{q_k}^2 + \mu_{p_k}^2 - 2}{4}.$$
A Fock engine at total-photon cutoff $c$ keeps $|n_1, n_2\rangle$ with $n_1 + n_2 \le c - 1$. Its observables differ from the exact Gaussian values by an amount controlled by the population that leaks beyond the cutoff, which is reported as the minimum retained norm. For low-energy two-mode states (coherent amplitudes $|\alpha| \le 0.25\sqrt2$, squeezing $|r| \le 0.4$) at $c = 14$, the leaked weight is tiny. When the retained norm stays within $10^{-6}$ of 1, the photon-number and $\langle q\rangle$ errors should be far below $10^{-3}$.

### What the code does
- **Workloads:** `N_RANDOM = 15` random two-mode patterns (seed 20260913), each with 4 gates drawn uniformly from:
  - rotation, $\phi \in [-\pi, \pi]$;
  - squeezing, $r \in [-0.4, 0.4]$;
  - beamsplitter, $\theta \in [-\pi/2, \pi/2]$;
  - CZ, $g \in [-1, 1]$;
  - displacement, $q, p \in [-0.3, 0.3]$.

  The inputs are coherent with real and imaginary parts in $[-0.25, 0.25]$.
- **Exact comparison:** `piquasso` and `native-gaussian` are each compared with NumPy `gaussian` on mean and covariance error. The photon-number error per mode is also checked for `piquasso`.
- **Fock comparison:** `native-fock`, `piquasso-fock` and `native-mixed-fock` at `FOCK_CUTOFF = 14`, plus the historical `piquasso-mixed-fock` (columns `mixed_fock_*`). Each reports $\max_k |\langle \hat n_k\rangle - \langle \hat n_k\rangle_G|$, $\max_k |\langle q_k\rangle - \langle q_k\rangle_G|$ and the minimum retained norm. A run stopped by the norm guard is caught by `common.safe_cutoff_run` and recorded with its error message, not dropped.
- **Outputs:** `R18_gaussian_backends.csv/.json` (seed, cutoff, engine list, tolerances), metadata, and the figure `R18_gaussian_backends`.

### Expected outcome
The exact engines agree with each other to rounding, about $10^{-15}$ or exactly zero. The Fock engines should show photon-number and $\langle q\rangle$ errors well below $10^{-3}$, consistent with the small truncation tail, and should normally never trip the norm guard on these low-energy inputs.

Two acceptance criteria are declared in the code:

1. **Exact:** every exact-engine mean and covariance error (both engines) and every `piquasso` photon-number error is at most $10^{-8}$.
2. **Fock:** for `native-fock`, `piquasso-fock` and `native-mixed-fock`, whenever the minimum retained norm is at least $1 - 10^{-6}$, $\max(\text{photon error}, q\text{ error}) \le 10^{-3}$. Cells below that norm are reported but not asserted.

`piquasso-mixed-fock` is reported but not asserted. Any failure is written to `R18_gaussian_backends_FAILURES.csv`.

The figure has two panels:

- **Left:** $\log_{10}$ covariance error relative to NumPy for each sample, for `piquasso` (circles) and `native-gaussian` (crosses). The y range ends at $-10$ and exact zeros are drawn at $-17$. All points should lie near the floor.
- **Right:** $\log_{10}$ photon-number error at $c = 14$ for the four Fock engines, slightly offset per engine. Each legend entry gives the number of successful runs, and a dotted line marks the $10^{-3}$ tolerance. All points should lie well below that line.

<!-- core:R19 -->
**Title:** Heralded photon subtraction and the exact binomial law

**Why this experiment.** Photon subtraction is the basic non-Gaussian operation used to make cat-like and other non-classical resources. PhotoGraphiQ models it physically, as a beamsplitter tap into a vacuum ancilla followed by photon-number-resolving detection. It does not use the unphysical normalised annihilation operator. R19 tests this primitive, together with fixed-outcome postselection and the reported branch probabilities, against an exact closed form. An error in the beamsplitter convention, in the branch probability, or in the conditional state would make every later non-Gaussian result, including R20, untrustworthy.

### Theory
A beamsplitter of angle $\theta$ mixes the signal mode with a vacuum ancilla. Because it conserves photon number, a Fock input splits as
$$|n\rangle|0\rangle \;\longmapsto\; \sum_{k=0}^{n} \sqrt{\binom{n}{k}}\, \cos^{n-k}\theta\, \sin^{k}\theta\; |n-k\rangle|k\rangle,$$
up to phases. Counting $k$ photons on the ancilla therefore occurs with probability
$$P(k \mid n, \theta) = \binom{n}{k} \sin^{2k}\theta\, \cos^{2(n-k)}\theta,$$
a binomial distribution with single-photon reflection probability $\sin^2\theta$. The heralded signal is exactly $|n-k\rangle$, so its conditional photon number equals $n - k$ with no spread. Written as an operator, the branch is the Kraus map
$$M_k = \frac{\sin^k\theta}{\sqrt{k!}}\, \cos^{\hat n}\theta\; \hat a^k.$$
This is an attenuated, lossy subtraction, not the ideal normalised $\hat a$.

### What the code does
- **System under test:** `pg.non_gaussian.photon_subtraction(theta)`. It prepares the ancilla `tap` in $|0\rangle$, applies `BeamSplitter(in, tap, theta)` and then a `PhotonNumber` measurement with key `count`.
- **Execution:** each branch runs with `measurement_outcomes={"count": k}`, input `FockInput.number(n)` and total-photon cutoff 8. The main engine is `native-fock`; `piquasso-fock` and the density-matrix engine `native-mixed-fock` serve as independent references.
- **Oracle:** the binomial law, evaluated with `scipy.special.comb`, independently of PhotoGraphiQ.
- **Sweep:** $n \in \{1, 2, 3\}$ and 15 tap angles evenly spaced in $[0.05, \pi/2 - 0.05]$, with every $k = 0, \dots, n$. That gives $15 \times (2 + 3 + 4) = 135$ branches.
- **Metrics:** for each branch and each engine, the absolute probability error (plus the relative error for the main engine), the conditional photon-number error $|\langle \hat n\rangle - (n-k)|$, and the minimum retained Fock norm over all engines.
- **Outputs:** `R19_photon_subtraction_probability.csv/.json`, metadata, a figure caption file, and the figure `R19_photon_subtraction_probability`.

### Expected outcome
The input never exceeds 3 photons, far below the cutoff, so nothing is truncated: retained norms equal 1, and all deviations should be at double-precision rounding level. The acceptance criterion is that, for all 135 branches and all three engines, the absolute probability error is at most $10^{-9}$ and the conditional photon-number error is at most $10^{-9}$. Otherwise the failures go to `R19_photon_subtraction_probability_FAILURES.csv` and the run aborts.

The figure has three panels sharing the y axis, one for each $n = 1, 2, 3$. Each panel plots herald probability against tap angle $\theta$. For every $k$ it shows the analytic binomial curve as a line and the `native-fock` branch probabilities as circles. The circles should sit on the lines. At small $\theta$, $k = 0$ dominates; as $\theta \to \pi/2$, $k = n$ dominates; intermediate $k$ peak in between. Each reported value is a postselected branch probability, not a deterministic operation.

<!-- core:R20 -->
**Title:** Photon subtraction on squeezed vacuum: even-to-odd parity flip

**Why this experiment.** R19 verifies subtraction on number states. R20 applies the same physical primitive to a Gaussian, non-classical input and checks the standard signature of the squeezed-light route to optical cat states: the parity flips from even to odd and the Wigner function becomes negative. The experiment also exercises the Gaussian-to-Fock input conversion, Wigner evaluation and the `cutoff_convergence` tool on a postselected branch. It is cross-checked across three Fock engines.

### Theory
In $\hbar = 2$ units, q-squeezed vacuum $\hat S|0\rangle$ with squeezing $r$ has support only on even photon numbers,
$$|\psi_r\rangle = \frac{1}{\sqrt{\cosh r}} \sum_{m} \frac{\sqrt{(2m)!}}{2^m m!}\, (\pm\tanh r)^m\, |2m\rangle,$$
so its parity is $\langle(-1)^{\hat n}\rangle = +1$ and its mean photon number is $\sinh^2 r$. The physical tap herald $k = 1$ applies the Kraus operator $M_1 = \sin\theta\, \cos^{\hat n}\theta\, \hat a$ (see R19). Since $\hat a$ maps even numbers to odd ones and $\cos^{\hat n}\theta$ is diagonal, the heralded state has only odd support and parity $-1$, up to truncation. More precisely, $\hat a|\psi_r\rangle \propto \hat S(r)|1\rangle$, and the attenuation $\cos^{\hat n}\theta$ only rescales $\tanh r \to \cos^2\theta \tanh r$. The output is therefore a squeezed single photon $\hat S(r')|1\rangle$ with
$$\tanh r' = \cos^2\theta\, \tanh r,\qquad \langle \hat n\rangle_{\rm after} = 1 + 3\sinh^2 r'.$$
Its Wigner function is negative at the origin, $W(0,0) < 0$, as for any odd state. For small $\theta$ the herald probability is approximately $\sin^2\theta\, \sinh^2 r$, so it is small and the result is a postselected branch.

### What the code does
- **Input and protocol:** `GaussianInput.squeezed(-0.6)`, i.e. q-squeezed vacuum with $r = 0.6$, passed through `non_gaussian.photon_subtraction(0.15)` with `measurement_outcomes={"count": 1}`.
- **Main engine:** `native-fock` at total-photon cutoff 24. The before state is the same input passed through an identity pattern.
- **Physics check:** the parity before must exceed $+0.5$ and the parity after must be below $-0.5$. The script also reports the mean photon number before and after, the herald probability and the full photon-number distributions.
- **Engine cross-check:** the same branch runs on `piquasso-fock` and `native-mixed-fock`. Their herald probability, parity, mean photon number and total-variation distance of the photon-number distribution from `native-fock` must all be within `REFERENCE_TOLERANCE` $= 10^{-6}$. The only expected difference is the Gaussian-input preparation route (exact Bargmann column versus Piquasso's decomposition).
- **Wigner functions:** evaluated on a $121 \times 121$ grid over $[-5, 5]^2$. The minima before and after are recorded.
- **Cutoff convergence:** `pg.cutoff_convergence` on `native-fock` for the fixed branch `count = 1` at cutoffs $\{12, 16, 20, 28, 36\}$. It records the norm, the minimum retained norm, the boundary population, and the fidelity and trace distance to the previous cutoff.
- **Outputs:** `R20_subtracted_squeezed_state_summary.json`, `_distribution.csv`, `_engines.csv`, `_convergence.csv`, metadata, a caption file, and the figure `R20_subtracted_squeezed_state`.

### Expected outcome
The parity should go from about $+1$ (even support only) to about $-1$ (odd support only). The mean photon number should rise from $\sinh^2(0.6) \approx 0.40$ to $1 + 3\sinh^2 r'$. The herald probability should be of order $10^{-2}$. The Wigner minimum should be non-negative up to rounding before the subtraction and clearly negative after. The reference engines should agree with `native-fock` to far better than $10^{-6}$. In the convergence table, the fidelity to the previous cutoff should approach 1 and the trace distance should approach 0 as the cutoff grows, with the retained norm close to 1 throughout.

Acceptance requires both conditions: the parity flip (before $> 0.5$, after $< -0.5$) and the engine agreement within $10^{-6}$. Wigner negativity and convergence are reported, not asserted.

The figure has three panels:

- **Left:** photon-number distributions before and after the subtraction, as bars. Even bars before, odd bars after.
- **Centre:** Wigner function of the squeezed vacuum, a positive ellipse.
- **Right:** Wigner function of the subtracted state, showing a negative central region.

The colour map is symmetric about zero.

<!-- core:R21 -->
**Title:** Finite-Fock cat states against an independent coherent-state expansion

**Why this experiment.** Cat states $|\alpha\rangle \pm |-\alpha\rangle$ are standard non-Gaussian resources in CV quantum computing and error correction, and PhotoGraphiQ provides them as `CatResource` (built on `FockInput.cat`). R21 checks the prepared amplitudes, not just a few moments, against an expansion written independently of the library. It checks three Fock engines in the same way. A sign or normalisation error, a wrong parity, or a truncation problem at larger $\alpha$ would affect every protocol that injects a cat resource.

### Theory
A coherent state has the Fock expansion
$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n} \frac{\alpha^n}{\sqrt{n!}}\, |n\rangle,$$
so for real $\alpha$ the cat state of parity $s = \pm 1$ is
$$|\mathrm{cat}_s(\alpha)\rangle \propto |\alpha\rangle + s\,|-\alpha\rangle = e^{-\alpha^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}} \big(1 + s(-1)^n\big)\, |n\rangle.$$
The factor $1 + s(-1)^n$ removes odd $n$ for the even cat and even $n$ for the odd cat. The parity is therefore exactly $\langle(-1)^{\hat n}\rangle = s$, and the mean photon numbers are $\alpha^2 \tanh\alpha^2$ (even) and $\alpha^2 \coth\alpha^2$ (odd). The reference evaluates the coefficients in log space ($n\ln\alpha - \tfrac12\ln n! - \alpha^2/2$, using `gammaln`) and normalises them over the retained basis $n = 0, \dots, c-1$. The fidelity with the library state $\psi$ (or $\rho$) is
$$F = |\langle \mathrm{cat}_s | \psi\rangle|^2 \qquad \text{or} \qquad F = \langle \mathrm{cat}_s|\rho|\mathrm{cat}_s\rangle.$$
For $\alpha \le 2$ the Poisson-like weight above $n = 31$ is negligible, so truncation should not limit the fidelity. In $\hbar = 2$ units the Wigner function of a cat shows two Gaussian lobes at $q = \pm 2\alpha$, with interference fringes along $p$ between them. The sign of the fringes at the origin follows the parity.

### What the code does
- **System under test:** a `Pattern` that prepares `CatResource(alpha, parity)` on mode 0, executed on `native-fock` (main engine) and on `piquasso-fock` and `native-mixed-fock`, all at total-photon cutoff 32.
- **Sweep:** $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$, each with both parities, which gives 8 states.
- **Oracle:** the independent NumPy expansion `analytic_cat`, which does not call `FockInput.cat`.
- **Metrics:**
  - fidelity to the oracle for each engine (vector overlap for `native-fock`, $\langle \mathrm{cat}|\rho|\mathrm{cat}\rangle$ for the references);
  - measured parity and its sign;
  - mean photon number, norm, retained norm and boundary population;
  - parity and photon-number differences between each reference engine and `native-fock`.
- **Outputs:** `R21_cat_state.csv/.json`, metadata, a caption file (with the Wigner minima), and the figure `R21_cat_state`.

### Expected outcome
All fidelities should equal 1 to near rounding precision. Parities should be essentially $\pm 1$, with the declared sign. Mean photon numbers should follow $\alpha^2 \tanh\alpha^2$ and $\alpha^2 \coth\alpha^2$, which are nearly equal at $\alpha = 2$ and very different at $\alpha = 0.5$. The boundary population should stay negligible even at $\alpha = 2$.

Acceptance requires all three of the following:

1. The parity sign matches the declared parity for all 8 states.
2. The fidelity of every engine is at least $1 - 10^{-6}$.
3. The reference engines agree with `native-fock` on parity and mean photon number to `ENGINE_AGREEMENT_TOLERANCE` $= 10^{-9}$.

Failures are written to `R21_cat_state_FAILURES.csv`.

The figure has two panels: the Wigner functions of the even and odd cats at $\alpha = 1.5$ on $[-6, 6]^2$, with a colour scale symmetric about zero. Both show two positive lobes near $q = \pm 3$ and a fringe pattern between them, with a positive central fringe for the even cat and a negative one for the odd cat. The negative regions show that the states are non-Gaussian.

<!-- core:R22 -->
**Title:** Native cubic phase gate against a dense matrix-exponential reference

**Why this experiment.** Adding the cubic phase gate $\exp(i\gamma \hat q^3/6)$ to the Gaussian gates gives a universal CV gate set, and it is the non-Gaussian element that cubic injection (R23, R34) is meant to approximate. Unlike Gaussian gates, it has no finite-dimensional exact form: every Fock-basis simulation applies some finite-cutoff version of it. R22 checks that the engines apply exactly the finite-cutoff operator they claim to apply. Each engine is compared with an independent dense reference at the same cutoff. This separates implementation errors from truncation effects, whose convergence is studied in R34.

### Theory
In $\hbar = 2$ units, with $\hat q = \hat a + \hat a^\dagger$, let $a_{jk} = \sqrt{k}\, \delta_{j, k-1}$ for $j, k = 0, \dots, c-1$ and $Q = a + a^T$. The truncated-generator definition of the gate is
$$U_c(\gamma) = \exp\!\Big(\frac{i\gamma}{6}\, Q^3\Big),$$
which is unitary on the $c$-dimensional space and equals the Gauss-Hermite (DVR) quadrature of $e^{i\gamma q^3/6}$ at the cutoff nodes. It never reports leakage. `native-fock` (by default, `cubic_phase="truncated_generator"`) and `piquasso-fock` both claim to implement exactly $U_c(\gamma)$. Their output should therefore equal $U_c(\gamma)|\psi_{\rm in}\rangle$ to rounding, at every cutoff.

The alternative native definition `cubic_phase="projected"` uses the exact projected matrix elements $\langle m|e^{i\gamma\hat q^3/6}|n\rangle$ for $m, n < c$, computed by converged quadrature. That operator, $P V P$, is not unitary, and leakage above the cutoff appears in its retained norm. It differs from $U_c$ at finite $c$, and both converge to the exact gate as $c \to \infty$. Their mutual infidelity is thus a truncation diagnostic, not an error. The fidelity used throughout is $F = |\langle \psi_{\rm ref}|\psi\rangle|^2$ with $\psi_{\rm ref} = U_c\psi_{\rm in}/\lVert U_c\psi_{\rm in}\rVert$.

### What the code does
- **System under test:** a `Pattern` applying `CubicPhase(0, gamma)` to the coherent input $\alpha = 0.3 + 0.1i$ on three engines: `native-fock`, `piquasso-fock`, and `NativeFockBackend(cutoff, cubic_phase="projected")` (labelled `native-fock-projected`).
- **Oracle:** `independent_cubic_unitary`, which builds the dense ladder matrix in NumPy and applies `scipy.linalg.expm`. It is applied to the truncated input vector that each engine itself produces at that cutoff, so both sides start from the same amplitudes.
- **Sweep:** $\gamma \in \{0.02, 0.05, 0.1, 0.2\}$ crossed with cutoffs $c \in \{16, 24, 32, 48\}$ and the three engines, which gives 48 cells.
- **Metrics:** fidelity to the reference, the phase-aligned state-vector error $\lVert\psi - e^{i\varphi}\psi_{\rm ref}\rVert$, the estimate $\sqrt{1 - F}$, and the final and minimum retained norms.
- **Outputs:** `R22_cubic_phase_direct.csv/.json` (records which engines are asserted and which are only reported), metadata, and the figure `R22_cubic_phase_direct`.

### Expected outcome
For `native-fock` and `piquasso-fock`, $1 - F$ should be at the rounding floor, about $10^{-16}$ to $10^{-14}$, for every $(\gamma, c)$, with retained norm 1. The two definitions are the same finite matrix. For `native-fock-projected`, $1 - F$ should be clearly nonzero at the smallest cutoff and larger $\gamma$, and should fall rapidly with $c$. This reflects the different finite-cutoff operator and its leakage, not a bug. Retained norms slightly below 1 should be reported where leakage occurs.

The acceptance criterion is $F \ge 1 - 10^{-12}$ (`FIDELITY_TOLERANCE`) for every cell of the two asserted engines. The projected engine is not asserted. Failures go to `R22_cubic_phase_direct_FAILURES.csv`.

The figure has one panel: a semilog plot of $1 - F$ against Fock cutoff. Colours encode $\gamma$. Solid circles show `native-fock`, dotted crosses `piquasso-fock` and dashed squares `native-fock-projected`. Values are clipped at $10^{-16}$, the rounding floor, and the y axis spans $5\times10^{-17}$ to $10^{-4}$. The first two engines should lie flat on the floor, and the projected curves should decrease with cutoff, more slowly for larger $\gamma$.

<!-- core:R23 -->
**Title:** Finite-resource cubic-phase injection in MBQC

**Why this experiment.** In measurement-based CV computing, the cubic gate is not applied directly. A non-Gaussian resource is consumed by gate teleportation, followed by nonlinear feed-forward. The paper must show that PhotoGraphiQ's `cubic_injection` implements this protocol correctly. It must also make clear that, with a finite-energy resource, the protocol gives a conditional filter times the cubic phase rather than an exact unitary. R23 measures how close the injected gate comes to the ideal gate as the resource squeezing grows, and how that depends on cutoff and engine. It also records the regime where the finite Fock basis can no longer hold the resource.

### Theory
In $\hbar = 2$ units ($[q, p] = 2i$), the resource is $|\phi\rangle = CP(\gamma)\, \hat S(-r)|0\rangle$, with $CP(\gamma) = e^{i\gamma\hat q^3/6}$. Its wavefunction is $\phi(x) \propto e^{i\gamma x^3/6} e^{-x^2/(4e^{2r})}$, a q-wide Gaussian whose envelope widens with $r$. `resource_injection` applies an inverse SUM between the input and the ancilla, then a homodyne $q$ measurement of the ancilla with outcome $m$. This maps
$$\psi(q) \;\longmapsto\; \psi(q)\, \phi(q + m).$$
Expanding $(q + m)^3$, the corrections `QuadraticPhase(-2 gamma m)` $= e^{-i\gamma m q^2/2}$ (since $QP(s) = e^{isq^2/4}$) and the momentum shift $-\gamma m^2$ cancel the $q^2$ and $q$ terms. What remains is
$$\psi_{\rm out}(q) \propto e^{i\gamma q^3/6}\, e^{-(q+m)^2/(4e^{2r})}\, \psi(q),$$
up to a global phase. Fixing $m = 0$ makes both corrections vanish and isolates the Gaussian filter $G(q) = e^{-\varepsilon q^2/4}$ with $\varepsilon = e^{-2r}$. Since $G$ commutes with $CP$, in the infinite-cutoff limit the fidelity to the ideal output depends only on $|\psi|^2$:
$$F = \frac{\langle G\rangle^2}{\langle G^2\rangle} \approx 1 - \frac{\varepsilon^2}{16}\, \mathrm{Var}(q^2) = 1 - \frac{e^{-4r}}{8}\, (1 + 2q_0^2),$$
for a coherent input with $|\psi(q)|^2 = \mathcal{N}(q_0, 1)$. Here $q_0 = 2\,\mathrm{Re}\,\alpha = 0.4$. The fidelity therefore rises monotonically towards 1 with $r$. The ancilla's photon number grows as $\sinh^2 r$, so a larger $r$ needs a larger cutoff, and at fixed cutoff the retained norm can leave the guard band.

### What the code does
- **System under test:** `pg.non_gaussian.cubic_injection(0.15, squeezing=r)` with input `GaussianInput.coherent(0.2)` and `measurement_outcomes={"m": 0.0}` (postselected branch). It runs on `native-fock` (main engine) and `piquasso-fock`. The two engines treat the two-mode CZ differently at finite cutoff: exact projected Gaussian elements versus Euler factors truncated after each step.
- **Oracle:** the same NumPy/SciPy $\exp(i\gamma Q^3/6)$ as in R22, applied to each engine's own truncated input vector and normalised.
- **Sweep:** $r \in \{0.1, 0.3, 0.6, 1.0, 1.5, 2.0\}$ crossed with cutoffs $\{24, 32, 48\}$ and the two engines, which gives 36 cells. Norm-guard failures are caught by `safe_cutoff_run` and recorded as `ok = False` with the error message.
- **Metrics:** fidelity to the ideal gate, $\sqrt{1 - F}$, the outcome probability density at $m = 0$, the retained norms, and the native-versus-Piquasso output fidelity where both engines succeed.
- **Outputs:** `R23_cubic_injection.csv/.json` (with norm-guard failure counts per engine), metadata, a caption file, and the figure `R23_cubic_injection`.

### Expected outcome
In successful cells, the fidelity should rise with $r$ towards 1, approximately following the filter formula above, until the cutoff stops representing the wide ancilla. Cells with large $r$ at small cutoff are expected to hit the norm guard, so the number of successful cells grows with cutoff. Where both engines succeed, their outputs should agree closely.

The criteria are declared before evaluation:

1. At the largest cutoff (48), the successful `native-fock` fidelities must not decrease with $r$ by more than `MONOTONE_SLACK` $= 10^{-6}$ between consecutive values.
2. Wherever both engines succeed, the native-versus-Piquasso output fidelity must be at least `ENGINE_FIDELITY_FLOOR` $= 1 - 10^{-3}$.

Failures go to `R23_cubic_injection_FAILURES.csv`.

The figure has one panel: fidelity to the ideal $CP(0.15)$ against resource squeezing $r$. Solid circles show `native-fock` and dashed crosses `piquasso-fock`, with one colour per cutoff. Each legend entry gives the number of successful cells out of 6, and guard-stopped cells are omitted. This is a postselected, finite-energy approximation, and no claim about a deterministic or fault-tolerant cubic gate is made.

<!-- core:R24 -->
**Title:** Native Kerr gate and the exact Fock-diagonal phase rule

**Why this experiment.** The Kerr interaction $\exp(i\kappa\hat n^2)$ is the standard non-Gaussian phase gate of CV quantum optics. It is the target of the Lie-product synthesis in R25. Because it is diagonal in the number basis, it is one of the few non-Gaussian gates with an exact finite-dimensional action. A Fock engine can therefore be checked against it with no truncation ambiguity. A sign error in $\kappa$, a confusion between $\hat n^2$ and $\hat n(\hat n - 1)$, or a wrong global or relative phase would show up immediately. R24 checks all three Fock engines on a superposition chosen so that relative phases matter.

### Theory
Since $\hat n^2 |n\rangle = n^2 |n\rangle$,
$$e^{i\kappa\hat n^2}\, |n\rangle = e^{i\kappa n^2}\, |n\rangle,$$
so for any $|\psi\rangle = \sum_n c_n |n\rangle$ the exact output is $\sum_n c_n e^{i\kappa n^2} |n\rangle$. For the test state $(|0\rangle + |1\rangle + |2\rangle)/\sqrt3$ the output is
$$|e(\kappa)\rangle = \tfrac{1}{\sqrt3}\big(|0\rangle + e^{i\kappa}|1\rangle + e^{4i\kappa}|2\rangle\big).$$
The relative phases $\kappa$ and $4\kappa$ distinguish $\hat n^2$ from other diagonal generators. For example, $\hat n(\hat n - 1)$ would give phases $0$ and $2\kappa$, and $\hat n$ would give $\kappa$ and $2\kappa$. The state lies in a 3-dimensional subspace, far below the cutoff 8, and the gate does not couple different $n$. There is therefore no truncation error at all: every deviation must be rounding or a gate error.

For pure-state engines the script compares the full vector without phase alignment, $\lVert \psi - e\rVert$, so even the global phase must match. For the density-matrix engine it uses $F = \langle e|\rho|e\rangle$ and the Frobenius distance $\lVert \rho - |e\rangle\langle e|\rVert_F$.

### What the code does
- **System under test:** a `Pattern` applying `Kerr(0, kappa)` to `FockInput((1, 1, 1)/sqrt(3))` at cutoff 8, on `native-fock` (main engine, where Kerr is an exact diagonal phase), `piquasso-fock` and `native-mixed-fock`.
- **Oracle:** the phase rule above, evaluated amplitude by amplitude in NumPy without calling the gate.
- **Sweep:** 17 values of $\kappa$ evenly spaced in $[-2, 2]$ for each of the three engines, which gives 51 cases.
- **Metrics:** fidelity to the exact phase rule, and the state-vector error (or the density-matrix Frobenius error for the mixed engine).
- **Outputs:** `R24_kerr_direct.csv/.json` (input amplitudes, cutoff, engines), metadata, and the figure `R24_kerr_direct`.

### Expected outcome
All three engines should reproduce the phase rule to double-precision rounding, with errors of about $10^{-16}$ to $10^{-15}$ that do not depend on $\kappa$ beyond rounding. The acceptance criterion requires, for every case, fidelity $\ge 1 - 10^{-10}$ and state-vector (or density-matrix) error $\le 10^{-8}$. Otherwise the failures go to `R24_kerr_direct_FAILURES.csv` and the run aborts.

The figure has one panel: a semilog plot of the state-vector error against $\kappa$ for the three engines (circles with a solid line for `native-fock`, crosses with a dashed line for `piquasso-fock`, squares with a dotted line for `native-mixed-fock`). Values are clipped at $10^{-17}$. All curves should lie flat near the rounding floor, with no trend in $\kappa$.

<!-- core:R25 -->
**Title:** Convergence of Lie-product Kerr synthesis from Gaussian and cubic primitives

**Why this experiment.** The Kerr gate $e^{i\kappa\hat n^2}$ is native on the Fock engines (R24), but a measurement-based compiler only has Gaussian steps and cubic-phase injection at its disposal. `photographiq.synthesis.synthesize_kerr` therefore approximates the Kerr gate by a product formula over rotation, quadratic-phase and cubic-phase primitives. The paper presents this as an approximate, refinable synthesis; R25 checks that the approximation error actually decreases as the number of product-formula steps grows. To measure the synthesis error alone, the synthesized gates are executed directly as physical Fock-engine commands, without compiling to an MBQC pattern, so finite-squeezing injection noise (R23) does not enter. Without this test a sign or ordering error in the synthesis could go unnoticed behind the separate finite-resource errors.

### Theory

Conventions: $\hbar = 2$, $[\hat q,\hat p] = 2i$, $\hat x_\theta = \hat q\cos\theta + \hat p\sin\theta$. The primitives are `Rotate`, `QuadraticPhase(s)` $= e^{is\hat q^2/4}$ and `CubicPhase(gamma)` $= e^{i\gamma\hat q^3/6}$, so a rotated quadrature power $e^{ic\hat x_\theta^d}$ ($d = 2, 3$) costs one gate between two rotations. The synthesis starts from the exact quartic decomposition

$$\hat n^2 = \frac{1}{24}\sum_{\theta\in\{0,\frac{\pi}{2},\frac{\pi}{4},-\frac{\pi}{4}\}} \hat x_\theta^4 \;-\; \frac{\hat q^2 + \hat p^2}{4},$$

which holds exactly as an operator identity at $\hbar = 2$ (no additive constant). With $N$ steps, each of the six terms is applied with strength $c/N$ (first-order Lie product). The cubic primitives cannot give a quartic directly, so each $e^{is\hat q^4}$ is built from a group commutator of two cubic generators, $A = \hat q^3$ and a Weyl-ordered $B \propto \hat q^2\hat p$, using $[\hat q^3, B] = 6i\hat q^4$:

$$e^{-itB}e^{-isA}e^{itB}e^{isA} = e^{6ist\,\hat q^4} + O(t^3),\qquad s = |t| = \sqrt{|c|/(6N)}.$$

$B$ is itself split symmetrically into three rotated cubics. Per step the commutator error is $O(t^3) = O(N^{-3/2})$, so the accumulated amplitude error is $O(N\cdot N^{-3/2}) = O(N^{-1/2})$; this is the order 0.5 that `SynthesisReport.order` records. Since $1 - F \approx \lVert\Delta\psi\rVert^2$ for small deviations, the infidelity is expected to fall roughly as $1/N$. The reference is the exact phase rule $c_n \mapsto c_n e^{i\kappa n^2}$, which needs no simulation.

### What the code does

- System under test: `synthesize_kerr(kappa=0.05, steps=N)`; each gate of the returned `Circuit` is translated to `pg.Rotate`, `pg.QuadraticPhase`, `pg.CubicPhase` or `pg.Displace` and appended to a one-mode `Pattern`, which is executed by `pg.simulate` on `native-fock` (main) and `piquasso-fock` (historical reference), cutoff 16.
- Input: $(|0\rangle + |1\rangle + |2\rangle)/\sqrt3$. Oracle: $\sum_n c_n e^{i\kappa n^2}|n\rangle$, written directly with NumPy; it is independent of both engines and of the synthesis code.
- Sweep: $N \in \{1, 2, 4, 8, 16, 32, 64\}$ on both engines.
- Metrics: fidelity $|\langle\psi_{\rm exact}|\psi_{\rm synth}\rangle|^2$, $\sqrt{1-F}$, primitive count, reported order, retained norms, wall time (provisional), and the fidelity between the two engines' outputs at each $N$. The engines realise `QuadraticPhase` differently at a finite cutoff (exact projected matrix elements vs Piquasso's truncated decomposition), so their results are reported side by side.
- Output: `R25_kerr_synthesis` (CSV/JSON with metadata) and figure `R25_kerr_synthesis` (PDF/PNG/SVG).

### Expected outcome

The infidelity should decrease steadily with $N$, roughly as $N^{-1}$ on log-log axes (amplitude error $\sim N^{-1/2}$), while the primitive count grows linearly in $N$: more accuracy costs proportionally more cubic-phase resources. Both engines should give nearly identical curves, and their mutual output fidelity should be close to 1. The retained norm should stay near 1 at cutoff 16 for this low-photon input, so the trend is a synthesis effect, not truncation. Timings are machine-dependent and labelled provisional. Declared acceptance criterion: on each engine, the infidelity at $N = 64$ must be smaller than at $N = 1$; otherwise a FAILURES CSV is written and the script raises. Faster-than-expected or slower convergence is reported, not asserted.

Figure: a single log-log panel of $1 - F$ versus synthesis steps, `native-fock` (circles, solid) and `piquasso-fock` (crosses, dashed), $\kappa = 0.05$, direct physical execution.

<!-- core:R26 -->
**Title:** Characterisation of finite-energy GKP resources in a truncated Fock space

**Why this experiment.** PhotoGraphiQ provides finite-energy square-lattice GKP codewords (`GKPResource`) as input states for encoded experiments (R27, R28). Before they are used, the paper must show what these states are: that their Fock projection reproduces the intended Gaussian-comb wavefunction, how much weight the projection discards at a given cutoff, how far the stabilizer expectations are from the ideal value 1, and how strongly the two codewords overlap. R26 provides these numbers and makes no fault-tolerance or threshold claim. It also checks that the two Fock engines agree on these observables, so that later GKP results do not depend on the engine.

### Theory

Conventions: $\hbar = 2$, $[\hat q,\hat p] = 2i$, lattice spacing $L = \sqrt{2\pi}$. The finite-energy codeword $|\mu_L\rangle$, $\mu\in\{0,1\}$, has the real wavefunction

$$\psi_\mu(q) \propto \sum_{k=-8}^{8} \exp\!\Big[-\frac{(q - c_k)^2}{4\sigma^2} - \frac{c_k^2\Delta^2}{4}\Big],\qquad c_k = L(2k + \mu),$$

so each tooth has probability standard deviation $\sigma$ (`peak_width`) and the teeth are weighted by a Gaussian envelope set by $\Delta$ (`envelope`, an inverse width). `GKPResource.project(cutoff)` computes $c_n = \int \psi(q)\phi_n(q)\,dq$ with the Hermite functions $\phi_n$ of `fock_measurements.wavefunctions` and Simpson quadrature, returns the normalised vector and the captured mass $\sum_n |c_n|^2$ before normalisation, so $1 - \text{mass}$ is the discarded weight.

The stabilizers are translations by $2L$: $\hat S_q = e^{-iL\hat p}$ shifts $q$ by $2L$ and $\hat S_p = e^{iL\hat q}$ shifts $p$ by $2L$. For an ideal codeword $|\langle\hat S\rangle| = 1$; finite peak width and envelope reduce it. `gkp.stabilizers` evaluates $\mathrm{Tr}[\rho\,\hat S]$ with displacement matrices built from padded ladder operators and cropped to the cutoff. `GKPCode.gram` is the $2\times2$ overlap matrix of the normalised projections; $|\langle 0_L|1_L\rangle|$ measures the residual non-orthogonality of the finite codewords. The probability densities are compared through

$$L_1 = \int \big|\,|\psi_{\rm target}(q)|^2 - |\textstyle\sum_n c_n\phi_n(q)|^2\,\big|\,dq .$$

### What the code does

- System under test: `pg.GKPResource(mu, peak_width, envelope).project(48)`, loaded through an identity pattern on `native-fock` (main) and `piquasso-fock` (reference); `pg.gkp.stabilizers`, `state.photon_number`, `pg.GKPCode(...).gram`.
- Parameter sets: $(\sigma, \Delta) = (0.3, 0.3)$ "sharp", $(0.4, 0.4)$ "default", $(0.6, 0.6)$ "fuzzy", each for $\mu = 0, 1$, at cutoff 48.
- Reconstruction: the Fock vector is mapped back to $q$ on 2000 points with the package's own Hermite functions and compared with the target density (interpolated). This is a self-consistency check of the projection, not an independent oracle.
- Engine cross-check: maximum difference of both stabilizer expectations and $\langle\hat n\rangle$ between the engines.
- Cutoff sensitivity: default parameters, $\mu = 0$, cutoffs $\{24, 32, 48, 64, 96\}$ on `native-fock`: captured mass, $|\langle\hat S_q\rangle|$, $|\langle\hat S_p\rangle|$, $\langle\hat n\rangle$.
- Output: `R26_gkp_resource` and `R26_gkp_resource_cutoff_sensitivity` (CSV/JSON with metadata), figure `R26_gkp_resource` (PDF/PNG/SVG) and a caption file.

### Expected outcome

The captured mass at cutoff 48 should be close to 1 for all three widths, and the discarded weight should fall rapidly with cutoff. Stabilizer magnitudes should be below 1 and decrease from "sharp" to "fuzzy"; the codeword overlap should grow with fuzziness, while the mean photon number (resource cost) grows as the code sharpens. The $L_1$ distance should be small wherever the mass is captured. Declared acceptance criterion: `native-fock` and `piquasso-fock` agree on both stabilizers and $\langle\hat n\rangle$ to $10^{-9}$ for every parameter set and codeword (same normalised input, identity evolution); otherwise a FAILURES CSV is written and the script raises.

Figure: (left) $q$-probability density of the default $|0_L\rangle$, target wavefunction (solid) and cutoff-48 Fock reconstruction (dashed), on $q\in[-12, 12]$; (right) discarded weight $1-\text{mass}$ versus cutoff (circles, log scale) with $|\langle\hat S_q\rangle|$ on `native-fock` on a twin axis (squares).

<!-- core:R27 -->
**Title:** GKP logical shifts, nearest-cell decoding and exact readout error probabilities

**Why this experiment.** Encoded GKP workflows in PhotoGraphiQ rest on three elementary pieces: logical Pauli operations as lattice displacements (`gkp.logical_displacement`), stabilizer estimation (`gkp.stabilizers`), and classical nearest-cell decoding of homodyne outcomes (`NearestCellDecoder`). R27 checks that the decoder implements exactly the documented rule, that logical displacements act on finite-energy codewords as the stabilizer algebra predicts, and that a $q$-homodyne readout of a codeword recovers the expected logical bit with small error. These are properties of finite-energy codewords under ideal detection, not logical error rates of a fault-tolerant scheme. Engine agreement is checked so that the numbers do not depend on the Fock engine.

### Theory

Conventions: $\hbar = 2$, $[\hat q,\hat p] = 2i$, $L = \sqrt{2\pi}$. Logical $X$ is $\hat D_q(L)$ ($q \to q + L$, generated by $e^{-iL\hat p/2}$) and logical $Z$ is $\hat D_p(L)$ ($p \to p + L$, generated by $e^{iL\hat q/2}$). The stabilizers $\hat S_q = e^{-iL\hat p}$ and $\hat S_p = e^{iL\hat q}$ are translations by $2L$. Conjugation gives

$$\hat D_q(L)^\dagger\,\hat S_p\,\hat D_q(L) = e^{iL(\hat q + L)} = e^{iL^2}\hat S_p = e^{2\pi i}\hat S_p = \hat S_p ,$$

and similarly $\hat D_p(L)^\dagger\hat S_q\hat D_p(L) = e^{-2\pi i}\hat S_q = \hat S_q$, while $\hat S_q$ commutes with $\hat D_q$ and $\hat S_p$ with $\hat D_p$. So in the infinite-dimensional theory both stabilizer expectations are exactly invariant under either logical shift, even for finite-energy states; at a finite cutoff only truncation can change them. The script records this predicted phase factor $e^{\pm iL^2} = 1$ next to the measured stabilizers.

Nearest-cell decoding of a $q$ outcome $x$: cell $j = \lfloor x/L + 1/2\rfloor$ (half-cell ties go to the upper cell), residual $x - jL \in [-L/2, L/2)$, bit $j \bmod 2$. Because $|0_L\rangle$ has peaks at even multiples of $L$ and $|1_L\rangle$ at odd multiples, the single-shot readout error probability under ideal homodyne is

$$P_{\rm err}(b) = \int |\psi(q)|^2\,\mathbf 1\big[\lfloor q/L + 1/2\rfloor \bmod 2 \ne b\big]\,dq ,$$

computed here exactly by quadrature, with no sampling. The expected bit is $\mu$ for the codeword and after $Z$, and $1 - \mu$ after $X$. A $p$-shift only multiplies $\psi(q)$ by a phase, so in the untruncated theory $Z$ leaves $P_{\rm err}$ unchanged.

### What the code does

- System under test: `pg.GKPResource(mu, peak_width=0.35, envelope=0.35).project(48)`, evolved by an identity displacement or `logical_displacement(0, "X"/"Z")` on `native-fock` (main) and `piquasso-fock` (reference), cutoff 48; `stabilizers(state)`; `pg.NearestCellDecoder().decode`.
- Decoder oracle: the floor formula above written in NumPy, evaluated at 61 outcomes evenly spaced on $[-3L, 3L]$ and compared bit by bit and residual by residual.
- Readout error: the Fock vector is mapped to $|\psi(q)|^2$ with the Hermite functions on a grid of 14001 points over $q\in[-14, 14]$ and integrated by the trapezoid rule over wrong-parity cells; the integrated total mass is reported as a grid check.
- Cases: $\mu\in\{0,1\}$ times gate $\in\{X, Z\}$; recorded: base and shifted stabilizer moduli and phases, engine differences, base and shifted $P_{\rm err}$ on both engines, captured mass, retained norms.
- Output: `R27_gkp_stabilizers` and `R27_gkp_stabilizers_decode_sweep` (CSV/JSON with metadata), figure `R27_gkp_stabilizers` (PDF/PNG/SVG) and a caption file.

### Expected outcome

The decoded bit should form a staircase alternating between 0 and 1 at the cell boundaries $(k + 1/2)L$, identical to the oracle at all 61 points. Stabilizer moduli should be close to, but below, 1 and nearly unchanged by the logical shifts (differences are truncation effects). Readout error probabilities should be small, far below 1/2, for the codewords and after both shifts. Declared acceptance criteria: decoder bit equal and residual within $10^{-9}$ of the oracle at every point; engines agree on every stabilizer modulus and on every shifted readout error probability to $10^{-6}$; every base and shifted error probability $< 1/2$. Any failure writes a FAILURES CSV and raises.

Figure: a single panel of decoded logical bit versus raw homodyne outcome over $[-3L, 3L]$, with dotted vertical lines at the cell boundaries $(k + 1/2)L$, $k = -3,\dots,3$.

<!-- core:R28 -->
**Title:** Finite-ancilla GKP q-syndrome extraction: ensemble statistics

**Why this experiment.** Shift-error correction with GKP codes depends on one operation: measuring the displacement of a data codeword modulo the lattice spacing with a GKP ancilla, then displacing it back. PhotoGraphiQ implements this as an executable pattern, `gkp.correction_pattern`, with a SUM interaction, a $q$-homodyne on the ancilla and modular feed-forward. R28 runs this pattern on a codeword with a known imposed shift and checks that the extracted syndrome is centred on the true shift. Because the ancilla is itself a finite-energy comb, single shots are noisy, so the test is statistical: ensemble circular means over seeded trajectories, with failure rates and Wilson intervals. It is one round with a finite ancilla and finite energy, not a fault-tolerance claim.

### Theory

Conventions: $\hbar = 2$, $[\hat q,\hat p] = 2i$, $L = \sqrt{2\pi}$. The data mode holds $\hat D_q(\delta)|0_L\rangle$. The ancilla is the normalised finite superposition $|+_L\rangle \propto |0_L\rangle + |1_L\rangle$ (`gkp.superposition(1, 1)`), whose $q$-comb has peaks at every multiple of $L$. The pattern conjugates a CZ (`Entangle`, $p_u \to p_u + w q_v$, $w = 1$) by $\pm\pi/2$ rotations of the ancilla, which realises a SUM gate $\hat q_a \to \hat q_a + \hat q_d$. A $q$-homodyne on the ancilla then returns $x = q_a + q_d$. For ideal combs $q_a \in L\mathbb Z$ and $q_d \in 2L\mathbb Z + \delta$, so $x \bmod L = \delta \bmod L$. The decoder (`decode_shift`) maps $x$ to its residual $\rho \in [-L/2, L/2)$, and the feed-forward applies $\hat D_q(-\rho)$ to the data. The expected residual is

$$\rho^\star = \big((\delta + L/2) \bmod L\big) - L/2 .$$

With finite peak width each shot samples one ancilla tooth smeared by its width, and the data's own peaks are smeared too, so $\rho$ scatters around $\rho^\star$. Because $\rho$ lives on a circle of circumference $L$, the ensemble is summarised by circular statistics with $\phi_j = 2\pi\rho_j/L$:

$$\bar\rho = \frac{L}{2\pi}\arg\Big\langle e^{i\phi_j}\Big\rangle,\qquad \sigma_\rho = \frac{L}{2\pi}\sqrt{-2\ln\big|\langle e^{i\phi_j}\rangle\big|}.$$

A shot counts as a syndrome error when its circular distance to $\rho^\star$ exceeds $L/4$. Rates are given with the Wilson 95% score interval for a binomial proportion.

### What the code does

- System under test: `pg.gkp.correction_pattern(quadrature="q", resource=ancilla_plus)` on `native-fock`, with a total-photon cutoff of 80 shared by both modes (two-mode dimension $\binom{81}{2} = 3240$); codewords with peak width = envelope = 0.35.
- Input preparation: the cutoff-80 projection of $|0_L\rangle$, displaced by $\delta$ on `native-fock` and renormalised.
- Sweep: 9 shifts $\delta$ evenly spaced on $[-0.6, 0.6]$, 16 seeded trajectories each (seeds $3000 + 97s$), 144 trajectories in total. Runs stopped by the retained-norm guard are counted separately, never averaged in.
- Engine cross-check: for $\delta\in\{-0.6, 0, 0.45\}$ the same seeds are rerun on `piquasso-fock` (48 trajectories). The engines share the Python sampling path, so records should agree up to the small difference of their evolved states.
- Recorded per $\delta$: circular mean and spread of $\rho$, mean-error $|\bar\rho - \rho^\star|$, mean $|\langle q\rangle|$ after correction, syndrome error count, rate and Wilson interval, norm-guard failures; pooled rate and interval.
- Output: `R28_gkp_syndrome` and `R28_gkp_syndrome_reference` (CSV/JSON with metadata), figure `R28_gkp_syndrome` (PDF/PNG/SVG) and a caption file.

### Expected outcome

The ensemble circular mean should follow the ideal line $\rho^\star = \delta$ (all tested $|\delta| < L/2$) within its standard error, while single-shot spreads are broad (a few tenths, set by the finite peak width). The syndrome error rate should be a small fraction of shots with a Wilson interval that reflects only 144 trials. The mean $|\langle q\rangle|$ after correction is not expected to fall below $|\delta|$: it stays roughly constant across $\delta$, plausibly because the finite-envelope ancilla partially measures the data $q$. The script reports it but does not use it as a criterion and does not explain its origin. Declared acceptance criteria: for each $\delta$, $|\bar\rho - \rho^\star| \le 5\,\sigma_\rho/\sqrt{n_{\rm ok}} + 0.05$; on the reference deltas the `native-fock` and `piquasso-fock` syndrome records agree to $10^{-3}$, a tolerance justified in the script as truncation-level differences in the two-mode gate, below 0.3% of the single-shot spread. Any failure writes a FAILURES CSV and raises.

Figure: (left) circular mean decoded residual $\pm$ SEM versus $\delta$, with the ideal $\delta \bmod L$ dashed; (right) mean $|\langle q\rangle|$ after correction versus $\delta$, with the uncorrected $|\delta|$ dashed.

<!-- core:R29 -->
**Title:** Finite resource squeezing sets the noise floor of compiled MBQC circuits

**Why this experiment.** A central message of the paper is that finite squeezing in measurement-based CV computation does not distort the logical operation; it adds Gaussian noise. PhotoGraphiQ therefore reports the compiled channel as a pair (logical map, added noise) and never counts finite squeezing as compilation error. R29 shows this for two fixed logical circuits over a wide range of resource squeezing: the map stays exact while the noise falls as $e^{-2r}$. It also checks the two-mode noise against a closed form, and executes the compiled patterns shot by shot to confirm that the analytic channel is what the trajectory engines actually produce. Without R29, a reader could not tell compiler error from physical noise.

### Theory

Conventions: $\hbar = 2$, interleaved ordering $(q_0, p_0, q_1, p_1)$, vacuum covariance $I$. A compiled pattern with all outcome-dependent corrections applied acts unconditionally as the Gaussian channel

$$\mu \mapsto S\mu + d,\qquad \Sigma \mapsto S\Sigma S^{T} + N .$$

Each teleportation step $T(k) = \begin{pmatrix} -k & -1 \\ 1 & 0\end{pmatrix}$ consumes a momentum-squeezed node and adds $\mathrm{diag}(0, e^{-2r})$, propagated by the later steps:

$$N \leftarrow T(k)\,N\,T(k)^{T} + \mathrm{diag}(0, e^{-2r}).$$

$S$ depends only on the shears, so it is independent of $r$, while $N(r) = e^{-2r}N_1$ with a circuit-dependent $N_1$; hence $\ln\mathrm{Tr}\,N$ falls with slope $-2$ in $r$. The targets are built by hand: $S_{\rm single} = S_{\rm sq}(0.3)\,R(0.5)$ (rotation applied first) with $R(\theta)$ the rotation matrix and $S_{\rm sq}(r) = \mathrm{diag}(e^{-r}, e^{r})$, and $S_{\rm CZ}(0.6)$, the identity with $p_0 \mathrel{+}= 0.6\,q_1$ and $p_1 \mathrel{+}= 0.6\,q_0$. The CZ is compiled as an instantaneous `Entangle` (no noise) followed by four $T(0)$ steps per mode. Since $T(0)$ is a quarter turn, four steps give

$$N_{\rm CZ} = 2e^{-2r}\,I_4,\qquad \mathrm{Tr}\,N_{\rm CZ} = 8e^{-2r},$$

which the code obtains independently from `gaussian.wire_channel([0, 0, 0, 0], r)` per mode. For a coherent input the output covariance is $SS^{T} + N$; for the single-mode circuit $SS^{T} = \mathrm{diag}(e^{-0.6}, e^{0.6})$, the large-$r$ limit of the output variances.

### What the code does

- System under test: `pg.Circuit(1).rotate(0, 0.5).squeeze(0, 0.3)` and `pg.Circuit(2).cz(0, 1, 0.6)`, compiled with `circuit.compile(squeezing=r)`; `pg.gaussian_channel(pattern)` returns $(S, N, d)$.
- Sweep: 20 values of $r$, geometrically spaced on $[0.1, 3.0]$.
- Metrics: Frobenius error of $S$ against the NumPy targets; $\mathrm{Tr}\,N$, $\lVert N\rVert_2$, $\lambda_{\max}(N)$; output $q$ and $p$ variances of the single-mode circuit for coherent input $\alpha = 0.3 + 0.2i$; Frobenius error of $N_{\rm CZ}$ against the closed form.
- Execution check at $r \in \{0.3, 1.0, 2.0\}$: both compiled patterns are run with `pg.run_shots` (600 trajectories, seed 20261113) on NumPy `gaussian` and C++ `native-gaussian`, inputs $\alpha = 0.3 + 0.2i$ (and $-0.1 + 0.25i$ on the second CZ mode). The ensemble mean and covariance are compared with `channel.apply` using z-scores, with standard errors from the analytic spread of the conditional means ($\Sigma_{\rm uncond} - \Sigma_{\rm cond}$); NumPy and native records are compared trajectory by trajectory.
- Output: `R29_finite_squeezing_single_mode`, `R29_finite_squeezing_cz`, `R29_finite_squeezing_execution` (CSV/JSON with metadata), figure `R29_finite_squeezing` (PDF/PNG/SVG) and a caption file.

### Expected outcome

The map error should stay at rounding level at every $r$ for both circuits. $\mathrm{Tr}\,N$ should fall exactly as $e^{-2r}$ for both circuits, with different prefactors set by the number and shears of the teleportation steps. On the log-log axes of the figure this appears as a curve that steepens with $r$, not a straight line. The output variances should approach $e^{-0.6}$ and $e^{0.6}$ from above. Sampled ensembles should agree with the analytic channel within a few standard errors, and the two engines should give the same records. Declared acceptance criteria: map Frobenius error $\le 10^{-7}$; CZ noise oracle error $\le 10^{-9}$; all execution z-scores (mean and covariance, both engines) $\le 5$; native vs NumPy record difference $\le 10^{-9}$. Any failure writes a FAILURES CSV and raises.

Figure: (left) $\mathrm{Tr}\,N$ versus $r$ on log-log axes for the rotate+squeeze circuit (circles) and the CZ circuit (squares); (right) output $q$ and $p$ variances of the single-mode circuit versus $r$ (log $r$ axis).

<!-- core:R30 -->
**Title:** Pure-loss channel against the analytic attenuation formulas on five engines

**Why this experiment.** Photon loss is the dominant imperfection in photonic hardware, and every noisy study in the paper uses the `Loss` command. Its implementation must therefore be exact on the phase-space engines, and consistent with the Kraus-operator implementation on the density-matrix (mixed-Fock) engines. R30 tests the simplest case, pure loss ($\bar n = 0$) of a coherent state, where the output is known in closed form. A wrong square root (amplitude vs intensity transmissivity), a missing vacuum term or an inconsistent Kraus truncation would show up here directly.

### Theory

Conventions: $\hbar = 2$, $\hat q = \hat a + \hat a^\dagger$, vacuum covariance $I$, coherent state $|\alpha\rangle$ with mean $\mu = (2\,\mathrm{Re}\,\alpha,\, 2\,\mathrm{Im}\,\alpha)$ and covariance $I$. A beam splitter of intensity transmissivity $\eta$ mixing the mode with an environment of mean occupation $\bar n$ gives the Gaussian channel

$$\mu \mapsto \sqrt\eta\,\mu,\qquad \Sigma \mapsto \eta\,\Sigma + (1 - \eta)(2\bar n + 1)\,I .$$

For pure loss ($\bar n = 0$) a coherent state stays coherent, $|\alpha\rangle \mapsto |\sqrt\eta\,\alpha\rangle$, so

$$\alpha_{\rm out} = \sqrt\eta\,\alpha_{\rm in},\qquad \langle\hat n\rangle_{\rm out} = \eta\,|\alpha_{\rm in}|^2,\qquad \mathrm{Var}(\hat q)_{\rm out} = 1 ,$$

using $\langle\hat n\rangle = (\mathrm{Tr}\,\Sigma + |\mu|^2)/4 - 1/2$. In the Fock basis the same channel has Kraus operators $\hat K_k = \sqrt{(1-\eta)^k/k!}\;\eta^{\hat n/2}\hat a^k$; on a truncated space this is exact up to the weight of the input beyond the cutoff.

### What the code does

- System under test: a one-mode pattern with `pg.Loss(0, eta, 0.0)` acting on `pg.GaussianInput.coherent(0.6 + 0.4j)` ($|\alpha|^2 = 0.52$).
- Phase-space engines: NumPy `gaussian`, Piquasso `piquasso`, C++ `native-gaussian`. Metrics: Euclidean error of the output mean against $(2\sqrt\eta\,\mathrm{Re}\,\alpha,\, 2\sqrt\eta\,\mathrm{Im}\,\alpha)$ and absolute error of $\langle\hat n\rangle$ against $\eta|\alpha|^2$.
- Density-matrix engines: `native-mixed-fock` and `piquasso-mixed-fock` at cutoff 20. Metrics: errors of $\langle\hat n\rangle$ and of $\mathrm{Var}(\hat q)$ (expected 1), minimum retained norm, wall time per run (provisional, shared machine).
- Oracle: the closed-form expressions above, written directly in NumPy and independent of every engine.
- Sweep: 20 values of $\eta$ evenly spaced on $[0.05, 1.0]$.
- Output: `R30_loss_validation` (CSV/JSON with metadata) and figure `R30_loss_validation` (PDF/PNG/SVG).

### Expected outcome

All three phase-space engines should reproduce the analytic mean and photon number at rounding level ($\sim 10^{-16}$), since the channel is a finite matrix operation. The mixed-Fock engines should agree almost as closely, because the coherent state with $|\alpha| \approx 0.72$ has negligible weight above 20 photons, and the retained norm should be 1 to within rounding. Wall times of the two mixed-Fock engines are recorded for comparison; absolute values are machine-dependent. Declared acceptance criteria: for every $\eta$, mean and photon-number errors $\le 10^{-9}$ on `gaussian`, `piquasso` and `native-gaussian`; on each mixed-Fock engine whose run succeeded with minimum retained norm $\ge 1 - 10^{-9}$, photon-number and $q$-variance errors $\le 10^{-8}$. Any failure writes a FAILURES CSV and raises.

Figure: a single panel of output mean photon number versus $\eta$, the analytic line $\eta|\alpha|^2$ (grey) with the NumPy (circles), Piquasso (crosses) and `native-gaussian` (plus signs) values on top of it. The mixed-Fock results are in the CSV only.

<!-- core:R31 -->
**Title:** Thermal attenuation channel against the analytic thermal-loss formula

**Why this experiment.** Real optical channels couple a mode to an environment that is not in vacuum. The `Loss` command supports this through a thermal occupation `thermal_photons`, and the noise studies of the paper rely on it. R31 generalises R30 to the full thermal-loss channel over a grid of transmissivities and occupations, compares the full output mean and covariance with the closed-form channel on the three phase-space engines, and tests the density-matrix engines on a subset. The mixed-Fock tests show a real limitation: a thermal environment spreads the state over many Fock levels, so truncation error grows with $\bar n$. R31 reports this error together with the retained norm instead of hiding it.

### Theory

Conventions: $\hbar = 2$, vacuum covariance $I$, statistical covariance $\Sigma_{ij} = \tfrac12\langle\{\Delta x_i, \Delta x_j\}\rangle$. Mixing the mode on a beam splitter of intensity transmissivity $\eta$ with a thermal mode of mean occupation $\bar n$ (covariance $(2\bar n + 1)I$) and tracing out the environment gives

$$\mu_{\rm out} = \sqrt\eta\,\mu_{\rm in},\qquad \Sigma_{\rm out} = \eta\,\Sigma_{\rm in} + (1-\eta)(2\bar n + 1)\,I .$$

The input covariance is attenuated by $\eta$, and the environment's full covariance $(2\bar n + 1)I$, vacuum part included, enters with weight $1 - \eta$. For a coherent input ($\Sigma_{\rm in} = I$),

$$\mathrm{Var}(\hat q)_{\rm out} = 1 + 2\bar n\,(1 - \eta),$$

so for $\bar n = 0$ the variance stays at the vacuum value for every $\eta$, and for $\bar n > 0$ it grows linearly as $\eta$ decreases, towards the thermal value $2\bar n + 1$. The mean photon number used for the mixed-Fock check is $\langle\hat n\rangle = (\mathrm{Tr}\,\Sigma_{\rm out} + |\mu_{\rm out}|^2 - 2)/4$. In the Fock basis the channel is a thermal-loss Kraus map; on a truncated space its accuracy is limited by the output population above the cutoff, which grows with $\bar n$.

### What the code does

- System under test: a one-mode pattern with `pg.Loss(0, eta, nbar)` acting on `pg.GaussianInput.coherent(0.5 - 0.3j)`.
- Oracle: `analytic_thermal_channel`, the two formulas above written directly in NumPy from the input mean and covariance.
- Grid: $\eta \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$ times $\bar n \in \{0, 0.3, 0.8, 1.5, 3.0\}$, 25 cases, on NumPy `gaussian`, Piquasso `piquasso` and C++ `native-gaussian`. Metrics: Euclidean error of the mean, Frobenius error of the covariance.
- Mixed-Fock subset: $\eta \in \{0.4, 0.8\}$ times all five $\bar n$ (10 cases) on `native-mixed-fock` and `piquasso-mixed-fock` at cutoff 30. Metrics: errors of $\langle\hat n\rangle$ and $\mathrm{Var}(\hat q)$, minimum retained norm, wall time (provisional).
- Output: `R31_thermal_noise` (CSV/JSON with metadata) and figure `R31_thermal_noise` (PDF/PNG/SVG).

### Expected outcome

The three phase-space engines should match the analytic channel at rounding level in all 25 cases, with no corner of the grid behaving differently. On the mixed-Fock engines the errors should be at rounding level for small $\bar n$ and grow with $\bar n$ (and with decreasing $\eta$, which raises the thermal admixture), together with a retained norm that falls slightly below 1. This is truncation at cutoff 30, not an implementation error. Declared acceptance criteria: mean and covariance errors $\le 10^{-9}$ on `gaussian`, `piquasso` and `native-gaussian` in every case; on each mixed-Fock engine, photon-number and $q$-variance errors $\le 10^{-6}$ wherever the run succeeded with minimum retained norm $\ge 1 - 10^{-9}$. Cases below that norm are reported, not asserted. Any failure writes a FAILURES CSV and raises.

Figure: a single panel of the analytic output $q$ variance versus $\eta$, one curve per $\bar n$: a flat line at 1 for $\bar n = 0$ and straight lines rising towards small $\eta$ for $\bar n > 0$. The plotted curves come from the analytic formula; the agreement of the engines with these values is in the CSV.

<!-- core:R32 -->
**Title:** Inefficient and noisy homodyne detection against the added-variance formula

**Why this experiment.** Real homodyne detectors have finite quantum efficiency and electronic noise, and in measurement-based computation both broaden every outcome that is fed forward. `Homodyne(efficiency, noise)` models them with outcomes calibrated to the incident quadrature, and its documented consequence is a fixed added variance. R32 checks this formula statistically on the phase-space engines, checks that the C++ engine reproduces the NumPy samples exactly for the same seed (the RNG contract), and checks the mixed-Fock implementation deterministically through its outcome density. A wrong calibration (for example forgetting the $1/\sqrt\eta$ rescaling) would change the variance by an amount these tests resolve.

### Theory

Conventions: $\hbar = 2$, vacuum quadrature variance 1. An inefficient detector is a beam splitter of transmissivity $\eta$ followed by an ideal homodyne; rescaling the outcome by $1/\sqrt\eta$ calibrates it to the incident quadrature:

$$x_{\rm meas} = x + \sqrt{\frac{1-\eta}{\eta}}\;x_{\rm vac} + \sqrt{\nu_{\rm el}}\;z ,$$

with $x_{\rm vac}$ a vacuum quadrature (variance 1) and $z$ standard normal electronic noise of calibrated variance $\nu_{\rm el}$ (`noise`). Hence

$$\mathrm{Var}(x_{\rm meas}) = \mathrm{Var}(x) + \nu,\qquad \nu = \frac{1-\eta}{\eta} + \nu_{\rm el}.$$

For a coherent input $|\alpha\rangle$ the $q$ outcome is Gaussian, $x_{\rm meas} \sim \mathcal N(2\,\mathrm{Re}\,\alpha,\; 1 + \nu)$. A sample variance from $M$ Gaussian draws has standard error $\mathrm{Var}\sqrt{2/(M-1)}$; the code uses six times this as its band.

### What the code does

- System under test: a one-mode pattern measuring `pg.Homodyne(angle=0, efficiency, noise)` on `pg.GaussianInput.coherent(0.4 + 0.2j)`.
- Powered sweep: $\eta \in \{1.0, 0.9, 0.7, 0.5\}$ times $\nu_{\rm el} \in \{0, 0.2, 0.6\}$ (12 cases), `pg.run_shots` with 6000 shots, seed 20260913, on NumPy `gaussian` and C++ `native-gaussian`. Metrics: sample variance (ddof = 1) against $1 + \nu$, the 6-sigma band, and the maximum native-vs-NumPy sample difference.
- Mixed-Fock density check (deterministic): $\eta \in \{1.0, 0.8\}$ times $\nu_{\rm el} \in \{0, 0.3\}$ on `native-mixed-fock` and `piquasso-mixed-fock` at cutoff 10. The postselected outcome density (`measurement_outcomes`) is evaluated at $x \in \{-1.5, -0.5, 0, 0.5, 0.8, 1.5, 3.0\}$ (56 points) and compared with the analytic $\mathcal N(0.8,\, 1 + \nu)$ density from SciPy.
- Mixed-Fock sampling (publication configuration, `--full`): 200 shots for each noiseless case and 6 shots for each noisy case, per engine. Noisy sampling on the mixed-Fock path costs several seconds per shot (an integral inside a root find), so the noisy runs are a coarse cross-check: they are recorded with seconds per shot but not asserted.
- Oracle: the closed-form variance and density above, written independently of the measurement code.
- Output: `R32_noisy_measurement`, `R32_noisy_measurement_mixed_fock`, `R32_noisy_measurement_mixed_fock_density` (CSV/JSON with metadata) and figure `R32_noisy_measurement` (PDF/PNG/SVG).

### Expected outcome

The sampled variances should scatter around $1 + (1-\eta)/\eta + \nu_{\rm el}$ by about one standard error (about 2% of the variance for 6000 shots), rising steeply as $\eta$ drops to 0.5. The native samples should equal the NumPy samples. The mixed-Fock densities should match the analytic Gaussian well inside the $10^{-6}$ tolerance (the script comment bounds the truncated tail below $10^{-9}$), since the input's weight beyond 10 photons is negligible. The 6-shot noisy runs give only rough variances. The `--full` run is dominated by the noisy mixed-Fock sampling; its wall time is machine-dependent and can reach minutes. Declared acceptance criteria: in each of the 12 cases, $|\widehat{\mathrm{Var}} - (1+\nu)| \le 6(1+\nu)\sqrt{2/5999}$ and native-vs-NumPy sample difference $\le 10^{-10}$; every mixed-Fock density error $\le 10^{-6}$; every completed noiseless mixed-Fock sampling run within its own 6-sigma band. Any failure writes a FAILURES CSV and raises.

Figure: a single panel of outcome variance versus detector efficiency, the analytic curves (grey lines) and the sampled `gaussian` variances (markers), one series per noise level.

<!-- core:R33 -->
**Title:** Fock-cutoff convergence of non-Gaussian and high-energy Gaussian workloads

**Why this experiment.** Every non-Gaussian result in the paper is computed in a Fock space truncated at a finite total photon number. The paper's rule is that truncation is always reported, never assumed negligible. R33 applies this rule to representative workloads with `cutoff_convergence`, and checks that the two Fock engines agree at every cutoff. The original low-energy workloads converge at their smallest cutoff, so R33 adds two demanding Gaussian workloads with exact references ($\langle\hat n\rangle \approx 16$ and $\approx 9$). These show whether an engine stays accurate at large cutoffs, where displacement and squeezing matrix elements are numerically delicate.

### Theory

Conventions: $\hbar = 2$, vacuum covariance $I$. A cutoff $c$ keeps $m$-mode states with total photon number $\le c - 1$. For a state $|\psi\rangle$ the truncation error is governed by the population above the cutoff, $\epsilon_c = \sum_{n \ge c}|\langle n|\psi\rangle|^2$; `cutoff_convergence` reports the retained norm after each operation, the boundary population and fidelities between adjacent cutoffs. Here the highest tested cutoff serves as reference, with $F_c = F(\rho_c, \rho_{c_{\max}})$; it is not the exact state, so convergence is judged from the decreasing trend of $1 - F_c$, not from the last value alone. For states with Gaussian or Poissonian photon-number tails, $1 - F_c$ falls rapidly once $c$ exceeds the mean photon number by a few standard deviations of the photon-number distribution.

For Gaussian states the exact moments give an absolute reference. With mean $\mu$ and covariance $\Sigma$,

$$\langle\hat n\rangle = \frac{\mathrm{Tr}\,\Sigma + |\mu|^2 - 2}{4}.$$

Workload (a): vacuum, `Squeeze(0.5)` then `Displace(q = 8)` ($\alpha = 4$), so $\Sigma = \mathrm{diag}(e^{-1}, e^{1})$, $\mu = (8, 0)$ and $\langle\hat n\rangle = (e^{-1} + e + 62)/4 \approx 16.27$. Workload (b): a mixed Gaussian input with $\Sigma = 1.4\,\mathrm{diag}(e^{-0.6}, e^{0.6})$ (thermal 0.2, $r = 0.3$) and $\mu = (6, 0)$ ($\alpha = 3$), so $\langle\hat n\rangle = (1.4(e^{-0.6} + e^{0.6}) + 34)/4 \approx 9.33$.

### What the code does

- System under test: `pg.cutoff_convergence(pattern, cutoffs, backend=...)`, which runs `pg.simulate` at strictly increasing cutoffs.
- Original workloads, each on `native-fock` (main) and `piquasso-fock` (reference):
  - even cat state `CatResource(1.5, parity=1)`, cutoffs $\{16, 24, 32, 48, 64\}$;
  - heralded photon subtraction `non_gaussian.photon_subtraction(0.3)` from $|2\rangle$, outcome `count = 1`, cutoffs $\{8, 12, 16, 24, 32\}$;
  - `CubicPhase(0.15)` on a coherent state $\alpha = 0.3$, cutoffs $\{16, 24, 32, 48, 64\}$;
  - finite-energy GKP $|0_L\rangle$ (peak width = envelope = 0.4), cutoffs $\{24, 32, 48, 64, 96\}$.
- Recorded for these: retained norm, boundary population, $\langle\hat n\rangle$, fidelity and trace distance to the previous and to the highest cutoff, and the fidelity between the two engines' states at each cutoff.
- Demanding workloads: (a) on `native-fock` and `piquasso-fock`, cutoffs $\{40, 60, 80, 100, 120\}$; (b) on `native-mixed-fock` and `piquasso-mixed-fock`, cutoffs $\{40, 60, 80\}$. Metrics: relative error of $\langle\hat n\rangle$, errors of $\langle\hat q\rangle$ and $\mathrm{Var}(\hat q)$ against the exact moments, retained norm, boundary population.
- Output: `R33_fock_cutoff` and `R33_fock_cutoff_demanding` (CSV/JSON with metadata), figure `R33_fock_cutoff` (PDF/PNG/SVG) and a caption file.

### Expected outcome

For the cat, subtraction and cubic workloads $1 - F_c$ should already be small at the smallest cutoffs and fall quickly (the script's docstring notes that these low-energy workloads converge already at their lowest cutoff). The GKP comb, with its broad photon-number distribution, should show a visible decrease over 24 to 96. The two engines should agree at every cutoff to rounding level. For the demanding workloads the relative $\langle\hat n\rangle$ error should decrease with cutoff and reach rounding level on the native engines at the highest cutoff. Piquasso's values are reported for comparison but not asserted, because its large-cutoff matrices are documented to lose accuracy. Declared acceptance criteria: native-vs-Piquasso fidelity $\ge 1 - 10^{-6}$ at every cutoff of the four original workloads; the native engines complete the demanding workloads and reach relative $\langle\hat n\rangle$ error $\le 10^{-8}$ at the highest cutoff (120 for (a), 80 for (b)). Any failure writes a FAILURES CSV and raises.

Figure (2 x 3): the first four panels show $1 - F$ to the highest cutoff versus cutoff (log scale) for cat, photon subtraction, cubic phase and GKP, `native-fock` circles and `piquasso-fock` crosses; the last two show the relative $\langle\hat n\rangle$ error versus cutoff for workloads (a) and (b) on their two engines. Values below $10^{-16}$ are exact zeros drawn at the floor.

<!-- core:R34 -->
**Title:** Cubic-phase injection fidelity over Fock cutoff and resource squeezing

**Why this experiment.** In measurement-based CV computation, the cubic phase gate is applied by injecting a finite-squeezing cubic-phase resource and correcting with outcome-dependent Gaussian feed-forward. Two separate limits control its accuracy: the resource squeezing $r$, which sets a physical Gaussian filter, and the Fock cutoff, which must hold the anti-squeezed ancilla. A one-dimensional sweep at fixed cutoff (R23) cannot separate them. R34 scans the full (cutoff, $r$) grid on both Fock engines and records where the simulation is converged in cutoff and where it hits the truncation guard. It confirms that the remaining infidelity is the physical finite-squeezing envelope, which disappears only as $r \to \infty$.

### Theory

Conventions: $\hbar = 2$, `CubicPhase(gamma)` $= e^{i\gamma\hat q^3/6}$. The resource is $\mathrm{CP}(\gamma)\,S(-r)|0\rangle$, a $q$-anti-squeezed vacuum ($q$ variance $e^{2r}$) with a cubic phase, $\phi(x) \propto e^{-x^2/(4e^{2r})}e^{i\gamma x^3/6}$. `resource_injection` couples input and ancilla by an inverse SUM and measures the ancilla $q$, so for outcome $m$ the input is filtered as $\psi(q) \mapsto \psi(q)\phi(q + m)$. The feed-forward `QuadraticPhase(-2 gamma m)` and `Displace(p = -gamma m^2)` removes the $m$-dependent quadratic and linear phases, leaving

$$\psi_{\rm out}(q) \propto e^{i\gamma q^3/6}\,E_r(q + m)\,\psi(q),\qquad E_r(x) = \exp\!\Big(-\frac{x^2}{4e^{2r}}\Big),$$

up to a global phase. R34 fixes $m = 0$, a postselected branch, so that the output differs from the ideal gate only by the envelope. Because $E_r$ commutes with $\mathrm{CP}(\gamma)$, the fidelity to $\mathrm{CP}(\gamma)|\psi\rangle$ is independent of $\gamma$ in the untruncated theory:

$$F(r) = \frac{|\langle\psi|E_r|\psi\rangle|^2}{\langle\psi|E_r^2|\psi\rangle} = \frac{\sqrt{1+4a}}{1+2a}\exp\!\Big[-2a\mu^2\Big(\frac{1}{1+2a} - \frac{1}{1+4a}\Big)\Big],\qquad a = \frac{e^{-2r}}{4},$$

for a coherent input with real $\alpha$ ($q$ distribution $\mathcal N(\mu, 1)$, $\mu = 2\alpha$). For $\alpha = 0.2$ this evaluates to about 0.947 at $r = 0.1$, 0.9974 at $r = 1$ and $1 - 5\times10^{-5}$ at $r = 2$, rising monotonically in $r$. The cost is in the cutoff: the ancilla's photon number grows like $\sinh^2 r$ and its cubic phase spreads $p$ by about $\gamma q^2/2$ with $q \sim e^{r}$, so large $r$ needs large cutoffs.

### What the code does

- System under test: `pg.non_gaussian.cubic_injection(0.15, squeezing=r)` with input `GaussianInput.coherent(0.2)` and `measurement_outcomes={"m": 0}`, run on `native-fock` (main) and `piquasso-fock` (reference).
- Reference: the engine's own truncated coherent input multiplied by $\exp(i\gamma\hat Q^3/6)$, with $\hat Q = \hat a + \hat a^\dagger$ built at the same cutoff and exponentiated with SciPy `expm`, then normalised. It is independent of the injection circuit.
- Grid: cutoffs $\{24, 32, 48, 64, 96\}$ times $r \in \{0.1, 0.3, 0.6, 1.0, 1.5, 2.0\}$, 30 cells per engine. Cells stopped by the retained-norm guard ($|\text{norm} - 1| > 10^{-3}$ after an operation) are recorded with the error message, not renormalised.
- Metrics: fidelity, $\log_{10}(1-F)$, retained norms, and the native-vs-Piquasso output fidelity wherever both engines succeed.
- Output: `R34_cubic_convergence` (heatmap-ready CSV/JSON with metadata), figure `R34_cubic_convergence` (PDF/PNG/SVG) and a caption file.

### Expected outcome

Along each row (fixed $r$) the fidelity should be constant in the cutoff once the cell runs: the result is converged in cutoff. Along each column it should rise with $r$, following the envelope formula above. The upper-left region (large $r$, small cutoff) should be hatched by the norm guard, and the boundary should move to larger $r$ as the cutoff grows. Both engines should give nearly identical heatmaps. The native engine is expected to run some large-$r$ cells at cutoff 96 where Piquasso stops. Declared acceptance criteria: native-vs-Piquasso output fidelity $\ge 1 - 10^{-3}$ wherever both succeed (truncation-level differences of the two-mode gate only), and at cutoff 96 the successful native fidelities must not decrease with $r$ by more than $10^{-6}$. Any failure writes a FAILURES CSV and raises.

Figure: two heatmaps (left `native-fock`, right `piquasso-fock`) of fidelity to the ideal cubic phase, cutoff on the horizontal axis and $r$ on the vertical axis, colour scale 0.9 to 1 with the value printed in each cell; norm-guard cells are hatched. One shared colour bar.

<!-- core:R35 -->
**Title:** Fock-space dimension, memory and measured cost of the Fock engines

**Why this experiment.** The cost of Fock-basis simulation is set by the size of the truncated Hilbert space, and the paper's performance section must say plainly how quickly that size grows and where the package's allocation guard stops a run. R35 evaluates the exact dimension formula used by both Fock engines over a modes x cutoff grid, cross-checks it against an independent binomial, converts it to memory, and reads the default `max_dimension` guard directly from the backend. It then measures the wall time of one two-mode non-Gaussian workload on the pure-state engines and the native density-matrix engine, with a fidelity check so that a fast engine with a wrong state cannot pass.

### Theory

With a total-photon cutoff $c$, an $m$-mode basis keeps all occupations $(n_1,\dots,n_m)$ with $\sum_i n_i \le c - 1$. Counting by stars and bars,

$$D(m, c) = \sum_{N=0}^{c-1}\binom{N + m - 1}{m - 1} = \binom{m + c - 1}{m}.$$

For fixed $m$ this grows like $c^m/m!$, and for fixed $c$ like $m^{c-1}/(c-1)!$, so neither variable can be raised freely. A dense complex128 state vector costs $16D$ bytes, and a dense density matrix $16D^2$ bytes. The latter is a rough upper-style estimate, since the mixed-Fock implementation need not store a single dense matrix of this size. For two modes, $D(2, c) = c(c+1)/2$, so a dense density matrix grows like $c^4$; this is why the mixed engine is timed only up to cutoff 32.

The timed workload is two coherent inputs ($\alpha_0 = 0.8 + 0.2i$, $\alpha_1 = -0.5 + 0.4i$), `BeamSplitter(0, 1, 0.4)`, `Kerr(0, 0.2)` ($e^{i\kappa\hat n^2}$) and `CubicPhase(1, 0.05)` ($e^{i\gamma\hat q^3/6}$). A Kerr gate is diagonal in the Fock basis, while the beam splitter and the cubic phase couple many basis states, so the gate cost grows faster than $D$. The time for state preparation alone is measured separately, and the difference estimates the gate-execution cost.

### What the code does

- System under test: `PiquassoFockBackend.dimension(modes)` and `NativeFockBackend.dimension(modes)`; oracle `common.fock_dimension` = `math.comb(modes + cutoff - 1, modes)`, written independently. Grid: modes $1..20$ times cutoffs $2..40$ (780 points); any disagreement raises immediately.
- Memory: pure-vector and dense density-matrix bytes, and a flag for points exceeding the default `max_dimension`, which is read from the `PiquassoFockBackend.__init__` signature.
- Timing (publication configuration; the notebook does not use `--quick`): cutoffs $\{8, 16, 24, 32, 48, 64\}$ on `native-fock` and `piquasso-fock`, and on `native-mixed-fock` for cutoffs $\le 32$. `common.benchmark` with 1 warm-up and 5 repeats each for a preparation-only pattern and the full workload; recorded medians, IQR, gate-time estimate, minimum retained norm, and fidelity of each engine's final state to `native-fock`.
- Timing metadata: `timing_status` (provisional unless `PGQ_IDLE_MACHINE=1`), CPU count, `OMP_NUM_THREADS`, native build information.
- Output: `R35_fock_dimension_scaling` and `R35_fock_dimension_scaling_timing` (CSV/JSON with metadata), figure `R35_fock_dimension_scaling` (PDF/PNG/SVG) and a caption file.

### Expected outcome

The two dimension formulas should equal the binomial at all 780 points (an exact integer identity). Memory should cross the default guard at modest cutoffs once several modes are involved, which is the reason the guard exists. Wall times should grow with dimension on all engines, steeply for the density-matrix engine, with `native-fock` expected to be clearly faster than `piquasso-fock` at large cutoffs. Absolute times are machine-dependent; on a shared or Colab machine they are labelled provisional, and only the trend and the ordering should be read. Declared acceptance criteria: exact equality of both engines' dimension formulas with the independent binomial at every grid point; fidelity of each timed engine's final state to `native-fock` $\ge 1 - 10^{-6}$ at every timed cutoff. Any failure raises (the timing failure also writes a FAILURES CSV).

Figure (three panels): (left) heatmap of $\log_{10} D$ over modes and cutoff; (centre) pure state-vector bytes versus cutoff for 1, 2, 4, 8, 16 and 20 modes on a log scale, with the default `max_dimension` guard as a red dashed line; (right) measured median wall time versus two-mode dimension on log-log axes, total (opaque) and preparation-only (faint) for each engine, titled with the timing status.

<!-- core:R36 -->
**Title:** Runtime of the Gaussian engines for line cluster states with and without feed-forward

**Why this experiment.** Gaussian MBQC patterns with hundreds of modes are the typical workload of PhotoGraphiQ, and the paper compares three Gaussian engines: the NumPy reference `gaussian`, the C++ `native-gaussian` and Piquasso's `piquasso`. R36 measures how one `simulate` call scales with the number of modes, and it separates the cost of preparing and entangling the cluster from the cost of measuring and feeding forward. Timings mean nothing if an engine computes something different, so every engine is first required to reproduce the NumPy records and output moments for the same seed. The results are local observations on one machine, not cross-machine performance claims.

### Theory

Conventions: $\hbar = 2$, interleaved ordering, vacuum covariance $I$. An $n$-mode Gaussian state is fully described by a $2n$ mean vector and a $2n \times 2n$ covariance, so the memory cost is polynomial, $O(n^2)$, in contrast with Fock simulation (R35). The line resource `CVGraph.line(n, squeezing=1.0, inputs=(0,))` prepares squeezed resource nodes ($r = 1$) and applies $n - 1$ nearest-neighbour CZ gates ($p_u \to p_u + q_v$, $p_v \to p_v + q_u$). A symplectic gate $S$ updates the state as

$$\mu \mapsto S\mu,\qquad \Sigma \mapsto S\Sigma S^{T}.$$

The NumPy engine embeds each gate into a dense $2n \times 2n$ matrix and multiplies full matrices, an $O(n^3)$ operation per gate, so its preparation cost can grow as fast as $O(n^4)$ for a line. An engine that updates only the affected rows and columns pays $O(n)$ per gate. In the measured workload every node except the last is $q$-homodyned and the next node is displaced by the outcome, $p_{i+1} \mathrel{+}= m_i$. Gaussian conditioning on a homodyne outcome is a Schur-complement update

$$\Sigma_B \mapsto \Sigma_B - \Sigma_{BA}\,(\Pi\Sigma_A\Pi)^{-}\,\Sigma_{AB},$$

with $\Pi = \mathrm{diag}(1, 0)$ selecting $q$, a rank-one update of cost $O(n^2)$ per measurement with dense storage, or less if measured modes are removed. Which regime each engine is in is an implementation property that the measured exponents reveal. They are empirical slopes over the tested range, not complexity proofs.

### What the code does

- System under test: `pg.simulate(pattern, backend=..., seed=0)` for backends `gaussian`, `native-gaussian`, `piquasso`.
- Workloads: `prepare_entangle` = `pg.Pattern(CVGraph.line(n, squeezing=1.0, inputs=(0,)))`; `measure_feedforward` = the same with `Measure(i, Homodyne.q())` on nodes $0..n-2$ and `Displace(i+1, p=Outcome(m_i))`.
- Correctness gate before timing: maximum absolute difference of all records, output means and covariances against NumPy `gaussian` with the same seed must be $\le 10^{-8}$; otherwise the script raises immediately.
- Sweep (publication configuration; the notebook does not use `--quick`): $n \in \{10, 20, 50, 100, 200\}$. `common.benchmark` (`time.perf_counter`) with 2 warm-ups and 9 repeats for $n \le 50$, 1 warm-up and 5 repeats for $n = 100, 200$; recorded median, standard deviation, IQR, min, max and command count.
- Metadata: full environment (`metadata.collect()`) and timing environment (status provisional unless `PGQ_IDLE_MACHINE=1`, CPU count, `OMP_NUM_THREADS`, native build info).
- Output: `R36_gaussian_runtime` (CSV/JSON with metadata), figure `R36_gaussian_runtime` (PDF/PNG/SVG) and a caption file.

### Expected outcome

All engines should pass the correctness gate with differences at rounding level. Runtime should grow polynomially with $n$, with the NumPy engine steepest because of its dense embedding. `native-gaussian` is expected to be fastest at large $n$, while at small $n$ fixed per-call overheads, not the matrix algebra, may dominate every engine. The measured workload should cost more than preparation alone, and the gap estimates measurement plus feed-forward. Absolute times depend on the machine (CPU, thread count, load); on Colab they are labelled provisional, and only slopes and engine ordering should be compared with the manuscript. The only acceptance criterion is the correctness gate ($\le 10^{-8}$); timings are not asserted.

Figure: a single log-log panel of median runtime versus number of modes, error bars of half the IQR, one colour per engine (`gaussian`, `native-gaussian`, `piquasso`), circles with solid lines for `prepare_entangle` and squares with dashed lines for `measure_feedforward`; the title states the timing status.

<!-- core:R37 -->
**Title:** Cost of compilation, channel analysis and execution versus logical gate count

**Why this experiment.** The paper presents `Circuit.compile` as a practical front end: a logical circuit is lowered to a measurement pattern on a finite-squeezing cluster, and the pattern can then be analysed or executed. That claim needs a measured cost per stage, because a compiler that is correct but scales badly with circuit length, or a channel extractor or executor that dominates the pipeline, would limit what the toolkit can be used for. Timing alone is not enough: a fast compiler that produces the wrong map would be meaningless. R37 therefore attaches correctness gates (exact logical map, native vs NumPy agreement) to every gate count before any timing is reported. R10 counts resources; R37 measures how the cost of each stage grows.

### Theory

Conventions: $\hbar = 2$, $[q,p] = 2i$, interleaved ordering $(q_0,p_0,q_1,p_1)$, vacuum covariance $V = I$. The compiler lowers every single-mode Gaussian gate to a product of teleportation steps

$$T(k) = \begin{pmatrix} -k & -1 \\ 1 & 0 \end{pmatrix},\qquad S = T(k_4)\,T(k_3)\,T(k_2)\,T(k_1),$$

where the shears $k_j$ come from a closed-form decomposition of the $2\times2$ symplectic $S$ (four steps, or five when $S_{10}\approx 0$). Each step prepares a momentum-squeezed resource node (squeezing $r$), applies a CZ, homodynes the old node at angle $\theta = \operatorname{atan2}(1,k)$ and feeds the outcome forward as a displacement. A CZ of weight $g$ ($p_0 \mapsto p_0 + g q_1$, $p_1 \mapsto p_1 + g q_0$) is an `Entangle` between the two frontier nodes followed by four Fourier steps $T(0)$ on each mode, since $T(0)^4 = I$.

For the circuit used here (gate $i$ is a rotation of mode 0 by $0.1 + 0.01 i$ when $i$ is even, a CZ of weight $0.2 + 0.01 i$ when $i$ is odd), every rotation costs 4 steps and every CZ 8 steps, so for $n$ logical gates

$$N_{\rm nodes} = 4\lceil n/2\rceil + 8\lfloor n/2\rfloor,\qquad N_{\rm commands} = 4 N_{\rm nodes} + \lfloor n/2\rfloor + 1,$$

each step contributing Prepare, Entangle, Measure and Displace (at $n = 100$: 600 nodes and 2451 commands). Compilation should therefore be $O(n)$.

`gaussian_channel` extracts the exact affine input-output map $x \mapsto M x + \xi$ of the pattern, treating the outcome-dependent correction displacements analytically (they are affine in the records). Finite squeezing ($r = 1$) only adds Gaussian noise to $\xi$; the transfer matrix $M$ must equal the logical symplectic exactly. The reference is the ordered product $M_{\rm ref} = S_n \cdots S_2 S_1$ of $4\times4$ rotation blocks on $(q_0,p_0)$ and CZ matrices; order matters because rotations of mode 0 do not commute with the CZ.

### What the code does

- System under test: `pg.Circuit(2)` with $n$ alternating `rotate`/`cz` gates, compiled by `Circuit.compile(squeezing=1.0, return_trace=True)`; then `pg.gaussian_channel(pattern)`; then one seeded trajectory (`seed=3`) on `gaussian` (NumPy) and `native-gaussian` (C++).
- Correctness gates, checked before timing at each $n$: relative Frobenius error $\lVert M - M_{\rm ref}\rVert / \lVert M_{\rm ref}\rVert \le 10^{-9}$, where $M_{\rm ref}$ is composed independently in the script from $2\times2$ rotations and CZ entries; and the maximum difference of all homodyne records between the NumPy and native engines with the same seed $\le 10^{-8}$. A failure raises `AssertionError`.
- Sweep: $n \in \{1, 2, 5, 10, 20, 50, 100\}$. Compilation is timed with `common.benchmark` (`time.perf_counter`), 2 warm-ups and 9 repeats for $n \le 20$, 1 warm-up and 5 repeats above; channel extraction and both executions use the same warm-up/repeat counts.
- Recorded: median, IQR and standard deviation of compile time, median channel and execution times, resource nodes, commands, measurements, map error, record difference, plus a timing-environment block (CPU count, `OMP_NUM_THREADS`, native build info). Timings are labelled provisional unless `PGQ_IDLE_MACHINE=1`.
- Output: `R37_compiler_runtime` (CSV/JSON with metadata) and figure `R37_compiler_runtime` (PDF/PNG/SVG).

### Expected outcome

Compile time should grow roughly linearly in $n$ (slope near 1 on the log-log plot once fixed overheads are amortised; at small $n$ the curve is flat because constant per-call overhead dominates). Channel extraction builds a dense source-space matrix whose width grows with the number of commands, so it can grow faster than linearly at large $n$. Execution on the native engine should be at or below the NumPy engine. Absolute timings are machine-dependent; on a Colab VM with few cores and shared load they will differ from the manuscript run and are flagged provisional. The map error should sit at rounding level ($\sim10^{-15}$ to $10^{-13}$), far below the $10^{-9}$ gate, and the native/NumPy record difference at $\lesssim 10^{-12}$, below the $10^{-8}$ gate.

Figure panels: (left) median time versus gate count on log-log axes for compile (with IQR/2 error bars), `gaussian_channel`, and execution on `gaussian` and `native-gaussian`; (right) resource nodes and commands versus gate count, two parallel straight lines of slope one on log-log axes. The figure title states whether the timings are final or provisional.

<!-- core:R38 -->
**Title:** Abstraction overhead of PhotoGraphiQ Gaussian engines relative to raw Piquasso

**Why this experiment.** PhotoGraphiQ wraps numerical engines in a graph/pattern/MBQC layer: labelled nodes, validated commands, per-command dispatch and record keeping. The paper reports what this layer costs, and how the package's own NumPy and C++ Gaussian engines compare with an external engine on the same physics. Without a matched-workload measurement a reader cannot tell whether the abstraction is negligible or dominant, nor whether a timing difference is caused by different physics. R38 fixes the physics (identical operations, checked to agree before timing) and varies only the execution path.

### Theory

Conventions: $\hbar = 2$, interleaved $(q_1,p_1,\dots,q_N,p_N)$, PhotoGraphiQ statistical covariance with vacuum $V = I$. Piquasso exports the anticommutator covariance $\sigma = 2V$ at the same $\hbar$, so the comparison uses $V_{\rm PQ} = \sigma/2$.

The workload on $N$ modes is a Gaussian unitary $S = S_{\rm CZ}\, \prod_i R_i(\phi_i) S_i(r_i)$ applied to vacuum: per mode $i$ (0-based) a squeezer $r_i = 0.1 + 0.01 i$ followed by a phase rotation $\phi_i = 0.2 + 0.01 i$, then a nearest-neighbour CZ chain of weight $s = 0.3$ on pairs $(i, i{+}1)$, $i = 0,\dots,N-2$. That is $3N - 1$ gates. The output covariance is $V = S S^{\mathsf T}$.

Each gate is a local $2\times2$ or $4\times4$ symplectic update. A dense update $V \mapsto S V S^{\mathsf T}$ with an embedded $2N\times 2N$ matrix costs $O(N^3)$ per gate, $O(N^4)$ for the circuit; a local update of the affected rows and columns costs $O(N)$ per gate, $O(N^2)$ overall. The measured ratio

$$\rho_{\rm engine}(N) = \frac{T_{\rm engine}(N)}{T_{\rm raw\ PQ}(N)}$$

is therefore governed by two effects: fixed per-gate overhead (validation, dispatch, object creation), which dominates at small $N$, and the asymptotic linear-algebra cost, which dominates at large $N$. The raw reference is the most efficient way to use Piquasso: one `Program` executed once by one `GaussianSimulator`. PhotoGraphiQ's `piquasso` backend instead runs one Piquasso program per command, by design, so that measurements and feed-forward can be interleaved; its ratio measures this architectural choice, not a deficiency of Piquasso.

### What the code does

- Raw reference: `raw_piquasso_run(N)` builds one Piquasso program (`Vacuum`, `Squeezing`, `Phaseshifter`, `ControlledZ(s=0.3)`) with `Config(hbar=2.0)`.
- System under test: the identical operations as a `pg.Pattern` (`Squeeze`, `Rotate`, `Entangle(i, i+1, 0.3)`, `Output`) executed by `pg.simulate(..., seed=0)` on `piquasso`, `gaussian` (NumPy) and `native-gaussian` (C++).
- Correctness gate before timing: absolute Frobenius error of each engine's covariance against the raw Piquasso covariance (divided by 2) must be $\le 10^{-9}$; otherwise `AssertionError`. This also checks the convention mapping between the packages.
- Sweep: $N \in \{5, 10, 20, 40\}$. `common.benchmark` with 2 warm-ups and 9 repeats for $N \le 10$, 1 warm-up and 5 repeats for $N \in \{20, 40\}$; medians and IQRs are recorded.
- Metrics: median runtimes of all four paths, `overhead_ratio` $= T_{\rm PG,piquasso}/T_{\rm raw}$, and the ratios of the NumPy and native engines to raw Piquasso; timing environment (CPU count, threads, native build) and provisional/final status.
- Output: `R38_piquasso_overhead` (CSV/JSON/metadata), figure `R38_piquasso_overhead` and a generated caption with the numbers at the largest $N$.

### Expected outcome

All engines must pass the $10^{-9}$ covariance gate; agreement is expected at rounding level. For timings only trends are meaningful, and they are machine-dependent (Colab runs on few, shared cores and will give different absolute numbers; timings are labelled provisional unless `PGQ_IDLE_MACHINE=1`). The `piquasso` backend is expected to be slower than raw Piquasso by a factor above one, because it pays program construction and simulator setup per gate ($3N-1$ times) instead of once. The native engine, with local $O(N)$ updates and no per-gate program construction, is expected to be the fastest path, possibly below raw Piquasso (ratio $< 1$). The NumPy engine performs dense per-gate updates with validation, so its ratio is expected to grow with $N$.

Figure panels: (left) log-log median runtime versus modes for raw Piquasso (one program), the PhotoGraphiQ `piquasso` backend, `gaussian` and `native-gaussian`; (right) the ratio of each PhotoGraphiQ path to raw Piquasso on a logarithmic axis, with a dotted line at parity (ratio 1). There is no timing acceptance criterion; the only pass/fail criterion is the covariance gate.

<!-- core:R39 -->
**Title:** Lossless JSON round trip of adaptive measurement patterns

**Why this experiment.** The paper claims that patterns are portable, archivable objects: a pattern written to disk with `Pattern.to_json` and read back with `Pattern.from_json` is the same computation. This underpins reproducibility (patterns stored next to results) and exchange between tools. A serializer can fail silently in ways that equality on gross structure does not catch: a symbolic `Outcome` turned into a constant, a lost edge weight, a reordered output tuple, a measurement efficiency reset to its default, or an expression that is restored but evaluates differently. Any such bug would make archived patterns compute something else. R39 checks the round trip at the level of structure, dependency graph, evaluated functions and execution on every Gaussian engine, and also for a non-Gaussian pattern with nonlinear feed-forward.

### Theory

Let $E$ = `Pattern.to_json` and $D$ = `Pattern.from_json` (versioned, allowlisted schema `photographiq`, version 1; parsing never evaluates Python code). The required invariant is

$$D(E(P)) \equiv P,$$

where $\equiv$ is checked at four levels:

1. Structure: equal inputs, outputs, command lists, declared parameters; equal resource-graph edges with weights, node squeezing values, and the typed IR `OpenGraph` dictionary after a JSON round trip.
2. Causality: equal edge sets of the classical dependency DAG (which measurement records feed which later commands, see R12).
3. Functions: a correction or adaptive angle is a function of records and parameters, e.g. $\theta_1(m_0;\vartheta) = 0.1\,m_0 + \vartheta$. Two expression trees can compare equal yet be evaluated differently if a node type is mis-restored, so the code evaluates every `Displace` $q$/$p$ expression and every measurement angle on random records $m_k \sim \mathcal N(0, 2^2)$ and parameters $\vartheta \sim U(-1,1)$, requiring exact equality.
4. Behaviour: with the same seed, inputs and parameters, original and restored patterns must yield identical outcomes and identical output moments on every Gaussian engine.

Test pattern: a three-node linear cluster `alice - bob - carol` (squeezing 1.0, input `alice`). `alice` is $q$-homodyned (record $m_0$); `bob` is displaced by $-m_0$ and measured at the adaptive angle $\theta_1 = 0.1\,m_0 + \vartheta$ (record $m_1$); `carol` receives $q \to q - m_1$, $p \to p + m_0 + m_1$. Conventions: $\hbar = 2$, homodyne at angle $\theta$ reads $q\cos\theta + p\sin\theta$. The non-Gaussian case is the cubic-phase injection gadget (`cubic_injection`, $\gamma = 0.1$, resource squeezing 0.5), whose corrections $\mathrm{QP}(-2\gamma m)$ and $Z(-\gamma m^2)$ are nonlinear in the record $m$.

### What the code does

- Round trip of the adaptive pattern; structural and DAG checks.
- Seeded execution on `gaussian` with coherent input $\alpha = 0.3 + 0.1i$ on `alice`, $\vartheta = 0.4$, seed 7: outcomes must be identical and output mean and covariance must agree to absolute Frobenius error $< 10^{-12}$.
- 25 random draws of records and $\vartheta$ (generator seed 39) to evaluate measurement metadata (node, key, measurement type, efficiency, noise, angle) and correction functions; exact equality required.
- Graph edges and weights, node squeezing, and the `OpenGraph.to_dict`/`from_dict` JSON round trip.
- Cross-engine: seeds 7, 8, 9 on `gaussian`, `native-gaussian` and `piquasso`; original vs restored must differ by exactly 0.0 in all records and means, and each engine must agree with the NumPy original to $< 10^{-10}$.
- Non-Gaussian: the injection pattern is round-tripped and executed on `native-fock` (cutoff 24, coherent input 0.2, branch fixed by `measurement_outcomes={"m": 0.7}`); restored commands must be equal and the two state vectors bit-identical (`allclose` with `atol=rtol=0`).
- Outputs: `R39_serialization.json` (all checks, with metadata), `R39_serialization_backends.csv` and the raw pattern JSON `R39_serialization_pattern.json`. No figure is produced.

### Expected outcome

Every check is Boolean and is expected to be `True`; any failure writes `R39_serialization_FAILURES.json` and raises `AssertionError`. Because serialization only stores and rebuilds the same objects, the seeded comparisons should give exact zeros, not merely small differences; the only numerical tolerances are $10^{-12}$ on the single-engine output moments and $10^{-10}$ for the cross-engine comparison with the NumPy original (different engines may differ by floating-point summation order). The printed summary lists each named check and its value.

<!-- core:R40 -->
**Title:** Seed reproducibility and statistical consistency of sampled shots

**Why this experiment.** Every stochastic result in the paper (homodyne trajectories, feed-forward branches, photon counts) is produced from a seed recorded with the result. That is only useful if two properties hold: the same seed reproduces the run bit for bit, and different seeds are independent draws from the same distribution. A bug in the random-number plumbing (a global generator, a seed that is ignored, or engines that consume random numbers differently) would break reproducibility or bias statistics without any visible error. R40 also checks a stronger claim that the code documents: the NumPy, C++ and Piquasso Gaussian engines give the same samples for the same seed.

### Theory

Conventions: $\hbar = 2$, $q = a + a^\dagger$, vacuum variance $\operatorname{Var}(q) = 1$. A coherent state $|\alpha\rangle$ measured in $q$ gives outcomes $m \sim \mathcal N(2\operatorname{Re}\alpha,\,1)$. With $\alpha = 0.3 + 0.2i$ the mean is $0.6$, and the sample mean of $M = 2000$ shots has standard error $1/\sqrt{2000} \approx 0.022$.

Different seeds are compared with the two-sample Kolmogorov-Smirnov statistic

$$D_{\rm KS} = \sup_x \bigl|\hat F_{42}(x) - \hat F_{43}(x)\bigr|,$$

whose null distribution (both samples from the same continuous law) is distribution-free; the test is accepted when its $p$-value exceeds 0.01. The self-comparison of two seed-42 runs must give $D_{\rm KS} = 0$ trivially.

Cross-engine RNG contract: `native-gaussian` draws outcomes in Python with the same `numpy.random.Generator` calls as `gaussian`, and `simulate` documents that `piquasso` and `gaussian` give identical seeded trajectories. The Gaussian samples are then deterministic functions of the same uniform stream and agree up to floating-point summation order. No such contract is written for the Fock engines: `native-fock` and `piquasso-fock` share the Python sampling path, so agreement is observed, not asserted. For photon counting on $|\alpha = 0.8\rangle$ the counts are Poissonian with mean $|\alpha|^2 = 0.64$ (cutoff 12 leaves a negligible tail).

### What the code does

- Pattern: one input mode measured with `Homodyne.q()`, input `GaussianInput.coherent(0.3+0.2j)`.
- `pg.run_shots(..., 2000, backend="gaussian")` with seeds 42, 42 and 43.
- Checks (all must hold): same seed bit-identical (`np.array_equal`); different seeds not identical; KS $p > 0.01$ for seed 42 vs 43; both sample means within 0.1 of $2\operatorname{Re}\alpha = 0.6$.
- Documented contract: the same 2000 shots with seed 42 on `native-gaussian` and on `piquasso`; the maximum absolute sample difference from `gaussian` must be $\le 10^{-12}$ for each.
- Observed only: 400 photon-number shots (`PhotonNumber`, cutoff 12, coherent 0.8, seed 42) on `native-fock` and `piquasso-fock`; the fraction of equal outcomes and both sample means (theory 0.64) are reported, not asserted.
- Outputs: `R40_seeded_shots.json` (summary, checks, metadata) and figure `R40_seeded_shots`. A failure writes `R40_seeded_shots_FAILURES.json` and raises `AssertionError`.

### Expected outcome

Same-seed runs are identical and different seeds are not; the KS $p$-value is a random number roughly uniform on $[0,1]$ under the null, so it should usually lie well above 0.01 (a failure at this level would occur by chance about 1% of the time for a fresh seed pair; with fixed seeds the outcome is deterministic). The mean errors should be of order the standard error, about 0.02, well inside the 0.1 tolerance (about $4.5$ standard errors). The native and Piquasso Gaussian samples are expected to match the NumPy samples to rounding level, typically exactly or at $\sim10^{-15}$. For the Fock engines the equal-outcome fraction is expected to be close to one and both counting means close to 0.64, but this is a reported observation.

Figure (single panel): overlaid density histograms (40 bins) of the seed-42 and seed-43 homodyne samples, both centred on the dashed vertical line at the theoretical mean $2\operatorname{Re}\alpha = 0.6$, with indistinguishable unit-variance Gaussian shapes.

<!-- core:R41 -->
**Title:** Automatic differentiation of Fock-space expectations against finite differences and native engines

**Why this experiment.** PhotoGraphiQ offers gradients of expectation values with respect to gate parameters through JAX (`photographiq.autodiff.expectation`), which is what variational and machine-learning workflows (PhotoGraphiQML) rely on. A wrong gradient is hard to notice: an optimiser still moves, just in the wrong direction or at the wrong rate. R41 tests the autodiff path in two ways. First, internal consistency: `jax.grad` must equal central finite differences of the same function. Second, independent realisation: the native C++ Fock engines, which have no autodiff and build their gates differently, must reproduce the forward value, and their finite-difference slope must reproduce the JAX gradient. This covers Gaussian gates, the non-Gaussian cubic phase and Kerr gates, and the non-unitary loss channel. The script runs only if JAX is installed and otherwise exits with SKIPPED.

### Theory

Conventions: $\hbar = 2$, $q = a + a^\dagger$, $p = -i(a - a^\dagger)$, Fock cutoff $c$ keeps photon numbers $n < c$. The differentiable path evolves a density matrix $\rho$ in the truncated space with projected generators: a gate $U(t) = \exp(t\,G)$ is realised as $\exp(t\,P G P)$, e.g. $G = i n$ (rotation), $G = \tfrac{i}{2}(p_d\, q - q_d\, p)$ (displacement by $(q_d, p_d)$), $G = \tfrac12(a^2 - a^{\dagger 2})$ (squeezing), $G = i q^3/6$ (cubic phase), $G = i n^2$ (Kerr). Loss with transmissivity $\eta$ is the Kraus channel

$$\rho \mapsto \sum_{\ell} K_\ell \rho K_\ell^\dagger,\qquad \langle n-\ell|K_\ell|n\rangle = \sqrt{\tbinom{n}{\ell}}\,(1-\eta)^{\ell/2}\,\eta^{(n-\ell)/2}.$$

The objective is $f(t) = \operatorname{Tr}[\rho(t)\,O]$ with $O \in \{n, q, p\}$, and JAX differentiates through `expm` and the Kraus sum.

Central differences satisfy

$$\frac{f(t_0+h) - f(t_0-h)}{2h} = f'(t_0) + \frac{h^2}{6} f'''(t_0) + O\!\left(\frac{\epsilon\,|f|}{h}\right),$$

with truncation error decreasing as $h^2$ and rounding error ($\epsilon \approx 2.2\times10^{-16}$) increasing as $1/h$, so the error is smallest at an intermediate $h$ (around $10^{-4}$ to $10^{-5}$ for $O(1)$ derivatives). The acceptance therefore requires only the best tested step to agree.

The native engines (`native-fock`, and `native-mixed-fock` for loss) use exact projected matrix elements $P\,e^{tG}P$ rather than $e^{tPGP}$. The two realisations differ at finite cutoff (e.g. by about $6\times10^{-5}$ for squeezing at $c = 12$) and converge as $c$ grows, so the cross-check is done at $c = 40$. Two closed forms for orientation (untruncated): under loss $\langle n\rangle = \eta|\alpha|^2$, so $d\langle n\rangle/d\eta = |\alpha|^2 = 0.13$ for $\alpha = 0.3 + 0.2i$; under displacement $\langle q\rangle = 2\operatorname{Re}\alpha + q_d$, slope 1.

### What the code does

- Six single-mode patterns, each with one `Parameter("t")`: `Rotate` (observable $q$), `Displace(q=t)` ($q$), `Squeeze` ($n$), `CubicPhase` ($p$), `Kerr` ($q$), `Loss(t, 0.0)` ($n$); input coherent $\alpha = 0.3 + 0.2i$; evaluation point $t_0 = 0.4$, except $t_0 = 0.5$ for the loss transmissivity.
- Autodiff vs finite differences of the same function at cutoff 12 (`jax_enable_x64`): steps $h \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$; relative error $|g_{\rm JAX} - g_{\rm FD}|/|g_{\rm JAX}|$.
- Native cross-check at cutoff 40: forward value $f(t_0)$ from `autodiff.expectation` vs `pg.simulate` on the native engine (absolute difference $\le 10^{-9}$), and central finite difference of the native observable with $h = 10^{-4}$ vs `jax.grad` at cutoff 40 (relative difference $\le 10^{-5}$). The forward difference at cutoff 12 is reported but not asserted. Finite differences are labelled as such and never presented as automatic differentiation.
- Acceptance: for every gate, the best of the four steps has relative error $\le 10^{-5}$; plus the two native tolerances above. Failures write `R41_jax_gradients_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R41_jax_gradients` (CSV/JSON with cutoff, native cross-check table and a supported-combinations note), `R41_jax_gradients_native_crosscheck.csv`, and figure `R41_jax_gradients`.

### Expected outcome

For each gate the finite-difference error should follow the V-shaped curve described above: at $h = 10^{-2}$ it is dominated by the $h^2$ term (roughly $10^{-5}$ to $10^{-6}$ relative, depending on $f'''$), it decreases about a hundredfold per decade of $h$, and at $h = 10^{-5}$ rounding begins to dominate. The best step is expected to reach $10^{-8}$ to $10^{-10}$, far below the $10^{-5}$ criterion. Loss is linear in $\eta$, so its truncation term vanishes and only rounding remains. The native forward values should agree with the autodiff values at the $10^{-12}$ level at cutoff 40, and the native finite-difference slopes should agree with `jax.grad` to about $10^{-8}$, limited by the $h = 10^{-4}$ truncation error.

Figure (single panel): log-log relative gradient error versus finite-difference step $h$, one line per gate (rotation, displacement, squeezing, cubic phase, Kerr, loss); errors are floored at $10^{-16}$ for display.

<!-- core:R42 -->
**Title:** Unbiasedness of score-function and pathwise gradient estimators

**Why this experiment.** Gradients of objectives that depend on measurement outcomes (heralding, feed-forward branches, homodyne records) cannot always be obtained by differentiating a deterministic simulation. PhotoGraphiQ provides two Monte Carlo primitives for this: `score_function_surrogate` (likelihood-ratio, REINFORCE) and `gaussian_sample` (reparameterisation, pathwise). Both are easy to implement with a subtle bias, for example by letting the gradient flow through the sampled values or through the baseline. R42 checks that each estimator is statistically unbiased on problems with exact gradients, including one where the samples are real homodyne records produced by the C++ `native-gaussian` engine. This functionality is labelled experimental; the script runs only if JAX is installed.

### Theory

Score-function estimator. For samples $x_i \sim p_\theta$ and payoff $f$,

$$\nabla_\theta\,\mathbb E_{p_\theta}[f(X)] = \mathbb E_{p_\theta}\!\left[(f(X) - b)\,\nabla_\theta \log p_\theta(X)\right],$$

for any baseline $b$ independent of $X$. The surrogate $L = \frac1N\sum_i \bigl[f_i + \mathrm{sg}(f_i - b)\,(\log p_\theta(x_i) - \mathrm{sg}(\log p_\theta(x_i)))\bigr]$, with $\mathrm{sg}$ = stop-gradient, has the value of the sample mean and the gradient of the estimator above (here $b = 0$).

- Bernoulli case: $X \sim \mathrm{Bernoulli}(\sigma(\theta))$, $f(1) = 2$, $f(0) = -1$. Exact gradient $\frac{d}{d\theta}\mathbb E f = (f_1 - f_0)\,\sigma(\theta)(1-\sigma(\theta))$, about $0.733$ at $\theta = 0.3$.
- Homodyne case ($\hbar = 2$, vacuum variance 1): a coherent state $\alpha = 0.3$ displaced by $q_d = \theta$ and measured in $q$ gives $m \sim \mathcal N(\mu, 1)$ with $\mu = 2\operatorname{Re}\alpha + \theta$. With $f(m) = m^2$, $\mathbb E f = \mu^2 + 1$ and the exact gradient is $2\mu = 2(0.6 + 0.3) = 1.8$. The log-likelihood used is $\log p_\theta(m) = -\tfrac12 (m - \mu)^2$ + const. The engine only supplies samples; JAX differentiates the surrogate, not the simulator.

Pathwise estimator. Writing $X = \mu_\theta + L z$ with $LL^{\mathsf T} = \Sigma$ and fixed base noise $z \sim \mathcal N(0, I)$, $\nabla_\theta\mathbb E f(X) = \mathbb E_z[\nabla_\theta f(\mu_\theta + L z)]$. For $X = \theta + \sigma z$, $\sigma = 0.7$, $f = X^2$: exact gradient $2\theta = 0.6$ at $\theta = 0.3$.

Unbiasedness is tested statistically. With $R$ independent repeats giving estimates $\hat g_r$, bias $= \bar g - g_{\rm exact}$ and standard error $\mathrm{SE} = s/\sqrt R$; an unbiased estimator has $|\mathrm{bias}|/\mathrm{SE}$ of order one.

### What the code does

- Bernoulli score function: $N = 20000$ samples per repeat, $R = 10$ repeats (NumPy generators seeded $1000 + r$), gradient via `jax.grad` of the surrogate at $\theta_0 = 0.3$.
- Pathwise: $N = 20000$ base-noise draws per repeat, $R = 10$ (seeds $2000 + r$), samples built with `gaussian_sample` under `jax.vmap`, gradient of the sample mean of $X^2$.
- Native homodyne score function: `pg.run_shots` of `Displace(0, q=0.3)` followed by `Homodyne.q()` on `native-gaussian`, 3000 shots per repeat, $R = 10$ (seeds $3000 + r$), coherent input $\alpha = 0.3$.
- Acceptance (each estimator): $|\mathrm{bias}| \le 5\,\mathrm{SE} + 10^{-6}$. A failure writes `R42_stochastic_gradients_FAILURES.json` and raises `AssertionError`.
- Outputs: `R42_stochastic_gradients.json` (exact gradient, mean estimate, bias, SE, sample sizes per estimator, with metadata) and figure `R42_stochastic_gradients`.

### Expected outcome

All three biases should be a fraction of a few standard errors. The pathwise estimator has the smallest variance (its per-sample gradient $2(\theta + \sigma z)$ has standard deviation $2\sigma = 1.4$, so each repeat has spread $1.4/\sqrt{20000} \approx 0.01$ and the SE of the mean over 10 repeats is about $0.003$). The score-function estimators have larger variance; the homodyne case, with fewer samples (3000) and a payoff $m^2$ with heavy spread, has the largest SE of the three. The 5-SE band corresponds to a false-alarm probability well below $10^{-4}$ for a truly unbiased estimator with approximately normal repeat means; with fixed seeds the outcome is deterministic.

Figure (single panel): for each of the three estimators, the mean estimate with error bars of $\pm 5$ SE and the exact gradient as a cross; the cross should lie inside each error bar.

<!-- core:R43 -->
**Title:** Validation of the native Gaussian engine against first-principles phase-space evolution

**Why this experiment.** The compiled `native-gaussian` engine (C++, in-place $O(nk)$ updates of the $k$ affected modes) is the default fast path for Gaussian simulation, and the performance results (R44) only mean something if it computes the same physics as an independent reference. Errors that matter here are subtle: a wrong sign in one gate symplectic, loss noise added in the wrong frame, a Schur-complement conditioning that loses symmetry or positivity, rounding that grows unexpectedly with depth, or a native trajectory that diverges from the NumPy one for the same seed. R43 tests states, gates, loss, homodyne and heterodyne conditioning and adaptive feed-forward against references that share no code with either engine, across 1 to 128 modes and depths up to 2560 gates.

### Theory

Conventions: $\hbar = 2$, $x = (q_1,p_1,\dots,q_n,p_n)$, $[x_j,x_k] = 2i\Omega_{jk}$ with $\Omega = \bigoplus_n \begin{pmatrix}0&1\\-1&0\end{pmatrix}$, statistical covariance with vacuum $V = I$; physicality is $V + i\Omega \succeq 0$.

States (closed forms): coherent mean $(2\operatorname{Re}\alpha, 2\operatorname{Im}\alpha)$, $V = I$; squeezed $V = R(\theta)\,\mathrm{diag}(e^{-2r}, e^{2r})\,R(\theta)^{\mathsf T}$; thermal $V = (2\bar n + 1)I$; displaced squeezed thermal as the product; mean photon number $\langle n\rangle = (\operatorname{tr}V - 2)/4 + |d|^2/4$.

Gates: a quadratic unitary $U = \exp(-\tfrac{i}{2}x^{\mathsf T}Mx)$ acts in the Heisenberg picture as $x \mapsto Sx$ with

$$S = \exp(2\Omega M),\qquad \mu \mapsto S\mu,\quad V \mapsto S V S^{\mathsf T},$$

with generators for rotation $e^{i\phi n}$, squeezing $e^{r(a^2 - a^{\dagger2})/2}$, shear $e^{isq^2/4}$, CZ $e^{igq_uq_v/2}$ and beam splitter $e^{\theta(a_v^\dagger a_u - a_u^\dagger a_v)}$ (the helpers are themselves checked against Fock-space Heisenberg action in the test suite). Thermal loss: $V \mapsto XVX^{\mathsf T} + Y$, $X = \sqrt\eta\, I$, $Y = (1-\eta)(2n_{\rm th}+1)I$.

Conditioning: homodyne at angle $\theta$ reads $q\cos\theta + p\sin\theta$. The reference builds the physical dilation (rotate by $-\theta$; for efficiency $\eta$, mix with a vacuum ancilla on a beam splitter with $\cos^2 = \eta$ and rescale the reading by $1/\sqrt\eta$, add calibrated noise; for heterodyne, a 50:50 splitter with vacuum and ideal $q$ and $p$ homodynes scaled by $\sqrt2$) and conditions through the joint precision matrix $\Lambda = \Sigma^{-1}$:

$$V_{x|y} = (\Lambda_{xx})^{-1},\qquad \mu_{x|y} = \mu_x - (\Lambda_{xx})^{-1}\Lambda_{xy}(y - \mu_y),$$

which is algebraically equal to, but numerically independent of, the Schur-complement formula $V_{x|y} = V_{xx} - V_{xy}V_{yy}^{-1}V_{yx}$ the engines use.

Error model for deep circuits (declared heuristic, not proven): each local update costs a few $k^2$ flops per entry, so the forward relative error is bounded as $\le C\,D\,\epsilon$ with $C = 64$, depth $D$, $\epsilon = 2^{-52}$. To keep the covariance well scaled at large $D$, active gate parameters are drawn with scale factor $\kappa = \min(1, \sqrt{40n/D})$: squeezing $r \sim \mathcal N(0, (0.15\kappa)^2)$, CZ and shear weights $\sim \mathcal N(0, (0.3\kappa)^2)$, so the accumulated random-walk squeezing stays $O(1)$.

### What the code does

- Engines under test: `native-gaussian` and NumPy `gaussian` via `registry.create_backend`, driven gate by gate.
- States: coherent ($\alpha = 0, 0.7-0.3i, -1.5+2i$), squeezed ($(r,\theta) = (0.3,0), (1,0.7), (2,-1.2)$), thermal ($\bar n = 0.1, 1, 5$), and one displaced squeezed thermal state; mean, covariance and $\langle n\rangle$ errors $\le 10^{-13}$.
- Random circuits (`--full`, seed 20261143): $n \in \{1,2,4,8,16,32,64,128\}$, $D \in \{10,40,160,640,2560\}$, two families (unitary: pure start, gates rotate/squeeze/beam splitter/CZ/shear/displace; noisy: mixed start and thermal loss with $\eta \in [0.6,1]$, $n_{\rm th} \in [0,0.2]$), 80 circuits. Random start states $V = S\,\mathrm{diag}(\nu)S^{\mathsf T}$ with the generator scaled by $1/\sqrt{4n}$. Reference: dense $2n\times2n$ propagation with $S = \exp(2\Omega M)$. Metrics: mean error, covariance error relative to $\max|V_{\rm ref}|$, physicality margin $\lambda_{\min}(V + i\Omega)/\lVert V\rVert_2$; for the unitary family the composed native map $S$ is extracted column by column from the native mean update and its symplectic residual $\lVert S\Omega S^{\mathsf T} - \Omega\rVert_{\max}/\lVert S\rVert_2^2$ and its error against the reference $S$ are computed.
- Conditioning: random mixed states on $n = 2, 4, 8$; homodyne $(\theta,\eta,\text{noise},\text{outcome}) = (0,1,0,0.8), (1.1,1,0,-1.3), (2.3,0.7,0.15,0.4)$ and heterodyne with outcome $(0.7,-1.1)$; conditional moments (relative to $\max(1,\max|V|)$) and outcome density compared with the dilation/precision reference; native vs NumPy for the same outcome.
- Feed-forward (`--full`: 20 seeds): a two-step teleportation wire ($r = 0.6$, coherent $\alpha = 0.3 - 0.2i$) whose second homodyne angle $\operatorname{atan2}(1, 0.2 + 0.5a)$ depends on the first outcome $a$, compared with hand propagation of each step and the analytic log-likelihood; and a 24-step Fourier wire with shears evenly spaced in $[-0.6, 0.6]$ (squeezing 1.0). For both, native and NumPy runs with the same seed, and a NumPy replay with the native outcomes postselected.
- Acceptance: states $\le 10^{-13}$; circuit mean/covariance error, symplectic residual and native $S$ error $\le 64\,D\,\epsilon$; physicality margin $\ge -10^{-12}$; conditional moments $\le 10^{-10}$; density $\le 10^{-10}$ relative; native vs NumPy same seed $\le 10^{-11}$; feed-forward vs hand and log-likelihood $\le 10^{-11}$. Failures go to `R43_native_gaussian_validation_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R43_native_gaussian_states`, `_conditional`, `_feedforward`, `R43_native_gaussian_validation` (CSV/JSON/metadata), figure `R43_native_gaussian_validation` and a generated caption.

### Expected outcome

Closed-form states should be reproduced exactly or to a few ulp. Circuit errors should grow slowly with depth and lie one to three orders of magnitude below the $64D\epsilon$ line (from $\sim 10^{-13}$ at $D = 10$ to $\sim 10^{-12}$ at $D = 2560$; the largest values in the pure single-mode family come as much from the dense `expm` reference as from the engine). The native symplectic residual should stay at $\sim10^{-15}$, showing the composed map is symplectic to rounding. Physicality margins should be at or just below zero by rounding ($\sim -10^{-15}$), since pure states lie on the boundary. Conditional moments and densities should agree at $10^{-16}$ to $10^{-14}$, and native and NumPy trajectories should coincide to $\sim10^{-15}$.

Figure panels: (a) covariance error of the noisy family versus depth, one line per mode count, with the dashed $64D\epsilon$ bound; (b) native symplectic residual of the unitary family versus depth with the same bound; (c) for each conditioning case (labelled hom, hom(eta=0.7), het with $n$), the native conditional-moment error, relative density error and native-vs-NumPy difference on a log axis, with the $10^{-10}$ acceptance line (exact zeros drawn at $10^{-17}$).

<!-- core:R44 -->
**Title:** Performance of the native Gaussian engine, measured together with accuracy

**Why this experiment.** The paper claims that the compiled Gaussian engine makes large multimode simulations and large trajectory batches practical. A speedup is only a valid claim when the competing implementations compute the same result, and when it is measured in a way that isolates the quantity of interest (warm-up excluded, repeats, per-configuration memory, controlled thread count). R44 times the native engine against the NumPy backend, a dense symplectic propagation written in the script and Piquasso, across modes, gate count, number of conditional measurements, batch size and threads, and checks the accuracy of every timed run against the first-principles reference of R43. A timing that came with an accuracy failure would be rejected.

### Theory

Conventions as in R43: $\hbar = 2$, interleaved quadratures, vacuum $V = I$; Piquasso stores $\sigma = 2V$.

Cost model. A gate acting on $k \le 2$ modes has a $2k\times2k$ symplectic $S_{\rm loc}$. The NumPy backend embeds it in $2n\times2n$ and computes $V \mapsto SVS^{\mathsf T}$ plus per-gate validation, $O(n^3)$ per gate. A local update only touches the $2k$ affected rows and columns,

$$V_{AA} \mapsto S_{\rm loc}V_{AA}S_{\rm loc}^{\mathsf T},\qquad V_{AB} \mapsto S_{\rm loc}V_{AB},$$

which is $O(nk)$ per gate. For $G$ gates the expected scalings are therefore $O(Gn^3)$ (dense) versus $O(Gn)$ (native) plus fixed per-call overhead. Homodyne conditioning of one mode by the Schur complement

$$V' = V_{BB} - \frac{V_{BA}h\,h^{\mathsf T}V_{AB}}{h^{\mathsf T}V_{AA}h},\qquad \mu' = \mu_B + \frac{V_{BA}h}{h^{\mathsf T}V_{AA}h}(m - h^{\mathsf T}\mu_A),\qquad h = (\cos\theta,\sin\theta),$$

is a rank-one update, $O(n^2)$. Batched kernels apply the same local gate and conditioning to $B$ independent states; they are embarrassingly parallel over $B$ and are the only native Gaussian kernels that use OpenMP (single-trajectory kernels are serial).

### What the code does

- Implementations: `native` (`native-gaussian` backend), `numpy-backend` (`gaussian` backend), `dense` (embedded NumPy matrices, using the same gate matrices as the reference, so its accuracy is a consistency floor, not independent evidence) and `piquasso` (Piquasso `GaussianSimulator` running the whole circuit as one program, with `Phaseshifter`, `Squeezing`, `Beamsplitter`, `ControlledZ`, `QuadraticPhase`, `Displacement`, `Attenuator`).
- Workload: R43's noisy random circuits (thermal loss included) on a random mixed state; start-state validation is excluded from timing by loading an unvalidated copy; the NumPy backend's own per-gate validation is included.
- Publication sweep (`--full`, seed 20261144): modes $n \in \{4,8,16,32,64,128,256\}$ at 200 gates (Piquasso skipped at $n = 256$); gates $\in \{50,200,800,3200\}$ at $n = 32$; sequential postselected homodynes ($\theta = 0.3$, outcome 0.2) $\in \{1,4,16,31\}$ at $n = 32$ (native and NumPy backend); batched beam splitter ($\theta = 0.7$) plus homodyne conditioning on $n = 16$ states, $B \in \{1,16,256,4096\}$ at one thread, native OpenMP kernels vs a vectorised NumPy formula; thread sweep `OMP_NUM_THREADS` $\in \{1,2,4,8,12\}$ at $B = 4096$.
- Method: each configuration runs in a fresh subprocess (so peak memory is per configuration and the thread setting takes effect, also for NumPy BLAS), with a 1800 s per-configuration timeout (a hang guard sized for slow 2-vCPU runtimes); `common.benchmark` with 2 warm-ups and 11 repeats (median, IQR); memory from `psutil` peak working set (Windows) or `ru_maxrss`, reported as absolute peak and increase over the pre-run peak.
- Accuracy: max of covariance error relative to $\max|V_{\rm ref}|$ and mean error, against R43's dense reference (circuits) or a repeated precision-matrix homodyne reference (measurements); acceptance $\le 10^{-10}$. Batched native vs vectorised NumPy outputs: per-entry relative difference $\le 10^{-12}$. Failures are written to `R44_native_gaussian_performance_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R44_native_gaussian_performance` (CSV/JSON with parameters, build info, batch cross-check), figure `R44_native_gaussian_performance` and a generated caption with speedups.

### Expected outcome

Accuracy: every implementation should agree with the reference at $10^{-14}$ to $10^{-12}$, and the batch kernels with NumPy at $\sim10^{-16}$. Timing trends: in the mode sweep the native runtime should rise slowly (near-linear in $n$ after the per-gate overhead), while the NumPy backend rises steeply (towards $n^3$), so the speedup grows with $n$ and is largest at $n = 256$; Piquasso, running one program per circuit, sits in between. In the gate sweep all curves are roughly linear in $G$. Measurement sequences show the same native advantage. In the batch sweep the native kernels should beat the vectorised NumPy update at large $B$. The thread curve need not be monotonic: absolute timings depend on the machine and its load. On Colab only a few cores exist, so thread counts above the core count oversubscribe and give no further gain, and all numbers are provisional unless the run declares `PGQ_IDLE_MACHINE=1`. The full sweep is long (tens of minutes).

Figure panels: (a) runtime vs modes at 200 gates, (b) runtime vs gate count at $n = 32$, (c) runtime vs number of sequential homodynes at $n = 32$, (d) accuracy of the runs in (a) with the $10^{-10}$ acceptance line, (e) batched runtime vs $B$ at one thread, (f) batched runtime vs `OMP_NUM_THREADS` at $B = 4096$; error bars are IQR/2, and the suptitle marks provisional timings.

<!-- core:R45 -->
**Title:** Accuracy and cutoff convergence of the native pure-state Fock engine

**Why this experiment.** Non-Gaussian simulation in PhotoGraphiQ runs on the compiled `native-fock` engine, which truncates Fock space and applies every gate as a projected operator. Two different errors must be kept apart: implementation error (the engine does not compute what the truncated model prescribes) and truncation error (the truncated model differs from infinite-dimensional physics). The paper claims the first is at rounding level and the second is always reported through a recorded retained norm, never hidden. R45 tests both, with separate references, for displacement, squeezing, beam splitters, Kerr, cubic phase and conditional photon counting, over cutoffs up to 96, and compares the same sequences with Piquasso's Fock simulator.

### Theory

Conventions: $\hbar = 2$, $q = a + a^\dagger$, $p = -i(a - a^\dagger)$, $\alpha = (q + ip)/2$. Truncation keeps the total photon number below the cutoff $c$, $n_1 + \dots + n_m < c$; for one mode the dimension is $c$, for $m$ modes $\binom{m + c - 1}{m}$. A gate $U$ acts as

$$|\psi\rangle \mapsto \frac{P U |\psi\rangle}{\lVert P U |\psi\rangle\rVert},\qquad \text{retained norm } \lVert P U\psi\rVert^2 \text{ recorded}.$$

Exact-in-truncation references give $PU\psi/\lVert PU\psi\rVert$ independently:

- coherent: $\langle n|\alpha\rangle = e^{-|\alpha|^2/2}\alpha^n/\sqrt{n!}$ (30-digit mpmath);
- squeezed vacuum: $\langle 2k|S(r)|0\rangle = (\cosh r)^{-1/2}(-\tanh r)^k\sqrt{(2k)!}/(2^k k!)$;
- displaced Fock state, $m \ge n$: $\langle m|D(\alpha)|n\rangle = \sqrt{n!/m!}\,\alpha^{m-n}e^{-|\alpha|^2/2}L_n^{(m-n)}(|\alpha|^2)$ (and the symmetric form for $m < n$), summed exactly at 50 digits to survive the cancellation in the Laguerre polynomial for large $|\alpha|$ and $n$;
- Kerr $e^{i\kappa n^2}$: exact phases on the coherent amplitudes;
- 50:50 beam splitter: `expm` of $\theta(a_2^\dagger a_1 - a_1^\dagger a_2)$ on an enlarged product space; being number conserving it leaks nothing;
- projected cubic phase $V = e^{i\gamma q^3/6}$: $\langle m|V|n\rangle = \int \psi_m(x)\,e^{i\gamma x^3/6}\,\psi_n(x)\,dx$ by quadrature with Hermite functions (8001 points on $[-22, 22]$);
- two-mode squeezed vacuum from two squeezers ($\pm r$) and a 50:50 splitter: $\sum_k \sqrt{1-\lambda^2}\,\lambda^k|k,k\rangle$, $\lambda = \tanh r$; within the cutoff the components with $2k < c$ survive, so $P(k) = (1-\lambda^2)\lambda^{2k}/\sum_{2j<c}(1-\lambda^2)\lambda^{2j}$ and the conditional state is exactly $|k\rangle$.

Untruncated observables: $\langle n\rangle = |\alpha|^2$ (coherent), $\sinh^2 r$ (squeezed), $\langle a\rangle$ under Kerr from a 400-level sum, $P(k) = (1-\lambda^2)\lambda^{2k}$, and for the cubic phase on vacuum $p \mapsto p + \gamma q^2$ (since $[q^3, p] = 6iq^2$ at $\hbar = 2$), so $\langle p\rangle = \gamma$ and $\langle p^2\rangle = 1 + 3\gamma^2$.

Cubic phase has two definitions in the engine: `projected` ($PVP$, leaks norm, has an exact-in-truncation reference) and `truncated_generator` ($\exp(i\gamma\,q_c^3/6)$ with the truncated $q_c$, unitary on the truncated space, never reports leakage).

### What the code does

- Engines: `native-fock` with the norm guard opened to $1 - 10^{-12}$ (truncation is studied, never rejected) and `piquasso-fock` as secondary comparison at cutoffs $\le 48$.
- Cases: $D(\alpha)|0\rangle$ for $\alpha = 1.2 + 0.5i$ ($|\alpha| = 1.30$) and $\alpha = 4$; $D(4)|c/2\rangle$; $S(0.8)|0\rangle$; Kerr $\kappa = 0.35$ on $|1.2 + 0.5i\rangle$; cubic phase $\gamma \in \{0.1, 0.3\}$ on $|0\rangle$, both definitions (native) or the default (Piquasso); 50:50 splitter on $|1,1\rangle, |2,1\rangle, |3,3\rangle, |4,2\rangle$ (when $n_1 + n_2 < c$); TMSV $r = 0.6$ with photon counting $k = 0..3$ (when $2k < c$).
- Cutoff sweep (`--full`): $c \in \{6, 8, 12, 16, 24, 32, 48, 64, 96\}$.
- Metrics: state distance with global phase aligned, retained-norm error, untruncated observable error, $|\lVert\psi\rVert - 1|$, TMSV conditional infidelity.
- Acceptance (native rows): exact-in-truncation state and retained norm $\le 10^{-12}$ ($\le 10^{-10}$ for the projected cubic phase, quadrature floor); HOM coincidence $P(1,1) \le 10^{-15}$; normalisation $\le 10^{-12}$; at the top cutoff, untruncated observable errors (excluding beam splitter and cubic cases) must satisfy $|{\rm err}| \le 10\,c\,(1 - \text{retained}) + 10^{-12}$, i.e. be explained by the recorded leaked norm; cubic-phase observable errors must be non-increasing over the last three cutoffs. Failures go to `R45_native_fock_accuracy_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R45_native_fock_accuracy`, `R45_native_fock_vs_piquasso` (CSV/JSON/metadata), figure `R45_native_fock_accuracy` and a generated caption.

### Expected outcome

Exact-in-truncation errors should sit at $10^{-16}$ to $10^{-15}$ at every cutoff, including $|\alpha| = 4$ and interior columns of $D(4)$, showing the engine computes the truncated model to rounding; the HOM coincidence should be exactly zero. Untruncated observable errors and leaked norm should fall rapidly (Gaussian tails, roughly super-exponentially in $c$ once $c$ exceeds a few times the mean photon number) and saturate at rounding; the $|\alpha| = 4$ coherent state ($\langle n\rangle = 16$) needs the largest cutoffs. The cubic phase has a heavier tail and converges more slowly, but monotonically. Piquasso should agree to rounding on $D(4)|0\rangle$, squeezing and the truncated-generator cubic phase, while its error on $D(4)|c/2\rangle$ grew with cutoff in the manuscript run (interior columns of a large displacement are numerically delicate), whereas the native engine stays at rounding.

Figure panels: (a) engine vs exact-in-truncation state error versus cutoff for each case, with the $10^{-12}$ line; (b) convergence of untruncated observables; (c) leaked norm $1 - \lVert PU\psi\rVert^2$ (number-conserving gates and the truncated-generator cubic phase leak nothing and are drawn at the $10^{-17}$ floor); (d) native vs Piquasso errors on $D(4)|c/2\rangle$, $S(0.8)|0\rangle$ (state errors) and the truncated-generator cubic phase with $\gamma = 0.3$ ($\langle p^2\rangle$ error). A shared legend below the panels names the cases.

<!-- core:R46 -->
**Title:** Implemented Fock-space optimisations and their measured effect

**Why this experiment.** The paper describes several design choices that keep multimode Fock simulation tractable: total-photon truncation, gate application on fibres without forming a global operator, block-diagonal construction of passive gates, phase multiplication for diagonal gates, and caching. Claims about such optimisations are only credible if each one is shown to leave the numerical output unchanged and its effect on runtime or memory is measured, not asserted. R46 restricts itself to optimisations that exist in the code (`cpp/src/fock`, `backends/native_fock.py`, `gpu/fock.py`), compares every variant on the same truncated operator, and labels as observations the effects that cannot be ablated from Python.

### Theory

Truncation keeps $n_1 + \dots + n_m < c$, giving the dimension

$$D_{\rm tot}(m,c) = \binom{m + c - 1}{m}\quad\text{versus}\quad D_{\rm prod} = c^m$$

for a per-mode product basis. These are different approximations (a model choice), so the product basis is reported only as a size; for fixed $c$, $D_{\rm tot}$ grows polynomially in $m$ ($\sim m^{c-1}/(c-1)!$) while $c^m$ grows exponentially.

Fibre-local application. A $k$-mode gate with local matrix $U_{\rm loc}$ (in the $k$-mode total-photon basis with cutoff $c$) acts independently on each fibre $f$, i.e. each fixed occupation $\bar n$ of the other $m-k$ modes. The fibre has remaining budget $c - |\bar n|$ and dimension $d_f = \binom{k + c - |\bar n| - 1}{k}$, and the gate applies the leading $d_f\times d_f$ block $P U_{\rm loc} P$. The cost is $\sum_f d_f^2$ and the only stored operator is $U_{\rm loc}$; the alternative is a global sparse $D\times D$ matrix $G$ (CSR) or a dense $D\times D$ matrix, both representing exactly the same $PUP$.

Passive gates. A passive Gaussian unitary (beam splitter, rotation) commutes with $\hat N = \sum_i a_i^\dagger a_i$, so its local matrix is block diagonal over sectors $N = 0,\dots,c-1$, sector $N$ of a two-mode gate having $N+1$ states. The non-zero fraction is $\sum_{N=1}^{c} N^2 / [c(c+1)/2]^2$ (about 2.1% at $c = 64$). With total-photon truncation a passive gate maps the truncated space onto itself and is exactly unitary there (retained norm 1).

Diagonal gates. Kerr $e^{i\kappa n^2}$, rotations and cross-Kerr are diagonal in the Fock basis, so they reduce to elementwise phase multiplication instead of a matrix product.

### What the code does

- Engine: `photographiq._native.fock` (`apply_pure`, `phase_pure`, `gaussian_unitary`, `dimension`, `basis`) and `photographiq.gpu.fock.apply_operator` (scipy.sparse on CPU; cuSPARSE via CuPy on GPU only if a CUDA device is usable, with `Execution.device` recorded).
- Gate variants (`--full` grid, seed 20261146): $(m,c) \in \{(2,30),(3,16),(4,12),(5,10),(6,8),(8,6),(4,20),(6,12),(8,9),(10,7)\}$; batches $B \in \{1, 32\}$ of random normalised states; gates: beam splitter $\theta = 0.7$ (passive), CZ $g = 0.4$ (active, two modes), squeezer $r = 0.3$ (one mode), all built by `gaussian_unitary`. Variants: native fibre kernel (for $B = 32$ a Python loop over states), global sparse $G$ with scipy, global sparse $G$ with cuSPARSE (incl. transfers), dense $D\times D$ matrix when $D \le 4000$. Recorded: median/IQR runtime (`common.benchmark`, 9 repeats), exact operator and state bytes (`.nbytes`), max difference from the native output, output norms.
- Phase study (same grid): Kerr phases $0.35\,n^2$ on mode 0 via `phase_pure` vs `apply_pure` of the diagonal matrix.
- Sector study at two-mode cutoffs $\{6,10,16,24,40,64\}$: off-sector entries and zero fraction of the beam-splitter local matrix, its unitarity error, and `gaussian_unitary` build time for the beam splitter vs the active CZ of the same local dimension $\binom{c+1}{2}$.
- Caches: first vs cached call for the basis table (6 modes, cutoff 12), the projected cubic-phase matrix ($\gamma = 0.5$, cutoff 60) and the sparse gate structure (6 modes, cutoff 10).
- Dimension table: $D_{\rm tot}$, $c^m$ and vector/density-matrix bytes for $m = 1..10$, $c \in \{4,6,8,12\}$.
- Acceptance: variants agree with the native kernel to $\le 10^{-12}$; beam-splitter output norm within $10^{-13}$ of 1; phase vs matrix $\le 10^{-14}$; passive off-sector entries exactly 0; any GPU row must report a `cuda` device. Failures go to `R46_fock_space_optimization_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R46_fock_space_optimization`, `R46_fock_phase_gates`, `R46_fock_passive_sectors`, `R46_fock_caches`, `R46_fock_basis_dimensions` (CSV/JSON/metadata), figure `R46_fock_space_optimization` and a generated caption.

### Expected outcome

All variants should agree at the $10^{-16}$ level, since they evaluate the same sums in different orders. Memory is the clearest gain: the native kernel stores only the local matrix (kilobytes), while CSR and dense global operators grow with $D$ and $D^2$ respectively. For single states the native kernel should be competitive with or faster than scipy.sparse; for $B = 32$ the batched sparse product can win because the native path loops over states in Python. The passive beam splitter should show exact zeros off-sector and build faster than the active CZ at large cutoff (an observation, not a controlled ablation). Phase multiplication may show little speedup at these sizes; caches should give large first-call/cached ratios, especially for the cubic-phase matrix. Timings are machine-dependent and provisional unless `PGQ_IDLE_MACHINE=1`; on Colab without a GPU runtime the cuSPARSE variant is simply not run, and few cores make absolute times differ from the manuscript run.

Figure panels: (a) basis dimension versus modes, total-photon (solid) vs product (dashed) for $c = 4, 6, 8, 12$; (b) runtime of one beam splitter on $B = 1$ and 32 states versus grid point ordered by $D$, for each variant; (c) bytes of the beam-splitter representation (local matrix, CSR, dense); (d) `gaussian_unitary` build time vs cutoff for the passive beam splitter and the active CZ.

<!-- core:R47 -->
**Title:** Validation of the native mixed-state Fock engine for lossy and conditional dynamics

**Why this experiment.** Realistic photonic MBQC involves mixed states: thermal or imperfectly prepared inputs, optical loss, partial traces over discarded modes, and measurements that condition the remaining modes. These are handled by the compiled density-matrix engine `native-mixed-fock`. A density-matrix code can fail in ways a pure-state code cannot: loss of Hermiticity, trace drift, small negative eigenvalues that grow over a long sequence, a Kraus channel with the wrong weights, or a partial trace over the wrong indices. The paper claims the engine produces physical density matrices that reproduce exact mixed-state physics and converge with the cutoff. R47 tests this against closed forms, channel invariants, Gaussian conditioning on the NumPy engine and the pure-state engine, and checks physicality after every operation of a long non-Gaussian sequence.

### Theory

Conventions: $\hbar = 2$, $q = a + a^\dagger$, vacuum $V = I$, total-photon cutoff $c$ ($\sum_i n_i < c$). Gates act as $\rho \mapsto P U\rho U^\dagger P/\operatorname{tr}(\cdot)$ with the retained norm recorded.

Exact-in-truncation references:

- thermal state: $\rho_{kk} = \bar n^k/(1+\bar n)^{k+1}$ restricted to $k < c$ and renormalised; retained norm $\sum_{k<c}\rho_{kk}$;
- displaced squeezed thermal state $\rho = U\rho_{\rm th}U^\dagger$, $U = D(\beta)\,e^{i\theta n}\,S(r)$, built with dense `expm` on a 90-level space, then truncated to $c$ and renormalised;
- reduced state of a two-mode squeezed vacuum (two squeezers $\pm r$ and a 50:50 splitter): $\operatorname{tr}_0|\Psi\rangle\langle\Psi| = \operatorname{diag}(\lambda^{2k})$, $\lambda = \tanh r$, kept for $2k < c$ (the passive splitter preserves the total-photon truncation of the product input).

Channel invariants. Thermal loss with transmissivity $\eta$ and environment occupation $n_{\rm th}$ maps Gaussian moments as $\mu \mapsto \sqrt\eta\,\mu$, $V \mapsto \eta V + (1-\eta)(2n_{\rm th}+1)I$, so for a coherent input $\langle n\rangle = \eta|\alpha|^2 + (1-\eta)n_{\rm th}$. Pure loss ($n_{\rm th} = 0$) maps $|\alpha\rangle$ to the pure coherent state $|\sqrt\eta\,\alpha\rangle$: fidelity 1 and purity $\operatorname{tr}\rho^2 = 1$. In the lossy TMSV the counted mode has a thermal marginal with $\bar n = \eta\sinh^2 r$, so $P(k) = \bar n^k/(1+\bar n)^{k+1}$.

Gaussian conditioning. Homodyne conditioning of a Gaussian state is again Gaussian (Schur complement on the NumPy engine). The Fock engine's conditional quadrature moments, obtained from $\langle q\rangle$, $\langle p\rangle$ and the variances at angles $0$, $\pi/2$, $\pi/4$ (the off-diagonal covariance is $V_{qp} = \operatorname{Var}_{\pi/4} - (\operatorname{Var}_q + \operatorname{Var}_p)/2$), and its outcome density must converge to the Gaussian ones.

Physicality: Hermiticity residual $\max|\rho - \rho^\dagger|$, $|\operatorname{tr}\rho - 1|$, and $\lambda_{\min}((\rho+\rho^\dagger)/2)$.

### What the code does

- Engine: `native-mixed-fock` with the norm guard opened to $1 - 10^{-12}$; the native OpenMP runtime is pinned to one thread (accuracy study; small Eigen GEMMs are slower when threaded). Secondary: `piquasso-mixed-fock` at cutoff 8 only. Gaussian reference: NumPy `gaussian` backend. Consistency reference: `native-fock`.
- Cases (`--full` cutoffs $c \in \{8,12,16,24,32,48\}$, seed 20261147): thermal inputs $\bar n \in \{0.3, 1.0\}$; displaced squeezed thermal input ($r = 0.4$, $\theta = 0.8$, $\beta = 0.6 + 0.3i$, $\bar n = 0.3$); pure loss $\eta \in \{0.3, 0.7\}$ on $|\alpha = 1.5\rangle$; thermal loss $\eta \in \{0.3, 0.7\}$, $n_{\rm th} = 0.4$; partial trace of TMSV $r = 0.6$ ($\langle n\rangle = \sinh^2 r$); lossy TMSV ($\eta = 0.8$ on mode 0) with photon counting $k = 0, 1, 2$ and $q$-homodyne conditioning at $x = 0.7$, compared with the Gaussian engine (conditional mean, covariance and outcome density of mode 1).
- Long sequence on 3 modes (inputs: coherent 0.4, $|1\rangle$, p-squeezed vacuum $r = 0.2$): (BS 0.7, CZ 0.3, Kerr 0.5, cubic 0.1, squeeze 0.1, displace (0.1, 0.2), thermal loss $\eta = 0.9$, $n_{\rm th} = 0.05$) repeated 3 times = 21 operations, at cutoffs $\{6, 8, 10, 12\}$; physicality recorded after every operation; the lossless version on the mixed engine vs $|\psi\rangle\langle\psi|$ from `native-fock`; native vs Piquasso mixed Fock reported at cutoffs 6 and 8.
- Acceptance (native rows): exact-in-truncation density matrix and retained norm $\le 10^{-12}$; Hermiticity $\le 10^{-13}$, $|\operatorname{tr} - 1| \le 10^{-12}$, $\lambda_{\min} \ge -10^{-12}$ for every case and every sequence step; mixed vs pure $\le 10^{-12}$; at the top cutoff, the homodyne-vs-Gaussian error $\le 10^{-6}$ and all other observable errors $\le 10\,c\,(1 - \text{retained}) + 10^{-12}$. Failures go to `R47_native_mixed_fock_validation_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R47_native_mixed_fock_validation`, `R47_native_mixed_fock_sequence_physicality`, `R47_native_mixed_fock_sequence_summary` (CSV/JSON/metadata), figure `R47_native_mixed_fock_validation` and a generated caption.

### Expected outcome

Exact-in-truncation errors should be at rounding ($\lesssim 10^{-15}$) at every cutoff. Invariant errors should decrease rapidly with cutoff: pure loss should reproduce $|\sqrt\eta\,\alpha\rangle$ with fidelity and purity deficits at rounding by $c = 48$; the homodyne-conditioned moments approach the Gaussian ones much more slowly (squeezed, conditioned states have heavier Fock tails), so this is the case that needs the largest cutoff to pass $10^{-6}$. Physicality residuals should stay at $10^{-16}$ to $10^{-15}$ over all 21 operations, with the smallest eigenvalue at most a few times $-10^{-16}$. The Piquasso mixed-Fock comparison is expected to differ at the truncation level ($10^{-4}$ to $10^{-3}$ at $c = 6, 8$) because its CZ is an Euler decomposition truncated after every factor; this is reported, not asserted. The full run is heavy (density matrices up to cutoff 48).

Figure panels: (a) density-matrix error vs exact-in-truncation reference versus cutoff for thermal, displaced squeezed thermal and reduced-TMSV cases, with the $10^{-12}$ line; (b) convergence of all physical invariants and Gaussian comparisons versus cutoff; (c) Hermiticity/trace residual and $\max(-\lambda_{\min}, 0)$ after each of the 21 operations, one colour per sequence cutoff (exact zeros at the $10^{-17}$ floor).

<!-- core:R48 -->
**Title:** Permanents, output distributions and exact samplers for Fock-state boson sampling

**Why this experiment.** The sampling layer (`photographiq.sampling`) computes boson-sampling probabilities through matrix permanents, with a native Gray-code Ryser kernel and a NumPy multiplicity-aware Ryser, and draws exact samples. Permanent algorithms are notoriously easy to get subtly wrong: sign conventions in inclusion-exclusion, the factorial normalisation for bunched (collision) inputs and outputs, row/column convention of the interferometer, and rounding at large $n$. A sampler can also produce plausible-looking but wrongly distributed samples. The paper claims the probabilities are correct, normalised and consistent with an independent quantum simulation of the same interferometer, and that the samplers are exact. R48 tests each claim against references that share no code with the kernels, and records how the kernels scale.

### Theory

An $m$-mode passive interferometer $U$ maps $a_i^\dagger \mapsto \sum_j U_{ji} a_j^\dagger$. For an input occupation $s$ and output $t$ with $n = \sum_i s_i = \sum_j t_j$ photons,

$$P(t\,|\,s) = \frac{|\operatorname{Perm}(U_{T,S})|^2}{\prod_i s_i!\,\prod_j t_j!},$$

where $U_{T,S}$ is the $n\times n$ matrix obtained by repeating row $j$ of $U$ $t_j$ times and column $i$ $s_i$ times. The permanent is $\operatorname{Perm}(A) = \sum_{\sigma\in S_n}\prod_i a_{i\sigma(i)}$ ($n!$ terms). Ryser's formula

$$\operatorname{Perm}(A) = (-1)^n \sum_{S\subseteq\{1..n\}} (-1)^{|S|} \prod_{i=1}^n \sum_{j\in S} a_{ij}$$

costs $O(2^n n)$ with Gray-code ordering of subsets; a multiplicity-aware version sums over $\prod_j (t_j+1)$ sub-multisets instead of expanding repeated rows. The independent reference used for $n > 8$ is Glynn's formula

$$\operatorname{Perm}(A) = \frac{1}{2^{n-1}} \sum_{\delta\in\{\pm1\}^n,\ \delta_1 = 1} \Bigl(\prod_k \delta_k\Bigr) \prod_{j=1}^n \sum_{i=1}^n \delta_i a_{ij}.$$

Because Ryser and Glynn sum terms of magnitude up to $\prod_i \sum_j |a_{ij}|$ that cancel to a much smaller permanent, the meaningful rounding measure is the error normalised by this scale; the relative error $|\Delta|/|\operatorname{Perm}|$ can be much larger when the permanent is small.

Fourier suppression law (Tichy et al., PRL 104, 220405): for the DFT $F_{jk} = e^{2\pi i jk/m}/\sqrt m$ and the input with one photon in every mode, $P(t) = 0$ unless $\sum_k k\,t_k \equiv 0 \pmod m$. This is an exact physical prediction, independent of any permanent code.

Fock-space reference: the passive unitary's Fock matrix in the total-photon basis with cutoff $n + 1$, from the native Bargmann/sector recurrence (`gaussian_unitary`), applied to $|s\rangle$. Passive gates conserve photon number, so the $n$-photon sector is complete and exact.

### What the code does

- System under test: `smp.permanent` (`backend="native"` and `"python"`), `smp.output_distribution`, `smp.collision_probabilities`, `smp.sample` (`method="clifford"` and `"enumerate"`).
- Haar unitaries from `scipy.stats.unitary_group` with fixed seeds (independent of the package's own `haar_unitary`); master seed 20261148.
- Permanents (`--full`): sizes $n = 1..14, 16, 18, 20$, 3 trials each, $A$ = top-left $n\times n$ block of a $2n$-mode Haar unitary times $\sqrt{2n}$ ($O(1)$ entries). Reference: permutation definition ($n \le 8$) or Glynn (this script). Native timed with the default OpenMP thread count and with 1 thread; NumPy timed for $n \le 18$. Bunched permanents: three repeated-row/column patterns of a scaled 4-mode Haar unitary against the definition on the explicitly repeated matrix.
- Distributions: 10 cases, $(m, s) = (4,(1,1,0,0))$, $(4,(2,0,0,0))$, $(6,(1,1,1,0,0,0))$, $(6,(2,1,0,0,0,0))$, $(8,(1^4,0^4))$, $(8,(2,2,0^6))$, $(10,(1^4,0^6))$, $(12,(1^4,0^8))$, $(10,(1^5,0^5))$, $(8,(3,2,0^6))$; compared with the Fock-space evolution and with Piquasso `PureFockSimulator` (`NumberState` + `Interferometer`, cutoff $n+1$).
- Suppression law: DFT on $m = 3..7$ with $s = (1,\dots,1)$; total mass of suppressed outputs.
- Samplers: 20000 seeded samples for $m = 6$, $s = (1,1,1,0,0,0)$ (Clifford-Clifford) and $m = 5$, $s = (2,1,0,0,0)$ (enumeration); Pearson chi-square with bins of expected count $< 5$ pooled; total variation distance and samples outside the support.
- Acceptance: permanent error $/\prod_i\sum_j|a_{ij}| \le 10^{-13}$ (both kernels); $|\sum_t P - 1| \le 10^{-12}$; vs Fock-space and vs Piquasso $\le 10^{-12}$; suppressed mass $\le 10^{-14}$; chi-square $p \ge 10^{-4}$ and no sample outside the support. Failures go to `R48_boson_sampling_FAILURES.csv` and raise `AssertionError`.
- Outputs: `R48_boson_sampling_permanents`, `R48_boson_sampling`, `R48_boson_sampling_suppression`, `R48_boson_sampling_samplers` (CSV/JSON/metadata), figure `R48_boson_sampling` and a generated caption.

### Expected outcome

Scaled permanent errors should be at $10^{-16}$; relative errors stay near $10^{-15}$ for $n \le 8$ (exact definition) and grow for larger $n$ against Glynn, partly because Glynn itself is evaluated in float64 (up to $\sim10^{-11}$ at $n = 20$ is plausible). Distributions should be normalised and agree with both references at $10^{-16}$ to $10^{-15}$; the suppressed DFT mass should be many orders below $10^{-14}$; sampler $p$-values should look like draws from $U(0,1)$. Runtimes should follow the exponential $2^n n$ law, roughly doubling per added photon; the NumPy kernel is orders of magnitude slower. Threading helps only for large $n$; at small $n$ OpenMP overhead can make the threaded kernel slower than one thread. Timings are machine-dependent and labelled provisional; on Colab's few cores the threaded and single-thread curves will be closer.

Figure panels: (a) relative permanent error versus $n$ for both kernels, with a dotted line separating the definition reference ($n \le 8$) from the Glynn reference; (b) median single-call runtime versus $n$ (native with default threads, native with 1 thread, NumPy) on a log axis; (c) for each distribution case, normalisation error and maximum difference from the Fock-space evolution and from Piquasso, with the $10^{-12}$ line.

<!-- core:R49 -->
**Title:** Gaussian boson sampling with photon-number-resolving detectors

**Why this experiment.** PhotoGraphiQ computes photon-number-resolving (PNR) probabilities of Gaussian states with `photographiq.sampling.pnr_probability`. The routine uses hafnians for undisplaced states, loop hafnians for displaced ones, and a reduced pure-state formula when the state is pure. These formulas carry several conventions (complex-amplitude ordering, conjugation of $A$ and $\gamma$, the vacuum prefactor, the factorials), and a convention slip gives probabilities that look plausible but are wrong, especially for displaced or lossy states. R49 tests the formulas against references that do not use any hafnian code.

### Theory

Conventions: $\hbar = 2$, interleaved mean $d$ and covariance $V$ (vacuum $V = I$). With $\xi = (a_1..a_m, a_1^\dagger..a_m^\dagger)$, $\beta = \langle\xi\rangle = (\alpha, \alpha^*)$, $\alpha_j = (d_{q_j} + i d_{p_j})/2$ and $\sigma_Q = \sigma + I/2$ (vacuum $\sigma_Q = I$), the library defines

$$A = X\,(I - \sigma_Q^{-1})^*, \qquad \gamma = \sigma_Q^{-1}\beta, \qquad p_0 = \frac{e^{-\beta^\dagger\sigma_Q^{-1}\beta/2}}{\sqrt{\det\sigma_Q}},$$

with $X = \begin{pmatrix}0 & I\\ I & 0\end{pmatrix}$. The probability of the pattern $n = (n_1..n_m)$ is

$$P(n) = \frac{p_0\,\mathrm{lhaf}\bigl(A_{n\oplus n};\,\gamma_{n\oplus n}\bigr)}{\prod_i n_i!},$$

where $A_{n\oplus n}$ repeats rows and columns $i$ and $i+m$ $n_i$ times and $\gamma$ fills the diagonal. The hafnian sums over perfect matchings $M$ of $2N$ indices, $\mathrm{haf}(A) = \sum_M \prod_{(i,j)\in M} A_{ij}$. The loop hafnian also allows loops $(i,i)$ weighted by $A_{ii}$. For $d = 0$ it reduces to $\mathrm{haf}(A_{n\oplus n})$. For a pure state $A = B \oplus B^*$ and $P(n) = p_0\,|\mathrm{lhaf}(B_n;\gamma_n)|^2/\prod_i n_i!$, with half the matrix dimension. Loss acts per mode as $V \to \eta V + (1-\eta)I$, $d \to \sqrt\eta\,d$.

Fock-space reference: $|\psi\rangle = U(u)\bigotimes_i S(r_i)|0\rangle$. The product of exact single-mode squeezed columns, kept only for total photon number $< c$, equals $P_{<c}|\psi_{\rm in}\rangle$. A passive interferometer conserves photon number, so $P_{<c}U = UP_{<c}$. For undisplaced lossless inputs, every amplitude with $|n| < c$ is therefore exact. This also gives a sum rule: $\sum_{|n|<c} P(n) = \|P_{<c}\psi\|^2$. A displacement applied after truncation does not commute with $P_{<c}$, so in that case agreement only appears as the cutoff grows.

### What the code does

- System under test: `squeezed_interferometer_state(r, U, eta, alpha)`, `gbs_matrices`, and `pnr_probability` (native C++ and NumPy hafnian backends, default and `method="mixed"`).
- Hafnian check: random complex symmetric matrices of size 1..12 (hafnian for even sizes, loop hafnian for all sizes). Each is compared with a memoised perfect-matching sum written in the script. The error is scaled by $\sum|\text{terms}|$.
- Fock reference (native sector recurrence, independent of hafnians). Each family uses one fixed seeded Haar unitary and q-squeezed vacua. Undisplaced: $(m, r, c)$ = (2, 0.5, 6/10/16), (3, 0.5, 10), (3, 1.0, 12), (4, 0.3, 9), plus `--full` (4, 0.6, 12) and (5, 0.4, 9). Displaced with $|\alpha_j| = 0.3$ and random phases: (2, 0.5, 8/14/22/30), plus `--full` (2, 0.5, 40) and (3, 0.4, 12/20). Patterns with more than 12 photons are skipped. Up to 6 photons, the pure formula is also compared with the mixed formula.
- Piquasso 8.0.1 `get_particle_detection_probability` on the identical program, with loss and displacement $\alpha_j = 0.4u\,e^{2\pi i v}$. Configurations $(m, r, \eta, \text{disp}, N_{\max})$: (4, 0.5, 1, no, 8), (6, 0.5, 0.9, no, 8), (6, 0.8, 1, yes, 6), (8, 0.6, 0.7, yes, 8), plus `--full` (10, 0.7, 0.9, no, 12), (12, 0.5, 1, no, 12), (10, 0.5, 0.8, yes, 10). Four random patterns are drawn per photon total.
- OpenMP is pinned to one thread. Seed 20261149.
- Outputs: `R49_gaussian_boson_sampling`, `R49_gbs_mass_sum_rules`, `R49_gbs_hafnians`, `R49_gbs_vs_piquasso`, and figure `R49_gaussian_boson_sampling`.

### Expected outcome

Acceptance (all must hold):

| Check | Criterion |
|---|---|
| Hafnian vs matching sum | scaled error $\le 10^{-12}$ |
| Undisplaced lossless $P(n)$ vs exact Fock | absolute error $\le 10^{-13}$ |
| Pure vs mixed formula | difference $\le 10^{-13}$ |
| Sum rule (cutoffs $\le 13$) | mass difference $\le 10^{-12}$ |
| Displaced, largest cutoff, $\vert n\vert \le 4$ | absolute error $\le 10^{-10}$ |
| Native vs Piquasso and native vs NumPy | relative difference $\le 10^{-9}$ for $P > 10^{-12}$ |

Undisplaced errors should sit at rounding level. Displaced errors should fall steadily as the reference cutoff increases.

Figure, four panels:

- (a) Hafnian vs Fock probabilities on log-log axes, lying on the diagonal.
- (b) Absolute error vs probability for the Fock comparison and for Piquasso, with the $10^{-13}$ line.
- (c) Worst displaced error ($|n| \le 4$) vs reference cutoff, decreasing.
- (d) Scaled hafnian and loop-hafnian errors vs matrix size for both backends, below $10^{-12}$.

<!-- core:R50 -->
**Title:** Threshold-detector boson sampling: torontonian cancellation and working precision

**Why this experiment.** Click probabilities for threshold detectors are alternating inclusion-exclusion sums. Their terms are $O(1)$ while the result can be many orders of magnitude smaller, so double precision silently loses digits for many clicks, weak squeezing or small probabilities. PhotoGraphiQ's `threshold_probability` returns a rounding bound $B$ and warns when $B/|P|$ is large. It also offers an mpmath path (`precision=<digits>`) and an opt-in `precision="auto"`. R50 tests whether the bound is reliable, whether the warnings fire, and whether the multiprecision paths recover the lost accuracy. Without this check a user could quote a double-precision click probability whose leading digit is wrong.

### Theory

Conventions: $\hbar = 2$, vacuum $V = I$, $M = V + I$. The probability that mode set $K$ is in vacuum is

$$\Pr[\text{vac}_K] = \frac{2^{|K|}\,\exp\!\bigl(-\tfrac12 d_K^T M_{KK}^{-1} d_K\bigr)}{\sqrt{\det M_{KK}}}.$$

Let $S$ be the clicked modes and $Z$ the dark modes. Inclusion-exclusion gives

$$P(S) = \sum_{T\subseteq S} (-1)^{|T|}\,\Pr[\text{vac}_{Z\cup T}] = p_0\,\mathrm{ltor}(O_S, \gamma'_S), \qquad O = I - \sigma_Q^{-1},$$

which is the torontonian for $d = 0$ and the loop torontonian otherwise. The library first conditions on $Z$. The Schur complement $C = M_{SS} - M_{SZ}M_{ZZ}^{-1}M_{ZS}$ and the shift $e = d_S - M_{SZ}M_{ZZ}^{-1}d_Z$ give

$$P(S) = P_Z \sum_T (-1)^{|T|}\, 2^{|T|} e^{-e_T^T C_T^{-1} e_T/2} / \sqrt{\det C_T}.$$

In floating point the error grows like $\epsilon\,\kappa$, where $\kappa = \sum_T|\text{term}_T|/|P|$ is the cancellation ratio. $B$ is a first-order rounding model: an alternating-sum part plus a conditioning part with Higham-type Cholesky backward-error bounds. It is calibrated empirically, not proven. `precision="auto"` recomputes in mpmath when $B > 10^{-6}|P|$. It uses $\lceil\log_{10}(X/\max(|P|,B)) + \log_{10}(1/\mathrm{rtol})\rceil + 10$ digits, where $X$ bounds the absolute sum.

### What the code does

- Cases (`--full`): modes $m \in \{4, 8, 12, 16, 20\}$, clicks $k \in \{1, 2, 3, 4, 6, 8, 10, 12\}$ with $k \le m$, and squeezing $r \in \{0.05, 0.2, 0.5, 1.0, 1.5\}$. That is 170 random cases. Each case has a seeded Haar unitary, transmissivity drawn from $\{1.0, 0.9, 0.5\}$, displacement $\alpha_j = 0.4u\,e^{2\pi i v}$ in half the cases, and a random click pattern.
- 15 zero-probability cases: an untouched vacuum mode clicks, with $k \in \{1, 3, \min(m,8)\}$.
- Builder, per-case evaluation and reference are reused from `benchmarks/gaussian_boson_sampling/threshold_reference_matrix.py`. The reference is a 40-digit mpmath sum over the $2^k$ subsets of the stored covariance. It uses a depth-first Cholesky-extension walk and shares no code with the library's `precision=` path. Its uncertainty is $10^{3-40}\sum|\text{terms}|$. It runs in an 8-process pool (`--procs`).
- Per case: native and Python double values, bound $B$ and warnings; `precision=50`; `precision="auto"` (path, digits, `ThresholdInfo`); Piquasso 8.0.1 `get_threshold_detection_probability`, scored against the reference of its own stored state.
- Precision sweep: the 8 highest-cancellation reliable cases at double, then 16, 20, 24, 30, 40, 50 and 60 digits.
- Normalisation: full $2^m$ click distributions for $m = 2, 4, 6, 8$ with $\eta = 0.9$, at $(r, \text{disp}) = (0.5, \text{no})$ and $(1.0, \text{yes})$.
- Outputs: `R50_threshold_gbs`, `R50_threshold_precision_sweep`, `R50_threshold_normalisation`, and figure `R50_threshold_gbs`.

### Expected outcome

Acceptance (seed 20261150):

| Check | Applies to | Criterion |
|---|---|---|
| Rounding bound holds | every case, native and Python | $\vert P_{\rm double} - P_{\rm ref}\vert \le B + $ reference uncertainty |
| Warning fires | every case | a `ThresholdPrecisionWarning` whenever $B/\vert P\vert > 10^{-6}$ |
| Zero cases | 15 zero-probability cases | $\vert P_{\rm ref}\vert \le 10^{-14}$, and both backends warn |
| `precision="auto"` | random cases with reference relative uncertainty $< 10^{-8}$ | relative error $\le 10^{-6}$ |
| `precision=50` | all non-zero cases | $\vert P_{50} - P_{\rm ref}\vert \le 10\,(\text{ref. abs. uncertainty} + \vert P_{50}\vert\cdot\text{reported error})$ |
| Normalisation | full distributions | $\vert\sum P - 1\vert \le 10^{-10}$ |

Double-precision relative errors (native, Python and Piquasso) are expected to exceed $10^{-6}$, and even reach $O(1)$, for small $P$ and many clicks. This is a property of the algorithm, not a failure. The reference is expensive for $k = 12$, so `--full` takes long. The per-case runtimes are not timing claims.

Figure, four panels:

- (a) Relative error vs $|P|$ for native, Python, Piquasso and `auto`, with a $10^{-6}$ guide.
- (b) Native error vs cancellation ratio, coloured by $r$, following the $\epsilon\times\kappa$ line.
- (c) Histogram of $B/|\text{error}|$, entirely to the right of 1.
- (d) Error vs working digits for the hardest cases, falling to float64 output level once enough digits are used.

<!-- core:R51 -->
**Title:** CV flow finding on weighted graphs against the defining linear algebra

**Why this experiment.** Deterministic CV measurement-based computation needs an order of homodyne measurements and a set of displacement corrections that cancel every outcome dependence in the ideal limit. `photographiq.flows.find_cv_flow` searches for this CV flow by backward real-linear layering (Booth and Markham, Quantum 7, 1146 (2023)). The search returns a certificate or an impossibility witness. A wrong positive would yield a non-deterministic pattern labelled deterministic. A wrong negative would reject valid resources. Because the library works in floating point, near-singular graphs also need an honest "uncertain" verdict. R51 re-derives every decision and certificate from the definition, using exact arithmetic wherever a decision is made.

### Theory

Let $A$ be the real weighted adjacency matrix of the open graph $(G, I, O)$. Measured nodes are $u \notin O$, read out by a sheared homodyne with record $m_u = k_u q_u + p_u$ (equal to the raw record divided by $\sin\theta$). A total order $\prec$ of the measured nodes admits CV flow iff, for every measured $u$, there is a real vector $c_u$ with the following two properties:

- it is supported on non-input nodes that are later than $u$ or are outputs;
- it satisfies $(A c_u)_v = \delta_{uv}$ for all $v \preceq u$.

Writing $M_u = A[\{v \preceq u\}, \{\text{later non-inputs}\}]$, the condition is $e_u \in \mathrm{range}(M_u)$, equivalently $\mathrm{rank}\,M_u = \mathrm{rank}\,[M_u\,|\,e_u]$. The certificate comes from a nullifier argument. The ideal graph state satisfies $p_t - (Aq)_t = 0$, so a record on $u$ can be pushed onto later nodes by the corrections

$$q_t \to q_t - c_{u,t}\,m_u, \qquad p_t \to p_t - (A c_u)_t\,m_u .$$

The condition depends only on the set of nodes measured up to $u$, so existence can be decided by a search over prefix sets. If $e_u \notin \mathrm{range}(M)$, the Fredholm alternative gives a witness $y$ with $y^T M = 0$ and $y_u \neq 0$. The flow is a statement about infinite squeezing. The resource squeezing $r = 1.0$ enters only the library's Gaussian determinism check.

### What the code does

- System under test: `find_cv_flow(og, seed=0)` on `OpenGraph(Model.CV_GAUSSIAN, ...)`. It returns a decision (certified, impossible or numerically uncertain), layers, layer vectors $c_u$, a `correction_map`, witnesses and a determinism report.
- 13 named cases with hand-derived expected decisions:
  - certified: wires of length 3, 6 and 10; weighted $2\times3$ and $3\times4$ grids; a branched Y; a branched tree; an invertible weighted hexagon;
  - impossible: unit $K_{2,2}$ bridge (det 0); singular weighted hexagon; $|O| < |I|$; an isolated measured node;
  - numerically uncertain: a bridge whose determinant is $10^{-10}$.
- Random families in `--full`: 120 weighted $G(n,p)$ graphs with $n = 4..7$, signed weights with $|w|$ log-uniform in $[0.2, 3]$ and random inputs/outputs; plus 60 random weighted brickwork clusters with $|I| = |O| = k$. The seed is 20261151.
- Independent references, none using library flow code:
  1. An exact forward search over prefix sets in `fractions.Fraction` arithmetic, with exact rank tests. It runs for all graphs here (up to 16 measured nodes allowed).
  2. A least-squares re-derivation of $c_u$ on the reported order, with an SVD rank gap.
  3. A re-check of the library layer vectors (support and residual) and of the map $(q, p) = (-c_t, -(Ac)_t)$.
  4. An exact rational re-check of every impossibility witness, $y^T A[W, C] = 0$ with $y_u \ne 0$.
- Uncertain cases are rerun with `exact=True`.
- Outputs: `R51_cv_flow_finding` (results), `R51_cv_flow_corrections` (CSV), table `R51_cv_flow_finding.md`, and figures `R51_cv_flow_examples` and `R51_cv_flow_summary`.

### Expected outcome

Every graph must pass all of the following:

- Each named case gets its expected decision.
- "Certified" never occurs where the exact search finds no order, and "impossible" never occurs where it finds one.
- For certified graphs:
  - least-squares residual $\le 10^{-9}$ with rank gap 0;
  - recomputed layer residual $\le 10^{-10}$ with valid support;
  - correction-map error $\le 10^{-12}$;
  - the library determinism check reports `ideal_limit_deterministic`.
- Every witness is exactly valid.
- The uncertain bridge becomes certified under `exact=True`.

The random graphs should show a mix of certified and impossible decisions, all agreeing with the exact search.

Example figure, two rows (grid_2x3 and branched_tree), three columns:

- the weighted resource graph;
- the same graph coloured by measurement layer (column = layer);
- a table of $(q, p)$ correction coefficients, records by corrected node.

Summary figure:

- (a) a stacked bar chart of decisions per family;
- (b) for certified graphs, the independent residual (left of the $10^{-9}$ line) against the largest correction coefficient.

<!-- core:R52 -->
**Title:** CV-MBQC determinism: ideal map, finite-squeezing noise and outcome dependence

**Why this experiment.** A CV flow (R51) promises that the corrected output does not depend on the homodyne records, but only in the limit of infinite squeezing. At finite squeezing the outcome-averaged channel still implements the intended map $S$. However, it adds noise $N$, and a residual outcome dependence of the conditional output remains. The paper keeps these three objects apart and never calls a finite-squeezing channel an ideal unitary. R52 checks quantitatively that the library's channel, its postselected trajectories and its sampled trajectories obey this decomposition. The check is against an independent linear model, and verifies the expected $e^{-2r}$ decay of both $N$ and the residual dependence.

### Theory

Conventions: $\hbar = 2$, coherent inputs with covariance $I$. Collect the Wigner variables $x = (x_{\rm in}, q_j, p_j, \ldots)$ with covariance

$$\Sigma = \mathrm{diag}\bigl(V_{\rm in}, e^{2r}, e^{-2r}, \ldots\bigr),$$

so each resource node is anti-squeezed in $q$ and squeezed in $p$. The pattern acts linearly on $x$:

- a weighted CZ maps $p_u \to p_u + w\,q_v$;
- a measurement produces the record $m_u = k_u q_u + p_u$;
- a correction maps $q_t \to q_t + c_q m_u$ and $p_t \to p_t + c_p m_u$.

The output quadratures are then $Lx$ and the records are $Rx$. The outcome-averaged channel is $V \to S V S^T + N$ with

$$S = L_{\rm in}, \qquad N = L_{\rm res}\Sigma_{\rm res}L_{\rm res}^T = e^{2r}N_+ + e^{-2r}N_-,$$

split exactly by the anti-squeezed and squeezed columns. For a unitary flow $N_+ = 0$. For an isometry ($|O| > |I|$), $N_+$ has rank $|O| - |I|$: these are the infinitely squeezed output directions of the ideal map. On $F = \mathrm{null}(N_+)$, $P_F N P_F \propto e^{-2r}$. Gaussian conditioning gives

$$\mathbb E[\text{out}\mid m] = L\mu + G\,(m - R\mu), \qquad G = L\Sigma R^T (R\Sigma R^T)^{-1},$$

with a conditional covariance that does not depend on $m$. Determinism means $P_F G \to 0$, asymptotically as $e^{-2r}$. Closed forms for $S$:

- unit teleportation step: $T(k) = \begin{pmatrix}-k & -1\\ 1 & 0\end{pmatrix}$;
- weighted step: $\begin{pmatrix}-k/w & -1/w\\ w & 0\end{pmatrix}$;
- graph teleportation: $C_Z(A)\bigoplus_i T(s_i)$.

### What the code does

- Five patterns, each with corrections from `find_cv_flow` and executed as `flows.cv.corrected_pattern`:
  - one-step teleportation (shear 0.35);
  - a weighted 4-node wire;
  - a 4-mode graph teleportation on a weighted cycle;
  - a weighted $2\times3$ cluster;
  - a branched $1\to2$ isometry.
- Squeezing sweep: $r \in \{0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5\}$.
- Library quantities: `gaussian_channel` $(S, N)$; `simulate(backend="gaussian", measurement_outcomes=...)` for 6 record sets (the first all zero, the rest random multiples of the record standard deviation), each one postselected branch; 40 seeded sampled trajectories; and `verify_cv_determinism` at $r = 1$.
- Reference: the linear Wigner model above, written in the script, plus the closed forms. Seed 20261152.
- Outputs: `R52_cv_mbqc_determinism` (rows, decay slopes, per-case metadata) and figure `R52_cv_mbqc_determinism`.

### Expected outcome

Acceptance criteria:

| Quantity | Criterion |
|---|---|
| $S$ vs model and closed form | $\le 10^{-10}$, relative to $\max(1, \vert S\vert)$ |
| $N$ vs model | $\le 10^{-9}$, relative to $\max(1, \vert N\vert)$ |
| Postselected moments vs model | $\le 10^{-9}$ (relative) |
| Outcome dependence of the conditional covariance | $\le 10^{-9}$ |
| Rank of $N_+$ | equals $\vert O\vert - \vert I\vert$ |
| Seeded spread of the corrected mean | within 40% of the model's $\sqrt{\mathrm{diag}(P_F G K G^T P_F)}$ |
| $P_F N P_F$ | strictly positive |
| Fitted slope of $\ln\max\vert P_F N P_F\vert$ ($r \ge 1$) | in $[-2.05, -1.95]$ |
| Fitted slope of $\ln\max\vert P_F G\vert$ ($r \ge 2$) | in $[-2.15, -1.85]$ |
| `verify_cv_determinism` | passes for every case |

The 40% band on the seeded spread is a statistical check (about 3.5 sigma with 40 samples). The slope fit for $G$ uses $r \ge 2$ because $G$ is pre-asymptotic at small $r$.

Figure, three panels:

- (a) $\max|P_F N P_F|$ vs $r$ for all five cases, parallel to the dashed $e^{-2r}$;
- (b) $\max|P_F G|$ vs $r$, approaching $e^{-2r}$;
- (c) for teleportation, the postselected conditional $\langle q_{\rm out}\rangle$ vs the record (in units of its standard deviation) at $r = 0.5, 1, 2, 3$. The lines flatten towards the ideal $(S\mu)_q$ as $r$ grows. Each point is a single conditional branch.

<!-- core:R53 -->
**Title:** Qubit causal flow, gflow and Pauli flow against Graphix 0.4

**Why this experiment.** PhotoGraphiQ implements the three standard sufficient conditions for deterministic qubit MBQC: causal flow, generalised flow (gflow) and Pauli flow (`find_causal_flow`, `find_gflow`, `find_pauli_flow`). These finders decide whether a measurement pattern on an open graph can be made strongly deterministic. A finder can fail in three ways:

- it can disagree with the established implementation on existence;
- it can return correction sets that violate the definition;
- it can return a flow whose pattern is not actually deterministic.

R53 tests all three against Graphix 0.4, an independent set-algebra checker and a brute-force statevector simulation. It also confirms that qubit certificates are never issued for CV graphs.

### Theory

Open graph $(G, I, O)$; measured nodes $u \notin O$ carry a plane $\lambda(u) \in \{XY, XZ, YZ\}$ or, for Pauli flow, a label $X$, $Y$ or $Z$. The observables are $\cos a\,X + \sin a\,Y$ (XY), $\cos a\,Z + \sin a\,X$ (XZ) and $\cos a\,Z + \sin a\,Y$ (YZ). The odd neighbourhood of a set $K$ is $\mathrm{Odd}(K) = \{v : |N(v)\cap K| \text{ odd}\}$. The strict order is $u < v$ iff $\mathrm{layer}(u) < \mathrm{layer}(v)$, with outputs last.

- **Causal flow** (Danos-Kashefi): an injective map $f$ into the non-inputs, with only XY measurements. F1: $f(u) \sim u$. F2: $u < f(u)$. F3: every neighbour $k \neq u$ of $f(u)$ satisfies $u < k$.
- **Gflow** (Browne et al.; three-plane form of Backens et al.): sets $g(u)$ disjoint from the inputs.
  - G1: $v \in g(u)\setminus\{u\} \Rightarrow u < v$.
  - G2: $v \in \mathrm{Odd}(g(u))\setminus\{u\} \Rightarrow u < v$.
  - G3 (self condition): XY needs $u \notin g$ and $u \in \mathrm{Odd}$; XZ needs $u \in g$ and $u \in \mathrm{Odd}$; YZ needs $u \in g$ and $u \notin \mathrm{Odd}$.
- **Pauli flow** (Simmons 2021):
  - P1 relaxes G1 for $v$ labelled X or Y.
  - P2 relaxes G2 for $v$ labelled Y or Z.
  - P3: if $\neg(u < v)$ and $\lambda(v) = Y$, then $v \in p(u) \Leftrightarrow v \in \mathrm{Odd}(p(u))$.
  - Self conditions: X needs $u \in \mathrm{Odd}$, Z needs $u \in p$, and Y needs exactly one of the two.

A flow gives the corrections $X_{p(u)\setminus\{u\}}\,Z_{\mathrm{Odd}(p(u))\setminus\{u\}}$ for outcome $s_u = 1$. Strong determinism means two things: every one of the $2^m$ branches has probability $2^{-m}$, and all branches give the same output map up to a phase.

### What the code does

- Finders under test: the three library finders, compared with Graphix's `OpenGraph.to_causalflow_or_none`, `to_gflow_or_none` and `to_pauliflow_or_none` on the identical graph. For Pauli flow, Pauli labels are passed as axes.
- 9 named graphs with literature or hand-derived expectations:
  - line and $2\times3$ cluster (all three flows exist);
  - a graph with gflow but no causal flow;
  - YZ state preparation;
  - an XZ node at the start of a line;
  - an XZ node inside an input line (no flow of any kind);
  - X measurements in the XZ plane (Pauli flow only);
  - Y and Z Pauli measurements;
  - $|I| > |O|$ (no flow).
- Random open graphs, 300 per notion in `--full`, with $n = 3..8$: XY only for causal flow, all three planes for gflow, and 40% Pauli labels for Pauli flow. Seed 20261153.
- Independent checker: plain set algebra implementing F1-F3, G1-G3 and P1-P9, applied to our flows and to Graphix's.
- Independent determinism check, for at most 12 simulated qubits. A dense Choi-vector simulator puts one Bell-paired reference qubit per input, projects onto the $\pm1$ eigenvector of each observable, and applies the X/Z corrections to live qubits. It runs every branch.
- CV guard: `analyze(cv_graph, notion)` must return `unsupported`.
- Outputs: `R53_qubit_flows_graphix` (results), table `R53_qubit_flows_graphix.md` and figure `R53_qubit_flows_graphix`.

### Expected outcome

Acceptance (a single failure fails the run):

- Existence agrees with Graphix on 100% of (graph, notion) pairs.
- Every named case matches its expectation.
- Every negative answer comes from a complete algorithm.
- Neither our flows nor Graphix's violate any defining condition.
- Every simulated pattern has branch-probability and overlap deviations $\le 10^{-12}$.
- Our depth never exceeds Graphix's (ours is maximally delayed).
- The CV guard returns `unsupported` for all three notions.

Correction sets and layer partitions need not coincide, because flows are not unique. Their coincidence counts are reported, not required.

Figure, two panels:

- (a) For each notion, bars of flow-exists and no-flow counts, annotated "agree N/N".
- (b) For jointly certified flows, counts of identical layers, identical correction sets and statevector-checked patterns.

<!-- core:R54 -->
**Title:** Qubit measurement calculus preserves branch semantics; CV command view

**Why this experiment.** The measurement calculus of Danos, Kashefi and Panangaden (DKP) rewrites MBQC patterns. Two of its rewrites are central here: standardization to NEMC form, and signal shifting, which reduces adaptive depth. `photographiq.calculus.qubit` implements both. A rewrite that is correct only on average, or only up to an outcome-dependent phase or frame, would silently change the computation. The comparison must therefore be made branch by branch, for every measurement outcome, including the classical dependencies.

For CV patterns, the library claims much less: it classifies commands and extracts displacement corrections as exact affine functions of the records. It implements no complete CV measurement calculus. R54 validates both claims and checks the scope boundary.

### Theory

Commands, listed in execution order:

- $N_i$ prepares $|+\rangle$;
- $E_{ij}$ applies $CZ$;
- $X_i^d$ and $Z_i^d$ apply the Pauli iff the parity of domain $d$ is 1;
- $S_i^d$ updates the signal, $s_i \leftarrow s_i \oplus \mathrm{par}(d)$;
- $M_i^{s,t}(\lambda, a)$ measures

$$O_{\rm eff} = Z^t X^s\, O(\lambda, a)\, X^s Z^t,$$

where outcome 0 is the $+1$ eigenvalue, $s$ and $t$ are the parities of the two domains, and $O_{XY} = \cos a\,X + \sin a\,Y$, $O_{XZ} = \cos a\,Z + \sin a\,X$, $O_{YZ} = \cos a\,Z + \sin a\,Y$.

Standardization rules, left before and right after, in execution order:

$$\mathrm{EX}: X_i^d E_{ij} \to E_{ij} X_i^d Z_j^d, \qquad \mathrm{EZ}: Z_i^d E_{ij} \to E_{ij} Z_i^d,$$
$$\mathrm{MX}: X_i^d M_i^{s,t} \to M_i^{s\oplus d,\,t}, \qquad \mathrm{MZ}: Z_i^d M_i^{s,t} \to M_i^{s,\,t\oplus d}.$$

The remaining rules are signal commutation and commutation on disjoint qubits. EX, EZ and the commutations are operator identities. MX and MZ hold up to a branch phase.

Signal shifting moves the outcome-flipping Pauli of each plane into the signal: the $t$-domain for XY, the $s$-domain for YZ and XZ (for XZ also $t \leftarrow t\oplus s$). Appended $S$ commands then restore the original signals.

The process is compared through branch maps. Each branch $b$, keyed by the final signals, gives a Kraus map $K_b$, represented by its Choi vector. Two patterns are equivalent when every $K_b$ agrees up to a phase and every probability $\|K_b\|^2$ agrees exactly.

### What the code does

- Patterns:
  - 120 seeded random valid non-standard patterns (`--full`) with 2 inputs, 3 ancillas and 2 outputs, built from 18 random steps that interleave N/E/M/X/Z with random planes, angles and $s$/$t$ domains;
  - 5 flow-derived standard patterns from `pattern_from_flow`: causal flow (line, $2\times3$ cluster), gflow (including XZ/YZ planes) and Pauli flow.
- Under test: `standardize` (with its rule trace), `signal_shift`, `branch_operators`, `is_deterministic`, `dependency_graph` and `schedule`.
- References:
  1. An independent dense simulator. It builds $O_{\rm eff}$ as a matrix, projects onto its eigenvector, and uses Bell-paired reference qubits to form Choi vectors.
  2. The Graphix 0.4 statevector simulator with `FixedBranchSelector`, on a random normalised input. Only branches with $\|K_b\psi\| \ge 10^{-3}$ are compared.
- Negative control: drop the last Pauli correction of the standard form.
- CV part:
  - `calculus.cv.command_view` on CV-flow-corrected patterns of a 4-node wire and a $2\times3$ cluster, comparing role counts with the command list;
  - every affine correction form evaluated at 5 random record sets drawn from $\mathcal N(0, 3^2)$, against direct expression evaluation;
  - a deliberately nonlinear feed-forward, $q = m_0^2$.
- Outputs: `R54_measurement_calculus` (results) and figure `R54_measurement_calculus`. Seed 20261154.

### Expected outcome

Acceptance criteria:

| Check | Criterion |
|---|---|
| Branch maps after standardize / signal shift | $\le 10^{-10}$ up to phase |
| Branch probabilities | $\le 10^{-12}$ |
| Library vs independent simulator | $\le 10^{-10}$ |
| Graphix, $1 - \vert\langle\cdot\vert\cdot\rangle\vert$ | $\le 10^{-10}$, with at least one branch checked |
| Standardized output | in standard form |
| Dependency edges | equal the $s$/$t$ domains, and every schedule layer depends only on earlier layers |
| Flow-derived patterns | strongly deterministic |
| Random patterns | library determinism verdict matches the simulator |
| Negative control | detected (deviation $> 10^{-3}$) in at least 80% of patterns |
| CV role counts | match the command list |
| CV affine forms | relative error $\le 10^{-12}$ |
| Nonlinear CV feed-forward | reported as non-affine |

Figure, three panels:

- (a) per-pattern maximum branch deviations for the four comparisons, all below the $10^{-10}$ line;
- (b) dependency depth before vs after standardization and signal shifting (marker area = count), typically on or below the diagonal (reported, not asserted);
- (c) bar chart of rule applications (EX, EZ, MX, MZ, S-commute, commute).

<!-- core:R55 -->
**Title:** Local complementation and local-Clifford equivalence of graph states

**Why this experiment.** Local complementation is the graph-level form of local Clifford (LC) equivalence for qubit graph states, and it underlies graph simplification and resource optimisation in MBQC. `photographiq.calculus.clifford` provides:

- `local_complement`, which returns the new graph;
- `lc_cliffords`, which returns the local unitaries;
- `lc_equivalence`, which searches for an LC sequence between two graphs and returns a Clifford witness.

A wrong edge toggle, a wrong Clifford, or a spurious "equivalent" or "not equivalent" verdict would propagate into every circuit derived from these tools.

For CV, the library implements the Gaussian graphical calculus of Menicucci, Flammia and van Loock (MFvL 2011) but not CV local complementation. R55 verifies both the qubit claims and the CV scope.

### Theory

Qubit graph state: $|G\rangle = \prod_{(u,v)\in E} CZ_{uv}\,|+\rangle^{\otimes n}$. Local complementation at $v$ toggles every edge inside the neighbourhood $N(v)$. The two graph states are related by

$$|\tau_v(G)\rangle = e^{i\phi}\,U_v|G\rangle, \qquad U_v = e^{-i\frac{\pi}{4}X_v}\prod_{w\in N(v)} e^{+i\frac{\pi}{4}Z_w},$$

following Van den Nest, Dehaene and De Moor (PRA 69, 022316 (2004)). States are compared up to the reported global phase $\phi$.

The LU invariant used is the vector of Schmidt ranks over all bipartitions. Local unitaries cannot change it, so if two graphs have different vectors, no LC sequence relates them.

Gaussian graphs. With $\hbar = 2$, a pure zero-mean state is $\psi(q) \propto \exp(\tfrac{i}{4} q^T Z q)$, where $Z = V + iU$ and $U > 0$. The covariance in block form is

$$\begin{pmatrix} U^{-1} & U^{-1}V \\ VU^{-1} & U + VU^{-1}V\end{pmatrix}.$$

The p-squeezed CV cluster with resource squeezing $r$ is $Z = A + i e^{-2r} I$. The graph rules are:

| Operation | Rule |
|---|---|
| $CZ(g)$ | $Z_{uv} \mathrel{+}= g$ |
| Shear $s$ | $Z_{kk} \mathrel{+}= s$ |
| General symplectic $\begin{pmatrix}A & B\\ C & D\end{pmatrix}$ | $Z \to (C + DZ)(A + BZ)^{-1}$ |
| Ideal $q$ homodyne | delete the vertex |
| Ideal $p$ homodyne | Schur complement $Z' = Z_{rr} - z z^T / Z_{kk}$ |

Displacements are tracked in a separate exact mean vector.

### What the code does

- Graphs: path $P_4$, star $S_4$, cycle $C_5$, complete $K_4$, the $2\times3$ cluster, and 6 seeded random connected graphs with $n = 4..6$. Seed 20261155.
- Single local complementations, at every vertex of every graph:
  - the library graph is compared with an independent edge toggle;
  - $U_v$ is built from cosines and sines of Pauli matrices (no library code), and the library's `lc_cliffords` are applied as well;
  - both are compared with the target state $|\tau_v G\rangle$, built independently from CZ phases.
- LC sequences: 4 random sequences of length 1-6 per graph. `lc_equivalence(G, target)` must find a sequence no longer than the generating one. The returned sequence is replayed on the edges, and the Clifford witness is applied to statevectors.
- Non-equivalences: $P_4$ vs $S_4$, $C_5$ vs $K_5$, and cluster vs $K_6$. The library must say "not equivalent" and the Schmidt-rank vectors must differ. Positive control: $S_4$ vs $K_4$ (both in the GHZ class) must be found equivalent, with equal invariants.
- CV: 6 trials with $n = 3..4$ modes, $r \in [0.3, 1]$, and random weighted adjacency (entries in $[-1.5, 1.5]$, density 0.6). The checks, in order:
  - `GaussianGraph.cluster`;
  - then, in sequence, `cz` (0.7), `quadratic_phase` ($-0.4$), `squeeze` (0.3), `rotate` (0.9), `beamsplitter` (0.6), each against $S V S^T$ and $S\mu$;
  - `measure_q` and `measure_p`, against Gaussian conditioning (Schur complement).
- `cv_local_complementation()` must raise `NotImplementedError`.
- Outputs: `R55_local_clifford` (results) and figure `R55_local_clifford`.

### Expected outcome

Acceptance:

- Edge sets match the independent toggle.
- $1 - |\langle\cdot|\cdot\rangle| \le 10^{-12}$ for the independent $U_v$, the library Cliffords and every sequence witness.
- Every sequence is found, with length at most the generating length.
- The non-equivalences are confirmed by both the library and the invariant, and the GHZ positive control passes.
- CV covariance and mean errors are $\le 10^{-10}$, relative to $\max(1, |V|)$.
- CV local complementation remains unimplemented.

The global phases are expected to be multiples of $\pi/4$; they are reported, not asserted.

Figure, three panels:

- (a) $|C_5\rangle$;
- (b) after $\tau_0$, with vertex 0 highlighted, added edges red solid, removed edges red dotted, and the measured global phase in the title;
- (c) after a further $\tau_2$.

<!-- core:R56 -->
**Title:** Scalar-exact soundness of the qubit ZX-calculus subset

**Why this experiment.** The ZX module `photographiq.calculus.zx` is used to reason about graph states and MBQC branches. It claims that every implemented rewrite preserves the diagram's linear map exactly, including the scalar. Many ZX implementations work "up to a non-zero scalar". An error in a scalar factor goes unnoticed in such a setting, yet it corrupts probabilities and branch weights once diagrams are composed or compared with Kraus operators. R56 checks each rule, random rule applications, the `simplify_spiders` fixed point, and the graph-state and MBQC-branch semantics. All comparisons are made with no freedom in phase or normalisation. The scope is the qubit ZX subset only. No CV-ZX identity is claimed.

### Theory

Semantics follow Coecke and Kissinger, with exact scalars. An $n$-leg Z spider with phase $\alpha$ is

$$Z_\alpha = |0\cdots0\rangle\langle0\cdots0| + e^{i\alpha}|1\cdots1\rangle\langle1\cdots1|,$$

and the X spider is the same tensor with $|\pm\rangle$ in place of $|0\rangle$ and $|1\rangle$. A simple edge is the identity and a Hadamard edge is $H$. The diagram denotes a $2^{n_{\rm out}}\times2^{n_{\rm in}}$ matrix (first boundary most significant) times its stored scalar.

The implemented rules and the exact factors they record:

| Rule | Factor |
|---|---|
| Spider fusion (phases add) | 1 |
| Identity removal (two Hadamard edges compose to a simple one, $HH = I$) | 1 |
| Colour change (toggles incident edges) | 1 |
| Pauli state copy | $e^{i a\alpha}\,2^{(1-n)/2}$, for state phase $a\pi$, $a \in \{0,1\}$, spider phase $\alpha$, $n$ remaining legs |
| $2\times2$ bialgebra | $\sqrt2$ |
| Hopf (two parallel simple Z-X edges removed) | $1/2$ |
| Parallel Hadamard-edge cancellation | $1/2$ |
| Self-loop removal | a simple loop gives factor 1; each Hadamard loop adds $\pi$ and a factor $2^{-1/2}$ |

`simplify_spiders` applies fusion, self-loop removal and identity removal to a fixed point. Soundness means $M_{\rm after} = M_{\rm before}$ exactly, and the reported proportionality constant must equal 1.

### What the code does

- Independent evaluator, written in the script. It reads only the vertex, edge and boundary data and the scalar of a diagram. Z spiders are built from computational-basis projectors. X spiders are built directly from $|+\rangle$/$|-\rangle$ outer products, not as Hadamard-conjugated Z spiders. The network is contracted with `numpy.einsum`.
- Textbook diagrams, one per rule (8 in total):
  - fusion of $Z(0.7)$ and $Z(1.1)$, compared with its closed form;
  - $H$-$Z(0)$-$H$ identity removal (equals $I$);
  - colour change of $X(0.4)$, compared with its closed form;
  - bialgebra, Hopf, Hadamard-pair cancellation, $X(\pi)$ copy through $Z(0.9)$, and Hadamard self-loop removal.
- Random rule applications: 600 random diagrams (`--full`), with 2-4 spiders, random colours, phases in $\{0, 0, \pi, U(0, 2\pi)\}$, random simple and Hadamard edges, and boundaries. In each, the first applicable rule at random vertex pairs is applied.
- Graph states: `graph_state_diagram` for random $G(n, 0.6)$ with $n = 3, 4, 5$ (3 each), before and after `simplify_spiders`, compared with a CZ-phase statevector.
- MBQC branches: all 4 branches of a 3-qubit pattern with XY and YZ measurements and X/Z corrections. Each `pattern_branch_diagram` is compared with the Kraus operator from `enumerate_branches`.
- Every library `Diagram.matrix()` is also compared with the independent evaluator.
- Outputs: `R56_zx_calculus` (results, rule counts) and figure `R56_zx_calculus`. Seed 20261156.

### Expected outcome

Acceptance criteria:

- Every comparison satisfies $\max|M_a - M_b|/\max(1, |M_b|) \le 10^{-12}$. This covers:
  - after vs before;
  - library vs independent evaluator, before and after;
  - closed forms;
  - graph-state and branch semantics.
- At least $\min(100, 600/2) = 100$ random rewrites are actually applied.

Deviations are expected at the level of float64 contraction rounding. The proportionality constants should read exactly 1.

Figure: a $4\times4$ grid. For each of the 8 textbook rules, the before diagram is drawn above the after diagram. Green dots are Z spiders and red dots are X spiders, with phases in units of $\pi$. Dashed blue edges with yellow boxes are Hadamard edges. Boundaries are black dots, with inputs on the left and outputs on the right. Each after-panel title gives the recorded scalar factor and the deviation, which is expected at rounding level.

<!-- core:R57 -->
**Title:** Interactive visualization: one validated graph, four views

**Why this experiment.** The paper's figures and the interactive viewer (`photographiq.interactive`) show weighted cluster graphs, flow dependencies, classical corrections and recorded trajectories. A visualization layer can misrepresent the physics in two ways:

- dragging nodes or switching layouts could alter the model it displays;
- arrows and labels could be produced by a separate code path that drifts from the semantics.

R57 shows that positions are display metadata only. It also checks that every drawn arrow and coefficient equals validated data, and that the exported files reproduce the model exactly.

### Theory

The displayed object is the compiled two-mode Gaussian circuit $R_0(0.4)\oplus S_1(0.3)$ followed by $CZ(0.8)$. It is compiled to a CV cluster pattern at resource squeezing $r = 1.5$ ($\hbar = 2$). The open graph carries the following data:

- node roles (input, output, measured);
- CZ weights;
- homodyne shears;
- the resource squeezing.

Its CV flow (R51) supplies layers and a standard-form correction map $u \to \{t: (c_q, c_p)\}$. The compiled pattern executes affine feed-forward displacements of the form

$$q_t \to q_t + \textstyle\sum_u c^{(q)}_{u t}\, m_u, \qquad p_t \to p_t + \textstyle\sum_u c^{(p)}_{u t}\, m_u .$$

The coefficient of record $m_u$ is recovered from each `Displace` expression $f(m)$ by finite differences at unit records, $c_{ut} = f(e_u) - f(0)$. Affinity is confirmed at a random point.

The flow's map (corrections after all CZs) and the compiled map (each correction applied before the next CZ) are different sets of arrows. They are equivalent operationally: a pattern built from exactly the flow map (`corrected_pattern`) must have the same Gaussian channel $(S, N)$ as the compiled pattern.

### What the code does

- Views: construct, flow, corrections and execution. They are built with `GraphView.from_pattern`, `flows.analyze`, `with_corrections(pattern)` and `execution_trace`. The execution view uses one seeded `native-gaussian` run (seed 57).
- Arrows vs data:
  - drawn flow arrows must equal `flow.correction_map`;
  - drawn correction arrows and labels must equal the $(u, t, c)$ table read directly from the `Displace` expressions (independent of `pattern_corrections`);
  - flow layers must respect every dependency;
  - the channel of the flow-map pattern must equal the channel of the compiled pattern.
- Layout invariance: all offered layouts (view, publication, layered, hierarchical, grid, circular, force) plus 5 random manual layouts, applied to all four stages. Each time, the following must be unchanged:
  - the graph digest;
  - the semantic payload with positions removed;
  - the arrow sets.

  A repeated seeded simulation must reproduce the outcomes and the pattern JSON.
- Static SVG per stage, parsed back: node ids, edge count, flow and correction arrow ids, and labels must equal the model. Every routed edge and overlay curve must keep at least the documented clearance of 0.34 layout units from non-endpoint nodes.
- Round trips: editor JSON (graph and layout, via `apply_edits`), the layout file (`export_layout`/`import_layout`), and the OpenGraph JSON metadata (measurements, squeezing, inputs/outputs).
- Browser check (optional):
  - It runs real mouse drag, stage and layout switching in headless Edge through Playwright, together with `scripts/check_interactive_browser.py`.
  - When Playwright is not installed (the default on Colab), this step is recorded as `not_run`, never as a pass.
  - Caveat: if Playwright is installed without an Edge browser, the launch error is recorded as a failure.
- Notebook check: nbclient executes a two-cell notebook and confirms that an iframe `srcdoc` is produced. Rendering that iframe in a browser also needs Playwright and is otherwise `not_run`.
- Outputs:
  - `R57_interactive_visualization` (check table and metadata);
  - per-stage SVGs and `layout.json` under `results/raw/R57_exports/`;
  - `results/raw/R57_interactive.html`;
  - figure `R57_interactive_visualization`.

### Expected outcome

Acceptance:

- The flow status is `certified`.
- All set equalities hold.
- Correction coefficients match to $\le 10^{-12}$.
- The two channels differ by $\le 10^{-9}$.
- Every layout/stage combination is invariant.
- The SVGs match the model.
- The minimum clearance is $\ge 0.34$.
- Position drift in the JSON round trip is $\le 10^{-12}$; exact equality is expected.
- Metadata round trips are exact.

Entries with status `not_run` do not fail the run unless `--require-browser` is given (the notebook does not pass it). The flow-analysis time is recorded as provisional only.

Figure, four stacked panels:

- (a) the complete weighted graph, with CZ weights on the edges, input squares, output diamonds and measured-node outlines;
- (b) flow arrows with the measurement order $\#k$;
- (c) orange dashed correction arrows with coefficients of the raw record;
- (d) the recorded `native-gaussian` outcomes $m$.

The flow-map and compiled arrow sets differ by a reported number of arrows, and the caption gives the channel equality.

<!-- core:R58 -->
**Title:** Batched Gaussian kernels on CPU and GPU: accuracy and separated stage timing

**Why this experiment.** `photographiq.gpu` offers CuPy implementations of batched Gaussian-state operations, alongside NumPy (vectorised CPU) and native C++/OpenMP batch kernels. GPU performance claims are often inflated in two ways: by not counting data transfers, or by silently falling back to the CPU. R58 makes three checks:

- all engines must agree with a float64 reference;
- each GPU row must prove it ran on the device;
- the GPU time is reported in separate stages: host preparation, host-to-device (H2D) copy, kernel, and device-to-host (D2H) copy.

**Requirements:** a CUDA GPU with CuPy, which on Colab means a GPU runtime. Without one, the script runs only the NumPy and native CPU rows and records that the GPU rows were not run. A CPU result is never labelled as GPU.

### Theory

Conventions: $\hbar = 2$, interleaved quadratures, vacuum $V = I$. The batch consists of $B$ independent $n$-mode states $(\mu_b, V_b)$. Four operations are tested:

- Gaussian unitary with symplectic $S$: $\mu \to S\mu$, $V \to S V S^T$.
  - Local version: a beam splitter with $\theta = 0.37$ on modes $(\lfloor n/3\rfloor, \lfloor 2n/3\rfloor)$, work $O(Bn)$.
  - Dense version: a different random symplectic per state, $S_b = O(U_b)\,D(r_b)$, a batched GEMM with work $O(Bn^3)$. Here $O(U)$ is the orthogonal symplectic of a Haar unitary $U$ (QR of a complex Ginibre matrix with phase fix), and $D$ applies single-mode squeezers with $|r| \le 0.3$.
- Homodyne conditioning on mode $k$ at angle $\phi$ with efficiency $\eta$, given an outcome $x$. The measured quadrature is $x_\phi = \cos\phi\,q_k + \sin\phi\,p_k$. With $v$ its variance, $\mu_x$ its mean and $c$ its cross-covariance with the other modes, the update is the Schur complement

$$V' = V_{\bar k\bar k} - \frac{c\,c^T}{v + (1-\eta)/\eta}, \qquad \mu' = \mu_{\bar k} + c\,\frac{x - \mu_x}{v + (1-\eta)/\eta},$$

  at work $O(Bn^2)$, with $\phi = 0.4$, $x = 0.1$, $\eta = 0.9$. These are postselected branches, not sampled trajectories.
- Purity: $\det(V)^{-1/2}$ from a batched `slogdet`. There is no native batch kernel for this.

Accuracy is measured as $\max|a - a_{\rm ref}|/\max(1, \max|a_{\rm ref}|)$ against the NumPy float64 result.

### What the code does

- Input: $B$ copies of one physical state, a squeezed CZ chain with thermal loss built by `GaussianBackend`, so every engine sees identical data. Seed 20261158.
- Grid (`--full`): $n \in \{4, 16, 32, 64\}$, $B \in \{64, 1024, 16384\}$, 21 timed repeats after 2 warm-ups, float64 plus explicitly labelled float32 GPU rows. Configurations with $B(2n)^2\cdot 8 > 512$ MB are skipped (`--max-bytes`). This removes $(32, 16384)$ and $(64, 16384)$.
- Engines:
  - NumPy CPU (`device="cpu"`), which also gives the reference;
  - native C++ (`batch_apply_symplectic`, `batch_condition_homodyne`);
  - CuPy GPU, timed end to end with NumPy in/out, plus stage timings from synchronised CUDA events: H2D; kernel on device-resident data (`on_device=True`, including the library's validity-check syncs); D2H.
- Medians of the repeats are reported. First calls (compilation, handle creation) are never timing samples.
- Device verification: `execution.device` starts with `cuda:`, `backend == "cupy"`, and `pointerGetAttributes(ptr).type == memoryTypeDevice` for device outputs.
- Outputs: `R58_gpu_gaussian` (rows, device info, skipped configurations) and figure `R58_gpu_gaussian`.

### Expected outcome

Acceptance:

| Rows | Criterion vs NumPy |
|---|---|
| float64 engines | $\le 10^{-12}$ |
| purity | $\le 10^{-11}$ (different LAPACK/cuSOLVER paths) |
| float32 GPU | $\le 10^{-3}$ |
| every GPU row | device verified |

Float64 differences should be at the $10^{-15}$ level.

Timing caveats:

- Timings are labelled PROVISIONAL unless `PGQ_IDLE_MACHINE=1` declares an idle machine; Colab GPUs are shared.
- Expectations: the GPU is slower than NumPy at small $B$. Transfers dominate the end-to-end GPU time for the $O(Bn)$ and $O(Bn^2)$ operations. The clearest kernel speed-up appears for the dense $O(Bn^3)$ case.
- Consumer GPUs have low FP64 throughput.

Figure, two panels:

- (a) Log-scale bars per operation at $n = \max$ and $B = \max$: host preparation, NumPy, native, GPU H2D/kernel/D2H, GPU end to end, and the float32 kernel. Note that this configuration (64 modes, 16384 states, about 2.1 GB of covariances) exceeds the default 512 MB cap. With the `--full` grid as coded, panel (a) is therefore empty unless `--max-bytes` is raised.
- (b) NumPy time divided by GPU kernel time vs $B$ for each operation and $n$. Points are below 1 at small $B$ and rise with $B$.

<!-- core:R59 -->
**Title:** PhotoGraphiQML physical execution of a finite-GKP MuTA on the native Fock engine

**Why this experiment.** PhotoGraphiQML trains measurement-based quantum neural networks, specifically the MuTA ansatz of Mantilla Calderon et al., at the logical qubit level. Its physical layer lowers a restricted MuTA to finite-energy GKP modes and executes it as a PhotoGraphiQ CV pattern. The restriction is to signed-X measurements, with angles 0 or $\pi$.

Before the compiled `native-fock` engine can replace the Piquasso-backed `piquasso-fock` reference, it must reproduce Piquasso's statistics, conditional states and sampled records, and it must converge in the Fock cutoff. The logical-to-physical interface (lowering, Pauli frame, decoding) must also stay consistent with the ideal logical model. The logical-physical gap is reported as a property of the finite resource, not hidden as a numerical error.

### Theory

Conventions: $\hbar = 2$, so the GKP logical spacing is $\ell = \sqrt{2\pi}$. Each finite-energy codeword is

$$\psi_\mu(q) \propto \sum_{s=-K}^{K} \exp\!\Bigl[-\frac{(q - c_s)^2}{4\Delta^2} - \frac{\kappa^2 c_s^2}{4}\Bigr], \qquad c_s = \ell\,(2s + \mu),\ \mu\in\{0,1\},$$

with peak width $\Delta$, envelope $\kappa$ and $K$ peaks per side. Codewords are projected onto a total-photon-number cutoff $c$ (exclusive). They are not orthogonal: $|\langle 0|1\rangle| \neq 0$ is reported.

The pattern is built from three ingredients:

- Logical CZ is the unit-gain `Entangle`, $p_u \to p_u + q_v$ and $p_v \to p_v + q_u$.
- Each measured node is a $p$-homodyne. Its record $m$ is decoded by the nearest lattice cell, giving the bit $b = \mathrm{round}(m/\ell) \bmod 2$.
- The decoded bits drive a classical logical Pauli frame (`lowering.node_frame`). The frame is applied to the final joint X/Z modular readout of the output modes.

Because the codewords overlap, the physical instrument is not the logical unitary. The logical-physical gap is quantified by the total variation distance

$$\mathrm{TV}(p, q) = \tfrac12\sum_x |p_x - q_x|,$$

measured from the ideal output distribution of the logical MuTA. Engine agreement is measured by $\max|\Delta P|$ of the decoded joint probabilities and by the final-state infidelity $1 - F$ (squared Uhlmann fidelity).

The two engines differ only in how CZ is truncated. Native applies the projected $P U P$ exactly, while Piquasso truncates after each Euler factor. Their difference is therefore bounded by the truncation errors of each engine.

For shots, the Rao-Blackwell estimate averages the conditional readout probabilities over sampled analog trajectories, $\hat P = n^{-1}\sum_s P(x\mid m^{(s)})$. Its standard error is reported together with the empirical frequencies and their binomial standard errors.

### What the code does

Configuration (`--full`, seed 20261159). Resources (peak width $\Delta$, envelope $\kappa$, peaks, grid points):

| Resource | $\Delta$ | $\kappa$ | Peaks | Grid points | Resolved cutoff |
|---|---|---|---|---|---|
| wide | 0.9 | 0.9 | 4 | 1025 | 32 |
| moderate | 0.6 | 0.6 | 6 | 2049 | 40 |
| narrow | 0.45 | 0.45 | 8 | 2049 | 56 |

**A. Engine agreement.** Fixed analog outcomes are drawn from $\mathcal N(0, 0.4^2)$, and parameters are random in $\{0, \pi\}$. The cases are:

- 1 wire, moderate resource, cutoffs 32, 40 and 48, seeds 0-4, output bases Z and X;
- 1 wire, wide resource, cutoff 32, seeds 0-4;
- 2 wires, wide resource, cutoff 32, seed 2.

**B. Cutoff convergence.**

- 1 wire, moderate resource: native at cutoffs 32-64, Piquasso at 32, 40 and 48.
- 2 wires, wide resource: native at cutoffs 32, 40 and 48.

All are compared against the largest native cutoff.

**C. Pauli frame.** All $2^4$ lattice-point branches (analog outcome 0 or $\ell$ per measured node) of a 1-wire MuTA are run at each resource level at its resolved cutoff. TV to the logical target is computed with and without the frame. Each branch is a postselected trajectory.

**D. Shots.** 32 physical shots (moderate resource, cutoff 40, seed 160) on both engines with identical seeds.

**E. Runtime and memory.** Hilbert dimension and `tracemalloc` peak (blind to native C++ buffers) are recorded.

**F. Interface checks.** The logical target is compared with `MuTA.unitary` applied to the input. Angle $\pi/2$ must be rejected before any Fock allocation on both backends, and `native-mixed-fock` must be rejected.

Retained-norm guard events are recorded as data, except in the validated wide domain, where they count as failures.

Outputs: `R59_photographiqml_physical` (all sections) and figure `R59_photographiqml_physical`.

### Expected outcome

Acceptance criteria:

| Check | Domain | Criterion |
|---|---|---|
| $\max\vert\Delta P\vert$ | wide only | $\le 2\times10^{-4}$ |
| $1-F$ | wide only | $\le 2\times10^{-6}$ |
| Both metrics at the largest cutoff | every moderate case | smaller than at the smallest cutoff |
| Engine identity | all cases | both engines report as executed |
| Decoded bits | all cases | identical |
| Logical-target error | all cases | $\le 10^{-12}$ |
| Probability sums | all cases | within $10^{-9}$ of 1 |
| Native negativity | all cases | $\le 10^{-9}$ |
| Native convergence | cutoffs $\ge$ the resolved cutoff | $\le 2\times10^{-4}$ from the reference; monotonicity reported, not asserted |
| Mean TV | every resource | frame-corrected below uncorrected |
| Worst corrected TV | wide to moderate to narrow | non-increasing |
| Shots | 32 shots | identical sampled bits, decoded bits and frequencies; Rao-Blackwell $\vert\Delta P\vert \le 2\times10^{-4}$ |
| Interface | both backends | both rejections occur |

Runtime numbers are provisional.

Figure, six panels:

- (a) Engine $|\Delta P|$ (filled markers: wide; open markers: moderate) and $1-F$ per case, with tolerance lines.
- (b) Distance to the native reference vs cutoff, for both engines.
- (c) TV over the 16 branches, corrected vs uncorrected, per resource.
- (d) Shot probabilities (Rao-Blackwell and empirical, with standard errors) for both engines, against the logical target.
- (e) Runtime vs peak Hilbert dimension.
- (f) Logical-physical TV, native vs Piquasso, on the diagonal.

<!-- core:R60 -->
**Title:** End-to-end showcase from a target Gaussian operation to validated output states

**Why this experiment.** The previous experiments validate components one at a time. R60 runs the complete user workflow on one two-mode example, with every number checked against an independent oracle:

1. target operation;
2. circuit;
3. compiled MBQC pattern;
4. complete weighted cluster graph;
5. CV flow certificate;
6. measurement schedules;
7. correction maps;
8. simulated trajectories;
9. output-state analysis;
10. finite-squeezing study;
11. interactive and static exports.

This demonstrates that the pieces compose correctly. It also shows the paper's central interpretation in one place: the intended map $S$, the finite-squeezing noise $N$ and the outcome-dependent conditional means are separate objects.

### Theory

Conventions: $\hbar = 2$, interleaved $(q_0, p_0, q_1, p_1)$, $R(t) = \begin{pmatrix}\cos t & -\sin t\\ \sin t & \cos t\end{pmatrix}$, $S(r) = \mathrm{diag}(e^{-r}, e^{r})$, and $CZ(g)$: $p_0 \to p_0 + g q_1$, $p_1 \to p_1 + g q_0$. The target symplectic map is

$$S = \bigl(I \oplus R(-0.6)\bigr)\, CZ(0.8)\, \bigl(R(0.4)\oplus S(0.3)\bigr), \qquad S\,\Omega\,S^T = \Omega .$$

At resource squeezing $r$, the compiled pattern averaged over outcomes is a Gaussian channel

$$\mu \to S\mu + d, \qquad V \to S V S^T + N(r).$$

Every resource mode contributes variance $e^{-2r}$ through linear maps that do not depend on $r$, so $N(r) = e^{-2r}N_0$ exactly. A single trajectory has a conditional covariance $V_c$ that does not depend on the outcomes, and a conditional mean $\bar x(m)$ that does. By the law of total covariance,

$$\mathrm{Cov}_{\rm ens} = V_c + \mathrm{Cov}_m[\bar x(m)] = S V S^T + N,$$

so the conditional means scatter around $S\mu + d$ with covariance $S V S^T + N - V_c$. The CV flow certificate (Booth-Markham) holds in the infinite-squeezing limit. Its determinism map must equal $S$.

The flow's standard-form correction map and the compiled feed-forward (each correction applied before the next CZ) are different but equivalent. A pattern that executes exactly the flow map must have the same channel.

### What the code does

- Target: `pg.Circuit(2)` with `rotate(0, 0.4)`, `squeeze(1, 0.3)`, `cz(0, 1, 0.8)` and `rotate(1, -0.6)`, compiled by `Circuit.compile(squeezing=1.5)`, with the compile trace kept.
- Graph and flow: `GraphView.from_pattern` builds the graph, and `flows.analyze` produces the certificate (status, layers, correction map, determinism matrix).
- Schedules: the flow order must respect every dependency of the flow map. The compiled command order must respect every executed feed-forward dependency (`command_dependencies`). The number of flow pairs that the compiled schedule orders differently is reported.
- Channels: `gaussian_channel` of the compiled pattern and of `flows.cv.corrected_pattern` (exact flow map). The independent oracle is the analytic product above, written in NumPy in the script.
- Trajectories: 4000 `native-gaussian` runs (`--full`) with seeds 60, 61, ..., and coherent inputs $\alpha = 0.6+0.2i$ and $-0.3+0.5i$ (mean $(2\,\mathrm{Re}\,\alpha, 2\,\mathrm{Im}\,\alpha)$, covariance $I$). The first 60 trajectories are re-run on the NumPy `gaussian` backend with the same seeds.
- Statistics:
  - z-scores of the ensemble mean, using standard error $\sqrt{\mathrm{diag}/K}$;
  - z-scores of the ensemble covariance, using the standard error of a sample covariance element, $\sqrt{(S_{ii}S_{jj} + S_{ij}^2)/(K-1)}$;
  - Gaussian fidelity between the ideal output and the ensemble output.
- Squeezing sweep: $r \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0\}$, giving $S(r)$, $\mathrm{Tr}\,N$, $1-F$ and a fitted slope of $\ln\mathrm{Tr}\,N$.
- Exports: an interactive HTML page (all four stages) and an SVG of the corrections stage. Timings come from 5 benchmark repeats and are provisional.
- Outputs: `R60_end_to_end_showcase` (checks, sweep rows, matrices, metadata) and figure `R60_end_to_end_showcase`.

### Expected outcome

All checks must pass:

| Check | Criterion |
|---|---|
| Flow status | `certified` |
| Determinism map, compiled $S$, every $S(r)$ vs analytic $S$ | $\le 10^{-9}$ |
| Symplecticity residual | $\le 10^{-9}$ |
| Schedule violations | 0 for both schedules |
| Flow-map channel vs compiled channel ($S$, $N$, $d$) | $\le 10^{-9}$ |
| Outcome dependence of the conditional covariance | $\le 10^{-10}$ |
| Ensemble mean and covariance z-scores | $\le 5$ |
| `native-gaussian` vs NumPy (same seeds) | $\le 10^{-10}$ |
| Relative spread of $N(r)e^{2r}$ | $\le 10^{-9}$ |

The fitted log-slope of $\mathrm{Tr}\,N$ should be $-2$ to high accuracy.

Figure, panels A-F:

- A: the target circuit.
- B: the complete weighted cluster graph.
- C: the flow, with each node labelled by its layer $L_k$ and all dependencies drawn.
- D: the executed corrections, with the output correction formulas.
- E: the conditional output means of the 4000 runs in the $(\langle q\rangle, \langle p\rangle)$ plane for both modes, inside the predicted $2\sigma$ ellipses, with $S\mu$ (+) and the sample mean (x).
- F: $\mathrm{Tr}\,N(r)$ on the $e^{-2r}$ line, $1-F$ falling with $r$, and $\max|S(r) - S_{\rm analytic}|$ at rounding level.

<!-- ml:R1 -->
**Title:** Triangle-neuron and MuTA graph anatomy against closed-form counts

**Why this experiment.** Every later result (unitaries, kernels, gradients, physical lowering) is computed on the resource graph that `TriangleNeuron` and `MuTA` assemble. If a node, a cross edge or a trainable measurement were silently missing or duplicated, all downstream numbers would refer to a different ansatz from the one defined in the paper (Fig. 1, Table I). This experiment pins the constructed graph to hand-derived counting formulas over the full configuration space (wire count, requested layers, `one_column`, `restrict_trainable`) and checks that the causal order used for adaptive measurement is a valid topological order.

### Theory
A triangle neuron, one paper $(n,p)$ layer with pivot wire $p$, is a graph state on nodes $(w,c)$ with wire $w\in\{0,\dots,n-1\}$ and local column $c\in\{0,\dots,4\}$. Each wire carries a 5-node chain (4 edges). Every connected non-pivot wire $w$ is joined to the pivot by the two cross edges $(p,1)-(w,0)$ and $(p,1)-(w,2)$. With $k$ connected wires (default $k=n-1$),
$$|V_\triangle| = 5n,\qquad |E_\triangle| = 4n + 2k,\qquad N^{\triangle}_{\text{train}} = 4n ,$$
since columns $0$-$3$ are measured and column 4 is the output.

A MuTA stacks $d$ paper layers. Block $b$ is relabelled $c\mapsto 4b+c$, so the output column of one block is identified with the input column of the next. The pivot sequence is $(0)^{L}$ if `one_column` is set, otherwise $(0,1,\dots,n-1)$ repeated $L$ times, so $d=L$ or $d=nL$. Hence
$$|V| = n(4d+1),\qquad |E| = 4nd + 2d(n-1),\qquad N_{\text{param}} = \begin{cases}4nd & \text{full model}\\ 3nd & \text{restricted}\end{cases}$$
where the restricted model freezes the column $c\equiv 3 \pmod 4$ measurements. The flow is $f(w,c)=(w,c+1)$ for every non-output node. The correction set of node $v$ is $\big(f(v),\,N(f(v))\setminus\{v\}\big)$, where $N(\cdot)$ is the graph neighbourhood. The dependency DAG has edges $v\to f(v)$ and $v\to u$ for $u\in N(f(v))\setminus\{v\}$, and the measurement order is a lexicographic topological sort keyed by (column, wire). The ordering is consistent if every DAG edge $u\to v$ satisfies $\mathrm{pos}(u)<\mathrm{pos}(v)$ in the sequence (measurement order followed by outputs).

### What the code does
- System under test: `TriangleNeuron(n, pivot=0, connections)` (`graph`, `input_nodes`, `output_nodes`, `trainable_nodes`) and `MuTA(n, L, one_column, restrict_trainable)` (`graph`, `dependency_graph`, `measurement_order`, `n_parameters`, `paper_depth`, I/O labels).
- Oracle (class A): the closed-form formulas above, coded in `analytic_triangle_counts` / `analytic_muta_counts` without using package bookkeeping. Acyclicity is checked with `networkx.is_directed_acyclic_graph`.
- Sweep: triangle level $n\in\{1,2,3,4\}$ with `connections` in {None, (), first non-pivot wire only}. Invalid combinations raising `ValueError` are skipped. MuTA level: $n\in\{1,2,3\}$, $L\in\{1,2,3\}$, both booleans, giving 36 configurations. Fully deterministic, no seeds.
- Metrics: absolute node, edge and trainable-count errors, `paper_depth == d`, input labels $(w,0)$, output labels $(w,4)$ or $(w,4d)$, DAG acyclicity and order consistency.
- Figure `R1_triangle_anatomy` (three panels).

### Expected outcome
All integer counts should match exactly. The declared acceptance is `max_node_error == max_edge_error == max_trainable_error == 0`, with every dependency graph acyclic, every measurement order topologically consistent and all input/output labels as specified. Any mismatch raises an `AssertionError`. Figure: three side-by-side scatter panels (node-count error, edge-count error, trainable-parameter error) plot $|\text{analytic}-\text{actual}|$ for each of the 36 MuTA configurations, coloured as passing when zero, with a reference line at 0. All markers are expected to lie on that line. Triangle-level rows are saved in the CSV/JSON but not plotted.

<!-- ml:R2 -->
**Title:** Single-wire Table-I Euler identities against analytic unitaries

**Why this experiment.** Table I states that the four measured columns of a single-wire paper layer implement a specific alternating Euler rotation sequence. That identity is what makes a MuTA wire a universal single-qubit gate and fixes the meaning of every trained angle. A sign error in $R_z$, a transposed gate product or a wrong rotation axis on one column would leave the model trainable but make its parameters incompatible with the paper and with MentPy. This experiment compares `MuTA.unitary` for one wire with matrix exponentials built from locally defined Pauli matrices.

### Theory
In the ideal logical (Appendix B) picture, measuring a cluster node at angle $\alpha$ and applying the flow correction teleports the wire through
$$J(\alpha) = H\,R_z(\alpha),\qquad R_z(\alpha)=e^{i\alpha Z/2}=\mathrm{diag}\!\left(e^{i\alpha/2},e^{-i\alpha/2}\right).$$
A single-wire layer measures columns $c_0,c_1,c_2,c_3$ in order (the input node $c_0$ is also measured), so
$$U = J(c_3)J(c_2)J(c_1)J(c_0) = (H R_{z}(c_3) H)\,R_z(c_2)\,(H R_z(c_1) H)\,R_z(c_0).$$
Using $H R_z(t) H = R_x(t) = e^{itX/2}$, the four Hadamards telescope pairwise and
$$U = R_x(c_3)\,R_z(c_2)\,R_x(c_1)\,R_z(c_0),$$
an X-Z-X-Z Euler decomposition read right to left. Setting all angles but one to zero (so that $J(0)=H$) isolates each column's sub-identity: $R_z(c_0)$, $R_x(c_1)$, $R_z(c_2)$, $R_x(c_3)$. Exact agreement is expected up to floating-point rounding. The error metric is the Frobenius norm $\lVert U_{\text{MuTA}}-U_{\text{ref}}\rVert_F$, with no global-phase optimisation, so the phase convention is tested too.

### What the code does
- System under test: `MuTA(1, 1).unitary(angles)`, which runs the logical frontier contraction `logical.execute` on each basis state.
- Oracle (class A): `scipy.linalg.expm(0.5j*a*P)` for Pauli matrices `X_REF`, `Z_REF` defined in the script, independent of the package's `H`, `X`, `Z` arrays and `local_gate`/`cz` code.
- Sweep: 40 trials with `common.rng(0)`. Each draws four angles uniformly in $[-2\pi,2\pi]$ for $(c_0,c_1,c_2,c_3)$ and evaluates the full sequence plus the four isolated single-column identities.
- Metrics: `full_sequence_error`, plus `c0_error` to `c3_error` for the isolated identities.
- Figure `R2_table_one_identities` (two panels).

### Expected outcome
All errors should sit at the level of double-precision rounding. The tolerance is the canonical `common.CANONICAL_R2_TOLERANCE` $=10\,\varepsilon_{64}\approx 2.22\times10^{-15}$, and acceptance requires $\max(\text{full-sequence error},\ \text{single-column error}) <$ tol over all 40 trials. Otherwise the script raises an `AssertionError`. Figure: (left) semilog plot of the full-sequence Frobenius error per trial with the dashed tolerance line; (right) the four isolated single-column errors per trial on the same scale. All points should lie below the tolerance line.

<!-- ml:R3 -->
**Title:** Pivot coupling implements the Ising-XX entangling gate

**Why this experiment.** Entanglement in MuTA enters only through the cross edges that tie each non-pivot wire to column 1 of the pivot wire. Appendix B and Table I claim that one trainable angle on this coupling produces the two-qubit Ising gate $e^{i\phi X\otimes X/2}$. That is the basis of the paper's statement that MuTA is universal and can learn nonlocal gates (R21). If the coupling realised a different generator ($ZZ$, $XZ$, a sign-flipped angle) or picked up stray local rotations, entangling-gate learning would be mis-specified and the expressivity results misread. This experiment checks the identity over a dense angle grid against an external matrix exponential, and checks the induced entanglement through the concurrence.

### Theory
Each measured node applies $J(\alpha)=H R_z(\alpha)$ with $R_z(\alpha)=e^{i\alpha Z/2}$, and each graph edge is a $CZ$. With all angles zero, every wire of the one-column, two-wire layer composes to the identity. The coupling claim is that binding only the cross-coupled angle `alpha.w1.c1` to $\phi$ (pivot $p=0$, all other angles left at their default 0) gives
$$U(\phi) = \exp\!\left(\tfrac{i\phi}{2}\,X\otimes X\right) = \cos\tfrac{\phi}{2}\,I\otimes I + i\sin\tfrac{\phi}{2}\,X\otimes X .$$
Acting on $|00\rangle$ (the first column of $U$), it produces $\cos\frac{\phi}{2}|00\rangle + i\sin\frac{\phi}{2}|11\rangle$. For a pure two-qubit state $a|00\rangle+b|01\rangle+c|10\rangle+d|11\rangle$ the concurrence is $C=2|ad-bc|$, so
$$C\big(U(\phi)|00\rangle\big) = 2\left|\cos\tfrac{\phi}{2}\sin\tfrac{\phi}{2}\right| = |\sin\phi| ,$$
maximal (a Bell state) at $\phi=\pm\pi/2$ and zero at multiples of $\pi$. The comparison is a Frobenius norm with no global-phase freedom.

### What the code does
- System under test: `MuTA(2, one_column=True).unitary({"alpha.w1.c1": phi})` and `photographiqml.diagnostics.concurrence` applied to the first column of that unitary.
- Oracle (class A): `scipy.linalg.expm(0.5j*phi*kron(X, X))` with a Pauli $X$ defined in the script, and the closed form $|\sin\phi|$. Neither uses the package's `cz`/`local_gate` code.
- Sweep: 61 equally spaced values $\phi\in[-2\pi,2\pi]$, deterministic.
- Metrics: `unitary_error` $=\lVert U_{\text{MuTA}}-U_{\text{ref}}\rVert_F$ and `concurrence_error` $=|C_{\text{MuTA}}-|\sin\phi||$.
- Figure `R3_entangling_identity` (two panels).

### Expected outcome
Both errors should be at rounding level across the whole grid. The tolerance is `declare_tolerance(scale=1.0)` $=10\,\varepsilon_{64}\approx 2.22\times10^{-15}$, and acceptance requires $\max(\text{unitary error},\ \text{concurrence error}) <$ tol. Otherwise the script raises an `AssertionError`. Figure: (left) semilog Frobenius error against $\phi$ with the dashed tolerance line; (right) the analytic curve $|\sin\phi|$ and the measured MuTA output concurrence (dashed). The two curves should coincide, with zeros at multiples of $\pi$ and maxima of 1 at odd multiples of $\pi/2$.

<!-- ml:R4 -->
**Title:** Exhaustive adaptive-branch and byproduct-correction verification

**Why this experiment.** The production simulator `MuTA.run` never branches. It uses the Appendix B frontier translation, in which each measurement is replaced by its deterministic logical gate $J(\alpha)$. That shortcut is valid only if, for every measurement outcome, the Pauli byproduct corrections defined by the flow restore the same logical output. This is the determinism theorem of measurement-based quantum computation with flow. If the correction sets $\big(f(v), N(f(v))\setminus\{v\}\big)$ were wrong, the shortcut would still return a plausible state but would no longer describe what an adaptive MBQC device computes. This experiment enumerates every branch of small graphs with a separate full graph-state contraction.

### Theory
Prepare the open graph state: the logical input on the input nodes, $|+\rangle$ on every other node, and $CZ$ on every edge,
$$|G\rangle = \prod_{(u,v)\in E} CZ_{uv}\;\big(|\psi_{\text{in}}\rangle\otimes|+\rangle^{\otimes(|V|-n)}\big).$$
Each non-output node $v$ is measured in the XY-plane basis
$$|\pm_\alpha\rangle = \tfrac{1}{\sqrt2}\big(|0\rangle \pm e^{i\alpha}|1\rangle\big),$$
with outcome $s_v\in\{0,1\}$ corresponding to $\pm$. The projection is $\langle\pm_\alpha| = \tfrac{1}{\sqrt2}(\langle 0| + (-1)^{s_v} e^{-i\alpha}\langle 1|)$, and its branch weight $p(s_v)$ is accumulated multiplicatively. If $s_v=1$, the byproduct is undone by $X$ on the flow successor $f(v)$ and $Z$ on every node of $N(f(v))\setminus\{v\}$. Flow determinism states that for each of the $2^m$ outcome strings $\mathbf{s}$ ($m$ measured nodes)
$$\rho_{\text{out}}(\mathbf{s}) = U\rho_{\text{in}}U^\dagger,\qquad \sum_{\mathbf{s}} p(\mathbf{s}) = 1,\qquad p(\mathbf{s})\ge 0,$$
where $U=\prod_v J(\alpha_v)\cdot\prod CZ$ is the frontier-translation unitary computed by `MuTA.run`. Branch probabilities are generally not uniform for nonzero angles, so only target agreement, normalisation and nonnegativity are tested.

### What the code does
- System under test: `MuTA.run` (frontier translation in `logical.execute`), giving the target density matrix.
- Oracle (class D): `photographiqml.validation.contract_branch`, an algorithmically separate path that builds the full Kronecker graph state, projects node by node in `measurement_order`, applies explicit $X$/$Z$ corrections and returns the normalised output plus its branch probability. It is capped at 16 graph nodes.
- Configurations: `MuTA(1,1,one_column=True)` (5 nodes, 4 measured, 16 branches) and `MuTA(2,1,one_column=True)` (10 nodes, 8 measured, 256 branches), each with seeds 0, 1, 2. The input is a Haar-random state `haar_states(n,1,seed)`, and the angles come from `model.initialize(seed+1000, scale=1.5)`.
- Metrics per configuration: maximum branch error $\lVert\rho(\mathbf{s})-\rho_{\text{target}}\rVert_F$ over all branches, $|\sum p-1|$, and the minimum branch probability.
- Figure `R4_adaptive_branches` (two panels).

### Expected outcome
Every corrected branch should reproduce the target to rounding level, and the probabilities should sum to one. The tolerance is `declare_tolerance(scale=1.0, safety_factor=100.0)` $=100\,\varepsilon_{64}\approx 2.22\times10^{-14}$, looser than R2/R3 because the full contraction involves more floating-point operations. Acceptance requires max branch error $<$ tol, max $|\sum p - 1| <$ tol and min probability $> -$tol. Figure: (left) semilog maximum corrected-branch error for each of the 6 (wire count, seed) configurations against the dashed tolerance; (right) bar chart of $|\sum p-1|$ per configuration with the tolerance line.

<!-- ml:R5 -->
**Title:** Logical execution against an independently written brute-force MBQC contraction

**Why this experiment.** R4 checks the frontier-translation simulator exhaustively, but only on graphs of up to 10 nodes and against `validation.contract_branch`, which shares the package's `cz`/`local_gate` primitives. A bug common to the package's primitives, or one that only appears once several paper layers are composed, would escape it. This experiment adds a third implementation, written from scratch in the script with different NumPy primitives, and compares it with `MuTA.run` on larger multi-layer, multi-wire graphs with Haar-random inputs and generic angles.

### Theory
The resource state is the open graph state
$$|G\rangle = \prod_{(u,v)\in E} CZ_{uv}\,\big(|\psi_{\text{in}}\rangle\otimes|+\rangle^{\otimes(|V|-n)}\big),$$
and each non-output node $v$ is measured by the bra $\langle m_v| = \tfrac{1}{\sqrt2}\big(\langle 0| + e^{-i\alpha_v}\langle 1|\big)$, the $s_v=0$ outcome of the XY-plane basis $|\pm_{\alpha}\rangle=(|0\rangle\pm e^{i\alpha}|1\rangle)/\sqrt2$. Byproduct corrections ($X$ on $f(v)$, $Z$ on $N(f(v))\setminus\{v\}$) are applied only when $s_v=1$, so the all-zero branch $\mathbf{s}=\mathbf{0}$ needs no correction. By flow determinism (shown exhaustively for small graphs in R4), this single branch must already equal the deterministic target:
$$\frac{\big(\bigotimes_v\langle m_v|\big)|G\rangle}{\big\lVert\big(\bigotimes_v\langle m_v|\big)|G\rangle\big\rVert} \;\propto\; U(\boldsymbol{\alpha})|\psi_{\text{in}}\rangle,\qquad U(\boldsymbol\alpha)=\textstyle\prod_v J(\alpha_v)\prod CZ .$$
Comparing density matrices $\rho=|\psi\rangle\langle\psi|$ removes the branch's global phase. The dense contraction costs memory $\propto 2^{|V|}$, so graphs are capped at 20 nodes.

### What the code does
- System under test: `MuTA.run(state, angles).density_matrix` (frontier translation).
- Oracle (class D): `independent_contraction` in the script. It embeds $|+\rangle$ ancillas with `np.kron`, applies each $CZ$ as an elementwise sign flip on the reshaped $2^{\otimes|V|}$ tensor (not the package's bit-parity `cz`), measures by `tensordot` against the zero-outcome bra (not `local_gate`), renormalises, and permutes the remaining output qubits into wire order. The measurement equations are shared physics, not shared code.
- Configurations (fixed list): (n, L, `one_column`) = (1,1,T) 5 nodes; (1,3,T) 13; (2,1,T) 10; (2,1,F) 18; (2,2,T) 18; (3,1,T) 15. None exceeds the 20-node cap.
- Each configuration uses seeds 0-3 (24 cases): Haar input `haar_states(n,1,seed)` and angles uniform in $[-\pi,\pi]$ from `common.rng(seed+500)`.
- Metrics: density-matrix Frobenius error; the zero-branch probability is recorded for information.
- Figure `R5_brute_force_agreement`.

### Expected outcome
All 24 cases should agree to rounding level. The tolerance is `declare_tolerance(scale=1.0, safety_factor=200.0)` $=200\,\varepsilon_{64}\approx 4.44\times10^{-14}$, which allows for the extra operations of a dense contraction over up to $2^{18}$ amplitudes. Acceptance requires the maximum density-matrix error to be below this tolerance; otherwise the script raises an `AssertionError`. Figure: a single semilog panel with one marker per (configuration, seed), labelled e.g. `2w1Lnc s3`, against the dashed tolerance line. All markers should lie below it.

<!-- ml:R6 -->
**Title:** Batched execution consistency, including the empty batch

**Why this experiment.** The classical-ML wrappers (classifier, regressor, kernel, gate-learning losses) evaluate MuTA on whole datasets through `MuTA.run_batch`. The API contract states that a batch of normalised complex rows of shape $(N,2^n)$ with shared parameters returns outputs of shape $(N,2^n)$, and that an empty batch is supported. If batching reordered rows, broadcast parameters wrongly, or failed on $N=0$ (as happens with empty cross-validation folds or filtered datasets), every dataset-level metric would be silently corrupted while single-sample tests still passed. This is a contract test, not a physics test.

### Theory
For a fixed parameter vector $\boldsymbol\alpha$, the logical model is the linear map $|\psi\rangle\mapsto U(\boldsymbol\alpha)|\psi\rangle$. Stacking inputs as rows of $\Psi\in\mathbb{C}^{N\times 2^n}$, the batched output must be
$$\Psi_{\text{out}} = \Psi\,U(\boldsymbol\alpha)^{\mathsf T},\qquad (\Psi_{\text{out}})_{k,:} = \texttt{run}(\Psi_{k,:},\boldsymbol\alpha).\texttt{state},$$
row by row. Because `run_batch` is implemented by calling `run` on each row, the two paths execute identical floating-point operations, so the expected discrepancy is exactly zero (bitwise), not merely small. For $N=0$ the output must be a well-formed empty array of shape $(0,2^n)$.

### What the code does
- System under test: `MuTA(n, 2).run_batch(states, parameters)` (default `one_column=False`, hence $2n$ paper layers).
- Reference (class E, self-consistency): the same model's scalar `MuTA.run` applied to each row. This is deliberately not an independent oracle; R1-R5 cover the physics.
- Sweep: $n\in\{1,2,3\}$ and $N\in\{0,1,5,17\}$, 12 cases in total. Parameters come from `model.initialize(seed=100n+N, scale=1.0)`, and inputs are Haar-random `haar_states(n, N, seed=100n+N+7)`, or `np.empty((0, 2**n))` for $N=0$.
- Metrics: Frobenius error between the batched and stacked scalar outputs (defined as 0 for $N=0$), and a shape check against $(N,2^n)$.
- Figure `R6_batch_consistency`.

### Expected outcome
The declared acceptance is `max_error == 0.0` exactly (bitwise) and a correct output shape in all 12 cases, including the three empty batches, which must return without raising. Any nonzero error or wrong shape fails the experiment. No floating-point tolerance is declared, because identical code paths must give identical bits. Figure: a single scatter panel with one marker per case, labelled `n<wires>b<batch>`, plotting the Frobenius error. All markers should sit at 0 and be coloured as passing.

<!-- ml:R7 -->
**Title:** Semantic topology agreement with MentPy MuTA templates

**Why this experiment.** PhotoGraphiQML reimplements the MuTA ansatz rather than wrapping MentPy, the reference MBQC-QML library in which MuTA was first expressed (`mentpy.templates.muta`, here `mentpy==0.1.0a15`). Claiming that both libraries describe the same model requires more than graph isomorphism. Two graphs can be isomorphic while assigning a given vertex a different role (wire, column, input or output), which would make trained angles non-transferable and numerical comparisons meaningless. This experiment establishes, for the full configuration sweep, that the two resource graphs are edge-for-edge identical in a shared semantic coordinate system. Every later MentPy cross-check (R8-R13, R20, R21) relies on this result.

### Theory
A MuTA resource graph is an open graph $(G, I, O)$ with flow $f$. PhotoGraphiQML labels nodes semantically as $(w,c)$, with wire $w$ and global column $c\in\{0,\dots,4d\}$, and its flow is $f(w,c)=(w,c+1)$. MentPy uses opaque integer labels. The semantic map $\mu$ from MentPy labels to $(w,c)$ is built without guessing an isomorphism. It starts at MentPy's $w$-th input node and follows MentPy's own flow $4d$ times:
$$\mu\big(f_{\text{MP}}^{\,c}(i_w)\big) = (w,c),\qquad c=0,\dots,4d,$$
and requires that the walk terminate at MentPy's $w$-th output. The relevant invariant is then labelled equality
$$\Delta = \big|\,\{\{\mu(u),\mu(v)\} : (u,v)\in E_{\text{MP}}\}\ \triangle\ E_{\text{MuTA}}\,\big| = 0,$$
together with $\mu(I_{\text{MP}}) = I_{\text{MuTA}}$ and $\mu(O_{\text{MP}}) = O_{\text{MuTA}}$ as ordered tuples. This is strictly stronger than $G_{\text{MP}}\cong G_{\text{MuTA}}$.

### What the code does
- System under test: `MuTA(n, L, one_column, restrict_trainable).graph`, `input_nodes`, `output_nodes`.
- Oracle (class B, independent external implementation): `mentpy.templates.muta` built through `photographiqml.validation.mentpy_reference`, which returns the MentPy graph state and the flow-walk map $\mu$. MentPy is separate, independently written code. It validates only the ideal logical qubit layer, never the GKP layer.
- Sweep: $n\in\{1,2,3\}$, $L\in\{1,2\}$, `one_column` and `restrict_trainable` in {False, True}, giving 24 configurations. Custom `connections` are excluded because MentPy has no counterpart. The sweep is deterministic.
- Metrics: node and edge counts of both graphs, the size of the edge symmetric difference, and input/output ordering agreement.
- Figure `R7_semantic_topology`.
- Requires MentPy. Without it the script stops with `ModuleNotFoundError`, which counts as an unavailable reference, not a failure.

### Expected outcome
The declared acceptance is `symmetric_difference == 0` and matching input and output orderings for every one of the 24 configurations. This is an integer comparison with no tolerance, and any discrepancy raises an `AssertionError`. Figure: a single scatter panel with one marker per configuration (labels `n<wires>L<layers>{1c|nc}{r|f}`) showing $|\Delta|$. All markers should sit at 0 and be coloured as passing.

<!-- ml:R8 -->
**Title:** Causal flow, correction sets and dependency order against MentPy

**Why this experiment.** R7 shows that PhotoGraphiQML and MentPy build the same labelled graph. A graph alone does not define an MBQC computation, though: the flow, the Pauli correction sets and the resulting causal order determine which measurement angles must be adapted to which earlier outcomes. They also fix the classical feed-forward schedule that the physical lowering later turns into Pauli frames (R37). If PhotoGraphiQML's correction map differed from MentPy's flow, the deterministic logical simulator would still run, but adaptive execution on hardware would be wrong. This experiment compares the corrections node by node and checks causal consistency, deliberately without demanding identical tie-breaking between causally independent measurements.

### Theory
A causal flow $(f,\prec)$ on an open graph $(G,I,O)$ assigns each measured node $v\in O^c$ a successor $f(v)\in I^c$ with $v\sim f(v)$, $v\prec f(v)$, and $v\prec u$ for all $u\in N(f(v))\setminus\{v\}$. An outcome $s_v=1$ is corrected by the Pauli frame
$$C_v = X_{f(v)}\prod_{u\in N(f(v))} Z_u .$$
For MuTA, $f(w,c)=(w,c+1)$. MentPy's `Flow.correction_op(v)` returns the full frame over all graph qubits. Because $v\in N(f(v))$, it contains a formal $Z_v$ on the node being measured, which has no physical effect after the destructive measurement. PhotoGraphiQML stores
$$\texttt{corrections}[v] = \big(f(v),\ N(f(v))\setminus\{v\}\big).$$
The comparison therefore removes $v$ from MentPy's mapped $Z$-set and records each removal in `self_correction_dropped`. A flow admits many total orders, so the invariant tested is causal validity: MentPy's own `measurement_order` must place $v$ before $f(v)$ and before every $u\in N(f(v))\setminus\{v\}$. These are exactly the edges of PhotoGraphiQML's `dependency_graph`.

### What the code does
- System under test: `MuTA.corrections` and `MuTA.dependency_graph`.
- Oracle (class B): MentPy's `reference.flow(node)`, `reference.flow.correction_op(node)` (whose binary symplectic matrix is decoded into $X$ and $Z$ supports) and `reference.measurement_order`, all mapped to semantic $(w,c)$ labels through `validation.mentpy_reference`.
- Sweep: $n\in\{1,2,3\}$, $L\in\{1,2\}$, `one_column` in {False, True}, giving 12 configurations (full, unrestricted models). Deterministic.
- Metrics per configuration: `order_violations` (dependency edges violated by MentPy's order), `correction_mismatches` (nodes whose $X$-support is not exactly $\{f(v)\}$ or whose $Z$-support differs from `z_targets` after removing the self term), and `self_correction_dropped`. The convention difference is stored as a `convention_note` in the JSON, not hidden.
- Figure `R8_flow_dependency_order` (two panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
The declared acceptance is `order_violations == 0` and `correction_mismatches == 0` for every configuration. These are exact discrete comparisons with no tolerance. `self_correction_dropped` is expected to equal the number of measured nodes, since every node is a neighbour of its own successor. This reflects the documented MentPy convention, not a defect. Figure: (left) causal-order violations and (right) X/Z correction-set mismatches, one marker per configuration (labels `n<wires>L<layers>{1c|nc}`). All markers should be at 0.

<!-- ml:R9 -->
**Title:** Density-matrix agreement with MentPy pattern simulation

**Why this experiment.** This is the headline numerical cross-validation of the logical layer (Fig. 3). R7 and R8 show that PhotoGraphiQML and MentPy agree structurally (graph, flow, corrections), but structural agreement does not guarantee that the two libraries assign the same quantum state to the same angles. Measurement-basis sign conventions, angle conventions, qubit ordering and output permutations can all differ while the structure matches. This experiment compares the output density matrices of `MuTA.run` and MentPy's `PatternSimulator` on random inputs and random angles over several model sizes.

### Theory
Both libraries simulate the same open-graph pattern: prepare $|\psi_{\text{in}}\rangle\otimes|+\rangle^{\otimes(|V|-n)}$, entangle with $\prod_{E} CZ$, then measure each node in the XY plane at angle $\alpha_v$. PhotoGraphiQML replaces every measurement-plus-correction by its deterministic logical action $J(\alpha)=H R_z(\alpha)$ (frontier translation). MentPy's `numpy-sv` backend simulates the pattern itself on a sliding window of qubits, in the zero-outcome branch, which by flow determinism requires no correction (R4, R5). The two outputs must coincide:
$$\rho_{\text{PGQ}} = U(\boldsymbol\alpha)\,|\psi_{\text{in}}\rangle\langle\psi_{\text{in}}|\,U(\boldsymbol\alpha)^\dagger = \rho_{\text{MentPy}} .$$
The metric is the maximum entrywise deviation
$$\epsilon = \max_{jk}\big|(\rho_{\text{MentPy}}-\rho_{\text{PGQ}})_{jk}\big| .$$
Using density matrices removes global phases. The MentPy schedule is PhotoGraphiQML's `measurement_order` followed by the outputs, mapped back to MentPy labels. The window size is the largest schedule distance spanned by any edge, and at least $n+1$, so every required neighbour is resident when a node is measured. Model-frozen columns are fixed explicitly in the reference.

### What the code does
- System under test: `MuTA.run(state, parameters).density_matrix`.
- Oracle (class B): `photographiqml.validation.compare_mentpy`, which builds `mentpy.templates.muta` via `mentpy_reference(..., fix_measurements=True)`, passes the angles in MentPy's `trainable_nodes` order through the semantic map, and runs `mp.PatternSimulator(backend="numpy-sv", schedule=..., window_size=...)` with `output_form="dm"`. The MentPy commit is recorded as `mentpy_commit`.
- Configurations: (n, L, `one_column`) = (1,1,T), (1,3,T), (2,1,T), (2,1,F), (2,2,T), (3,1,T).
- 8 seeds per configuration (48 cases): input `haar_states(n,1,seed=1000*seed+n)`, angles `model.initialize(seed=2000*seed+L, scale=2.0)`.
- Metric: maximum absolute density-matrix entry difference.
- Figure `R9_state_density_agreement`.
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
All 48 cases should agree to near machine precision. The tolerance is `declare_tolerance(scale=1.0, safety_factor=1000.0)` $=1000\,\varepsilon_{64}\approx 2.22\times10^{-13}$. The larger safety factor allows for MentPy's independent linear-algebra pipeline, its different operation ordering and its windowed simulation. Acceptance requires the maximum error over the seed-by-configuration grid to be below this tolerance. Only deterministic (zero-branch) execution is validated here, not adaptive branch sampling. Figure: a single log-scale strip plot with the 8 seed errors per configuration (x labels `n<wires>L<layers>{1c|nc}`) and the dashed tolerance line. All points should lie well below it.

<!-- ml:R10 -->
**Title:** Trainability audit of frozen column-3 measurements

**Why this experiment.** The restricted MuTA (`restrict_trainable=True`) fixes the column-3 measurement of every paper layer and trains only $3nd$ of the $4nd$ angles (R1). The restricted model is a different hypothesis class, so any claim about its expressivity, gradient landscape or parameter count requires that the frozen angles actually stay fixed. The MentPy audit (`docs/ml/research/mentpy-audit.md`) documents that MentPy's `restrict_trainable` only removes column-3 nodes from a `trainable_nodes` list and never fixes their `Ment` angles. That list is also rebuilt whenever `add_edge` or `hstack` re-runs the circuit constructor, which happens for any stacked or multi-wire template. This experiment records MentPy's behaviour as an upstream finding, not a pass/fail criterion, and checks that PhotoGraphiQML's own freezing never leaks, in every configuration.

### Theory
Let $\mathcal{P}=\{\alpha_{w,c}\}$ be the measured-node angles and $\mathcal{F}=\{\alpha_{w,c}: c\equiv 3 \pmod 4\}$ the frozen set of the restricted model. A correct restriction defines the parameter map
$$\boldsymbol\theta\in\mathbb{R}^{|\mathcal P\setminus\mathcal F|}\ \longmapsto\ \boldsymbol\alpha(\boldsymbol\theta),\qquad \alpha_v = \begin{cases}\theta_v & v\notin\mathcal F\\ \bar\alpha_v\ (\text{fixed}) & v\in\mathcal F,\end{cases}$$
so that the model $U(\boldsymbol\alpha(\boldsymbol\theta))$ cannot move along frozen directions. In MentPy a measurement is trainable exactly when its `Ment.angle is None`. A restriction that only edits a list, without assigning angles, is therefore lost as soon as the attribute list is rebuilt from the `Ment` objects. PhotoGraphiQML stores the frozen names in a `ParameterStore`. Its `bind` raises `ValueError` whenever a frozen name is rebound to a value different from its stored value (default $0$), and `trainable_parameters()` excludes frozen names. Two invariants are checked: (i) no column-3 name appears among the trainable parameters; (ii) every attempted override of a column-3 angle is rejected.

### What the code does
- System under test (class E): `MuTA(n, L, one_column, restrict_trainable=True)` with its `ParameterStore`. For each column-3 name, `model.run(|0...0>, {name: 1.2345})` must raise `ValueError`, and the name must be absent from `trainable_parameters()`.
- External audit (class B): the unmodified MentPy template from `mentpy_reference(model, fix_measurements=False)`. For each column-3 node, mapped semantically, the script records whether it is absent from `reference.trainable_nodes` (list restricted) and whether `reference[node].angle is not None` (angle actually fixed).
- Configurations (n, L, `one_column`): (1,1,T), (1,2,T), (2,1,T), (2,1,F), (2,2,T), (3,1,T), (1,1,F), (3,2,T). There are 8 in total, deterministic.
- Metrics: `mentpy_list_restricted`, `mentpy_angle_actually_fixed`, `mentpy_restriction_effective` (all column-3 angles fixed), `photographiqml_leaked_into_trainable`, `photographiqml_override_rejected`, and `photographiqml_robust`.
- Figure `R10_trainability_audit` (two panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
Acceptance applies to PhotoGraphiQML only: `photographiqml_robust` must be True in all 8 configurations, meaning zero leaked column-3 names and 100% of overrides rejected. This is an exact discrete check. The MentPy columns are descriptive. Following the audit document, column-3 `Ment` angles are expected to remain unfixed, and the list restriction to survive only for the unstacked one-wire, one-layer block. The number of configurations in which MentPy's restriction is fully effective is saved as `n_configs_mentpy_restriction_actually_effective`. Figure: (left) the fraction of column-3 MentPy measurements whose angle is actually fixed, per configuration, coloured grey where the restriction is lost; (right) bars giving the fraction of frozen overrides PhotoGraphiQML correctly rejects, expected to be 1 everywhere.

<!-- ml:R11 -->
**Title:** Eq. 5 kernel feature states and Gram matrix against MentPy

**Why this experiment.** The MuTA quantum kernel of paper Eq. 5 is one of PhotoGraphiQML's learning primitives (Fig. 4; used in R25-R28). Its value depends on an exact assignment of data features to specific measurement angles of a two-wire triangle, including the nonlinear coupling angle $\cos x_0\cos x_1$. If a feature went to the wrong node, or the coupling angle were mis-wired, the kernel would remain a valid PSD fidelity kernel, so a Gram-matrix sanity check would still pass, but it would no longer be the kernel defined in the paper. This experiment rebuilds the same circuit with the same angles in MentPy and compares the feature density matrices and the resulting Gram matrix.

### Theory
The Eq. 5 feature map embeds $\mathbf{x}=(x_0,x_1)\in\mathbb{R}^2$ into a one-column, two-wire, one-layer MuTA acting on $|00\rangle$:
$$|\psi(\mathbf x)\rangle = U\big(\boldsymbol\alpha(\mathbf x)\big)|00\rangle,\qquad \alpha_{0,0}=x_0,\ \alpha_{1,0}=x_1,\ \alpha_{1,1}=\cos x_0\cos x_1,\ \alpha_{0,2}=x_0,\ \alpha_{1,2}=x_1,$$
with all other angles at $0$. Here $\alpha_{w,c}$ is the angle of node $(w,c)$, and $\alpha_{1,1}$ drives the pivot cross-coupling, the Ising-XX entangler of R3. The kernel is the state fidelity
$$k(\mathbf x,\mathbf y) = |\langle\psi(\mathbf x)|\psi(\mathbf y)\rangle|^2 = \mathrm{Tr}\,\big[\rho(\mathbf x)\rho(\mathbf y)\big],\qquad \rho(\mathbf x)=|\psi(\mathbf x)\rangle\langle\psi(\mathbf x)|,$$
and the Gram matrix is $K_{ij}=k(\mathbf x_i,\mathbf x_j)$. PhotoGraphiQML computes $K$ from state overlaps, $|\Psi^{*}\Psi^{\mathsf T}|^{2}$ elementwise. The reference computes $\mathrm{Tr}(\rho_i\rho_j)$ from MentPy density matrices. For pure states the two expressions are equal, so agreement of the feature states implies agreement of the kernel independently of global phases.

### What the code does
- System under test: `MuTAKernel()` (`kernel.model` is `MuTA(2,1,one_column=True)`). Per-sample states come from `kernel.model.run([1,0,0,0], angles)` with the Eq. 5 angles; the Gram matrix comes from `kernel.gram_matrix(X)`.
- Oracle (class B): the helper `mentpy_density_matrix` in the script. It mirrors `validation.compare_mentpy`, which only returns a scalar error, and returns the MentPy `PatternSimulator(backend="numpy-sv")` density matrix for the semantically mapped template with the same angles and a flow-respecting schedule and window. The MentPy Gram matrix is assembled as $\mathrm{Re}\,\mathrm{Tr}(\rho_i\rho_j)$.
- Data: 8 points $\mathbf x_i$ uniform in $[-2\pi,2\pi]^2$ from `common.rng(0)`.
- Metrics: per-sample Frobenius density-matrix error, and the Frobenius norm of the Gram difference. Both Gram matrices are stored in the JSON.
- Figure `R11_kernel_gram_agreement` (three panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
States and Gram matrix should agree to near machine precision. The tolerance is `declare_tolerance(scale=1.0, safety_factor=1000.0)` $=1000\,\varepsilon_{64}\approx 2.22\times10^{-13}$, and acceptance requires both the maximum per-sample state error and the Gram error to be below it. The Gram matrix itself should be symmetric with a unit diagonal and entries in $[0,1]$. Figure: (left) semilog per-sample density-matrix error with the dashed tolerance line; (centre) heat map of the PhotoGraphiQML Gram matrix on a $[0,1]$ colour scale; (right) heat map of $|K_{\text{PGQ}}-K_{\text{MentPy}}|$ with the Gram error in the title.

<!-- ml:R12 -->
**Title:** Quantum-instrument branches and conditional states against MentPy

**Why this experiment.** `QuantumInstrumentModel` turns a logical MuTA into a quantum instrument: after the circuit, one wire is measured destructively in the computational basis, and the model returns each outcome's probability together with the normalised post-measurement state of the remaining wires. Quantum-output models of this kind are needed for teleportation-like and state-preparation tasks. Two failure modes would not be visible in a simple state comparison: a measured-wire index or qubit-ordering error, which permutes branches, and an incorrect normalisation of the conditional states. This experiment computes both quantities independently from a MentPy-executed density matrix and checks trace preservation on both sides.

### Theory
Let $\rho = U|\psi_{\text{in}}\rangle\langle\psi_{\text{in}}|U^\dagger$ be the $n$-wire output, and $\Pi^b_w = |b\rangle\langle b|_w\otimes I$ the computational-basis projector on wire $w$. The instrument $\{\mathcal I_b\}_{b=0,1}$ gives
$$P(b) = \mathrm{Tr}\big(\Pi_w^b\rho\big),\qquad \rho_{b} = \frac{\mathrm{Tr}_w\big(\Pi^b_w\rho\,\Pi^b_w\big)}{P(b)} = \frac{\langle b|_w\,\rho\,|b\rangle_w}{P(b)},\qquad \sum_b P(b)=1 .$$
On the reference side, $\langle b|_w\rho|b\rangle_w$ is read off by fixing both the ket and bra index of axis $w$ to $b$ in the reshaped $2^{\otimes 2n}$ density tensor. Because those indices are fixed rather than summed, this equals the partial trace of the projected operator. PhotoGraphiQML instead slices the pure output vector, $|\phi_b\rangle = \langle b|_w|\psi_{\text{out}}\rangle$, so that $P(b)=\langle\phi_b|\phi_b\rangle$ and $\rho_b = |\phi_b\rangle\langle\phi_b|/P(b)$. Branches with $P(b)\le 10^{-15}$ carry no conditional state, following the model's convention.

### What the code does
- System under test: `QuantumInstrumentModel(MuTA(n,1,one_column=True), measured_wire=0).run(state, parameters)`, which returns `InstrumentBranch(outcome, probability, state)`.
- Oracle: the MentPy output density matrix from the script helper `mentpy_density_matrix` (`PatternSimulator`, `numpy-sv`, `output_form="dm"`; class B, independent external simulation), followed by a fresh tensor projection `project_wire` (class E, textbook quantum mechanics applied to that independent state).
- Sweep: $n\in\{1,2,3\}$, 4 seeds each (12 cases). Inputs are `haar_states(n,1,seed=10n+seed)` and angles `model.initialize(seed=20n+seed, scale=2.0)`. Wire 0 is measured.
- Metrics: maximum $|P_{\text{PGQ}}(b)-P_{\text{MP}}(b)|$, maximum conditional-state Frobenius error (only for branches with $P_{\text{MP}}>10^{-15}$), and $|\sum_b P(b)-1|$ for both computations.
- Figure `R12_instrument_agreement` (two panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
All quantities should agree to near machine precision. The tolerance is `declare_tolerance(scale=1.0, safety_factor=1000.0)` $=1000\,\varepsilon_{64}\approx 2.22\times10^{-13}$, and acceptance requires the maximum probability error, the maximum conditional-state error and the maximum trace-preservation error (PhotoGraphiQML and MentPy-derived) all to be below it. For $n=1$, the conditional state is a trivial one-dimensional block. This validates only the ideal logical instrument, since no physical quantum-output instrument exists. Figure: (left) semilog per-case maximum probability error (circles) and conditional-state error (squares) with the dashed tolerance; (right) trace-preservation errors $|\sum P-1|$ for PhotoGraphiQML and for the MentPy-derived projection.

<!-- ml:R13 -->
**Title:** Matched-workload runtime of PhotoGraphiQML and MentPy

**Why this experiment.** Once R9-R12 have shown that PhotoGraphiQML and MentPy compute the same logical states, a reader will ask what the reimplementation costs or saves in practice. This experiment reports a local, descriptive timing of one full logical MuTA evaluation in both libraries on the same semantic circuit with the same angles, together with the structural resource counts that also feed the aggregate performance figure (R_PERF). It makes no correctness claim and no general speed claim. Its role is to document observed costs honestly, so that runtime statements in the manuscript are backed by recorded evidence rather than anecdote.

### Theory
There is no correctness equation. The two algorithms differ structurally. PhotoGraphiQML's frontier translation keeps only the $n$ logical wires resident and replaces each measured node by a $2\times2$ gate $J(\alpha)=H R_z(\alpha)$ (plus diagonal $CZ$ sign flips for junction edges). One evaluation therefore costs roughly $O\big((|V|-n)\,2^{n}\big)$ for a graph with $|V|$ vertices. MentPy's `PatternSimulator` simulates the measurement pattern itself on a sliding window of $W\ge n+1$ physical qubits, so each step acts on a $2^{W}$-dimensional vector. For the one-column, one-layer MuTA used here, the resource counts are
$$|V| = 5n,\qquad |E| = 4n + 2(n-1),\qquad N_{\text{param}} = 4n .$$
Timings are summarised by the median and the interquartile range (IQR) of repeated wall-clock measurements, after warm-up runs that exclude one-off costs (imports, caches).

### What the code does
- Workloads: `MuTA(n,1,one_column=True).run(state, angles)` versus a closure that builds a fresh `mp.PatternSimulator(reference, backend="numpy-sv", schedule, window_size)` and calls `run(values, output_form="dm")`. MentPy simulators are single-use, so construction is part of each MentPy evaluation. The MentPy circuit comes from `validation.mentpy_reference(..., fix_measurements=True)` with the same flow-respecting schedule and window as R9.
- Oracle class N/A: MentPy is a comparison workload, not ground truth (correctness is covered by R7-R12).
- Sweep: $n\in\{1,2,3\}$. The input is `haar_states(n,1,seed=n)` and the angles are uniform in $[-\pi,\pi]$ from `common.rng(n)`.
- Timing: `common.benchmark(fn, warmup=5, repeats=11)` with `time.perf_counter`, recording median, IQR, min and max seconds.
- Also recorded: graph vertices, edges and trainable parameters.
- Figure `R13_runtime_scaling` (two panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
There is no acceptance threshold. The JSON records `acceptance_condition = "N/A (descriptive performance measurement)"`, and the status is set to pass once the script completes. Timings are single-host, single-process observations (`machine_note`) and must not be read as complexity-class or general performance claims. Neither library was tuned for this workload, and the MentPy closure includes simulator construction and density-matrix output, whereas PhotoGraphiQML returns a state vector. Figure: (left) median runtime against $n$ on a log axis for both libraries, with error bars of $\pm$IQR/2; (right) graph vertices, edges and parameters against $n$, which grow linearly as $5n$, $6n-2$ and $4n$.

<!-- ml:R14 -->
**Title:** Parameter ordering, binding and freezing contract

**Why this experiment.** Every optimiser, gradient routine and serialised model in PhotoGraphiQML maps a flat real vector to named measurement angles through `ParameterStore`. If the ordering of that vector drifted from the documented paper-block, wire, column convention, or if a frozen angle could be overwritten quietly, then trained parameters would be applied to the wrong nodes, published angle tables would not be reproducible, and restricted models (R10) would not be restricted. The failure would be silent, because any permutation of angles still yields a valid unitary. This experiment checks the public parameter contract item by item.

### Theory
There is no physics equation. The contract defines the parameter map formally. For measured nodes $v=(w,c)$ in paper block $b=\lfloor c/4\rfloor$ with local column $c \bmod 4$, the canonical order is lexicographic in $(b, w, c \bmod 4)$, and node $v$ carries the name `alpha.w{w}.c{c}`. Let $\mathcal F$ be the frozen set with stored values $\bar\alpha_v$ (default $0$). Binding is the map
$$\mathrm{bind}(\boldsymbol\theta)_v = \begin{cases}\theta_{\iota(v)} & v\notin\mathcal F\\ \bar\alpha_v & v\in\mathcal F\end{cases},\qquad \boldsymbol\theta\in\mathbb R^{|\mathcal P\setminus\mathcal F|},$$
where $\iota$ enumerates the trainable names in canonical order. A vector of the wrong length is an error. A dict binding updates only the given names, and unknown names are an error. Supplying a value $\ne\bar\alpha_v$ for a frozen $v$ is an error, not a silent override. Initialisation draws $\theta\sim\mathcal N(0,s^2)$ from `numpy.random.default_rng(seed)`, so equal seeds give identical vectors and distinct seeds give different vectors with probability 1.

### What the code does
- System under test (class E, self-consistency with the documented API contract): `MuTA(2, 2, one_column=True)` and its `ParameterStore` (`parameters`, `trainable_parameters`, `_parameters.bind`, `freeze`, `unfreeze`, `initialize`, `n_parameters`, `run`).
- Thirteen deterministic sub-checks:
  - `measured_nodes` sorted by (block, wire, local column), and `parameters()` order matching them;
  - partial dict binding changes only the named entry (others stay 0.0);
  - vector binding `arange(1, N+1)` follows the `trainable_parameters()` order, and a vector of length $N+1$ raises `ValueError`;
  - `freeze(name, 0.0)` removes the name from the trainable set and reduces `n_parameters` by one;
  - `run` with an override of 1.0 for the frozen name raises `ValueError`;
  - `unfreeze` restores both the trainable set and the count;
  - the unknown name `alpha.w9.c9` is rejected;
  - `initialize(seed=42, scale=0.5)` is reproducible and differs from `seed=43`.
- Metric: the number of failed sub-checks.
- Figure `R14_parameter_contracts`.

### Expected outcome
The declared acceptance is 0 failed sub-checks out of 13, an exact discrete criterion with no tolerance. Any failure raises an `AssertionError` that names the failed checks. This validates only the interface. Physical and numerical correctness are covered elsewhere (R1-R12). Figure: a horizontal checklist with one unit bar per named sub-check, coloured as passing or failing, and the number of failures in the title. All bars are expected to be coloured as passing.

<!-- ml:R15 -->
**Title:** Logical and physical model persistence round trips

**Why this experiment.** A trained model is only scientifically useful if it can be saved, shared and reloaded into exactly the same hypothesis: the same topology, the same stored angles, the same frozen set and the same physical resource configuration. A lossy serialiser, such as one that drops frozen flags, rounds floats or reinterprets a physical model as a logical one, would make published models irreproducible without raising any error. A loader that accepts unknown schema versions could likewise misread future or corrupted files. This experiment tests the documented schema-1 (logical) and schema-2 (GKP-physical) round trips and the rejection of unsupported schemas.

### Theory
There is no physics equation. Serialisation is required to be the identity on the model state $\mathcal S = (\text{config}, \{\bar\alpha_v\}, \mathcal F)$, together with the GKP resource configuration (cutoff, peak width, envelope, peaks, grid points, decoder, backend) for physical models:
$$\mathrm{load}\circ\mathrm{save}(\mathcal S) = \mathcal S,\qquad U_{\mathrm{load}(\mathrm{save}(\mathcal S))} = U_{\mathcal S}.$$
Values are written with `json.dumps`, whose float representation round-trips binary64 exactly, and the reloaded model executes the same code, so the execution difference is expected to be exactly zero rather than merely small. Schema handling must be closed: only versions $\{1,2\}$ are accepted. A schema-2 file with representation `gkp-physical` must be reconstructed as a `PhysicalMuTA`, and anything else must raise `ValueError` rather than be guessed. No legacy schema exists, so "migration" here is limited to that rejection.

### What the code does
- System under test (class E, the model against its own round-tripped copy): `MuTA.save` / `MuTA.load` / `state_dict` and `PhysicalMuTA.save` with reload through `MuTA.load`.
- Schema 1, configurations (n, L, `one_column`, `restrict_trainable`) = (2,1,T,F), (1,2,T,T), (3,1,F,F). Half of the parameter names (at least one) are frozen at values uniform in $[-2,2]$ from `common.rng(7)`, which gives nontrivial persisted values through the public API only. Each configuration is saved, reloaded and run with stored values on $|0\cdots0\rangle$.
- Schema 2: `PhysicalMuTA(1)` with `GKPPhysicalConfig(cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513)`. Saved and reloaded, it is compared by `physical_config` equality, recorded as an execution error of 0, or NaN on mismatch. No physical simulation is run.
- Rejection: a schema-1 payload with `schema = 99` must make `MuTA.load` raise `ValueError`.
- Metrics: output-state Frobenius error (logical), and equality of `values`, `frozen` and `config` in `state_dict`.
- Files go to `results/raw/R15_roundtrip/`. Figure `R15_persistence_roundtrip`.

### Expected outcome
Declared acceptance: `max_execution_error == 0` exactly, all `values`/`frozen`/`config` fields equal in every case, and `unsupported_schema_rejected == True`. Any violation raises an `AssertionError`. Figure: one bar per case (three logical configurations plus `physical_1wire`), height 1 if the round trip is exact, coloured as passing when the state fields match, with the schema-rejection flag in the title. All four bars are expected at height 1.

<!-- ml:R16 -->
**Title:** Fail-fast matrix for invalid inputs and unsupported capabilities

**Why this experiment.** PhotoGraphiQML deliberately exposes a narrow, validated surface. Logical inputs must be normalised statevectors (never silently renormalised). The physical GKP path supports only the signed-X measurement subset, where the XY angle must be $0$ or $\pi$ modulo $2\pi$ within the upstream absolute tolerance $10^{-14}$. Resource budgets are enforced before any Fock allocation. The manuscript's capability table (Table III) and its anti-claims depend on the library refusing everything outside this surface, rather than returning a plausible but wrong answer such as a renormalised state, a rounded angle or a truncated Hilbert space. This experiment enumerates representative invalid inputs across the logical, parameter, lowering and physical layers and checks that each fails fast with the documented exception type.

### Theory
There is no physics equation. Each case encodes a precondition of the formal model:
- Logical input: $|\psi\rangle\in\mathbb C^{2^n}$, finite, with $\big|\langle\psi|\psi\rangle-1\big|\le 10^{-10}$.
- Parameters: finite reals bound by name or as a vector of exactly $N_{\text{train}}$ entries.
- Physical lowering: each measured node must be an X-basis measurement with $\alpha\in\{0,\pi\}\ (\mathrm{mod}\ 2\pi)$, a GKP Pauli measurement realisable by homodyne detection with Pauli-frame tracking. A generic angle such as $\alpha=0.37$ would require a non-Clifford measurement synthesis that is not implemented, so it must raise `NotImplementedError` (`UnsupportedPhysicalMeasurement`).
- Physical execution modes (`physical-shots`, `physical-conditional`), shot counts and seeds: conditional execution needs one analog outcome per measured node and exactly one shot.
- Fock resources: the live-mode Hilbert dimension $\binom{\text{cutoff}+m-1}{m}$ for peak live modes $m$ must not exceed `max_dimension`.

`ValueError` signals a malformed request. `NotImplementedError` signals a well-formed request outside the validated capability (an unsupported angle or output basis, or dense unitaries and batches on the physical model).

### What the code does
- System under test (class E, the package's own documented contract): `MuTA(2,1,one_column=True)` and `PhysicalMuTA(1, physical_config=GKPPhysicalConfig(cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513))`, plus the constructors of `MuTA`, `PhysicalMuTA` and `GKPPhysicalConfig`.
- The helper `case(name, fn, expected)` records a pass only if `fn` raises the expected exception type. No exception, or an exception of another type, is recorded as a failure.
- 29 cases:
  - malformed states: non-normalised, wrong shape, NaN;
  - parameters: infinite value, unknown name, wrong-length vector;
  - construction: representation `"cv"`, `n_wires` 0 or `True`, custom connections without `one_column`;
  - execution: legacy `gkp-resource` execution, physical angle 0.37, output basis `"Y"`, `shots=0`, negative seed, mode `"logical-exact"`;
  - conditional mode: empty outcomes, conditional mode with `shots=2`;
  - configuration: backend `"strawberryfields"`, decoder `"maximum-likelihood"`, `max_dimension=1`, measurement family `"Z"`;
  - physical model: continuous `initialize`, `run_batch`, `unitary`;
  - logical model: dense unitary for 11 wires, negative init scale, freezing an unknown name, freezing to $\infty$.
- Metric: number of cases in which the expected exception was not raised.
- Figure `R16_invalid_input_matrix`.

### Expected outcome
The declared acceptance is 0 failed cases out of 29, an exact criterion. Every call should raise before any expensive computation, so the run is light. The set is representative, not a proof of total input coverage. Figure: a vertical checklist of horizontal unit bars, one per named case in the order listed, coloured as passing or failing, with the failure count in the title. All bars are expected to be coloured as passing.

<!-- ml:R17 -->
**Title:** Seeded reproducibility of logical initialisation and physical shots

**Why this experiment.** Every multi-seed study in this suite (R19-R24, R44, R48, R57) relies on the guarantee that a declared seed fixes the random stream completely. That covers both the logical parameter initialisation and the physical GKP shot sampling, which draws analog homodyne outcomes and final readout bits. Without the guarantee, re-running a script could produce different "evidence", bootstrap intervals could not be reproduced, and seed-level comparisons would mix seed effects with nondeterminism. A seed that is ignored is the opposite failure, and would make different seeds look falsely identical. This experiment checks exact same-seed reproducibility and different-seed distinctness on both layers.

### Theory
There is no physics equation, but the statistical framing matters. Logical initialisation draws $\theta_k\sim\mathcal N(0,s^2)$ from `default_rng(seed)`, so equal seeds must give bitwise-equal vectors. A physical-shots run with $N$ shots draws, per shot, analog measurement records for every node, then a final bit string sampled from the joint conditional output POVM. It returns two estimators of the decoded joint distribution: the Rao-Blackwell mean of the conditional POVM probabilities over shots, $\hat P_{\text{RB}}(\mathbf b)=\frac1N\sum_i P(\mathbf b\mid\text{records}_i)$, and the empirical frequency of sampled bits, $\hat P_{\text{emp}}(\mathbf b)=\frac{1}{N}\#\{i:\mathbf b_i=\mathbf b\}$. With equal seeds both must be identical. For two independent seeds the empirical distributions differ by sampling noise, measured by the total-variation distance
$$\mathrm{TV}(\hat P_a,\hat P_c) = \tfrac12\sum_{\mathbf b}\big|\hat P_a(\mathbf b)-\hat P_c(\mathbf b)\big| ,$$
reported with the rough per-cell binomial standard error
$$\mathrm{SE}_{\mathrm{TV}} \approx \tfrac12\Big(\textstyle\sum_{\mathbf b}\big[\hat P_a(1-\hat P_a)+\hat P_c(1-\hat P_c)\big]/N\Big)^{1/2}.$$
A small TV value consistent with this error is an expected outcome, not a failure.

### What the code does
- System under test (class E, a run compared with a second run under the same declared seed): `MuTA(2,1,one_column=True).initialize(seed, scale=0.7)` and `PhysicalMuTA(1, physical_config=GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)).run([1,0], mode="physical-shots", shots=32, seed=...)`. Default angles are 0, so the model lies in the supported signed-X subset. The docstring explains that `cutoff=40` is needed so the finite-GKP resource passes the retained-norm guard at multiple shots.
- Logical: seeds $\{0,1,7,123\}$, each initialised twice and compared by maximum absolute difference. Distinctness compares seed 0 with seed 1.
- Physical: two runs with seed 2026 give the number of differing `sampled_output_bits` and the maximum difference in `decoded_joint_probabilities`. A third run with seed 4052 gives $\mathrm{TV}$ and $\mathrm{SE}_{\mathrm{TV}}$ between `empirical_probabilities`.
- Figure `R17_seeded_reproducibility` (two panels).

### Expected outcome
Declared acceptance: the same-seed maximum difference is exactly 0 in every logical and physical case, and the seed-0 and seed-1 logical initialisations differ. The physical different-seed TV distance and its standard error are recorded for information and are not part of the pass/fail criterion. Figure: (left) bars of the logical same-seed initialisation difference for the four seeds, all expected at 0; (right) two bars for the physical model with 32 shots, the same-seed bit-plus-probability difference (expected 0) and the different-seed TV distance with its $\pm$SE error bar.

<!-- ml:R18 -->
**Title:** Finite-difference gradients against the analytic parameter-shift rule

**Why this experiment.** By default, PhotoGraphiQML's `Trainer` computes gradients by central finite differences (`training.finite_difference`). Every gradient-based training result (R19-R24) therefore depends on those gradients being correct, with an error that behaves as numerical analysis predicts. A sign error, a factor-of-two error or a wrong parameter-to-node assignment in the gradient would still let optimisers move, possibly even converge, but towards the wrong place or more slowly, and the error would be hard to detect from loss curves alone. This experiment compares finite-difference gradients with the exact gradient from the parameter-shift rule, an independent closed-form theorem, over a sweep of step sizes.

### Theory
Each MuTA angle $\alpha_k$ enters the circuit exactly once, through $J(\alpha_k)=H R_z(\alpha_k)$ with $R_z(\alpha)=e^{i\alpha Z/2}=e^{i\alpha G}$, $G=Z/2$, whose eigenvalues are $\pm\frac12$. For any Hermitian observable $O$ and $f(\boldsymbol\alpha)=\langle\psi_{\text{out}}(\boldsymbol\alpha)|O|\psi_{\text{out}}(\boldsymbol\alpha)\rangle$, the dependence on a single $\alpha_k$ is a trigonometric polynomial of degree one, $f = a + b\cos\alpha_k + c\sin\alpha_k$. The two-term parameter-shift rule is therefore exact:
$$\frac{\partial f}{\partial \alpha_k} = \frac{1}{2}\Big[f\big(\boldsymbol\alpha+\tfrac{\pi}{2}\mathbf e_k\big) - f\big(\boldsymbol\alpha-\tfrac{\pi}{2}\mathbf e_k\big)\Big].$$
The central finite difference
$$D_h f = \frac{f(\boldsymbol\alpha+h\mathbf e_k)-f(\boldsymbol\alpha-h\mathbf e_k)}{2h} = \partial_k f + \frac{h^2}{6}\partial_k^3 f + O(h^4) + O\!\left(\frac{\varepsilon\,|f|}{h}\right)$$
has truncation error $\propto h^2$ and floating-point cancellation error $\propto\varepsilon/h$. The total error as a function of $h$ is U-shaped, with its optimum near $h\sim\varepsilon^{1/3}$ and a floor of order $\varepsilon^{2/3}\approx 3.6\times10^{-11}$, multiplied by the scale of $f$ and its derivatives.

### What the code does
- System under test: `photographiqml.training.finite_difference(objective, x, step)` applied to `objective(x) = MuTA(2,1,one_column=True).run(|00>, x).expectation(O)`, which has 8 parameters.
- Oracle (class A): the parameter-shift gradient, evaluated by fresh code in the script with two extra `run` calls per component. No package differentiation code is reused.
- Observable: $O=A+A^\dagger$, where $A$ is a $4\times4$ complex Gaussian matrix from `common.rng(0)`.
- Sweep: 5 random parameter points uniform in $[-2\pi,2\pi]^8$, and 15 step sizes `geomspace(1e-1, 1e-8, 15)`.
- Metric: $\lVert D_h f - \nabla_{\text{PS}} f\rVert_2$ for each (point, step), and the minimum over steps for each point.
- Figure `R18_parameter_shift_vs_fd`.

### Expected outcome
For each point the error should first fall as $h^2$ and then rise as $1/h$ at small steps. The tolerance is `declare_tolerance(scale=1.0, safety_factor=1e8)` $=10^{8}\,\varepsilon_{64}\approx 2.22\times10^{-8}$, set well above the $\varepsilon^{2/3}$ floor rather than near machine precision. Acceptance requires the minimum error over the step grid to be below this tolerance for every one of the 5 points; the worst such minimum is reported. Figure: a single log-log panel with one curve per point (error against step size), showing the U shape, and the dashed tolerance line. The minimum of every curve should lie below it.

<!-- ml:R19 -->
**Title:** Optimizer convergence on analytic test objectives

**Why this experiment.** All gradient-based results in the paper (gate learning, classifier, regressor) are produced by `photographiqml.training.Trainer`, which implements Adam and plain SGD itself and wraps SciPy's L-BFGS-B. If the optimiser loop had a bias-correction bug, a sign error, an off-by-one in the history, or mis-recorded losses or gradient norms, a failed or slow MuTA training run could be wrongly blamed on the ansatz (barren plateaus, poor expressivity), or a spurious success could be reported. This experiment isolates the optimiser from quantum circuits by running it on two classical objectives with known minima, so that convergence behaviour can be attributed unambiguously.

### Theory
The two objectives are the convex quadratic bowl
$$f_{\text{bowl}}(\mathbf x) = \lVert\mathbf x-\mathbf x^\star\rVert_2^2,\qquad \nabla f = 2(\mathbf x-\mathbf x^\star),\qquad \min f = 0 \text{ at } \mathbf x^\star,$$
and the non-convex Rosenbrock function in $d=4$ dimensions,
$$f_{\text{R}}(\mathbf x) = \sum_{i=1}^{d-1}\Big[100\,(x_{i+1}-x_i^2)^2 + (1-x_i)^2\Big],\qquad \min f_{\text{R}} = 0 \text{ at } \mathbf x=(1,\dots,1),$$
whose narrow curved valley is a standard stress test for first-order methods. SGD updates $\mathbf x\leftarrow\mathbf x-\eta\nabla f$; on the bowl this contracts the error by the factor $|1-2\eta|$ per step. Adam uses moment estimates $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ and $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ with $\beta_1=0.9$ and $\beta_2=0.999$, and the bias-corrected step $\Delta\mathbf x=-\eta\,\hat m_t/(\sqrt{\hat v_t}+10^{-8})$. L-BFGS-B is a quasi-Newton method capped at `epochs` iterations, for which the learning rate is unused. Exact analytic gradients are supplied, so the test isolates the optimiser from finite-difference error (covered by R18).

### What the code does
- System under test: `Trainer(optimizer, epochs, learning_rate).fit(objective, x0, gradient=...)` and its `History` (losses, gradient norms; the initial point is included).
- Oracle (class A): the closed-form minima and minimisers above, unrelated to MuTA.
- Setup (`common.rng(0)`): $\mathbf x^\star$ uniform in $[-2,2]^4$, bowl start uniform in $[-3,3]^4$, Rosenbrock start $(-1.5,1.5,-1.0,1.0)$, for which $f_{\text{R}}=1123$.
- Configurations (optimizer, epochs, $\eta$, threshold):
  - bowl: Adam (300, 0.1, $10^{-6}$), SGD (300, 0.1, $10^{-6}$), L-BFGS (100, 0.1, $10^{-6}$);
  - Rosenbrock: Adam (3000, 0.01, 5.0), SGD (3000, 0.001, 5.0), L-BFGS (200, 0.01, 0.5).
- Metrics: final loss (last `History.losses` entry), distance to the analytic minimiser, and final gradient norm.
- Figure `R19_optimizer_convergence` (two panels).

### Expected outcome
Declared acceptance: `final_loss < threshold` for every one of the 6 pairs. On the bowl, all three optimisers should reach a loss below $10^{-6}$ (for SGD the error contracts by $0.8$ per step). On Rosenbrock, L-BFGS is expected to approach the minimum (threshold 0.5). The fixed-step first-order methods are held only to substantial descent from the initial loss (threshold 5.0), because full convergence along the valley within the budget is not required. Figure: semilog loss against iteration for (left) the quadratic bowl and (right) Rosenbrock, with one curve per optimiser (adam, sgd, lbfgs).

<!-- ml:R20 -->
**Title:** Haar single-wire gate learning over 20 declared seeds

**Why this experiment.** The paper claims that MuTA models can be trained end to end to implement target quantum operations, and gate learning is the canonical demonstration (Fig. 6). A single successful run proves little: it may be a lucky seed, an overfit training set, or an implementation that drifts from the reference semantics as parameters move away from initialisation. This experiment learns a fresh Haar-random single-qubit gate for each of 20 seeds declared in advance, evaluates on held-out states, reports the median with a bootstrap confidence interval and all individual trajectories, and checks against an independent MentPy execution at training checkpoints that PhotoGraphiQML's simulator still agrees with the reference along the optimisation path.

### Theory
The target is $V = U_{\text{Haar}}\otimes I_2$ with $U_{\text{Haar}}\sim\mathrm{Haar}(U(2))$, acting on wire 0 of a two-wire, one-column, one-layer MuTA with 8 trainable angles. By the Table-I identity (R2), the four wire-0 angles alone, with every other angle at zero, implement $R_x(c_3)R_z(c_2)R_x(c_1)R_z(c_0)\otimes I$. This Euler form covers $U(2)$ up to a global phase, so the target lies inside the model class. Given Haar-random inputs $\{|\psi_i\rangle\}$, the training loss is the mean state infidelity
$$\mathcal L(\boldsymbol\theta) = 1 - \frac{1}{N}\sum_{i=1}^{N}\big|\langle \psi_i|U(\boldsymbol\theta)^\dagger V|\psi_i\rangle\big|^2 ,$$
which is insensitive to global phase. Training uses Adam with central finite-difference gradients (step $10^{-6}$), validated in R18. Generalisation is measured by the same quantity on held-out states. Across seeds the summary statistic is the median final test infidelity, with a percentile bootstrap 95% CI from 2000 resamples of the 20 per-seed values (`common.bootstrap_ci(..., statistic=np.median)`). The checkpoint oracle compares output density matrices, $\max_{jk}|\rho^{\text{MentPy}}-\rho^{\text{PGQ}}|_{jk}$, for the current angles (as in R9).

### What the code does
- System under test: `MuTA(2, one_column=True)`, `MuTA.unitary`, `Trainer(optimizer="adam", epochs=120, learning_rate=0.05).fit(...)` (default finite-difference gradient), `models.haar_states` and `models.infidelity`.
- Oracles: the target from `scipy.stats.unitary_group.rvs(2, random_state=default_rng(seed))` (class A), and `validation.compare_mentpy(model, states[-1], values)` evaluated at epochs 0, 40, 80 and 120 through the training callback (class B).
- Seeds 0-19. For each seed: 12 Haar states `haar_states(2, 12, seed+1000)`, of which the first 8 are for training and the last 4 held out; initial angles `model.initialize(seed, 0.3)`.
- Metrics per seed: final train and test infidelity, wall time and maximum MentPy checkpoint error. Aggregates: median test infidelity with its bootstrap CI, and the maximum checkpoint error over all seeds.
- Figure `R20_haar_gate_learning` (two panels).
- Requires MentPy (absent: `ModuleNotFoundError`, counted as unavailable).

### Expected outcome
Declared acceptance: median final test infidelity $<0.05$, and maximum MentPy checkpoint error below `declare_tolerance(scale=1.0, safety_factor=1000.0)` $=1000\,\varepsilon_{64}\approx 2.22\times10^{-13}$. The CI is reported descriptively and is not used as an equivalence test. Individual seeds may converge slowly. They are kept, never filtered. Figure: (left) all 20 held-out test-infidelity trajectories against Adam step on a log scale (faint), with the median trajectory in bold; (right) a histogram of final test infidelities with the median marked and the 95% bootstrap CI shaded.

<!-- ml:R21 -->
**Title:** Learning the Ising-XX entangling gate across twenty seeds

**Why this experiment.** R20 shows that a MuTA circuit can learn a generic single-wire Haar target. The paper's gate-learning claim also needs an entangling target, since this is where the pivot-wire cross-coupling of the triangle ansatz does the work. R21 trains a two-wire, one-layer MuTA toward the fixed Ising-XX gate $e^{-i\pi X\otimes X/4}$ and checks generalization to held-out states, using 20 declared seeds rather than one hand-picked run. Because training updates the parameters away from any value checked earlier, the experiment also re-validates the trained model against MentPy at regular checkpoints. Without that check, a training-time optimum could sit on a parameter configuration where the PhotoGraphiQML contraction and the independent MBQC simulator disagree.

### Theory
The target is the two-qubit Ising-XX unitary

$$U_\star = \exp\!\left(-\tfrac{i\pi}{4}\,X\otimes X\right) = \tfrac{1}{\sqrt2}\left(\mathbb{1}-i\,X\otimes X\right),$$

which maps $|00\rangle$ to a maximally entangled state (concurrence 1). R3 established that the pivot coupling of MuTA(2, one_column=True) realizes $e^{i\phi X\otimes X/2}$ through the single angle alpha.w1.c1. The target therefore lies inside the model family (at $\phi=-\pi/2$ with the other angles at zero).

Given Haar-random input states $|\psi_k\rangle$ (complex Gaussian vectors, normalized) and model unitary $U(\boldsymbol\theta)$, the loss is the mean state infidelity

$$\mathcal L(\boldsymbol\theta) = 1-\frac1N\sum_{k=1}^{N}\big|\langle\psi_k|U_\star^\dagger U(\boldsymbol\theta)|\psi_k\rangle\big|^2 ,$$

which ignores the global phase. The training set uses $N=8$ states and the held-out set uses the other 4. The parameters are updated by Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) using central finite-difference gradients with step $10^{-6}$, which is the Trainer default when no gradient callable is supplied. At the checkpoints, the density matrix $\rho_{\rm PGQ}$ from `MuTA.run` is compared with MentPy's `PatternSimulator` output $\rho_{\rm MP}$ for the same angles via $\max_{ij}|\rho_{\rm MP}-\rho_{\rm PGQ}|_{ij}$.

### What the code does
- **System under test:** `MuTA(2, one_column=True)` with 8 trainable angles, `MuTA.unitary`, `Trainer(optimizer="adam", epochs=120, learning_rate=0.05)`, `models.haar_states` and `models.infidelity`.
- **Oracles:** class A for the target, which is built with `scipy.linalg.expm` of a locally formed $X\otimes X$, independent of the ansatz code. Class B for the checkpoints: `validation.compare_mentpy` executes the trained parameters in MentPy, an external MBQC simulator.
- **Protocol:** seeds 0 to 19. For each seed, 12 Haar states are drawn with seed `seed+1000` (8 for training, 4 held out), and the initial angles are drawn from $\mathcal N(0,0.3^2)$ with the same seed. A MentPy comparison on the last held-out state runs whenever the history length minus one is a multiple of 40 (steps 0, 40, 80 and 120).
- **Metrics:** final train and test infidelity, the median final test infidelity with a 2000-resample percentile bootstrap 95% CI, and the worst MentPy checkpoint error over all seeds.
- **Figure:** `R21_ising_gate_learning`.

### Expected outcome
The target is exactly representable, so the held-out infidelity should fall by orders of magnitude for most seeds. The MentPy checkpoint error should stay at floating-point roundoff. The run passes when

$$\operatorname{median}_{\text{seeds}}\mathcal L_{\rm test}^{\rm final}<0.05 \quad\text{and}\quad \max\text{ checkpoint error}<1000\,\varepsilon_{\rm mach}\approx 2.22\times10^{-13},$$

where the tolerance comes from `declare_tolerance(scale=1, safety_factor=1000)` and is fixed before training.

- **Left panel:** held-out test-infidelity trajectories for all 20 seeds (faint lines) and their pointwise median (bold), on a logarithmic axis against Adam step.
- **Right panel:** histogram of final test infidelities, with a vertical line at the median and a shaded band for its 95% bootstrap CI.

<!-- ml:R22 -->
**Title:** Sensitivity of gate learning to data size, depth and initialization

**Why this experiment.** R20 and R21 report the median over many seeds at one training configuration, which hides how the optimization behaves at other settings. R22 maps how held-out gate-learning performance depends on three practitioner-controlled knobs: training-set size, circuit depth (paper layers) and initialization scale. The headline quantity is the failure rate, not only a central tendency. This places the main-text gate-learning results inside an honest operating envelope and shows where the ansatz under-fits (too few states, too shallow a circuit) or starts from a poor region (large initial angles). The study is descriptive and asserts no global performance bound.

### Theory
For a single-wire Haar target $V\sim\mu_{\rm Haar}(U(2))$, the two-wire target is $U_\star=V\otimes\mathbb{1}_2$, drawn with `scipy.stats.unitary_group`. Training minimizes the state infidelity on $n_{\rm train}$ Haar inputs,

$$\mathcal L(\boldsymbol\theta)=1-\frac{1}{n_{\rm train}}\sum_{k}\big|\langle\psi_k|U_\star^\dagger U_L(\boldsymbol\theta)|\psi_k\rangle\big|^2 ,$$

where $U_L$ is the $L$-layer, two-wire MuTA unitary with $8L$ trainable measurement angles. The angles are initialized as $\theta_j\sim\mathcal N(0,s^2)$, with $s$ the initialization scale. For a fixed target, the expected held-out infidelity is a monotone function of the average gate fidelity, so the held-out loss estimates how close $U_L$ is to $U_\star$ (up to phase) over the whole Hilbert space. With very few training states the minimizer is under-determined, and a low training loss need not imply a low test loss. For each grid cell, the failure rate over $S$ seeds is

$$f=\frac1S\sum_{s=1}^{S}\mathbb 1\!\left[\mathcal L_{\rm test}^{(s)}>0.2\right],$$

measured on 6 held-out Haar states.

### What the code does
- **System under test:** `MuTA(2, n_layers, one_column=True)`, `MuTA.unitary`, and `Trainer(optimizer="adam", epochs=60, learning_rate=0.05)` with the default central finite-difference gradient.
- **Oracle:** class A. The Haar target from `scipy.stats.unitary_group` is analytic and independent of the ansatz.
- **Grid:** training size $\in\{2,4,8\}$, $L\in\{1,2\}$ and $s\in\{0.1,0.5,1.5\}$, which gives 18 cells with seeds 0 to 5 in each. The seed fixes the target, the initial angles and (as `seed+1000`) the state draw `haar_states(2, n_train+6, seed+1000)`. The first $n_{\rm train}$ states train the model and the last 6 are held out.
- **Metrics per cell:** median final test infidelity, failure rate at threshold 0.2, and the raw list of 6 final test infidelities.
- **Figure:** `R22_gate_learning_sensitivity`.

### Expected outcome
The failure rate should generally fall with more training states and with the deeper circuit. Large initialization scales may raise it, but with only 6 seeds per cell every rate is a multiple of $1/6$ and carries wide sampling uncertainty. No performance threshold is asserted. The only enforced acceptance condition is structural: every cell must have a finite median test infidelity and a failure rate in $[0,1]$.

- **Two heatmaps,** one for `n_layers=1` and one for `n_layers=2`. Rows are training-set size, columns are initialization scale, and the colour (magma scale, 0 to 1, shared colorbar) is the failure rate, with the numeric value printed in each cell.

<!-- ml:R23 -->
**Title:** Verification of the logical MuTA classifier on repeated stratified splits

**Why this experiment.** `MuTAClassifier` is the supervised-learning entry point of PhotoGraphiQML: a MuTA circuit, angle encoding of classical inputs, and a logistic head on the first-wire $Z$ expectation. R23 checks that this wrapper (encoding, readout, loss, optimizer coupling and the predict/score path) learns a decision rule whose correct answer is known in closed form. Accuracy is estimated on several stratified held-out splits, so a single favourable split cannot pass the test. The experiment verifies the wrapper's correctness and makes no claim about expressivity or quantum advantage. A two-feature boundary was tried first and did not converge within the epoch budget; this is recorded in ISSUES_FOUND.md.

### Theory
A scalar feature $x\in[0,\pi]$ is encoded as

$$|\psi(x)\rangle=R_y(x)|0\rangle=\cos\tfrac{x}{2}|0\rangle+\sin\tfrac{x}{2}|1\rangle ,\qquad \langle Z\rangle_{\psi(x)}=\cos x .$$

The circuit applies the trainable single-wire MuTA unitary $U(\boldsymbol\theta)$. One paper layer realizes the Table-I Euler sequence $R_x(c_3)R_z(c_2)R_x(c_1)R_z(c_0)$, so `MuTA(1,1,one_column=True)` has four angles. The model then reads

$$z(x;\boldsymbol\theta)=\langle\psi(x)|U^\dagger Z_1 U|\psi(x)\rangle,\qquad f(x)=a\,z+b,\qquad p(y=1\mid x)=\sigma(f) .$$

Training minimizes the binary cross-entropy written in logit form,

$$\mathcal L=\frac1N\sum_i\Big[\log\!\big(1+e^{f_i}\big)-y_i f_i\Big],$$

over $(\boldsymbol\theta,a,b)$. The labels are $y=\mathbb 1[x<\pi/2]$, i.e. $y=1$ exactly when $\cos x>0$. At $U=\mathbb 1$, $a>0$ and $b=0$ the rule is therefore representable exactly, and it is linearly separable in $\cos x$. The prediction is $\arg\max$ over $(1-p,\,p)$.

### What the code does
- **System under test:** `MuTAClassifier(MuTA(1,1,one_column=True), trainer=Trainer("adam", epochs=80, learning_rate=0.1), seed=split)`, with `fit` and `predict`. Initialization: circuit angles $\sim\mathcal N(0,0.1^2)$, head $(a,b)=(1,0)$. Gradients are central finite differences.
- **Oracle:** class A. The labels come from a closed-form rule. A majority-class predictor (most frequent training label, scored on the test set) gives the sanity floor.
- **Data and protocol:** 60 points $x\sim U(0,\pi)$ from seed 0. Five splits use `train_test_split(test_size=0.3, stratify=y, random_state=split)` (42 train, 18 test), and split $k$ also seeds the training run.
- **Metrics:** held-out accuracy per split, its median with a bootstrap 95% CI, the paired median difference from the majority baseline (bootstrap over splits), the confusion matrix aggregated over splits, and the final training loss.
- **Figure:** `R23_classifier_verification`.

### Expected outcome
Held-out accuracy should be close to 1 on every split. The majority baseline sits near the class balance, about 0.5. The run passes when

$$\operatorname{median}(\text{accuracy})\ge 0.85\quad\text{and}\quad \operatorname{median}(\text{accuracy}-\text{baseline})>0 ,$$

where the second quantity is the point estimate of the paired bootstrap object.

- **Left panel:** held-out accuracy per split (circles) against the majority baseline (squares, dashed), with a dotted line at the median accuracy.
- **Right panel:** the aggregated $2\times2$ confusion matrix (rows true label, columns predicted label), with counts annotated. Almost all counts should fall on the diagonal.

<!-- ml:R24 -->
**Title:** Verification and residual calibration of the logical MuTA regressor

**Why this experiment.** `MuTARegressor` gives PhotoGraphiQML's logical models a continuous output: an affine head on the first-wire $Z$ expectation of an angle-encoded MuTA circuit. R24 checks that the wrapper can fit a smooth target that the bare encoding cannot represent. It also checks that the fit generalizes across repeated held-out splits and leaves residuals without systematic bias. A wrapper that fit the training data but produced biased or trend-carrying held-out residuals would pass a naive accuracy check while still being miscalibrated. This experiment guards against that failure.

### Theory
The input $x\in[0,\pi]$ is encoded as $R_y(x)|0\rangle$, so without a trainable circuit $\langle Z\rangle=\cos x$. The target is $y=\sin x$. No affine map $a\cos x+b$ reproduces $\sin x$ on $[0,\pi]$, because $\cos$ is monotone there while $\sin$ is not, so the trainable rotations have to reshape the effective feature. A rotation about $y$ shifts the angle: $\langle Z\rangle\to\cos(x+\delta)$, and $\delta=-\pi/2$ gives $\sin x$, so the target is representable in principle. The model is

$$\hat y(x)=a\,\langle\psi(x)|U(\boldsymbol\theta)^\dagger Z_1U(\boldsymbol\theta)|\psi(x)\rangle+b ,$$

where $U$ is a single-wire, two-layer MuTA with 8 Table-I measurement angles. The training loss is the mean squared error $\frac1N\sum_i(\hat y_i-y_i)^2$. On a test split the metrics are

$$R^2=1-\frac{\sum_i(\hat y_i-y_i)^2}{\sum_i(y_i-\bar y)^2},\qquad r_i=\hat y_i-y_i ,$$

together with the pooled mean residual $\bar r$ and the Pearson correlation $\operatorname{corr}(r,\hat y)$. A non-zero $\bar r$ indicates bias, and a large $|\operatorname{corr}(r,\hat y)|$ indicates a systematic error trend along the output range.

### What the code does
- **System under test:** `MuTARegressor(MuTA(1,2,one_column=True), trainer=Trainer("adam", epochs=150, learning_rate=0.1), seed=split)`, with `fit` and `predict`. Initialization: circuit angles $\sim\mathcal N(0,0.1^2)$, head $(a,b)=(1,0)$. Gradients are central finite differences.
- **Oracle:** class A. The analytic target $\sin x$ is independent of the code under test.
- **Data and protocol:** 60 points $x\sim U(0,\pi)$ from seed 0. Five random splits use `train_test_split(test_size=0.3, random_state=split)`, and the split index also seeds training.
- **Metrics:** per-split $R^2$, MSE and mean residual. Pooled over splits: the median $R^2$ with a bootstrap 95% CI, $\bar r$, $\operatorname{corr}(r,\hat y)$ and the overall MSE.
- **Figure:** `R24_regressor_verification`.

### Expected outcome
The fit should reach $R^2$ near 1 on every split, with residuals small and centred on zero. The run passes when

$$\operatorname{median}R^2\ge 0.9,\qquad |\bar r|<0.1,$$

and, only when the pooled MSE is at least $10^{-4}$, $|\operatorname{corr}(r,\hat y)|<0.5$. Below that MSE floor the residuals are at optimizer-noise level, so a correlation computed on them carries no calibration information, and the code records that the gate was skipped (`calibration_check_gated_by_mse_floor`).

- **Left panel:** held-out $R^2$ per split, with a dashed acceptance line at 0.9.
- **Right panel:** pooled residuals plotted against predictions, with a zero line. The title reports $\bar r$ and the correlation. The points should show no visible trend.

<!-- ml:R25 -->
**Title:** Kernel feature states against an independent graph-state contraction

**Why this experiment.** The PhotoGraphiQML quantum kernel `MuTAKernel` implements the paper's Eq. 5 feature map as a fixed two-wire MuTA layer whose measurement angles are functions of the data. Every kernel-learning result (R26 to R28, R54) assumes that the feature states $|\psi(\mathbf x)\rangle$ are computed correctly. R11 already checks them against MentPy (class B). R25 adds a second cross-check from a different source that needs no external package: a contraction written from scratch inside the script. An error in the angle-to-column assignment or in the MBQC contraction would otherwise go unnoticed in every downstream Gram matrix.

### Theory
For $\mathbf x=(x_0,x_1)$, the Eq. 5 feature map binds the following angles of MuTA(2, 1, one_column=True), leaving all others at 0:

$$\alpha_{w0,c0}=x_0,\quad \alpha_{w1,c0}=x_1,\quad \alpha_{w1,c1}=\cos x_0\cos x_1,\quad \alpha_{w0,c2}=x_0,\quad \alpha_{w1,c2}=x_1 .$$

The feature state is the logical output of the measurement pattern on input $|00\rangle$, on the deterministic all-zero outcome branch:

$$|\psi(\mathbf x)\rangle \propto \Big(\textstyle\bigotimes_{v\in M}\langle\pm_{\alpha_v}|\Big)\prod_{(u,v)\in E}CZ_{uv}\;\big(|00\rangle\otimes|+\rangle^{\otimes(|V|-2)}\big),$$

where $\langle\pm_{\alpha}|$ with outcome 0 is the projector onto $(|0\rangle+e^{i\alpha}|1\rangle)/\sqrt2$, applied as the row vector $(1,e^{-i\alpha})/\sqrt2$. The measurements follow the flow order $M$, and the state is renormalized after each one. Because the state is pure, comparing density matrices $\rho=|\psi\rangle\langle\psi|$ removes the global phase, and the metric is

$$\varepsilon=\big\|\rho_{\rm PGQ}-\rho_{\rm ind}\big\|_F .$$

### What the code does
- **System under test:** the PhotoGraphiQML execution of the kernel circuit, `MuTAKernel().model.run([1,0,0,0], angles).state`, with the Eq. 5 angle dictionary written out in the script. This dictionary is identical to the one inside `MuTAKernel.features`.
- **Oracle:** class D. `independent_contraction` embeds the input, appends $|+\rangle$ ancillas, applies each graph CZ as a sign flip on the $|11\rangle$ tensor slice, and contracts the projective measurements with `np.tensordot` in `model.measurement_order`, then permutes the output wires into place. It does not use `photographiqml.logical.cz`, `local_gate` or `validation.contract_branch`. It does read the graph, measurement order and parameter names from the model object, so the inputs are shared while the arithmetic is independent.
- **Samples:** 10 points $\mathbf x\sim U(-2\pi,2\pi)^2$ from RNG seed 1.
- **Metric:** density-matrix Frobenius error per sample, and its maximum.
- **Figure:** `R25_kernel_feature_state`.

### Expected outcome
The two contractions compute the same finite sequence of linear-algebra operations in different orders, so the errors should sit at the $10^{-16}$ to $10^{-15}$ roundoff level. Acceptance:

$$\max_{\text{samples}}\varepsilon<200\,\varepsilon_{\rm mach}\approx4.44\times10^{-14},$$

with the tolerance taken from `declare_tolerance(scale=1, safety_factor=200)` before the comparison.

- **Single panel:** per-sample density-matrix error on a logarithmic axis, with a dashed horizontal line at the tolerance. All points should lie well below the line.

<!-- ml:R26 -->
**Title:** Symmetry, positivity and conditioning of the MuTA fidelity kernel

**Why this experiment.** A kernel method is well posed only if its Gram matrix is a valid positive-semidefinite (PSD) kernel matrix. For a fidelity kernel this follows from theory, so the numerical Gram matrix returned by `MuTAKernel.gram_matrix` should satisfy the same invariants to roundoff at every sample size. R26 verifies symmetry, unit diagonal and PSD-ness against these analytic invariants. It also records how conditioning grows with $N$, which matters for SVM solvers that operate on a precomputed kernel (R27, R28). A violation would point to a broken feature map or Gram assembly, and it would make downstream SVM results unreliable.

### Theory
With pure feature states $|\psi_i\rangle=|\psi(\mathbf x_i)\rangle\in\mathbb C^4$ (R25), the fidelity kernel is

$$K_{ij}=|\langle\psi_i|\psi_j\rangle|^2=\operatorname{Tr}(\rho_i\rho_j),\qquad \rho_i=|\psi_i\rangle\langle\psi_i| .$$

Three properties follow.

1. **Symmetry.** $K_{ij}=K_{ji}\in\mathbb R$.
2. **Unit diagonal.** $K_{ii}=\langle\psi_i|\psi_i\rangle^2=1$.
3. **PSD.** $K$ is the Gram matrix of the $\rho_i$ under the Hilbert-Schmidt inner product, so $\mathbf c^{\mathsf T}K\mathbf c=\|\sum_i c_i\rho_i\|_{\rm HS}^2\ge0$. Equivalently, $K=G\circ\overline G$ with $G_{ij}=\langle\psi_i|\psi_j\rangle$, which is PSD by the Schur product theorem.

The $\rho_i$ lie in the 16-dimensional real space of $4\times4$ Hermitian matrices, so $\operatorname{rank}K\le16$. For $N>16$ the matrix is necessarily singular, and its smallest eigenvalues sit at the roundoff floor, possibly slightly negative. The diagnostics are

$$e_{\rm sym}=\max_{ij}|K_{ij}-K_{ji}|,\qquad e_{\rm diag}=\max_i|K_{ii}-1|,\qquad \lambda_{\min}(K),\qquad \kappa=\frac{\lambda_{\max}}{\max(\lambda_{\min},\varepsilon_{\rm mach})} .$$

### What the code does
- **System under test:** `MuTAKernel.gram_matrix` and `MuTAKernel.diagnostics`, with the eigenvalues computed by `np.linalg.eigvalsh`.
- **Oracle:** class A. The invariants above are theorems about fidelity kernels and do not depend on the implementation.
- **Sweep:** $N\in\{2,4,8,16,32,64\}$, with fresh inputs $\mathbf x\sim U(-2\pi,2\pi)^2$ drawn for each $N$ from one RNG stream (seed 2).
- **Metrics:** $e_{\rm sym}$, $e_{\rm diag}$, $\lambda_{\min}$, $\lambda_{\max}$ and the clipped condition number $\kappa$, each recorded against $N$.
- **Figure:** `R26_kernel_properties`.

### Expected outcome
Symmetry error should be zero or at roundoff level, since the elementwise modulus of a Hermitian overlap matrix is symmetric. The diagonal error should be at roundoff level. $\lambda_{\min}$ may be marginally negative at roundoff level, especially once $N>16$. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=100)`:

$$\max_N e_{\rm sym}<\tau,\qquad \max_N e_{\rm diag}<\tau,\qquad \min_N\lambda_{\min}>-\tau,\qquad \tau=100\,\varepsilon_{\rm mach}\approx2.22\times10^{-14}.$$

The condition number is descriptive only. It is expected to grow with $N$ and to reach the $\lambda_{\max}/\varepsilon_{\rm mach}$ ceiling once the matrix becomes rank-deficient.

- **Left panel:** symmetry and unit-diagonal errors against $N$ on a log axis, with values floored at $10^{-18}$ for display and a dotted tolerance line.
- **Right panel:** condition number $\kappa(K)$ against $N$ on a log axis.

<!-- ml:R27 -->
**Title:** MuTA-kernel support vector classification against classical baselines

**Why this experiment.** R25 and R26 establish that the MuTA fidelity kernel is computed correctly and is a valid PSD kernel. They do not show that it is usable in a standard learning pipeline. R27 inserts `MuTAKernel` into scikit-learn's precomputed-kernel SVM and measures held-out accuracy on three standard two-dimensional benchmarks, under repeated stratified splits. Classical models trained on the identical raw features serve as reference points. The comparison is descriptive by design. Its role is to demonstrate an end-to-end kernel workflow with honest uncertainty, and it supports no claim of quantum advantage. The structural part guards against an invalid (non-PSD) Gram matrix silently reaching the SVM solver.

### Theory
Each input $\mathbf x\in\mathbb R^2$ is used directly as a pair of angles in radians, with no scaling, in the Eq. 5 feature map $\mathbf x\mapsto|\psi(\mathbf x)\rangle$ (R25). The kernel is

$$k(\mathbf x,\mathbf x')=|\langle\psi(\mathbf x)|\psi(\mathbf x')\rangle|^2 .$$

A soft-margin SVM with a precomputed training Gram matrix $K_{ij}=k(\mathbf x_i,\mathbf x_j)$ solves the dual problem

$$\max_{\boldsymbol a}\ \sum_i a_i-\tfrac12\sum_{ij}a_ia_jy_iy_jK_{ij}\quad\text{s.t.}\quad 0\le a_i\le C,\ \ \sum_ia_iy_i=0,$$

with labels $y_i\in\{\pm1\}$, and predicts $\operatorname{sign}\big(\sum_ia_iy_ik(\mathbf x,\mathbf x_i)+b\big)$ from the test-versus-train cross-kernel. The problem is convex only if $K$ is PSD, which is why $\lambda_{\min}(K)$ is checked. The baselines see the same raw coordinates: an RBF-SVM with $k_{\rm RBF}=e^{-\gamma\|\mathbf x-\mathbf x'\|^2}$ (scikit-learn default $\gamma$), and $\ell_2$-regularized logistic regression. For paired comparisons the code bootstraps the per-split difference $\Delta_s=\mathrm{acc}^{\rm MuTA}_s-\mathrm{acc}^{\rm base}_s$ and reports its median.

### What the code does
- **System under test:** `MuTAKernel.gram_matrix(train_x)` and `MuTAKernel(test_x, train_x)`, fed to `sklearn.svm.SVC(kernel="precomputed", C=1.0)`.
- **Reference:** N/A, since no correct accuracy exists. `SVC(C=1.0)` (RBF) and `LogisticRegression(max_iter=1000)` act as comparison points, not ground truth. A class-A structural check is folded in: the PSD-ness of every training Gram matrix.
- **Datasets:** `make_circles(factor=0.5, noise=0.08)`, `make_moons(noise=0.1)` and `make_blobs(centers=2, cluster_std=1.2)`, each with 120 samples. Seeds 0 to 7 each generate the data and a stratified 70/30 split (84 train, 36 test).
- **Metrics:** held-out accuracy for each (dataset, model, seed); the median with a bootstrap 95% CI per model; the paired MuTA-minus-RBF and MuTA-minus-logistic median differences with CIs; the MuTA confusion matrix aggregated per dataset; and the minimum eigenvalue over all training Gram matrices.
- **Figure:** `R27_kernel_classification`.

### Expected outcome
The relative performance is reported, not asserted. The raw-coordinate angle encoding is periodic and not adapted to each dataset's geometry, so the MuTA kernel may trail the RBF-SVM on some datasets. The linear logistic baseline is expected to struggle on circles. The structural status passes when every accuracy lies in $[0,1]$ and

$$\min\lambda_{\min}(K)>-\tau,\qquad \tau=10^5\,\varepsilon_{\rm mach}\approx2.22\times10^{-11},$$

with $\tau$ from `declare_tolerance(scale=1, safety_factor=1e5)`, which is looser than R26 because these Gram matrices are larger. The scientific outcome is recorded as "descriptive".

- **Three panels,** one each for circles, moons and blobs. Each shows bars of median held-out accuracy for MuTA, RBF-SVM and logistic regression, with asymmetric 95% bootstrap CI error bars, on a 0 to 1.05 accuracy axis.

<!-- ml:R28 -->
**Title:** Robustness of the MuTA-kernel SVM and leakage-safe model selection

**Why this experiment.** One configuration (R27) cannot show whether the kernel SVM's behaviour is stable or a property of a single setting. R28 varies dataset size, input noise, the SVM regularization constant $C$ and a global coordinate scale, one axis at a time, on the moons task. The coordinate scale is especially informative because MuTAKernel encodes raw coordinates directly as angles. Rescaling the inputs therefore changes the encoded quantum states and the kernel bandwidth; it is not a neutral preprocessing step. The experiment also shows a hyperparameter-selection protocol in which $C$ is chosen on validation data only, so the reported test accuracy is free of selection leakage, a common pitfall in published kernel benchmarks.

### Theory
For inputs scaled as $\mathbf x\to s\,\mathbf x$, the Eq. 5 kernel becomes

$$k_s(\mathbf x,\mathbf x')=|\langle\psi(s\mathbf x)|\psi(s\mathbf x')\rangle|^2 .$$

The feature map is $2\pi$-periodic in each angle and also enters through $\cos(sx_0)\cos(sx_1)$, so $s$ acts like an inverse bandwidth. Small $s$ makes all states similar ($K\to\mathbf 1\mathbf 1^{\mathsf T}$, under-fitting). Large $s$ makes nearby points nearly orthogonal and folds distant points onto each other (over-fitting and aliasing). $C$ trades margin width against training errors in the soft-margin dual $0\le a_i\le C$ (R27). The `noise` parameter of `make_moons` adds Gaussian jitter of that standard deviation to the coordinates, which blurs the class boundary.

Leakage-safe selection uses a three-way split $\mathcal D=\mathcal D_{\rm tr}\cup\mathcal D_{\rm val}\cup\mathcal D_{\rm te}$:

$$C^\star=\arg\max_{C\in\mathcal C}\ \mathrm{acc}\big(\mathrm{SVM}_C[\mathcal D_{\rm tr}];\ \mathcal D_{\rm val}\big),\qquad \text{report } \mathrm{acc}\big(\mathrm{SVM}_{C^\star}[\mathcal D_{\rm tr}];\ \mathcal D_{\rm te}\big),$$

so $\mathcal D_{\rm te}$ is never seen during selection.

### What the code does
- **System under test:** `MuTAKernel.gram_matrix` and the cross-kernel call, with `SVC(kernel="precomputed", C)`.
- **Reference:** N/A. This is a descriptive sensitivity study with no correctness oracle.
- **One-axis sweep:** from the baseline (120 samples, noise 0.1, $C=1$, scale 1), the code varies `n_samples` over $\{60,120,240\}$, `noise` over $\{0.05,0.15,0.3\}$, $C$ over $\{0.1,1,10\}$ and `coordinate_scale` over $\{0.5,1,2\}$. Each point uses seeds 0 to 3, which set both the data and a stratified 70/30 split, and the median test accuracy is reported.
- **Selection demo:** 6 seeds, each with 150 moons points at noise 0.15, split stratified 60/20/20 into train, validation and test. $C^\star\in\{0.1,1,10\}$ is chosen by validation accuracy (ties go to the first candidate) and refit on the training set. The code records $C^\star$, the validation scores and the test accuracy.
- **Figure:** `R28_kernel_robustness`.

### Expected outcome
Accuracy should improve with more data and degrade as input noise grows. It should vary with $C$ and, most strongly, with coordinate scale, reflecting the bandwidth role of $s$. None of these trends is gated. The structural acceptance condition is that every median sweep accuracy and every selection-demo test accuracy is finite and in $[0,1]$. Validation-only selection holds by construction of the splits. The scientific outcome is recorded as "descriptive".

- **Panels 1 to 4:** median test accuracy against `n_samples`, `noise`, `C` and `coordinate_scale` (accuracy axis 0 to 1.05).
- **Panel 5:** bars of test accuracy for each of the 6 selection-demo split seeds.

<!-- ml:R29 -->
**Title:** Trace preservation and branch completeness of the logical quantum instrument

**Why this experiment.** `QuantumInstrumentModel` turns a MuTA circuit into a quantum-output model: it measures one wire destructively and returns each classical outcome with its probability and post-measurement state. For any downstream use (expected losses, sampling, the R12 MentPy comparison), the returned branches must form a complete, trace-preserving instrument. This means both outcomes are always reported, the probabilities sum to one, and a branch of zero probability is flagged explicitly rather than returning a state normalized by roughly zero. R29 checks these structural invariants on random circuits and on an engineered case with a branch of exactly zero probability.

### Theory
Let $|\Psi\rangle=U(\boldsymbol\theta)|\psi_{\rm in}\rangle$ be the $n$-wire output state, and measure wire $w$ in the computational basis. Writing $|\Psi\rangle=\sum_{b\in\{0,1\}}|b\rangle_w\otimes|\tilde\phi_b\rangle$, the instrument is

$$\mathcal I_b(\rho)=\big(\langle b|_w\otimes\mathbb 1\big)\rho\big(|b\rangle_w\otimes\mathbb 1\big),\qquad p_b=\langle\tilde\phi_b|\tilde\phi_b\rangle,\qquad |\phi_b\rangle=\frac{|\tilde\phi_b\rangle}{\sqrt{p_b}} .$$

Completeness of the projectors, $\sum_b|b\rangle\langle b|=\mathbb 1$, gives trace preservation, $\sum_bp_b=\langle\Psi|\Psi\rangle=1$. The number of branches equals the local dimension, 2. By the package convention, $|\phi_b\rangle$ is returned as `None` exactly when $p_b<10^{-15}$. When all angles are zero, the single-wire MuTA acts as the identity, so the input $|0\rangle$ gives $p_0=1$ and $p_1=0$ exactly. The surviving branch is then the trivial state of the empty remaining register, the scalar 1.

### What the code does
- **System under test:** `QuantumInstrumentModel(MuTA(n, 1, one_column=True), measured_wire=0).run(state, parameters)`.
- **Oracle:** class E (structural self-consistency). Trace preservation, the branch count and the `None` convention are invariants of any valid computational-basis instrument, so no external reference is needed.
- **Random cases:** $n\in\{1,2,3\}$ wires with 6 seeds each (18 cases). Each case uses a Haar input `haar_states(n,1,seed=100n+s)` and angles from `initialize(seed=200n+s, scale=2.0)`.
- **Engineered case:** a one-wire model with all angles at zero and input $|0\rangle$. The code checks that branch 1 has $p<10^{-15}$ and state `None`, and that branch 0's state equals $[1.0]$ to within tolerance.
- **Metrics per case:** $|\sum_bp_b-1|$, the number of branches, and whether the `None` convention is consistent.
- **Figure:** `R29_instrument_diagnostics`.

### Expected outcome
All probability-sum errors should be at or near machine precision. The random cases are not expected to produce negligible branches, while the engineered case must produce exactly one. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=100)`:

$$\max_{\text{cases}}\Big|\sum_bp_b-1\Big|<100\,\varepsilon_{\rm mach}\approx2.22\times10^{-14},$$

together with exactly 2 branches in every case, and state `None` if and only if $p_b<10^{-15}$ (including the engineered-branch checks).

- **Single panel:** probability-sum error for each case index on a logarithmic axis (floored at $10^{-18}$ for display), with markers coloured by whether the `None` convention held and a dashed line at the tolerance.

<!-- ml:R30 -->
**Title:** Pure-state concurrence against the Wootters spin-flip formula

**Why this experiment.** PhotoGraphiQML reports two-qubit concurrence as its entanglement diagnostic. Examples are the induced entanglement of the pivot coupling (R3) and the expressivity summaries of trained models. The production routine uses the amplitude shortcut $2|ad-bc|$. R30 cross-checks it against an independently coded evaluation of Wootters' spin-flip definition on random states and on the two boundary cases. Without this check, an amplitude-ordering or conjugation error in the shortcut could systematically misreport entanglement in every experiment that cites concurrence.

### Theory
For a pure two-qubit state $|\psi\rangle=a|00\rangle+b|01\rangle+c|10\rangle+d|11\rangle$, Wootters' concurrence is

$$C(\psi)=\big|\langle\psi|\,\sigma_y\otimes\sigma_y\,|\psi^*\rangle\big| ,$$

where $|\psi^*\rangle$ is the complex conjugate in the computational basis. Since

$$\sigma_y\otimes\sigma_y=\begin{pmatrix}0&0&0&-1\\0&0&1&0\\0&1&0&0\\-1&0&0&0\end{pmatrix},$$

expanding gives $\langle\psi|\sigma_y\otimes\sigma_y|\psi^*\rangle=-2(a^*d^*-b^*c^*)$, and therefore

$$C(\psi)=2|ad-bc| .$$

$C=0$ exactly for product states, where the coefficient matrix $\begin{pmatrix}a&b\\c&d\end{pmatrix}$ has rank one, and $C=1$ for maximally entangled states such as $|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$. The two expressions are identical in exact arithmetic. The experiment evaluates them by two different numerical routes: a direct amplitude combination, and an explicit $4\times4$ matrix contraction $\psi^\dagger(\sigma_y\otimes\sigma_y)\psi^*$.

### What the code does
- **System under test:** `photographiqml.diagnostics.concurrence`, which evaluates $2|ad-bc|$ after validating the statevector.
- **Oracle:** class A. `spin_flip_concurrence` builds $\sigma_y\otimes\sigma_y$ locally with `np.kron` and contracts it with the conjugated amplitudes, without reusing the determinant shortcut.
- **Inputs:** 40 Haar-random two-qubit states (`haar_states(2, 40, seed=5)`), plus $|00\rangle$ and $|\Phi^+\rangle$.
- **Metrics:** the absolute difference between the two routes for every state, and the production value at the separable and Bell cases.
- **Figure:** `R30_concurrence`.

### Expected outcome
The two routes should agree to roughly $10^{-16}$. The separable value should be exactly 0, and the Bell value should be 1 to within roundoff. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=100)`:

$$\max|C_{\rm prod}-C_{\rm SF}|<\tau,\qquad C_{\rm prod}(|00\rangle)<\tau,\qquad |C_{\rm prod}(\Phi^+)-1|<\tau,\qquad \tau=100\,\varepsilon_{\rm mach}\approx2.22\times10^{-14}.$$

- **Left panel:** the per-sample disagreement for the 40 random states on a log axis, with a dashed line at the tolerance.
- **Right panel:** bar heights of the production concurrence for the separable and Bell states, with reference lines at 0 and 1.

<!-- ml:R31 -->
**Title:** Pure-state quantum Fisher information against the variance trace formula

**Why this experiment.** The quantum Fisher information (QFI) measures how sensitively a state responds to a generator. PhotoGraphiQML uses it as an expressivity and sensitivity diagnostic (`diagnostics.pure_qfi`). The production code evaluates it via a state-and-image inner-product shortcut. R31 confirms that this shortcut equals the textbook variance formula evaluated by an independent density-matrix route, across dimensions and for generic (non-Pauli, non-commuting) Hermitian generators. A sign or conjugation slip in the shortcut would silently bias every QFI-based statement.

### Theory
For a pure state $|\psi\rangle$ evolving as $|\psi(t)\rangle=e^{-iHt}|\psi\rangle$, the QFI with respect to $t$ is

$$F_Q[\psi,H]=4\big(\langle\psi|H^2|\psi\rangle-\langle\psi|H|\psi\rangle^2\big)=4\,\mathrm{Var}_\psi(H).$$

The production route sets $|\eta\rangle=H|\psi\rangle$ and computes

$$F_Q=4\big(\langle\eta|\eta\rangle-(\mathrm{Re}\langle\psi|\eta\rangle)^2\big),$$

which is correct for Hermitian $H$ because $\langle\psi|H|\psi\rangle$ is real. The reference route forms $\rho=|\psi\rangle\langle\psi|$ and evaluates

$$F_Q^{\rm ref}=4\Big(\mathrm{Re}\,\mathrm{Tr}(\rho H^2)-\big(\mathrm{Re}\,\mathrm{Tr}(\rho H)\big)^2\Big).$$

The two are algebraically equal but numerically distinct: matrix-vector products and inner products in one case, dense matrix products and traces in the other. $F_Q$ is bounded by $(\lambda_{\max}-\lambda_{\min})^2$ of $H$ and carries an overall factor of 4, which motivates the scale factor in the tolerance.

### What the code does
- **System under test:** `photographiqml.diagnostics.pure_qfi(state, generator)`, which also checks that the generator is Hermitian and the state is normalized.
- **Oracle:** class A. `trace_qfi` is a fresh density-matrix trace implementation of $4\,\mathrm{Var}(H)$.
- **Inputs:** for $n\in\{1,2,3\}$ qubits, 10 Haar states (`haar_states(n, 10, seed=n)`), each paired with a random Hermitian generator $H=M+M^\dagger$, where $M$ has i.i.d. complex Gaussian entries (seed $100n+i$). This gives 30 cases.
- **Metric:** absolute difference $|F_Q-F_Q^{\rm ref}|$ per case, and its maximum.
- **Figure:** `R31_qfi`.

### Expected outcome
The two routes should agree to roundoff, with a disagreement that may grow mildly with dimension and with the norm of $H$. Acceptance, with the tolerance from `declare_tolerance(scale=4, safety_factor=100)`:

$$\max_{\text{cases}}|F_Q-F_Q^{\rm ref}|<400\,\varepsilon_{\rm mach}\approx8.88\times10^{-14}.$$

This is an absolute tolerance, and the random generators are not normalized, so the margin is tightest for the largest-norm generators.

- **Single panel:** per-sample disagreement on a log axis, with one line per qubit number (1, 2, 3) and a dashed line at the tolerance.

<!-- ml:R32 -->
**Title:** Exact Pauli Lie-closure dimension against a matrix-commutator closure

**Why this experiment.** The dynamical Lie algebra (DLA) generated by a circuit's Pauli generators determines which unitaries the circuit can reach at arbitrary depth. Its dimension also enters barren-plateau and trainability arguments. PhotoGraphiQML computes this dimension with an exact binary-symplectic algorithm (`expressivity.pauli_lie_dimension`) that avoids numerical rank tolerances. R32 checks that algorithm against a closure written from scratch that works with explicit matrices, on generator sets whose closures are known in closed form, from commuting sets up to the full $\mathfrak{su}(4)$. The table of DLA dimensions in the manuscript (Table IV) depends on this routine being correct.

### Theory
For Pauli strings $P_1,\dots,P_k$ on $n$ qubits, the DLA is

$$\mathfrak g=\mathrm{Lie}\langle iP_1,\dots,iP_k\rangle=\mathrm{span}_{\mathbb R}\big\{iP_j,\ [iP_j,iP_l],\ [iP_j,[iP_l,iP_m]],\ \dots\big\}.$$

Two Pauli strings either commute or anticommute. If they anticommute, $[P,Q]=2PQ$, which is proportional to a third Pauli string. The closure is therefore spanned by Pauli strings, and $\dim\mathfrak g$ equals the number of distinct strings reached. Writing each string as a bit pair $(\mathbf x,\mathbf z)\in\mathbb F_2^n\times\mathbb F_2^n$ (with $X=(1,0)$, $Z=(0,1)$ and $Y=(1,1)$), the strings $P$ and $Q$ anticommute iff the symplectic form

$$\langle(\mathbf x,\mathbf z),(\mathbf y,\mathbf w)\rangle=\mathbf x\cdot\mathbf w+\mathbf z\cdot\mathbf y \pmod 2$$

equals 1, and their product has bits $(\mathbf x\oplus\mathbf y,\ \mathbf z\oplus\mathbf w)$. The closed-form reference values are:

| Generators | $\dim\mathfrak g$ |
|---|---|
| a single string, or pairwise-commuting strings | number of strings |
| $\{X,Z\}$ | 3 ($\mathfrak{su}(2)$) |
| $\{XI,ZI,IX,IZ\}$ | 6 ($\mathfrak{su}(2)\oplus\mathfrak{su}(2)$) |
| $\{XI,ZI,IX,IZ,XX\}$ | 15 ($\mathfrak{su}(4)$; the entangling generator is required) |

### What the code does
- **System under test:** `pauli_lie_dimension(generators)`, the exact GF(2) closure that repeatedly adds products of anticommuting pairs.
- **Oracle:** class A. `matrix_lie_dimension` builds explicit $2^n\times2^n$ Pauli matrices with `np.kron`, forms the Hermitian commutators $i[A,B]$, and admits a commutator only if it raises the real rank (tolerance $10^{-9}$) of the stacked, real-flattened basis. It iterates until no commutator adds rank and returns the final rank. No bit encoding is shared with the production code.
- **Cases:** `X`; `ZI,IZ`; `X,Z`; `XI,IX`; `XI,ZI,IX,IZ`; `XI,ZI,IX,IZ,XX`; `XII`.
- **Metric:** $|\dim_{\rm prod}-\dim_{\rm ind}|$ per case.
- **Figure:** `R32_lie_closure`.

### Expected outcome
Both methods should return the analytic dimensions: 1, 2, 3, 2, 6, 15 and 1 for the seven cases in order. The acceptance condition is exact integer agreement in every case, $\max|\dim_{\rm prod}-\dim_{\rm ind}|=0$. These dimensions describe reachability at arbitrary depth, not the expressivity of a finite-depth circuit.

- **Single panel:** paired bars of the production and independent closure dimensions for each named case. The two bars in each pair should have identical heights.

<!-- ml:R33 -->
**Title:** Local quantum Fisher spectra: positivity, step convergence and rank

**Why this experiment.** The local quantum Fisher information matrix (QFIM) of the parametrized MuTA output state shows how many independent directions in state space the parameters move at a given point. Its rank and conditioning are standard local expressivity and redundancy diagnostics. PhotoGraphiQML builds it from central finite-difference state derivatives (`expressivity.state_fisher`). R33 checks the two properties any correct construction must have: positive semidefiniteness, and convergence as the step size shrinks. It then reports rank and conditioning against wire count and depth. Without the convergence check, a rank read off a QFIM dominated by truncation error would be meaningless.

### Theory
For $|\psi(\boldsymbol\theta)\rangle=U(\boldsymbol\theta)|0\cdots0\rangle$, the pure-state QFIM is

$$F_{ij}=4\,\mathrm{Re}\Big(\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\Big)=4\,\mathrm{Re}\,\langle\partial_i\psi|(\mathbb 1-|\psi\rangle\langle\psi|)|\partial_j\psi\rangle ,$$

which is invariant under global-phase gauge changes. Because $\mathbb 1-|\psi\rangle\langle\psi|$ is a projector, $F$ is the real part of a Gram matrix and is therefore PSD. This remains true when the derivatives are replaced by the central differences

$$|\partial_i\psi\rangle\approx\frac{|\psi(\boldsymbol\theta+h\mathbf e_i)\rangle-|\psi(\boldsymbol\theta-h\mathbf e_i)\rangle}{2h},$$

so any significantly negative eigenvalue signals an implementation error. The central-difference truncation error is $O(h^2)$, and roundoff contributes $O(\varepsilon_{\rm mach}/h)$. Moving $h$ from $10^{-2}$ toward $10^{-5}$ should therefore shrink the deviation from a small-step reference. The numerical rank is the number of eigenvalues above $10^{-8}\lambda_{\max}$, and it is bounded by both the parameter count and the real dimension of the state manifold, $2(2^n-1)$.

### What the code does
- **System under test:** `state_fisher(model, input_state, parameters, step)` on `MuTA(n, L, one_column=True)` with input $|0\cdots0\rangle$.
- **Oracles:** class A for positivity (an analytic invariant of any Fisher matrix). Class E for step convergence (self-consistency against a smaller reference step, $h_{\rm ref}=10^{-6}$).
- **PSD and scaling sweep:** $(n,L)\in\{(1,1),(2,1),(2,2),(3,1)\}$, with parameters drawn from $U(-\pi,\pi)$ under RNG seed $10n+L$ and $h=10^{-5}$. For each configuration the code records $\lambda_{\min}$, the numerical rank, the parameter count and the condition number $\lambda_{\max}/\lambda_{\min}^{(>10^{-12})}$.
- **Convergence sweep:** MuTA(2,1) at seed-0 parameters, measuring $\|F(h)-F(10^{-6})\|_F$ for $h\in\{10^{-2},10^{-3},10^{-4},10^{-5}\}$.
- **Figure:** `R33_fisher_spectra`.

### Expected outcome
The minimum eigenvalues should be non-negative up to roundoff. The convergence error should fall roughly as $h^2$ until roundoff in the reference begins to matter. Rank is expected to saturate below the parameter count whenever parameters are redundant, or when the state manifold has fewer dimensions than there are parameters (for one wire, at most 2). Acceptance, with the tolerance from `declare_tolerance(scale=4, safety_factor=1e4)`:

$$\min_{\text{configs}}\lambda_{\min}(F)>-4\times10^4\,\varepsilon_{\rm mach}\approx-8.88\times10^{-12},$$

and the convergence errors must be non-increasing along $h=10^{-2},10^{-3},10^{-4},10^{-5}$ (the code checks `errors[i] >= errors[i+1]`).

- **Left panel:** $\lambda_{\min}$ bars for each configuration (labelled nXLY), with a zero line.
- **Centre panel:** log-log plot of $\|F(h)-F_{\rm ref}\|$ against step size.
- **Right panel:** numerical rank bars with the parameter count overlaid as a dashed line, for each configuration.

<!-- ml:R34 -->
**Title:** Finite-energy GKP codewords under Fock-cutoff and grid refinement

**Why this experiment.** Every physical PhotoGraphiQML result (R36 to R59) runs on finite-energy GKP qubits represented in a truncated Fock space. Before any physical gate or measurement is trusted, the codeword resource itself has to be characterized. Three questions matter here: how much of each codeword's weight survives the Fock cutoff, how strongly the finite codewords overlap, and whether the resulting two-dimensional code basis stays a valid (PSD) Gram system as numerical resolution changes. R34 sweeps cutoff and quadrature grid independently. It also confirms that `GKPBridge.diagnostics` never presents a resource diagnostic as a validated physical MuTA implementation. Without this experiment, truncation artefacts in the codewords could be mistaken for logical errors downstream.

### Theory
In the convention $[\hat q,\hat p]=2i$ used by PhotoGraphiQ, the square-lattice GKP logical spacing is $\sqrt{2\pi}$. The finite-energy codeword $\mu\in\{0,1\}$ is a Gaussian comb with a Gaussian envelope,

$$\psi_\mu(q)\propto\sum_{s=-P}^{P}\exp\!\Big[-\frac{(q-c_s)^2}{4\sigma^2}-\frac{\kappa^2c_s^2}{4}\Big],\qquad c_s=\sqrt{2\pi}\,(2s+\mu),$$

where $\sigma$ is the peak width (`peak_width`), $\kappa$ the inverse envelope width (`envelope`) and $P$ the number of cells on each side (`peaks`). The function is sampled on a $q$ grid of `grid_points` points. Projection onto the first $N_c$ Fock states uses the position-space number-state wavefunctions $\varphi_n(q)$ and Simpson quadrature:

$$a_n^{(\mu)}=\int\psi_\mu(q)\varphi_n(q)\,dq,\quad n<N_c,\qquad w_\mu=\sum_{n<N_c}|a_n^{(\mu)}|^2\in[0,1] .$$

The captured weight $w_\mu$ is the fraction of the codeword inside the cutoff. With normalized truncated codewords $|\tilde\mu\rangle=\mathbf a^{(\mu)}/\sqrt{w_\mu}$, the code Gram matrix is

$$G=\begin{pmatrix}1&\langle\tilde0|\tilde1\rangle\\\langle\tilde1|\tilde0\rangle&1\end{pmatrix},\qquad \lambda_\pm=1\pm|\langle\tilde0|\tilde1\rangle| .$$

Finite peaks overlap, so $G\neq\mathbb 1$ in general. Being a Gram matrix, however, $G$ must be Hermitian and PSD, and since $|\langle\tilde0|\tilde1\rangle|\le1$ by Cauchy-Schwarz, $\lambda_-\ge0$.

### What the code does
- **System under test:** `GKPBridge(cutoff, peak_width=0.4, envelope=0.4, peaks=8, grid_points).diagnostics()`. This calls PhotoGraphiQ's `GKPCode.resource(bit).project(cutoff)` and returns the captured weights, $|\langle\tilde0|\tilde1\rangle|$, the eigenvalues of $G$, and the flag `physical_muta_validated`.
- **Oracles:** class A for PSD-ness, an algebraic invariant of any Gram matrix. Class E for the refinement sweep, which is a self-consistency study with no external ground truth.
- **Sweeps:** cutoff $\in\{12,20,32,48\}$ at `grid_points=4097`, and `grid_points` $\in\{2049,4097,8193\}$ at cutoff 32. All other resource parameters are held at the bridge defaults.
- **Metrics:** $w_0$, $w_1$, codeword overlap, and $\lambda_{\min}$ and $\lambda_{\max}$ of $G$ at each point.
- **Figure:** `R34_gkp_codeword_projection`.

### Expected outcome
The captured weight should grow toward 1 as the cutoff increases; this is a qualitative trend, not a gated convergence claim. The codeword overlap should settle once both cutoff and grid are resolved, and it should be insensitive to the grid size at the resolved settings. The run passes when, at every point,

$$-\tau\le w_\mu\le1+\tau,\qquad \lambda_{\min}(G)>-\tau,\qquad \tau=10^3\,\varepsilon_{\rm mach}\approx2.22\times10^{-13},$$

with $\tau$ from `declare_tolerance(scale=1, safety_factor=1e3)`, and `physical_muta_validated` is `False` everywhere. No fault-tolerance or gate-validation claim is made.

- **Left panel:** captured weight of codeword 0 against cutoff.
- **Centre panel:** codeword overlap against cutoff (bottom axis) and against `grid_points` (twinned top axis, distinct colour).
- **Right panel:** the PSD violation $\max(0,-\lambda_{\min})$, floored at $10^{-18}$, against cutoff on a log axis, with the tolerance line.

<!-- ml:R35 -->
**Title:** The signed-X capability boundary of physical MuTA lowering

**Why this experiment.** The physical GKP bridge of PhotoGraphiQML is deliberately restricted. With PhotoGraphiQ 0.3.1 as the backend, the only logical XY-plane measurements a finite GKP qubit can perform are $X$-basis measurements, at angles $0$ and $\pi$ (modulo $2\pi$) and up to an absolute tolerance of $10^{-14}$. The manuscript's claim that unsupported models are rejected before any resources are allocated, rather than silently rounded, depends on `physical_capabilities` drawing this boundary exactly. R35 probes the boundary at sub-tolerance and super-tolerance offsets, at canonical unsupported angles (the $Y$ basis, $\pi/4$, a generic angle) and with non-finite inputs. A trainable angle of $10^{-13}$ rounded to $X$, for example, would execute a physical circuit different from the logical model while reporting success.

### Theory
In MBQC, measuring node $v$ in the XY plane at angle $\alpha$ projects onto

$$|\pm_\alpha\rangle=\tfrac{1}{\sqrt2}\big(|0\rangle\pm e^{i\alpha}|1\rangle\big).$$

For a GKP qubit, the $X$ basis ($\alpha=0$) is a homodyne measurement of $\hat p$ followed by nearest-lattice-cell binning. $\alpha=\pi$ is the same measurement with the outcome relabelled, since $|\pm_\pi\rangle=|\mp\rangle$; PhotoGraphiQ returns it as `PhysicalGKPReadout("X", flip=1)`. Any other angle, including the $Y$ basis at $\alpha=\pi/2$, needs a non-Clifford resource such as magic-state injection, which the backend does not provide. The supported set is therefore

$$\mathcal S=\Big\{\alpha\in\mathbb R:\ \min\big(|\bar\alpha|,\,|\bar\alpha-2\pi|\big)\le10^{-14}\ \text{ or }\ |\bar\alpha-\pi|\le10^{-14}\Big\},\qquad \bar\alpha=\alpha \bmod 2\pi .$$

The angle audit calls `GKPCode.logical_measurement("XY", alpha)` without projecting any codeword. The audit is therefore free of Fock allocation, and the tolerance is the upstream PhotoGraphiQ contract, not a value chosen by PhotoGraphiQML.

### What the code does
- **System under test:** `MuTA(1,1,one_column=True).physical_capabilities({"alpha.w0.c1": angle})`, which returns a per-node `angle_audit`. The supported flag of node $(0,1)$ is inspected.
- **Oracle:** class E. Each result is compared with the documented upstream boundary, $|\text{offset}|\le10^{-14}$ counting as supported. The tolerance itself is not re-derived.
- **Cases:** offsets $\{-10^{-13},-10^{-15},0,10^{-15},10^{-13}\}$ around both $0$ and $\pi$ (10 cases); the angles $\pi/2$, $\pi/4$ and $1.23456789$, which must be unsupported; and NaN and $\infty$, for which the call must raise `ValueError` rather than return an audit. That gives 15 cases in total.
- **Metric:** the number of misclassified cases.
- **Figure:** `R35_capability_map`.

### Expected outcome
The offsets $0$ and $\pm10^{-15}$ around $0$ and $\pi$ should be classified supported, and the offsets $\pm10^{-13}$ unsupported. $\pi/2$, $\pi/4$ and $1.23456789$ should be unsupported, and the two non-finite inputs rejected with an error. The acceptance condition is exact: zero misclassifications over the 15 cases. The audit does not allocate Fock space, but the script relies on that by construction and does not measure it.

- **Single panel:** one unit-height bar per case, labelled by centre and offset (or by angle name). Colour shows the actual classification (supported or unsupported), and a red edge would mark a misclassification. No bar should have a red edge.

<!-- ml:R36 -->
**Title:** Structural preservation and Fock-space accounting under physical lowering

**Why this experiment.** `lower_muta_to_gkp` translates a logical MuTA model whose angles have passed the capability audit (R35) into an inspectable PhotoGraphiQ `Pattern` of GKP preparations, CZ gates, homodyne measurements and classical signals. Physical results mean something only if this translation preserves the logical structure exactly. Every logical vertex must become one optical mode, every graph edge one CZ, inputs and outputs must map to the same nodes, every measured node must get a measurement record, and every node's Pauli-frame dependencies must match the flow corrections. The reported resource cost (peak live modes and Fock dimension) must also be the true combinatorial count, because it drives the allocation guards and the scaling claims in R47 and R58. R36 checks all of this on small signed-X models.

### Theory
A MuTA graph $G=(V,E)$ with flow $f$ yields correction sets

$$\text{source }s\ \mapsto\ \big(f(s),\ Z_s=N(f(s))\setminus\{s\}\big),$$

so node $u$ depends on the sources $\{s: u=f(s)\}\cup\{s: u\in Z_s\}$. The lowering uses a just-in-time schedule. Before node $v$ is measured, each unprepared neighbour is prepared in the finite GKP $|+\rangle$ state, the edges incident on $v$ are entangled with a unit-weight CV CZ, and $v$ is then measured. The number of simultaneously live modes peaks at some value $m$. With an exclusive total-photon cutoff $K$, meaning that states with $\sum_in_i<K$ are kept, the joint Fock space of $m$ modes has dimension

$$D(K,m)=\#\Big\{\mathbf n\in\mathbb N^m:\ \sum_{i=1}^m n_i\le K-1\Big\}=\binom{K+m-1}{m}.$$

By stars and bars, this binomial coefficient is the number of compositions of at most $K-1$ photons into $m$ modes.

### What the code does
- **System under test:** `lower_muta_to_gkp(model, angles, config)`, which returns a `PhysicalLoweringResult` with fields `node_map`, `input_modes`, `output_modes`, `measurement_keys`, `frame_dependencies`, `audit` and `pattern`. The config is `GKPPhysicalConfig(cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513)`.
- **Oracles:** class A for the Fock dimension, recomputed independently as `common.fock_dimension(peak, cutoff)` $=\binom{K+m-1}{m}$. Class E for structure: the model's public `graph`, `input_nodes`, `output_nodes`, `measurement_order` and `corrections`, which R1, R7 and R8 validate independently.
- **Configurations:** $(n,L)\in\{(1,1),(1,2),(2,1)\}$ with signed-X angles drawn uniformly from $\{0,\pi\}$ (RNG seed $10n+L$).
- **Checks per configuration:** violations of the identity node map; input and output tuple equality; measurement keys equal to the measured nodes; frame-dependency mismatches against `model.corrections`; that `pattern.validate()` succeeds; that peak live modes is at least $n$; that the CZ count in the audit equals $|E|$; and the exact Fock-dimension error.
- **Figure:** `R36_lowering_preservation`.

### Expected outcome
The lowering should be structure-preserving by construction, so every check should give zero violations, and the reported Hilbert dimension should equal $\binom{K+m-1}{m}$ exactly. The acceptance condition is that, in every configuration, all violation and mismatch counts are 0, all boolean checks hold, the Fock-dimension error is exactly 0 and the pattern validates.

- **Single panel:** marker plots, offset per check, of the node-map violations, frame-dependency mismatches and Fock-dimension error for each configuration (labelled nXLY). Markers are used so that zero values stay visible, and all markers should sit at 0.

<!-- ml:R37 -->
**Title:** Exhaustive validation of virtual Pauli-frame tracking over all outcome branches

**Why this experiment.** In the physical layer, MBQC byproduct operators are never applied as optical gates. Each measured outcome is reinterpreted through a classical Pauli frame (`lowering.node_frame`), and only the final frame is applied, or folded into the readout, at the outputs. If this bookkeeping is wrong for even one outcome pattern, physical results are correct on average over some branches and wrong on others, which is hard to see in sampled statistics. R37 removes all finite-GKP effects and checks the frame convention exhaustively in the qubit Hilbert space. For each of the $2^8=256$ raw outcome branches of the two-wire signed-X model, the frame-corrected output must equal the deterministic logical target.

### Theory
On the open graph state $|G\rangle=\prod_{(u,v)\in E}CZ_{uv}\,|\psi_{\rm in}\rangle\otimes|+\rangle^{\otimes(|V|-n)}$, node $v$ is measured with raw outcome $r_v$, which projects onto

$$\langle\pm_{\alpha_v}|\ \propto\ \big(1,\ (-1)^{r_v}e^{-i\alpha_v}\big)/\sqrt2 .$$

The flow corrections give each node $u$ a Pauli frame $X^{x_u}Z^{z_u}$, with

$$x_u=\bigoplus_{s:\,f(s)=u}s_s,\qquad z_u=\bigoplus_{s:\,u\in N(f(s))\setminus\{s\}}s_s .$$

Here $s_s$ is the interpreted bit of an earlier source. For an XY-plane measurement, an $X$ byproduct maps $\alpha\to-\alpha$, and a $Z$ byproduct maps $\alpha\to\alpha+\pi$, which flips the outcome. In the signed-X family $\alpha\in\{0,\pi\}$ we have $-\alpha\equiv\alpha$, so no measurement basis needs to adapt. Only the outcome is reinterpreted:

$$s_v=r_v\oplus z_v\qquad(\texttt{frame.correction("X")}=z_v).$$

After all measurements, each output wire $u$ is corrected by first applying $X^{x_u}$ and then $Z^{z_u}$, i.e. by the operator $Z^{z_u}X^{x_u}$. Since the projection-based contraction keeps exactly what the frame accounts for, every branch must reproduce $\rho_{\rm target}=U|\psi_{\rm in}\rangle\langle\psi_{\rm in}|U^\dagger$, up to a global phase removed by comparing density matrices.

### What the code does
- **System under test:** `photographiqml.lowering.node_frame`, the same function the physical execution layer uses, together with `LogicalPauliFrame.correction`.
- **Oracle:** class D. The script embeds the graph state with `logical.cz` and contracts each branch's projective measurements by its own `moveaxis` and reshape loop. This is a different algorithm from the frontier translation in `logical.execute`, which `MuTA.run` uses to produce the target.
- **Setup:** `MuTA(2, one_column=True)` (8 measured nodes), signed-X angles drawn from $\{0,\pi\}$ with RNG seed 0, and a Haar input `haar_states(2,1,seed=17)`. All 256 raw branches `itertools.product((0,1), repeat=8)` are enumerated.
- **Metric:** $\max_{\text{branch}}\|\rho_{\rm branch}-\rho_{\rm target}\|_F$, with the worst branch recorded.
- **Figure:** `R37_pauli_frame_validation`.

### Expected outcome
Every branch should reproduce the target to roundoff. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=100)`:

$$\max_{256\ \text{branches}}\|\rho_{\rm branch}-\rho_{\rm target}\|_F<100\,\varepsilon_{\rm mach}\approx2.22\times10^{-14}.$$

The experiment validates the discrete frame convention only. It does not test finite-GKP accuracy, which R43 to R48 address.

- **Single panel:** one log-scale bar for the maximum branch error, with a dashed tolerance line. The bar should lie well below the line.

<!-- ml:R38 -->
**Title:** Physical MuTA against an independently assembled public PhotoGraphiQ pattern

**Why this experiment.** R36 and R37 test the lowering's structure and its frame logic separately. Neither runs the finite-GKP physics end to end against a second, independently orchestrated implementation. R38 does this for the largest subproblem that is practical to assemble by hand: a one-wire, one-layer MuTA chain executed physically. The decoded output distribution of `PhysicalMuTA(1).run` is compared with that of a pattern built directly from public PhotoGraphiQ commands. The independent pattern never calls `lower_muta_to_gkp` or `node_frame`, and it derives its own frame recursion. Agreement shows that PhotoGraphiQML's lowering, scheduling, signal wiring and output-frame handling form a faithful orchestration of the public physical API.

### Theory
The one-wire chain has nodes $(0,0),\dots,(0,4)$, with output $(0,4)$ and edges between consecutive columns. Its flow is $f(c)=c+1$, with $Z$ targets $N(c+1)\setminus\{c\}=\{c+2\}$ for $c+2\le4$. For $X$-basis readout, the interpreted bits therefore obey

$$s_c=r_c\oplus s_{c-2},\qquad s_{-1}=s_{-2}=0,$$

and the output frame is $X^{s_3}Z^{s_2}$. Physically, each node is a finite-energy GKP mode. Ancillas are prepared in $|\tilde+\rangle$ and entangled by the unit-weight CV gate

$$CZ=\exp\!\big(i\,\hat q_u\hat q_v/2\big)\qquad([\hat q,\hat p]=2i),$$

which applies $(-1)^{\mu\nu}$ on ideal lattice states. Each measured node undergoes a homodyne measurement followed by nearest-cell decoding. In `physical-conditional` mode the analog outcomes are fixed rather than sampled; here all are postselected at $0$. The conditional output state $\rho_{\rm out}$ is read out with exact modular $Z$ POVM effects $E_b$, folding in the frame bit:

$$P(b)=\mathrm{Tr}\big[\rho_{\rm out}\,E_{b\oplus x_{\rm out}}\big].$$

Both sides use the same finite resource and the same analog outcomes, so any nonzero difference beyond the numerical floor would indicate an orchestration or frame disagreement, since sampling noise plays no part.

### What the code does
- **System under test:** `PhysicalMuTA(1, physical_config=config).run([1,0], mode="physical-conditional", analog_outcomes={node: 0.0})`, which returns `decoded_joint_probabilities`. The config is `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)` on the default `piquasso-fock` backend.
- **Oracle:** class C. The independent pattern is built from `pg.Prepare`, `GKPCode.logical_cz`, `pg.Measure`, `pg.Signal` with `pg.CallableExpression` (encoding the recursion above), and `pg.Output`. It runs with `pg.simulate(..., measurement_outcomes=0.0 per node, seed=11)` on the same backend and is read out by `pg.multimode_readout` with the hand-derived output frame. The GKP code parameters and the input encoding `code.encode(1,0)` are shared inputs; the construction, schedule (at most two live modes) and frame arithmetic are independent.
- **Angles:** all measurement angles are 0, i.e. $X$ readout.
- **Metric:** the maximum absolute difference over output labels between the two decoded joint distributions.
- **Figure:** `R38_public_pattern_comparison`.

### Expected outcome
The two decoded distributions should agree to within accumulated Fock-space roundoff. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=1e6)`:

$$\max_b|P_{\rm PGQML}(b)-P_{\rm indep}(b)|<10^6\,\varepsilon_{\rm mach}\approx2.22\times10^{-10}.$$

The looser factor allows for two separately constructed Fock contractions. The comparison covers one wire, zero angles and one conditional trajectory, and it is not a general independent backend.

- **Single panel:** paired bars of the decoded output probabilities from PhysicalMuTA and from the independent pattern for each output label. The title reports the maximum error.

<!-- ml:R39 -->
**Title:** Finite-GKP resource state round-trip through raw Piquasso

**Why this experiment.** PhotoGraphiQML's physical layer runs its Fock simulations on Piquasso, reached through PhotoGraphiQ's backend dispatch. Before any gate is examined, the most basic interface has to be lossless: a finite-GKP resource vector handed to Piquasso must be represented as exactly the same Fock-space state. R39 takes the finite-energy GKP $|+\rangle$ amplitudes produced by PhotoGraphiQ and prepares them in a Piquasso program written directly against Piquasso's public API, with no PhotoGraphiQ or PhotoGraphiQML orchestration. It then reads the state back. A basis-ordering, truncation or normalization mismatch at this interface would corrupt every physical result while looking like ordinary finite-resource error.

### Theory
The truncated GKP codewords $|\tilde\mu\rangle=\sum_{n<K}a^{(\mu)}_n|n\rangle$ (see R34) define the finite plus state

$$|\tilde+\rangle=\frac{|\tilde0\rangle+|\tilde1\rangle}{\big\||\tilde0\rangle+|\tilde1\rangle\big\|}=\sum_{n<K}c_n|n\rangle ,$$

which is what `GKPCode.plus()` returns. Because the finite codewords overlap, the normalization is not simply $1/\sqrt2$. Piquasso's `FockStateVector` prepares $\sum_nc_n|n\rangle$ from an occupation-to-amplitude map in a pure single-mode Fock simulator with cutoff $K$. The figure of merit is the state fidelity

$$\mathcal F=\big|\langle\mathbf c|\boldsymbol\phi_{\rm pq}\rangle\big|^2,$$

where $\boldsymbol\phi_{\rm pq}$ is Piquasso's normalized tensor representation. The test is a state round-trip through the backend API. It does not re-derive the GKP wavefunction and does not test gate dynamics.

### What the code does
- **System under test:** the Fock-basis interface between PhotoGraphiQ GKP resources and Piquasso. The vector comes from `photographiq.GKPCode(cutoff, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097).plus()`.
- **Oracle:** class C. The amplitude vector is a shared input, since Piquasso has no GKP constructor. The preparation and read-back are written independently: `pq.Program` with `pq.Q() | pq.FockStateVector(fock_amplitude_map=...)`, executed by `pq.PureFockSimulator(d=1, config=pq.Config(cutoff=K))`, and read via `result.state.get_tensor_representation()`.
- **Sweep:** cutoff $K\in\{16,24,32\}$.
- **Metric:** $|1-\mathcal F|$ per cutoff, and its maximum.
- **Figure:** `R39_raw_piquasso_state_prep`.

### Expected outcome
The round-trip should be exact up to Piquasso's internal normalization arithmetic, giving infidelities at the roundoff floor for every cutoff. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=1e4)`:

$$\max_K|1-\mathcal F|<10^4\,\varepsilon_{\rm mach}\approx2.22\times10^{-12}.$$

- **Single panel:** $|1-\mathcal F|$ against cutoff on a log axis (floored at $10^{-18}$ for display), with a dashed tolerance line.

<!-- ml:R40 -->
**Title:** The finite-GKP controlled-Z gate against a raw Piquasso Gaussian transform

**Why this experiment.** The CV controlled-Z is the only entangling operation in the physical MuTA lowering, and every graph edge becomes one. In production it reaches Piquasso through several layers: `GKPCode.logical_cz`, then a `Pattern`, then `pg.simulate`, then backend dispatch. R40 checks that this stack applies exactly the intended Gaussian unitary by comparing it with a direct call to `piquasso.GaussianTransform` using hand-derived Bogoliubov matrices. All four finite-GKP basis inputs $|\tilde\mu\rangle|\tilde\nu\rangle$ are tested at three cutoffs. A wrong sign, weight, quadrature convention or mode ordering in the dispatch would corrupt every entangled physical result.

### Theory
In the convention $[\hat q,\hat p]=2i$ (so $\hat q=\hat a+\hat a^\dagger$), the unit-weight CV CZ is

$$\hat U_{CZ}=\exp\!\big(\tfrac{i}{2}\hat q_1\hat q_2\big),\qquad \hat p_1\mapsto\hat p_1+\hat q_2,\quad \hat p_2\mapsto\hat p_2+\hat q_1,\quad \hat q_{1,2}\ \text{unchanged}.$$

In terms of ladder operators this is the Bogoliubov map $\hat{\mathbf a}\mapsto P\hat{\mathbf a}+A\hat{\mathbf a}^\dagger$, with

$$P=\begin{pmatrix}1&\tfrac i2\\\tfrac i2&1\end{pmatrix},\qquad A=\begin{pmatrix}0&\tfrac i2\\\tfrac i2&0\end{pmatrix}.$$

On ideal lattice states with $q_1=\sqrt{2\pi}(2s+\mu)$ and $q_2=\sqrt{2\pi}(2t+\nu)$, the phase is $e^{i\pi(2s+\mu)(2t+\nu)}=(-1)^{\mu\nu}$, i.e. a logical CZ. For finite, overlapping codewords this identity holds only approximately. R40 does not test the identity. It tests that both execution paths apply the same finite-dimensional Gaussian unitary to the same truncated input. The two-mode input is the product state $\sum_{n+m<K}a^{(\mu)}_na^{(\nu)}_m|n,m\rangle$. The gate does not conserve photon number, so a generous cutoff is needed for the retained norm to stay above PhotoGraphiQ's guard.

### What the code does
- **System under test:** the production CZ path. `GKPCode.logical_cz(0,1)` is placed in a `pg.Pattern` and run with `pg.simulate(..., backend="piquasso-fock", cutoff=K)`. The raw two-mode tensor is read through `result.state.native`.
- **Oracle:** class C. The codeword amplitudes from `GKPCode.zero()` and `GKPCode.one()` are shared inputs. The gate is applied independently in a hand-written `pq.Program`: `FockStateVector` (total photon number $<K$), then `pq.Q(0,1) | pq.GaussianTransform(passive=P, active=A)`, executed by `pq.PureFockSimulator(d=2)`.
- **Sweep:** $K\in\{48,64,80\}$ with `peak_width=0.9`, `envelope=0.9`, `peaks=4` and `grid_points=1025`, and all four $(\mu,\nu)\in\{0,1\}^2$, for 12 cases.
- **Metrics:** the Frobenius distance between the normalized $K\times K$ amplitude tensors (the primary metric) and the state fidelity.
- **Figure:** `R40_raw_piquasso_cz`.

### Expected outcome
Both paths call the same Piquasso Fock kernel with identical parameters, so the outputs should agree to roundoff in all 12 cases. Acceptance, with the tolerance from `declare_tolerance(scale=1, safety_factor=1e4)`:

$$\max_{K,\mu,\nu}\big\|\boldsymbol\phi_{\rm raw}-\boldsymbol\phi_{\rm prod}\big\|_F<10^4\,\varepsilon_{\rm mach}\approx2.22\times10^{-12}.$$

This metric is sensitive to global phase, so it also confirms that the two paths use the same phase convention.

- **Single panel:** amplitude error for each case, labelled c$K$($\mu$,$\nu$), on a log axis (floored at $10^{-18}$), with marker colour showing pass or fail against the tolerance and a dashed tolerance line.

<!-- ml:R41 -->
**Title:** Fixed-outcome conditional execution against an independently assembled pattern

**Why this experiment.** R38 established, for a single baseline, that `PhysicalMuTA.run` in physical-conditional mode reproduces a one-wire GKP pattern assembled by hand from public PhotoGraphiQ commands. One case can hide bugs in the parts that only matter away from it: how analog homodyne values are mapped to measurement keys, how a nonzero analog outcome is binned, and how a $\pi$ angle becomes a readout flip. R41 repeats the comparison over several conditioning vectors and both members of the signed-X family, so the lowering's correction bookkeeping is checked beyond the trivial all-zero branch.

### Theory
The physical path lowers one MuTA wire to a five-mode linear chain of finite-energy GKP $|+\rangle$ resources joined by logical CZ gates. Columns $c=0,\dots,3$ are measured in the logical $XY$ basis at angle $\alpha_c\in\{0,\pi\}$. The adaptive angle of a flow-based pattern is $(-1)^{s_X}\alpha_c+s_Z\pi$. For $\alpha_c\in\{0,\pi\}$ we have $(-1)^{s_X}\alpha_c\equiv\alpha_c \pmod{2\pi}$, so X dependencies have no effect and Z dependencies only flip the decoded bit. With flow $f(c)=c+1$ the interpreted bits obey
$$s'_c = s_c \oplus s'_{c-2},\qquad s'_{-1}=s'_{-2}=0,$$
where $s_c$ is the raw nearest-cell bit. The output mode receives the Pauli frame $X^{s'_3}Z^{s'_2}$. In physical-conditional mode the analog outcomes $m=(m_0,\dots,m_3)$ are fixed rather than sampled, so the result is the normalized conditional state $\rho_m\propto K_m\rho K_m^\dagger$ ($K_m$ the product of homodyne Kraus operators). The decoded output distribution is the frame-corrected modular POVM
$$P(b\mid m)=\mathrm{Tr}\!\left[\Pi^{Z}_{b\oplus x}\,\rho_m\right],\qquad \Pi^Z_b=\sum_{s\in\mathbb Z}\int_{\text{cell }(2s+b)L}|q\rangle\langle q|\,dq,\quad L=\sqrt{2\pi}\ (\hbar=2),$$
where $x$ is the frame's X-type correction. Both sides use the same finite GKP code and the same fixed outcomes, so they should agree to rounding.

### What the code does
- System under test: `PhysicalMuTA(1, physical_config=config).run([1, 0], parameters=..., mode="physical-conditional", analog_outcomes=...)`, with `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Reference, oracle class C (shared finite-GKP inputs, independent orchestration): `build_and_run_independent_pattern` builds the chain from `pg.Prepare`, `code.logical_cz`, `code.logical_measurement("XY", alpha)`, `pg.Measure`, `pg.Signal` and `pg.Output`, writes the recursion above as its own callable, and never calls `lower_muta_to_gkp` or `node_frame`. It runs `pg.simulate(..., measurement_outcomes=...)` and `pg.multimode_readout` with the frame $X^{s'_3}Z^{s'_2}$.
- Four cases: zero angles with outcomes $0$; zero angles with uniform outcomes $0.3$; zero angles with mixed outcomes $(0.2,-0.1,0.15,0.05)$; $\alpha_0=\pi$ with zero outcomes.
- Metric: $|P_{\rm PGML}(b)-P_{\rm ind}(b)|$ for every case and output label.
- Figure `R41_conditional_execution_comparison`.

### Expected outcome
Every per-label error should sit at the level of rounding in the shared Fock simulation. The tolerance is declared before the run as `declare_tolerance(scale=1, safety_factor=1e6)` $=10^6\,\varepsilon_{64}\approx2.22\times10^{-10}$, the same as R38. The script passes if the maximum error over all cases and labels is below it and raises `AssertionError` otherwise. Figure: one semilog panel showing the error for each (case, label) pair on the x axis (floored at $10^{-18}$ for display), with a dashed tolerance line. All points should lie well below the line. The scope is one wire only. The $\pi$ case covers only the first column, not every placement of a $\pi$ angle.

<!-- ml:R42 -->
**Title:** Abstraction overhead of the physical pipeline at matched workload

**Why this experiment.** PhotoGraphiQML wraps the physical simulation in a capability audit, a lowering pass, a frame-tracking decoder and result bookkeeping. A reader needs to know whether these layers cost much compared with the Fock simulation they wrap, and where the time goes. R38 and R41 have already shown that the paths agree, so R42 measures only cost. It is descriptive: it supports no speed claim, and its numbers feed panel (c) of the aggregate figure (R_PERF).

### Theory
There is no physical model under test. The quantity is wall-clock time. For each pipeline $k$ the script records repeated timings $t_{k,1},\dots,t_{k,R}$ after $W$ warm-up calls and reports robust statistics:
$$\tilde t_k=\operatorname{median}_r t_{k,r},\qquad \mathrm{IQR}_k=Q_{75}(t_k)-Q_{25}(t_k).$$
The wrapper overhead is defined from medians as
$$\Delta_{\rm wrap}=\tilde t_{\rm full}-\left(\tilde t_{\rm lower}+\tilde t_{\rm sim}\right),$$
which counts the audit, input encoding, frame decoding, joint readout, leakage diagnostics and the logical-target computation inside `run_physical`. A difference of medians is not the median of a difference. $\Delta_{\rm wrap}$ is therefore a point estimate with no interval and can come out negative when timing noise exceeds the true overhead. For scale, a single-wire chain has a peak of a few live modes, so the total-photon Fock dimension $\binom{K+m-1}{m}$ at $K=24$ is small and per-call times are dominated by fixed costs and readout contractions.

### What the code does
- Workload: one wire, all angles $0$, input $|0\rangle$, physical-conditional with all analog outcomes $0$. The resource is `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Pipelines timed with `common.benchmark(..., warmup=1, repeats=5)`:
  1. `photographiqml_full_run`: `PhysicalMuTA.run` (audit + lowering + simulate + decode).
  2. `photographiqml_lowering_only`: `lower_muta_to_gkp(model, config=config)`.
  3. `photographiq_direct_simulate`: `pg.simulate` on the pre-lowered pattern with fixed outcomes.
  4. `raw_piquasso_independent_construct_and_simulate`: the R38/R41 hand-assembled pattern, built and simulated each call.
- The internal `simulation_seconds` and `readout_seconds` from the last full run are saved as a cross-check.
- Oracle class N/A. Metrics are the median, IQR, min and max seconds per pipeline and $\Delta_{\rm wrap}$.
- Figure `R42_abstraction_overhead`.

### Expected outcome
This is a descriptive, single-machine measurement and makes no claim about other hardware. The only enforced condition is that every pipeline's median time is strictly positive; otherwise the script raises `AssertionError`. One would expect the full run to be the slowest path and the lowering-only path the cheapest, with $\Delta_{\rm wrap}$ a modest share of the total that is set mainly by joint readout and diagnostics. These are expectations, not checks. With only five repeats the IQRs are coarse. Figure: one bar chart of median seconds for the four pipelines with error bars of $\pm\mathrm{IQR}/2$. The title reports $\Delta_{\rm wrap}$ in seconds.

<!-- ml:R43 -->
**Title:** Decoded logical-physical statistics without an invented fidelity

**Why this experiment.** A finite-energy GKP simulation does not produce a logical qubit state, so its overlap with the ideal MuTA state is not well defined without an extra, assumption-laden projection. The library therefore compares the two levels only through decoded classical statistics. `compare_logical_physical` returns a total-variation distance, per-wire observable differences and code-subspace leakage, and it deliberately sets `finite_state_fidelity` to `None`. R43 checks this contract directly. If it broke, the paper could quote a fabricated "physical fidelity", or a single resource setting could be reported as converged.

### Theory
Let $p^{\rm L}(b)=|\langle b|\psi_{\rm L}\rangle|^2$ be the ideal logical output distribution over $b\in\{0,1\}^n$ in the readout basis, and $p^{\rm P}(b)$ the decoded physical joint distribution from the modular POVM. The decoded gap is
$$\mathrm{TV}(p^{\rm P},p^{\rm L})=\tfrac12\sum_b\left|p^{\rm P}(b)-p^{\rm L}(b)\right|\in[0,1].$$
The per-wire observable difference is
$$\delta_w=\left(p^{\rm P}_w(0)-p^{\rm P}_w(1)\right)-\sum_b(-1)^{b_w}p^{\rm L}(b),$$
the physical minus the ideal $\langle Z_w\rangle$. Marginal leakage is the weight of the reduced output-mode state $\rho_w$ outside the finite codeword span, $\ell_w=1-\mathrm{Tr}[P_{\rm code}\rho_w]$. The library clips it to $[0,1]$, so the leakage range sub-check holds by construction. The backend also reports the retained norm, the probability mass kept by the Fock truncation at each step. With all angles at $0$, the logical MuTA is the identity on the input (the Euler identity of R2/R50 with zero arguments), so for input $|0\cdots0\rangle$ the ideal distribution is a point mass on $b=0\cdots0$. Then, for a normalised $p^{\rm P}$, $\mathrm{TV}=1-p^{\rm P}(0\cdots0)$, and the gap measures finite-resource decoding error only.

### What the code does
- System under test: `PhysicalMuTA(n).run([1, 0, ...], mode="physical-conditional", analog_outcomes=all 0.0)` for $n=1,2$, followed by `compare_logical_physical(result)`.
- Resource: `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Oracle class E: the function's documented contract, not an external ground truth.
- Recorded per configuration: TV distance, observable differences, marginal leakage, whether `finite_state_fidelity is None`, minimum and maximum retained norm, and the `certified` flag of the convergence record.
- Figure `R43_decoded_statistics`.

### Expected outcome
The acceptance condition is a conjunction, all evaluated in code: TV $\in[0,1+10^{-9}]$; every marginal leakage value in $[0,1]$; `finite_state_fidelity` is `None` for both configurations; and `certified` is `False` for both. Any violation raises `AssertionError`. The TV values and leakages themselves are reported, not bounded beyond these ranges. At this weak-squeezing resource ($\sigma=0.9$ in $\hbar=2$ units, under 1 dB), a visibly nonzero decoded gap is expected, and the two-wire case can differ from the one-wire case because it has more measured modes. Figure: (left) bar chart of the decoded TV distance from the ideal logical target for the 1-wire and 2-wire models; (right) bar chart of the largest marginal code-subspace leakage for each.

<!-- ml:R44 -->
**Title:** Shot convergence of Rao-Blackwell and empirical decoded estimators

**Why this experiment.** In physical-shots mode, PhotoGraphiQML reports two different estimators of the decoded output distribution, each with its own standard error. One is a Rao-Blackwell average of conditional POVM probabilities over sampled homodyne trajectories. The other is the plain frequency of sampled output bits. The paper uses these errors as uncertainty statements, so they must shrink with the number of shots and give intervals that are not badly miscalibrated. A wrong normalisation (for example dividing by $N$ instead of $\sqrt N$, or using marginal instead of joint sampling) would make the physical error bars meaningless. R44 tests both properties at small, tractable shot counts.

### Theory
Each shot $i$ samples the analog outcomes $m^{(i)}$ of the measured modes and yields the conditional joint POVM row $P(b\mid m^{(i)})$. The Rao-Blackwell estimator and its reported error are
$$\hat p_{\rm RB}(b)=\frac1N\sum_{i=1}^N P(b\mid m^{(i)}),\qquad \widehat{\rm SE}_{\rm RB}(b)=\frac{s_b}{\sqrt N},$$
where $s_b$ is the sample standard deviation with $\mathrm{ddof}=1$. A final bit string $b^{(i)}$ is then drawn from each row, which gives
$$\hat p_{\rm emp}(b)=\frac{n_b}{N},\qquad \widehat{\rm SE}_{\rm emp}(b)=\sqrt{\frac{\hat p_{\rm emp}(1-\hat p_{\rm emp})}{N}}.$$
By the law of total variance, $\mathrm{Var}\,\mathbb 1[b^{(i)}=b]=\mathbb E\,\mathrm{Var}(\cdot\mid m)+\mathrm{Var}\,P(b\mid m)\ge\mathrm{Var}\,P(b\mid m)$, so the Rao-Blackwell estimator is never noisier than the empirical one. Both errors scale as $N^{-1/2}$ for a fixed resource. A nominal 95% normal interval $\hat p\pm1.96\,\widehat{\rm SE}$ should cover the true value about 95% of the time. Two caveats apply here: with 8 repetitions the binomial standard deviation of the coverage rate is about $\sqrt{0.95\cdot0.05/8}\approx0.08$, and the reference value is itself a finite-shot estimate.

### What the code does
- System under test: `PhysicalMuTA(1).run([1, 0], mode="physical-shots", shots=N, seed=...)` with `GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`, zero angles.
- Shot sweep: $N\in\{4,8,16,32\}$ at seed 2026. The script records $\max_b\widehat{\rm SE}_{\rm RB}$ and $\max_b\widehat{\rm SE}_{\rm emp}$.
- Reference: $N=64$, seed 999. Coverage: 8 repetitions at $N=16$, seeds 1000 to 1007. Each tests whether $\hat p_{\rm RB}(b)\pm1.96\,\widehat{\rm SE}_{\rm RB}(b)$ contains the reference for the first label $b=(0)$.
- Oracle class E (self-consistency plus a statistical coverage sanity check).
- Figure `R44_shot_convergence`.

### Expected outcome
Both maximal standard errors should fall roughly as $N^{-1/2}$, with the Rao-Blackwell curve at or below the empirical one. Small-$N$ fluctuations are expected because each point uses one seed. Acceptance as coded: the $N=32$ Rao-Blackwell SE is strictly below the $N=4$ one, and the coverage rate lies in $[0.5,1.0]$. This is a deliberately loose band, because exact nominal coverage cannot be resolved with $n=8$. Failure raises `AssertionError`. Figure: (left) log-log plot of maximal SE against shots for both estimators; (right) the 8 coverage repetitions as estimates with $\pm1.96\,$SE error bars and a dashed horizontal line at the 64-shot reference, with the coverage rate in the title.

<!-- ml:R45 -->
**Title:** Joint physical readout retains correlations beyond the product of marginals

**Why this experiment.** A multi-wire decoded distribution must be computed as the joint POVM on the multimode Fock state. Multiplying single-mode marginals is cheaper but silently discards every correlation between wires, which would make entangling MuTA layers look like product channels at the physical level. R45 confirms that `PhysicalMuTA(2)` returns a genuinely joint distribution. It also confirms that the reported two-point correlator `diagnostics["pair_correlations"]` matches a closed-form recomputation from that distribution.

### Theory
For decoded bits $(b_0,b_1)$ with joint law $P$ and marginals $P_0, P_1$, the two-point $Z$ correlator and single-wire expectations are
$$\langle Z_0Z_1\rangle=\sum_{b_0,b_1}(-1)^{b_0+b_1}P(b_0,b_1),\qquad \langle Z_w\rangle=P_w(0)-P_w(1).$$
The connected correlation $\langle Z_0Z_1\rangle_c=\langle Z_0Z_1\rangle-\langle Z_0\rangle\langle Z_1\rangle$ vanishes exactly when the two-bit joint law factorises. The script compares $P$ with $Q(b_0,b_1)=P_0(b_0)P_1(b_1)$ through
$$\mathrm{TV}(P,Q)=\tfrac12\sum_{b_0,b_1}\left|P(b_0,b_1)-P_0(b_0)P_1(b_1)\right|.$$
For two bits, $\mathrm{TV}(P,Q)>0$ is equivalent to $\langle Z_0Z_1\rangle_c\neq0$. Here all angles are $0$, so the logical target is the product state $|00\rangle$ and has no correlations. Any nonzero $\mathrm{TV}(P,Q)$ therefore comes from finite-energy effects in the shared CZ-coupled GKP resource and conditional decoding, which a product-of-marginals readout would hide. The check is structural; it does not say the correlation is large or useful.

### What the code does
- System under test: `PhysicalMuTA(2).run([1, 0, 0, 0], mode="physical-conditional", analog_outcomes=all 0.0)` with `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Oracle class A for the correlator: the formula above, evaluated fresh in the script from `decoded_joint_probabilities` and compared with `diagnostics["pair_correlations"][(0, 1)]`. Oracle class E for the comparison between the joint and the product of `decoded_marginals`.
- Metrics: $|\langle Z_0Z_1\rangle_{\rm indep}-\langle Z_0Z_1\rangle_{\rm reported}|$ and $\mathrm{TV}(P,Q)$. Per-label joint and product values are saved.
- Figure `R45_joint_readout_correlations`.

### Expected outcome
The two correlator evaluations compute the same finite sum, so they should agree to a few ulps. The tolerance, declared before the run, is `declare_tolerance(scale=1, safety_factor=100)` $=100\,\varepsilon_{64}\approx2.22\times10^{-14}$. Acceptance: correlation error < tol and $\mathrm{TV}(P,Q)>0$ (strictly positive, with no magnitude threshold). Failure raises `AssertionError`. Figure: (left) paired bars of the actual joint probability and the product of marginals for the four labels $(0,0),(0,1),(1,0),(1,1)$, with $\mathrm{TV}(P,Q)$ in the title; (right) bars of the independently computed and the reported $\langle Z_0Z_1\rangle$, with their difference in the title. The study uses one zero-angle configuration only and does not sweep entangling angles.

<!-- ml:R46 -->
**Title:** Hard versus soft GKP decoding and rejection of soft feed-forward

**Why this experiment.** PhotoGraphiQ offers two GKP decoders. The nearest-cell decoder makes a hard decision and claims no confidence. The soft decoder returns a Bayesian posterior, but that posterior is valid only for a declared equal-prior ensemble of single-mode codeword preparations, not for the conditional node states of an adaptive MuTA graph. R46 checks both halves of this contract. If the nearest-cell decoder reported a confidence, that number would be invented. If `PhysicalMuTA` accepted `decoder="soft"` for flow-based execution, uncalibrated posteriors would enter feed-forward unnoticed.

### Theory
With $\hbar=2$ the logical lattice spacing is $L=\sqrt{2\pi}$. Ideal bit-0 peaks sit at $q=2sL$ and bit-1 peaks at $q=(2s+1)L$. The finite codewords used by the code are
$$\psi_b(q)\propto\sum_s\exp\!\left[-\frac{(q-c_s)^2}{4\sigma^2}-\frac{\kappa^2c_s^2}{4}\right],\qquad c_s=(2s+b)L,$$
with peak standard deviation $\sigma$ (`peak_width`), inverse envelope width $\kappa$ (`envelope`) and a finite number of peaks, then projected to Fock cutoff $K$. The nearest-cell rule is $\hat b=\mathrm{round}(q/L)\bmod2$, with no posterior attached. The soft decoder evaluates the Fock-space homodyne likelihoods $|\langle q|\psi_b\rangle|^2$ and returns
$$P(b\mid q)=\frac{\pi_b|\langle q|\psi_b\rangle|^2}{\sum_{b'}\pi_{b'}|\langle q|\psi_{b'}\rangle|^2},\qquad \pi_0=\pi_1=\tfrac12,$$
with confidence $\max_bP(b\mid q)$. Moving $q$ from the bit-0 site at $0$ to the cell boundary at $L/2$, the bit-0 likelihood falls and the bit-1 likelihood rises, so the confidence should decrease from near 1 toward $1/2$. Deep in the cell interior it saturates near 1, where pointwise differences are at the level of numerical noise.

### What the code does
- Part (a): `GKPBridge(cutoff=32, peak_width=0.4, envelope=0.4, peaks=8, grid_points=4097).code`, with `pg.NearestCellDecoder()` and `pg.SoftDecisionDecoder(code, basis="Z")` called through `code.decode` at six equally spaced raw outcomes $q\in[0,L/2]$ (`np.linspace(0, L/2, 6)`).
- Part (b): `physical_capabilities(MuTA(1, 1, one_column=True), config=GKPPhysicalConfig(cutoff=16, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513, decoder="soft"))`. This is an audit that does no Fock projection.
- Oracle class E: the documented decoder contract.
- Metrics: nearest confidence `is None` at every point; soft confidence per point; the audit's `supported` flag and `reasons`.
- Figure `R46_hard_vs_soft_decoding`.

### Expected outcome
Four conditions, all computed in code, must hold: (i) the nearest-cell confidence is `None` at all six points; (ii) the soft confidence is non-increasing with slack $10^{-3}$, i.e. $c_i\ge c_{i+1}-10^{-3}$, which absorbs noise on the saturated plateau; (iii) the overall trend decreases, $c_0>c_5$; (iv) the audit reports `supported=False` with a reason mentioning "soft". Any failure raises `AssertionError`. The confidence is expected to stay close to 1 over most of the sweep and drop sharply toward $0.5$ at $q=L/2$. Figure: soft-decoder confidence against $q/L$ from $0$ to $0.5$, with a dashed line at the 50/50 level. The calibration statement is limited to one isolated mode with an equal-prior 0/1 ensemble.

<!-- ml:R47 -->
**Title:** One-axis-at-a-time resource convergence of decoded probabilities

**Why this experiment.** Every decoded physical number depends on five settings. Three are numerical approximations to a fixed resource: the Fock cutoff $K$, the quadrature grid used to project codewords, and the number of comb peaks. Two change the physical resource itself: peak width $\sigma$ (squeezing) and envelope $\kappa$ (energy). A result at one setting means little unless the numerical axes are shown to stabilise. Changes along the physical axes must also be reported as changes of resource, not as convergence. R47 runs each axis in isolation and checks that the library never certifies convergence automatically.

### Theory
For a sweep $v_1<v_2<\dots<v_k$ of one axis, with all other settings fixed at the baseline, let $P_j(b)$ be the decoded conditional distribution at $v_j$. The successive-point change is
$$\Delta_j=\max_b\left|P_j(b)-P_{j-1}(b)\right|,\qquad j=2,\dots,k.$$
For a numerical axis, $\Delta_j\to0$ as the discretisation is refined. Fock truncation, for example, discards the tail weight $\sum_{n\ge K}|\langle n|\psi\rangle|^2$, which falls rapidly once $K$ exceeds the resource's mean photon number. Successive deltas are therefore expected to shrink. A small $\Delta$ does not certify the absolute error, because it bounds only the change between neighbouring settings. For the physical axes, $\sigma$ and $\kappa$ enter the codeword
$$\psi_b(q)\propto\sum_s e^{-(q-c_s)^2/4\sigma^2-\kappa^2c_s^2/4},$$
so changing them changes the target itself. Their deltas describe sensitivity to the resource, not numerical error. Because the analog outcomes are fixed, every delta is a conditional difference with no sampling noise.

### What the code does
- System under test: `PhysicalMuTA(1).physical_convergence([1, 0], values, axis=..., mode="physical-conditional", analog_outcomes=all 0.0)`, which calls `validate_physical_model`. It audits the whole sweep before running any point and never sets `certified=True`.
- Baseline: `GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`, zero angles.
- Sweeps: cutoff $\{20,24,28\}$; grid_points $\{513,1025,2049\}$; peaks $\{3,4,5\}$; peak_width $\{0.85,0.9,0.95\}$; envelope $\{0.85,0.9,0.95\}$. The library labels the first three axes "numerical refinement" and the last two "physical resource change".
- Oracle class E (self-consistency; no independent finite-GKP simulator at this resolution).
- Metrics: $\Delta_j$, the argmax prediction at each point, and the `certified` flag.
- Figure `R47_resource_axis_convergence`.

### Expected outcome
Acceptance as coded: `certified` is `False` for all five axes, and for the two numerical axes cutoff and grid_points the finest-pair delta is strictly below the coarsest-pair delta, $\Delta_3<\Delta_2$, since each sweep has three points. Peaks is plotted as numerical but is not part of the acceptance, and the physical axes carry no trend requirement. Failure raises `AssertionError`. Figure: five panels, one per axis, each plotting $\Delta$ against the axis value. Numerical axes and physical-resource axes use different colours. The first point of each axis has no predecessor and is drawn at $0$, so only the last two points carry information. Results cover one wire at zero angles only.

<!-- ml:R48 -->
**Title:** Instability of discrete physical training (a negative result)

**Why this experiment.** On the physical path, measurement angles are restricted to the categorical signed-X family $\{0,\pi\}$, so training is a discrete search rather than gradient descent. Each objective evaluation is also a finite-shot Monte Carlo estimate. A classifier "trained" this way can easily fit the noise of the particular trajectories it saw. R48 is included so that the paper reports this failure mode openly and does not present a lucky run as physical learning. Its declared scientific outcome is **negative**: it documents instability and supports no claim of classifier performance.

### Theory
The classifier encodes a scalar input as $|\psi(x)\rangle=R_y(x)|0\rangle=\cos\tfrac x2|0\rangle+\sin\tfrac x2|1\rangle$. It reads the decoded first-wire expectation $z(x;\boldsymbol\alpha)=P_0(0)-P_0(1)$ and applies a logistic head $\hat y=\sigma(wz+c)$. For each candidate $\boldsymbol\alpha\in\{0,\pi\}^4$, the head $(w,c)$ is fitted by BFGS to the cross-entropy plus $10^{-4}\lVert(w,c)\rVert^2$. `DiscreteSearch(sweeps=1)` starts at $\boldsymbol\alpha=\mathbf 0$ (the classifier passes this initial point) and flips one coordinate at a time, keeping any flip that lowers the fitted loss, for five evaluations in total. With $S$ shots the decoded $z$ is a Rao-Blackwell mean whose standard error scales as $S^{-1/2}$. At $S=2$ the loss landscape over the 16 configurations is noisy enough that the chosen configuration and the predictions can depend on the shot seed. The instability rate is
$$r=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}\mathbb 1\!\left[\mathrm{acc}_{\rm train}(s)\neq\mathrm{acc}_{\rm fresh}(s)\right],$$
where "fresh" means re-predicting the same four inputs with an independent trajectory seed $s+1000$. This is not a held-out generalisation estimate.

### What the code does
- System under test: `MuTAClassifier(PhysicalMuTA(1), trainer=DiscreteSearch(sweeps=1, seed=s), physical_options={"mode": "physical-shots", "shots": 2, "seed": s})`, with `GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Data: $X=\{0,0.2,2.9,\pi\}$, $y=\{0,0,1,1\}$.
- Seeds $\mathcal S=\{7,19,23,31,42,101\}$. Training accuracy uses physical seed $s$, and fresh accuracy is re-predicted with physical seed $s+1000$.
- Oracle class N/A (descriptive). Metrics: per-seed selected configuration, both accuracies, an exact-categorical flag, $r$, and the number of distinct selected configurations.
- Figure `R48_physical_training_stability`.

### Expected outcome
The enforced structural condition is only that every accuracy lies in $[0,1]$ and every selected $\alpha_j$ is (to `np.isclose`) exactly $0$ or $\pi$, never a rounded continuous value; failure raises `AssertionError`. Scientifically, $r>0$ and more than one distinct selected configuration across seeds are the anticipated finding, recorded with `scientific_outcome="negative"`. Six seeds give no population-level interval. Figure: for each training seed, paired bars of training-trajectory and fresh-trajectory accuracy, each labelled with its value so that a zero-height bar is not misread as missing data. The title reports $r$ and the number of distinct configurations. The setup (2 shots, 4 points) is deliberately small, and R57 extends it with separate validation and test shot streams.

<!-- ml:R_PERF -->
**Title:** Aggregate performance and resource-scaling figure from saved evidence

**Why this experiment.** The manuscript's performance figure brings together results from four experiments: logical runtime scaling against MentPy (R13), the Fock-dimension growth of the lowered pattern (R36), the overhead of the physical abstraction layers (R42), and numerical-refinement convergence (R47). Rebuilding the figure by rerunning those simulations would be costly and could silently mix outputs from different runs. R_PERF runs no simulation. It rebuilds the figure from saved JSON and marks any missing source with an explicit placeholder rather than leaving it out.

### Theory
R_PERF introduces no new physics. It plots the quantities defined in its sources:
- R13: median wall-clock time per matched logical evaluation against the number of wires, with $\pm\mathrm{IQR}/2$ error bars, for PhotoGraphiQML and MentPy.
- R36: the total-photon Fock dimension of the lowered pattern,
$$D=\binom{K+m-1}{m},$$
the number of $m$-mode occupation vectors with total photon number below $K$. Here $K$ is the cutoff and $m$ is the peak number of simultaneously live modes in the just-in-time schedule. Memory for a dense state vector grows as $16D$ bytes.
- R42: median time per pipeline and the wrapper overhead $\Delta_{\rm wrap}=\tilde t_{\rm full}-(\tilde t_{\rm lower}+\tilde t_{\rm sim})$.
- R47: successive-point changes $\Delta_j=\max_b|P_j(b)-P_{j-1}(b)|$ along the cutoff and grid-point axes.
All timings are local single-machine observations. Combining them into one figure adds no comparative claim.

### What the code does
- Loads `R13_runtime_scaling.json`, `R36_lowering_preservation.json`, `R42_abstraction_overhead.json` and `R47_resource_axis_convergence.json` from `results/json/`. A missing file yields `None`, and that panel shows "missing: run R<N> first".
- R13 needs MentPy. In an environment without MentPy, panel (a) is expected to show the placeholder.
- Oracle class N/A (reporting only). Status category: descriptive.
- Saves one row, `sources_present` out of `sources_total = 4`, and the figure `R_PERF_aggregate_performance`.

### Expected outcome
The declared acceptance condition is that all four source JSON files exist and are readable. The saved status is `"pass"` only when `sources_present == 4`, otherwise `"fail"`. The script does not raise on a missing source; it records the failure and still draws the figure with placeholders. Figure, a 2x2 grid:
- **(a)** R13: log-scale median seconds against `n_wires` for PhotoGraphiQML (circles) and MentPy (squares), with IQR/2 error bars.
- **(b)** R36: log-scale bars of the reported Fock dimension $D$ for each `n{wires}L{layers}` configuration.
- **(c)** R42: bars of median seconds for the four physical pipelines, with $\Delta_{\rm wrap}$ in the panel title.
- **(d)** R47: log-scale $\Delta_j$ for the cutoff and grid-points axes. Each has its own x axis (a twin top axis) because their value ranges are incompatible, and the first point of each sweep has no predecessor and is left out instead of being plotted as a false zero.

Expected visual trends follow the sources: $D$ grows combinatorially with the live modes, and $\Delta_j$ falls along the numerical axes. These are inherited expectations, not new checks.

<!-- ml:R50 -->
**Title:** Canonical Euler and Ising-XX identities reproduced at several depths

**Why this experiment.** The source MuTA paper describes its circuits by triangle count and Euler/coupling structure, while PhotoGraphiQML exposes named angle parameters. Any reproduction claim therefore depends on an explicit paper-to-API mapping. R2 and R3 check a single layer. R50 checks that the mapping composes correctly when blocks are stacked: depths 1-3 for the single-wire Euler sequence and depths 1-2 for the two-wire XX coupling. It also records which source-paper claims are reproduced, which are only partially reproduced, and which are unsupported. An ordering error between blocks, or a coupling parameter placed at the wrong index, would pass a single-layer test and fail here.

### Theory
With $R_P(a)=e^{iaP/2}$, one single-wire paper block with angles $(a_0,a_1,a_2,a_3)$ realises $R_X(a_3)R_Z(a_2)R_X(a_1)R_Z(a_0)$. For $B$ blocks, with parameters stored consecutively in groups of four,
$$U_{\rm ref}=\prod_{k=B-1}^{0}\Big[e^{\frac i2a_{4k+3}X}\,e^{\frac i2a_{4k+2}Z}\,e^{\frac i2a_{4k+1}X}\,e^{\frac i2a_{4k}Z}\Big],$$
where later blocks multiply on the left and $B$ is `MuTA.paper_depth`, equal to the requested depth for one wire. For two wires in one-column mode, setting every angle to zero except the coupling angle $\phi_k$ at index $8k+5$ (parameter `alpha.w1.c(4k+1)`) leaves one coupling $e^{\frac i2\phi_kX\otimes X}$ per block (the R3 identity). These couplings commute, giving
$$U_{\rm ref}=\exp\!\Big(\tfrac i2\big(\textstyle\sum_k\phi_k\big)X\otimes X\Big).$$
The error metric is $\lVert U_{\rm MuTA}-U_{\rm ref}\rVert_F$ with no global-phase freedom, so phase conventions are tested as well.

### What the code does
- System under test: `MuTA(1, depth).unitary(values)` for depth $\in\{1,2,3\}$ and `MuTA(2, depth, one_column=True).unitary(values)` for depth $\in\{1,2\}$.
- Oracle class A: `scipy.linalg.expm` with Pauli matrices defined locally in `extended_studies.py`.
- Seeds $\{3,11,29,47\}$. Euler angles come from `common.rng(100*depth + seed)`, uniform on $[-\pi,\pi]$, and coupling angles from `common.rng(200*depth + seed)`. That gives 12 Euler and 8 Ising-XX configurations.
- Metric: Frobenius error per configuration. The JSON also stores `maximum_frobenius_error` and a `reproduction_scope` record (reproduced: Euler and Ising-XX identities; partially reproduced: multi-triangle composition; unsupported: unmapped paper learning curves and arbitrary-angle physical MuTA).
- Figure `R50_canonical_reproduction`.

### Expected outcome
All errors should be at double-precision rounding level. The tolerance is declared before evaluation as `declare_tolerance(scale=2.0, safety_factor=20.0)` $=40\,\varepsilon_{64}=8.881784197001252\times10^{-15}$. If the maximum error reaches it, the script raises `AssertionError` before writing any evidence. Figure: one semilog panel with two series, Euler and Ising-XX, plotting the Frobenius error against the index of the predeclared configuration (depth-major, then seed), and a dashed line at the declared tolerance. All points should lie below the line. The scientific outcome is recorded as positive and covers only the mapped identities.

<!-- ml:R51 -->
**Title:** Gate learning across local and nonlocal target families, failures retained

**Why this experiment.** R20 and R21 study Haar single-wire and Ising-XX gate learning in depth. R51 asks a broader question with equal weight given to failures: how does direct operator learning behave over a predeclared family of local rotations, Haar-local two-qubit targets and a nonlocal XX target, at two depths? Every seed and every optimisation trace is kept, so a high success rate cannot come from selecting favourable targets. The study documents the optimiser's behaviour; it is not a universality proof.

### Theory
For a target $V\in U(d)$ and model unitary $U(\boldsymbol\theta)$, the loss is the operator root-mean-square error
$$\mathcal L(\boldsymbol\theta)=\frac{\lVert U(\boldsymbol\theta)-V\rVert_F}{\sqrt{d^2}},$$
minimised by L-BFGS-B. This loss is *not* invariant under global phase. A target $V=e^{i\varphi}W$ with $W\in SU(d)$ is reachable with zero loss only if the model can also produce the phase. MuTA unitaries are built from fixed gates and rotations $e^{iaP/2}$ of unit determinant, so the global phase is not a free parameter. A Haar target drawn from $U(2)$ generally carries a determinant phase the model cannot match, which can leave a residual $\mathcal L>0$ even when the target is learned exactly up to phase. A phase-invariant check is therefore reported alongside: the mean state fidelity
$$\bar F=\frac1M\sum_{i=1}^M\left|\langle\psi_i|U^\dagger V|\psi_i\rangle\right|^2$$
over $M=12$ Haar test states, computed as $1-$`infidelity`. The families are $R_X(\theta)=e^{i\theta X/2}$ and $R_Z(\theta)$ on one wire, $\theta\sim U[-\pi,\pi]$; `haar_local`, $V=q\otimes I$ with $q$ a QR-based Haar $U(2)$ matrix; and `ising_xx`, $V=e^{i\theta X\otimes X/2}$.

### What the code does
- Models: `MuTA(1, depth)` for rx and rz, and `MuTA(2, depth, one_column=True)` for haar_local and ising_xx, with depth $\in\{1,2\}$.
- Oracle class A: analytic and SciPy target matrices.
- Seeds $\{2,7,19,31\}$. The generator is `common.rng(1000*depth + 10*family_index + seed)`, and initial parameters are drawn from $\mathcal N(0,0.2^2)$. That gives 4 families x 2 depths x 4 seeds = 32 runs.
- Optimiser: `scipy.optimize.minimize(..., method="L-BFGS-B", options={"maxiter": 60, "ftol": 1e-10})`.
- Per run: final operator RMSE, $\bar F$ on 12 Haar states (seed `seed + 8000`), `success` $=\mathcal L<0.1$, parameter count, runtime and the optimiser's own success flag. All objective traces are stored for plotting.
- Summary: bootstrap 95% CI (median, 2000 resamples, seed 51) of the final RMSE, and the success rate.
- Figure `R51_expanded_gate_learning`.

### Expected outcome
The declared structural acceptance is finite, labelled metrics for all 32 runs. In the shared `_finish` helper the structural status is written as `pass` without being recomputed. The scientific outcome is computed: **positive** if the success rate (RMSE < 0.1) is at least 0.5, otherwise **negative**. Rotations and XX targets lie inside the model's reachable set and should usually succeed. Haar-local targets may show a phase-related RMSE floor and some failures within 60 iterations, which the boxplot keeps visible. Figure: (left) every optimisation trace (operator RMSE against objective evaluation, semilog, translucent); (right) a boxplot of final operator RMSE per family titled "Failures retained".

<!-- ml:R52 -->
**Title:** Leakage-safe repeated classification against classical baselines

**Why this experiment.** QML classification results are easy to inflate. Scaling on the full data set, choosing hyperparameters on the test labels, reporting a single favourable split, or omitting trivial baselines can each make a quantum model look better than it is. R52 is a descriptive benchmark built to avoid all four. It compares the Eq. 5 MuTA kernel (SVM) and a variational `MuTAClassifier` with majority, logistic and RBF-SVM baselines over repeated stratified splits. Scaling is fit on training data only, and the kernel's $C$ is chosen on validation data only. No claim of quantum advantage is made or implied.

### Theory
The MuTA kernel is the fidelity kernel of the Eq. 5 feature state, $k(\mathbf x,\mathbf x')=|\langle\phi(\mathbf x)|\phi(\mathbf x')\rangle|^2$. Here $|\phi(\mathbf x)\rangle$ is the output of a two-wire one-column MuTA whose angles are $(x_0,x_1,\cos x_0\cos x_1,x_0,x_1)$ on five designated columns. The kernel is PSD with unit diagonal and is used through a precomputed SVC. The variational classifier encodes $\mathbf x$ by product $R_y$ rotations, reads $z=\langle Z_0\rangle$ from a `MuTA(2, one_column=True)` model, and applies a logistic head $\sigma(wz+c)$ trained by Adam on the cross-entropy. On held-out test labels $y$, with predictions $\hat y$ and scores $s$, the script reports
$$\mathrm{Acc}=\tfrac1n\textstyle\sum_i\mathbb 1[\hat y_i=y_i],\quad \mathrm{BAcc}=\tfrac12(\mathrm{TPR}+\mathrm{TNR}),\quad \mathrm{F1}_{\rm macro}=\tfrac12\textstyle\sum_{c\in\{0,1\}}\frac{2P_cR_c}{P_c+R_c},$$
together with ROC-AUC computed from $s$ and the confusion matrix. A constant-score majority classifier has ROC-AUC exactly $0.5$. Uncertainty is a percentile bootstrap of the mean.

### What the code does
- Data sets: `simple` (150 Gaussian points, $y=\mathbb 1[x_0>0]$); `nonlinear` (`make_circles`, 150 points, factor 0.45, noise 0.1); `breast_cancer` (first two features of the scikit-learn data set).
- Seeds $\{5,13,23,41,59\}$. Each split is stratified: 25% test, then 25% of the remainder for validation (`random_state=seed+1`). `StandardScaler` is fit on the training part only.
- Quantum kernel: `MuTAKernel().gram_matrix(train)`. $C\in\{0.1,1,10\}$ is selected by validation accuracy, and the final model is `SVC(kernel="precomputed", C=best_C, probability=True)`.
- `MuTAClassifier(MuTA(2, one_column=True), trainer=Trainer(optimizer="adam", epochs=35, learning_rate=0.04), seed=seed)`.
- Baselines: majority class, `LogisticRegression(max_iter=2000)`, `SVC(C=1.0)` with RBF kernel. Oracle class N/A.
- Summary: bootstrap CI (mean, 2000 resamples, seed 52) of quantum-kernel accuracy pooled over 3 data sets x 5 seeds.
- Figure `R52_comprehensive_classification`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is that all metrics are finite in $[0,1]$ and that every split uses training-only preprocessing and validation-only selection. The split logic follows this by construction, but the metric ranges are not recomputed: the shared `_finish` helper records `pass`. Linear and RBF baselines should do well on `simple`; `nonlinear` separates linear from nonlinear models; and majority sets the floor on every data set. Whether the MuTA models match the classical baselines is reported, not asserted. Figure: 15 bars of mean held-out test accuracy (3 data sets x 5 methods: quantum_kernel, muta_model, linear, rbf, majority), with the MuTA methods coloured differently from the classical baselines.

<!-- ml:R53 -->
**Title:** Regression in interpolation, extrapolation and a real benchmark

**Why this experiment.** R24 checks that `MuTARegressor` can fit $\sin x$ on held-out points from the training domain. R53 asks the harder questions: how does the model behave outside the training domain, and on real data, compared with simple classical regressors under the same splits? Extrapolation is where the periodic inductive bias of angle encoding helps or hurts, and real targets with arbitrary scale test the affine readout head. The study is descriptive and makes no claim of regression superiority.

### Theory
`MuTARegressor` encodes a standardised scalar feature $\tilde x$ as $R_y(\tilde x)|0\rangle$, applies a single-wire MuTA layer $U(\boldsymbol\theta)$, and predicts
$$\hat y=w\,\langle Z\rangle_{\boldsymbol\theta}(\tilde x)+c,\qquad \langle Z\rangle\in[-1,1],$$
trained by Adam on the mean squared error. A single-qubit model of this kind is a degree-1 trigonometric function of $\tilde x$, $\hat y=a_0+a_1\cos\tilde x+b_1\sin\tilde x$. That suits a sine target, but it is periodic in $\tilde x$, so its extrapolation is fixed by the period in standardised units rather than learned. Only the features are standardised; targets keep their native scale, so $(w,c)$ must absorb the target's mean and range from the initial $(1,0)$. On test data the script reports
$$\mathrm{RMSE}=\sqrt{\tfrac1n\textstyle\sum_i r_i^2},\quad \mathrm{MAE}=\tfrac1n\textstyle\sum_i|r_i|,\quad R^2=1-\frac{\sum_ir_i^2}{\sum_i(y_i-\bar y)^2},\qquad r_i=y_i-\hat y_i,$$
together with the residual mean (bias) and residual standard deviation (calibration). $R^2$ uses the *test* mean, so the train-mean baseline can score $R^2<0$ out of domain.

### What the code does
- Tasks: `sine_interpolation` (100 points uniform on $[-\pi,\pi]$, 25% random test split); `sine_extrapolation` (120 points on a uniform grid over $[-1.5\pi,1.5\pi]$, trained on $|x|\le\pi$ and tested on the disjoint outer domain; this split does not depend on the seed); `diabetes_bmi` (scikit-learn diabetes data, BMI feature only, 25% test split).
- Seeds $\{3,17,37,67\}$. `StandardScaler` is fit on training features only.
- System under test: `MuTARegressor(MuTA(1), trainer=Trainer(optimizer="adam", epochs=45, learning_rate=0.035), seed=seed)`.
- Baselines (oracle class N/A): training-mean predictor, `LinearRegression`, `SVR(C=10.0, gamma="scale")`.
- Summary: bootstrap CI (mean, 2000 resamples, seed 53) of MuTA test RMSE pooled over all 12 runs.
- Figure `R53_comprehensive_regression`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is finite RMSE, MAE, $R^2$ and residual statistics for every task, seed and method; `_finish` records `pass` without recomputing it. Expectations: on the sine tasks the MuTA model should be competitive with the RBF-SVR in interpolation, and outside the training domain all models, MuTA included, extrapolate according to their own inductive bias. On `diabetes_bmi`, whose targets are unscaled, the short Adam schedule on the affine head may limit the MuTA fit. These are expectations; the code makes no such assertion. The pooled RMSE CI mixes target scales (sine versus diabetes) and should be read per task. Figure: (left) mean test RMSE per method (muta_regressor, mean, linear, rbf_svr), one line per task; (right) MuTA residual mean against residual SD for each run, with a dashed line at zero bias.

<!-- ml:R54 -->
**Title:** MuTA-kernel ablations over the supported axes

**Why this experiment.** Kernel results depend strongly on input scaling, regularisation and sample size, and a kernel's spectrum often explains its performance better than accuracy alone. R54 sweeps every axis the fixed Eq. 5 `MuTAKernel` actually exposes and records spectral diagnostics beside test accuracy. It also states plainly which axes it cannot sweep. The kernel has no depth, triangle-count, shot or physical-noise parameter, so those axes are recorded as "unsupported" rows instead of synthetic data points.

### Theory
For training inputs $\{\mathbf x_i\}_{i=1}^n$ scaled by a factor $\lambda$, the Gram matrix is $K_{ij}=|\langle\phi(\lambda\mathbf x_i)|\phi(\lambda\mathbf x_j)\rangle|^2$. As a fidelity kernel it is symmetric and PSD with unit diagonal; this is the analytic invariant, oracle class A. The diagnostics are as follows. Kernel-target alignment with $\pm1$ labels $\mathbf y$:
$$A(K,\mathbf y)=\frac{\mathbf y^\top K\mathbf y}{\lVert K\rVert_F\,\lVert\mathbf y\mathbf y^\top\rVert_F}=\frac{\mathbf y^\top K\mathbf y}{n\,\lVert K\rVert_F}.$$
Effective rank, the exponential of the spectral entropy of the positive eigenvalues $\lambda_k$ normalised to $p_k=\lambda_k/\sum_{k'}\lambda_{k'}$:
$$r_{\rm eff}=\exp\!\Big(-\sum_kp_k\ln p_k\Big)\in[1,n].$$
The condition number is evaluated as $\kappa(K+10^{-10}I)$, and the script also records the minimum eigenvalue and the symmetry error $\max|K-K^\top|$. Larger $\lambda$ makes the feature map vary faster, which pushes $K$ toward the identity (higher $r_{\rm eff}$, a risk of memorisation). Smaller $\lambda$ makes $K$ nearly constant (low rank, underfitting). The SVM regularisation $C$ changes only the classifier, not $K$.

### What the code does
- Data: `make_moons(n_samples=180, noise=0.12, random_state=seed)` with a stratified 70/30 split, giving 126 training and 54 test points. The features are not standardised.
- Seeds $\{7,19,43,71\}$. Sweep: training size $n\in\{40,80,126\}$ (the first $n$ training points), scale $\lambda\in\{0.5,1,2\}$, and $C\in\{0.1,1,10\}$, for 108 supported rows.
- System under test: `MuTAKernel().gram_matrix` and `kernel(test, train)`, used with `SVC(kernel="precomputed", C=C)`. Oracle classes A (PSD invariant) and N/A (accuracy).
- Four additional rows label `ansatz_depth`, `triangle_count`, `shot_count` and `physical_noise` as unsupported.
- Summary: bootstrap CI (mean, 2000 resamples, seed 54) of test accuracy over all supported rows.
- Figure `R54_kernel_ablations`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is that supported Gram matrices are symmetric and PSD within floating-point tolerance, scores are finite, and unsupported axes stay labelled. The symmetry error and minimum eigenvalue are saved per row, but no tolerance is computed or enforced in code; `_finish` records `pass`. The symmetry error should be essentially zero, and the minimum eigenvalue non-negative up to floating-point rounding. The Gram-only diagnostics repeat across the three $C$ values. Effective rank is expected to grow with $n$ and with $\lambda$. Figure: (left) effective rank against training size for $C=1$, one colour per scale; (right) kernel-target alignment against test accuracy over all supported rows, as a descriptive association only.

<!-- ml:R55 -->
**Title:** Finite-system expressivity and trainability diagnostics

**Why this experiment.** Expressivity and trainability claims for variational circuits are often drawn from a few random points. R55 records several local diagnostics side by side for small MuTA models of increasing width and depth, under five seeds: the quantum Fisher information (QFI) spectrum, two-wire entanglement, gradient magnitudes and short-horizon optimisation progress. The aim is to describe these finite systems honestly. The study does **not** claim, and its data cannot establish, asymptotic barren-plateau behaviour or global expressibility.

### Theory
For a pure state $|\psi(\boldsymbol\theta)\rangle$ the QFI matrix is
$$F_{ij}=4\,\mathrm{Re}\!\left[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right],$$
which is real symmetric and PSD (class A invariant). `state_fisher` evaluates it with central differences $\partial_i\psi\approx[\psi(\boldsymbol\theta+h\mathbf e_i)-\psi(\boldsymbol\theta-h\mathbf e_i)]/2h$, $h=10^{-5}$. Its numerical rank, the number of eigenvalues above $10^{-8}$, counts the locally independent state directions and is a local expressivity proxy bounded by $\min(P,\,2\cdot2^n-2)$. For two wires the pure-state concurrence is $C=2|ad-bc|$ for $|\psi\rangle=(a,b,c,d)$. Trainability is probed with the infidelity loss toward a Haar-random target $|t\rangle$,
$$\mathcal L(\boldsymbol\theta)=1-|\langle t|\psi(\boldsymbol\theta)\rangle|^2,$$
through the norm of its central-difference gradient ($h=10^{-5}$) and the reduction achieved by 30 L-BFGS-B iterations.

### What the code does
- Models: `MuTA(n, d, one_column=True)` for $(n,d)\in\{(1,1),(1,2),(2,1),(2,2),(3,1)\}$, input $|0\cdots0\rangle$.
- Seeds $\{2,11,29,53,83\}$. Parameters are drawn from $\mathcal N(0,0.4^2)$ via `common.rng(seed + 100*n + d)`, and the Haar target from `haar_states(n, 1, seed + 900)`. That gives 25 rows.
- Per row: QFI rank, minimum and maximum QFI eigenvalue, concurrence (two-wire models only, otherwise `None`), gradient norm, initial and final infidelity, and `optimization_success` = final < initial.
- Oracle class A/E. Summary: bootstrap CI (mean, 2000 resamples, seed 55) of the gradient norm pooled over configurations. The record states `barren_plateau_claim: "not made"`.
- Figure `R55_expressivity_trainability`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is that the QFI is PSD within a declared numerical tolerance and that all metrics are finite. The minimum QFI eigenvalue is saved, but no tolerance is declared or checked in this function; `_finish` records `pass`. The finite-difference QFI should have a minimum eigenvalue of order the differencing error, not significantly negative. The QFI rank should grow with the parameter count until it saturates at the state-manifold dimension, which is reached quickly for one wire. Gradient norms are reported per configuration without any threshold. Figure: (left) log-scale boxplots of gradient norm per configuration `n{n}d{d}`; (right) local QFI rank against parameter count, coloured by depth.

<!-- ml:R56 -->
**Title:** Decoded logical-physical gap along cutoff and shot axes

**Why this experiment.** R43 checks the contract of `compare_logical_physical` at one resource setting. R56 measures how the decoded gap between the ideal logical MuTA and its finite-GKP physical execution changes along two supported axes, varied one at a time: the Fock cutoff in conditional mode and the shot count in sampled mode. It also records explicitly what stays unsupported, namely a finite-state fidelity and soft adaptive decoding. This is the main-text evidence for the size of the logical-physical gap, and it is descriptive.

### Theory
With $p^{\rm L}$ the ideal logical distribution and $p^{\rm P}$ the decoded physical distribution, the gap is reported as
$$\mathrm{TV}=\tfrac12\sum_b|p^{\rm P}(b)-p^{\rm L}(b)|,\qquad \epsilon_{\rm obs}=\max_w\big|\langle Z_w\rangle_{\rm P}-\langle Z_w\rangle_{\rm L}\big|.$$
The script also records a captured weight, the minimum retained norm reported by the backend over the simulation steps, which measures how much probability survives Fock truncation. All angles are $0$ and the input is $|0\rangle$, so the logical target is $|0\rangle$ with $p^{\rm L}=(1,0)$. Then, up to the normalisation of $p^{\rm P}$, $\mathrm{TV}=p^{\rm P}(1)$ and $\epsilon_{\rm obs}=2p^{\rm P}(1)$, which is purely the decoding error of the finite resource. Along the cutoff axis the conditional result is deterministic. Once the cutoff resolves the resource, the gap should settle to the intrinsic finite-squeezing error and not tend to zero. Along the shot axis, $p^{\rm P}$ is the Rao-Blackwell mean over $N$ sampled trajectories. Its sampling error scales as $N^{-1/2}$, so the TV estimate combines the intrinsic gap with Monte Carlo noise.

### What the code does
- Cutoff axis: `PhysicalMuTA(1)` with `_physical_config(cutoff)` = `GKPPhysicalConfig(cutoff, peak_width=0.9, envelope=0.9, peaks=3, grid_points=513)` for cutoff $\in\{20,24,28\}$, run in physical-conditional mode with all analog outcomes $0$.
- Shot axis: `GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)` in physical-shots mode, $N\in\{2,8\}$, seeds $5600+N$ (5602, 5608).
- Each run is compared with `compare_logical_physical`. The script records TV, $\epsilon_{\rm obs}$, `finite_state_fidelity` (expected `None`), captured weight (cutoff axis only) and runtime. Runtime is effectively recorded only on the shot axis: the cutoff rows read `diagnostics.get("seconds")`, a key that does not exist, so they store `None`.
- One row labels `decoder = "soft adaptive flow"` as unsupported: the soft ensemble posterior is not calibrated for adaptive node states.
- Oracle class E. Figure `R56_logical_physical_gap`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is TV in $[0,1]$, finite-state fidelity remaining `None`, and unsupported configurations labelled. These quantities are saved, but `_finish` records `pass` without recomputing them. TV is expected to change little between cutoffs once the resource is resolved; any residual reflects the weak squeezing ($\sigma=0.9$). The two shot-axis points use a different resource from the cutoff axis (4 peaks, 1025 grid points, cutoff 40), so their levels are not directly comparable with the cutoff points. Figure: (left) TV against cutoff; (right) TV against shots. Both are line plots over the supported rows.

<!-- ml:R57 -->
**Title:** Physical categorical selection evaluated on disjoint shot streams

**Why this experiment.** R48 shows that discrete physical training is unstable when it is judged on the same trajectories used for selection. R57 sets out the stricter protocol a robustness claim would need. A categorical configuration is selected on one shot stream and then evaluated on two further streams with disjoint seeds and more shots: a validation stream and a fresh test stream. A robustness criterion is declared in advance. If it is not met, the recorded outcome is negative, and the paper must not claim a robust physical classifier.

### Theory
The model is the R48 classifier: $R_y(x)$ encoding, a one-wire signed-X `PhysicalMuTA` with $\boldsymbol\alpha\in\{0,\pi\}^4$, decoded $z=P(0)-P(1)$, and a logistic head fitted per candidate. Configurations are chosen by `DiscreteSearch(sweeps=1)` starting from $\boldsymbol\alpha=\mathbf 0$. With $S$ shots the decoded $z$ is a Rao-Blackwell mean with standard error $\propto S^{-1/2}$. Selection at $S=1$ is therefore the noisiest stage, and evaluation at $S=2$ and $S=4$ is progressively less noisy. For each selection seed $s$ the script reports
$$\mathrm{gap}(s)=\mathrm{acc}_{\rm train}(s)-\mathrm{acc}_{\rm test}(s),\qquad u(s)=\mathbb 1\!\left[\mathrm{acc}_{\rm val}(s)\neq\mathrm{acc}_{\rm test}(s)\right],$$
the instability rate $r=\overline{u}$, and a percentile-bootstrap 95% CI $[\ell,h]$ for the mean fresh-test accuracy. The predeclared robustness criterion is
$$\text{positive}\iff r=0\ \text{and}\ \ell\ge0.75,\qquad\text{negative otherwise}.$$
All three streams predict the same four inputs; "fresh" refers to independent quantum trajectories, not unseen inputs. The study therefore measures stability under shot noise, not input generalisation.

### What the code does
- System under test: `MuTAClassifier(PhysicalMuTA(1), trainer=DiscreteSearch(sweeps=1, seed=s), physical_options={"mode": "physical-shots", ...})`, with `GKPPhysicalConfig(cutoff=40, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)`.
- Data: $X=\{0,0.3,2.8,\pi\}$, $y=\{0,0,1,1\}$.
- Selection seeds $\{7,23,47,89\}$. Training uses 1 shot (seed $s$), validation 2 shots (seed $s+1000$) and fresh test 4 shots (seed $s+2000$).
- Per seed: selected configuration, the three accuracies, gap, $u$ and runtime. Summary: CI (mean, 2000 resamples, seed 57) of test accuracy and $r$.
- Oracle class N/A. Figure `R57_physical_training_evaluation`.

### Expected outcome
Declared acceptance: the structural part is exact categorical selections, disjoint shot seeds and finite metrics. Disjoint seeds hold by construction, but the categorical and finiteness checks are not recomputed here; `_finish` records `pass`. The scientific outcome is computed from the criterion above. With only four seeds and very few shots, the bootstrap CI is wide and any disagreement between validation and test makes $r>0$. A negative outcome is therefore a realistic and acceptable result, and it must be reported as such. Figure: for each selection seed, grouped bars of training, validation and fresh-test accuracy on $[0,1.05]$.

<!-- ml:R58 -->
**Title:** Operating boundary of the signed-X physical path

**Why this experiment.** The physical MuTA path is usable only in a bounded region of resource settings. With too small a Fock cutoff the truncated state loses probability mass. Changes to the GKP peak width alter both the resource quality and its photon-number content. Past a configured memory budget the library refuses to allocate at all. R58 maps this region over cutoff and peak width using a criterion fixed in advance, and runs the actual conditional simulation at every audited point. It also records the physical effects the path does not model, so the boundary is not read as covering them.

### Theory
The capability audit `physical_capabilities` reports the peak live-mode Hilbert dimension
$$D=\binom{K+m-1}{m},$$
the total-photon truncation at cutoff $K$ for $m$ simultaneously live modes. Alongside it are estimates of the joint-readout workspace ($64\,D_{\rm out}^2$ bytes, with $D_{\rm out}=\binom{K+n-1}{n}$ for $n$ output wires) and the projection workspace ($64\,G\,(K+2P+1)$ bytes for $G$ grid points and $P$ peaks). The audit rejects a point if $D$ exceeds `max_dimension` ($10^5$ by default) or either workspace exceeds `max_matrix_bytes` ($2.56\times10^8$ by default), and it rejects any angle outside $\{0,\pi\}$ or a soft decoder. For supported points, the backend's retained norm $\eta$ is the norm of the state after each Fock-truncated operation, before renormalisation; the script takes its minimum over steps. The Fock backend raises a `ValueError` itself whenever $|\eta-1|$ exceeds its `norm_tolerance` (default $10^{-3}$ in `backends/fock.py`), and R58 catches that. A finite-energy GKP state with peak width $\sigma$ and envelope $\kappa$ has a photon-number distribution that extends further as $\sigma$ decreases, so a narrower peak needs a larger $K$ to reach the same $\eta$. Each point is classified by the predeclared criterion:
$$\text{acceptable}\iff\text{audit supported},\ \eta\ge0.90,\ \text{and within the allocation budget}.$$
Supported points with $\eta<0.90$ are "unstable". Points rejected by the audit or by a `ValueError` during execution are "unconverged/resource-rejected", and those with an unsupported angle are "unsupported".

### What the code does
- Grid: cutoff $K\in\{8,12,16,20,24,28\}$ x peak width $\sigma\in\{0.75,0.9,1.05\}$, with `_physical_config` = envelope 0.9, peaks 3, grid_points 513, nearest decoder.
- For each point: `PhysicalMuTA(1).physical_capabilities(zeros)`. If supported, `run([1, 0], mode="physical-conditional", analog_outcomes=all 0.0)` is executed and the minimum `retained_norm` recorded.
- Four rows record unmodelled features as unsupported: loss channel, detector inefficiency, arbitrary physical angle, soft adaptive decoder.
- Oracle class E (capability audit plus retained-norm self-consistency); deterministic, no seeds.
- Figure `R58_physical_resource_boundary`.

### Expected outcome
The outcome is descriptive. The declared acceptance is that regions are assigned by the predeclared criterion, which the code does by construction; `_finish` records `pass`. Low cutoffs are expected to fail, with narrower peaks needing a larger $K$. Because the backend aborts any step with $\eta<1-10^{-3}$ (at default tolerance), such points should appear as "unconverged/resource-rejected" rather than "unstable". Completed runs then have $\eta\ge0.999$, and the 0.90 threshold is in practice never the binding constraint. At these single-wire sizes the allocation budget is not expected to bind either. Figure: minimum retained norm against Fock cutoff, one line per peak width, with a dashed line at the predeclared 0.90 boundary. Points without a retained norm (rejected) appear as gaps.

<!-- ml:R59 -->
**Title:** End-to-end runtime and allocation scaling over measured ranges

**Why this experiment.** Users need a realistic picture of what each PhotoGraphiQML pipeline costs as its natural size grows: logical execution with the number of wires, kernel Gram matrices with the number of samples, training with epochs, physical simulation with the Fock cutoff, and evidence serialisation with payload size. R59 measures wall time and peak traced allocation for all five under one protocol. The fitted scaling exponents apply only within the measured ranges. Extrapolating them, or comparing them across machines, would overstate what the data show, so the study is descriptive.

### Theory
For each pipeline and size $x$, the script performs one warm-up call and then $R$ timed repetitions $t_r$, reporting the median $\tilde t(x)$ and $\mathrm{IQR}(x)=Q_{75}-Q_{25}$. Peak memory is the largest `tracemalloc` peak across repetitions. This counts allocations traced by Python's allocator (NumPy buffers included), not process resident memory, and tracing adds overhead to the timings. Scaling is summarised by a least-squares fit in log-log space,
$$\ln\tilde t(x)=\beta\ln x+c,$$
with the empirical exponent $\beta$ and the measured range $[x_{\min},x_{\max}]$ stored and no extrapolation. Reference expectations: logical state-vector simulation of $n$ wires grows as $2^n$ per gate, which looks steep on a log-log axis. A Gram matrix of $N$ samples needs $N$ feature-state evaluations plus an $N\times N$ overlap, so it is between linear and quadratic in $N$. Adam training scales linearly in epochs. Physical simulation follows the total-photon dimension $\binom{K+m-1}{m}$, polynomial in $K$ of degree $m$. JSON serialisation is linear in payload size. With two or three sizes per pipeline (two for the physical one), $\beta$ is a coarse descriptor.

### What the code does
- Logical: `MuTA(n, one_column=True).run(|0...0>, zeros)` for $n\in\{1,2,3\}$, 3 repeats.
- Kernel: `MuTAKernel().gram_matrix` on $N\in\{20,40,80\}$ Gaussian points (`common.rng(N)`), 3 repeats.
- Training: `Trainer(epochs=E, learning_rate=0.05).fit` (default optimiser Adam) on $\sum_j(p_j-0.2)^2$ with 8 parameters, $E\in\{5,10,20\}$, 3 repeats.
- Physical and decoding: `PhysicalMuTA(1)` with `_physical_config(K)` (envelope 0.9, peak width 0.9, 3 peaks, 513 grid points), physical-conditional with zero outcomes, $K\in\{20,24\}$, 2 repeats.
- Serialisation: `json.dumps` of the accumulated rows repeated $\{10,100,1000\}$ times, 5 repeats.
- Instruments: `time.perf_counter` and `tracemalloc`. Fits use `np.polyfit` on log-log data. Oracle class N/A; no seeds.
- Figure `R59_end_to_end_scaling`.

### Expected outcome
The outcome is descriptive. The declared structural acceptance is positive finite timings, explicit warm-up and repetition counts, and fits restricted to the measured range. The counts and ranges are written into every row and fit, but positivity and finiteness are not re-checked; `_finish` records `pass`. At these small sizes, fixed per-call overheads can dominate, so the fitted exponents may lie well below the asymptotic ones listed above. Absolute values are specific to the host. Each serialisation payload also contains the serialisation rows appended by earlier sizes, so its block content grows slightly from one size to the next. Figure: (left) log-log median seconds against pipeline-specific size, one line per pipeline; (right) log-log peak traced bytes against the same sizes. The size units (wires, samples, epochs, cutoff, row blocks) differ between lines, so comparing slopes across pipelines is meaningful but comparing levels is not.
