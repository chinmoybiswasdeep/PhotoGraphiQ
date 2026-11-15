**End-to-end execution (compact version of manuscript experiment R60).** `Circuit(1).rotate(0,
0.5).squeeze(0, 0.3).compile(squeezing=1.5)` gives a 10-mode wire;
`flows.analyze` returns `certified`. (a) Execution stage of one `native-gaussian`
trajectory (outcomes `m`, executed corrections dashed). (b) Output phase space for a coherent
input alpha = (0.6-0.4j): ideal target S(0.3) R(0.5) (the compiled channel matrix equals it to
5.6e-16), the unconditional finite-squeezing channel (added noise Tr N =
0.5472) and the conditional output means of 200 trajectories, which
scatter because finite squeezing leaves an outcome-dependent shift. (c) Output covariance
entries: ideal S S^T, the channel prediction S S^T + N and the sampled ensemble (one conditional
covariance plus the sample covariance of the conditional means; statistical, not exact). The full pipeline with
independent validation is `paper_experiments/21_showcase/60_end_to_end_showcase.py`.

Source: `scripts/generate_demo_figures.py` (`14_end_to_end_execution`).
