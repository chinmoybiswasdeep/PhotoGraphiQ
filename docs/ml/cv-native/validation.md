# Validation

Every capability is mapped to the tests that check it and the experiment that measures
it. Tests are in `tests/ml/cv_native/` (48 tests, all passing on 2026-11-16 with
`.venv/Scripts/python -m pytest tests/ml/cv_native`). Experiments are in
`paper_experiments_ml/15_cv_native/`. Results are in
`paper_experiments_ml/results/json/R68_*.json` to `R74_*.json`, and every R68 to R74 file
records `status: pass`. Numbers below are taken from those files. R75 is reported in the
final report.

Test files:

- **G**: `test_cv_gaussian_programs.py` (11 tests)
- **C**: `test_cv_cluster.py` (10)
- **L**: `test_cv_learning.py` (17)
- **K**: `test_cv_kernels_nongaussian.py` (10)
- **F**: `tests/validation/cv_fock_reference.py`, the independent reference used by K

| Capability | Tests | Experiment | Status and evidence |
| --- | --- | --- | --- |
| Feature maps: moments, energy, input validation | G: `test_displacement_encoding_moments_and_energy`, `test_squeezing_and_rotation_encodings`, `test_feature_map_rejects_invalid_inputs` | (none) | Pass. Analytic means, covariances and photon numbers |
| Ideal `GaussianProgram` map | G: `test_ideal_channel_matches_independent_gate_products`, `test_ideal_moments_match_gate_level_pattern_execution` | R69 | Pass. Matches `photographiq.gaussian` gate products (1e-14) and `gaussian` / `native-gaussian` execution of `gate_pattern` (1e-12) |
| Compiled CV-MBQC channel (Path A) | G: `test_compiled_channel_matches_the_executable_pattern` (r = 0.4, 1.2), `test_finite_squeezing_noise_scales_as_exp_minus_two_r`, `test_compiled_pattern_trajectories_reproduce_the_ensemble_channel` | R69 | Pass. R69 (36 configurations): max \|X - S\| 7.6e-14; pattern matrix 2.4e-14 and noise 7.8e-15 (relative); \|Y e^{2r} - const\| at most 2.8e-15; trajectories (1000 shots) largest \|z\| 1.44 |
| Bounded parameters, serialisation, validation errors | G: `test_bounded_squeezing_and_serialisation_roundtrip`, `test_program_validation_errors` | (none) | Pass |
| `CVCluster` channel (Path B) vs `flow_pattern` + `gaussian_channel`; symplectic ideal map | C: `test_channel_matches_flow_pattern_and_ideal_map_is_symplectic` (1x5, 2x3, 2x4 with rungs (1, 3), 3x4) | R69, R68 | Pass (1e-11). R68: 2 x 4 cluster has 6 measurements = w(c - 1) |
| Single-wire teleportation product, pi-periodicity, singular-angle rejection, bounded/direct equivalence | C: `test_single_wire_is_a_product_of_teleportation_shears`, `test_angles_are_pi_periodic_and_singular_angles_are_rejected` | (none) | Pass |
| Trainable weights recertify the flow | C: `test_trainable_weights_recertify_the_flow` | (none) | Pass. The channel follows new weights. Complex-step derivatives cannot see weights, so `gradient="auto"` uses central differences for them and weights start at `rung_weight` (L: `test_trainable_rung_weights_use_real_derivatives_and_start_at_their_value`; [gradients](gradients.md)) |
| Homodyne conditioning, cluster trajectories | C: `test_homodyne_conditioning_matches_an_independent_schur_complement`, `test_cluster_trajectories_conditional_versus_ensemble` | R69 | Pass |
| Layout validation | C: `test_invalid_layouts_are_rejected` | (none) | Pass |
| Complex-step gradients | L: `test_complex_step_matches_closed_form_derivative`, `test_complex_step_matches_central_differences_on_physical_losses`, `test_gradient_near_the_homodyne_margin_is_finite_and_consistent` | R73 | Pass. Closed form 2.2e-16; complex step vs central differences, 180 cases, at most 8.8e-9 relative; finite to 0.050 rad from the singular angle |
| Shot estimator (exact in distribution, reproducible, CRN) | L: `test_shot_estimator_is_exact_in_distribution_and_reproducible` | R72, R73 | Pass. R72 sampled vs exact expected MSE, largest \|z\| 1.36; R73 CRN gradient std 0.0099 to 0.15 vs 1.6 to 2.0 for independent noise |
| Training: regression, reproducibility, history | L: `test_regressor_learns_an_affine_target_reproducibly` | R70 | Pass. R70 median test MSE: layered ideal 3.95e-4, compiled r = 1 3.95e-4, 2 x 3 cluster 3.90e-4, OLS 3.95e-4; 3 x 3 cluster 4.36e-4 vs OLS 4.36e-4 |
| Affine limit of Gaussian displacement models | L: `test_gaussian_mean_readout_is_affine_in_displacement_features` | R71 | Pass. R71 circles: Gaussian affine model 0.629 = linear logistic 0.629 |
| Optimisers and user gradients | L: `test_other_optimisers_and_user_gradients` | R70 | Pass. R70 2 x 3 cluster final train loss: Adam 2.97e-4, SGD 4.50e-4, L-BFGS 2.97e-4 (128 evaluations vs 968) |
| Classifiers: logistic, homodyne (Born), softmax | L: `test_classifiers_binary_homodyne_and_multiclass` | R71, R72 | Pass. R71 blobs: logistic and homodyne heads 0.975 = linear baseline 0.975 |
| Penalties; ideal = physical means | L: `test_penalties_and_finite_squeezing_semantics` | R72 | Pass. R72: noise-penalised added noise at most the unpenalised value at every r (e.g. 0.330 vs 0.375 at r = 1) |
| Resource report | L: `test_resource_report_and_configuration_errors` | R68, R74 | Pass. R74 structural checks pass (cluster counts, compiled counts, depth bound); timings provisional |
| Nonlinearity attribution | L (affine test), K (Kerr regressor) | R71 | Pass (structural checks). Accuracies are descriptive ([feature encoding](feature-encoding.md#where-the-nonlinearity-comes-from)) |
| Finite-squeezing learning | L: `test_penalties_and_finite_squeezing_semantics` | R72 | Pass. Expected MSE decreases in r and in S ([finite squeezing](finite-squeezing.md)) |
| Existing families untouched; CVCluster is not a MuTA representation | L: `test_existing_model_families_are_untouched` | (none) | Pass |
| Kernels: RBF identity, unitary invariance, finite-squeezing dependence, fidelity vs overlap | K: `test_coherent_overlap_kernel_is_the_gaussian_rbf`, `test_unitary_invariance_and_finite_squeezing_dependence`, `test_fidelity_versus_hilbert_schmidt_for_mixed_states` | (none) | Pass (RBF to 1e-13; PSD to 1e-12) |
| Fock engine vs independent expm reference | K: `test_fock_engine_matches_the_matrix_exponential_reference` (uses F) | R75 | Pass (means 2e-6, variances 2e-5 at cutoffs 40, 60). R75: see the final report |
| Cutoff convergence, retained norm | K: `test_cutoff_convergence_and_retained_norm_reporting` | R75 | Pass. R75: see the final report |
| Gaussian program on Fock engine = moments | K: `test_gaussian_only_program_agrees_between_fock_and_moment_execution` | (none) | Pass (1e-8) |
| Non-Gaussian learning (Kerr regressor beats affine on x^2) | K: `test_non_gaussian_regressor_expresses_a_nonlinear_map_gaussian_cannot` | R71, R75 | Pass. R71 Kerr classifier, circles: 0.829 (642 s). R75: see the final report |
| Cubic injection lowering | K: `test_cubic_injection_lowering_tracks_its_outcome_record` | R75 | Pass (outcome probability within 0.02 of 1). R75: see the final report |

## Experiments

| ID | Script | Question | Recorded outcome |
| --- | --- | --- | --- |
| R68 | `68_cv_architecture.py` | Are the pipeline's resource counts those of executable patterns? | pass; numbers in [resource analysis](resource-analysis.md) |
| R69 | `69_circuit_vs_mbqc.py` | Does the compiled channel equal the intended map plus e^{-2r} noise? | pass |
| R70 | `70_gaussian_regression.py` | Do Gaussian models reach OLS on affine targets? | pass (median test MSE within 2 x OLS + 1e-4 for every model) |
| R71 | `71_nonlinear_classification.py` | Where does nonlinear classification come from? | pass (structural); accuracies descriptive |
| R72 | `72_finite_squeezing_learning.py` | How do finite squeezing and shots affect learning? | pass |
| R73 | `73_gradients.py` | Are gradients exact and finite; how noisy are shot gradients? | pass |
| R74 | `74_resource_scaling.py` | Do resources follow from structure? Accuracy-resource trade-off | pass (structural); timings PROVISIONAL; trade-off descriptive |
| R75 | `75_non_gaussian_learning.py` | Fock convergence, Wigner negativity, non-Gaussian regression, cubic injection | see the final report |

## What is not validated

- No comparison with an external CV-ML library or hardware.
- Loss, detector inefficiency and mode mismatch are not modelled, so they are not
  validated.
- Learning results use one or a few seeds on small synthetic datasets. They show
  trainability and structure, not performance in general, and no quantum advantage.
