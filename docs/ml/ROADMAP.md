# Roadmap

1. Expand faithful logical MuTA validation and reproduce more published tasks.
   Implement tied measurement groups, bounds, mixed quantum inputs, noisier
   resource branches and the learned teleportation instrument. Independently
   train the pinned MentPy reference, rather than compare only checkpoints.
2. Extend the v0.2 signed-X bridge with validated arbitrary logical XY
   measurement/injection instruments, correlated input preparation and joint
   code-subspace leakage. Expand the independent resource and whole-pattern
   convergence studies. Legacy `representation="gkp"` remains resource-only;
   physical execution uses explicit `PhysicalMuTA`.
3. Derive and implement CVMuTA independently. Prove the ideal cell map,
   characterize its finite-resource channel, establish identity embeddings in
   the appropriate regime, and validate against direct PhotoGraphiQ physics.
4. Expand training with valid autodiff, stochastic estimators, mini-batches,
   noise-aware readouts, callbacks and optimizer/model checkpointing.
5. Complete full paper reproduction, ten larger demo projects and the full
   release acceptance checklist. Run the Python CI matrix on hosted runners.

Reservoir computing and recurrent models are future directions. None of these
items imply quantum advantage or an error-correction threshold.
