# Motion-geometry readout integration

MG is applied to `pred_world` and detached ground-truth world coordinates. The
released predictor output remains detached before the trajectory readout, so MG
gradients update the readout only. This is intentionally a
motion-geometry-regularized trajectory readout, not a physics-aware JEPA latent
predictor.

All five MG weights default to zero. In that state the skeleton is not initialized,
the MG module is not imported, and the original baseline loss graph is retained.

The integration-smoke configuration enables every component on the 8+4 train-only
membership. Its weights are historical exploratory scales used solely to exercise
the plumbing; they are not selected or frozen formal hyperparameters.

The trainer records the raw weighted MG aggregate, the warmup-scaled applied MG
loss, every component, and the scale. Non-finite MG loss terminates the run under
the formal fail-fast policy.
