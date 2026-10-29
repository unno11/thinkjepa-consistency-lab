# Direct32 context-only controls and cluster inference

## Frozen controls

- `LastPose`: repeat context frame 32 for all 32 future frames.
- `ConstVelocityK4`: estimate joint-wise velocity from the last four context
  frames and extrapolate it for 32 future frames.

Both controls receive only the 32-frame observed world-space joint trajectory.
The future target is passed only to the canonical metric function after a
prediction has been produced.  The locked test manifest remains forbidden.

The machine-readable protocol is
`configs/direct32_naive_controls_validation_v1.json`.  Formal output refuses to
overwrite an existing directory and records the manifest, protocol, Git commit,
sample count, and cluster audit.

## Paired comparisons

Each of the three independently trained B checkpoints is compared with the same
deterministic controls on the same 200 validation samples.  Sample-level paired
bootstrap and randomization tests are reported with Holm correction across both
pre-declared controls and all metrics in each Spatial/Geometry/Dynamics family.

Across training seeds, the unit of replication remains the trained B model seed
(`n=3`).  Cross-seed mean and standard deviation are descriptive only.

## Task-macro cluster inference

The frozen validation manifest contains 13 task directories with strongly unequal
sample counts.  Task-macro inference first averages paired deltas within each task,
then gives the 13 task means equal weight.  It reports a 13-task cluster bootstrap
CI and an exact two-sided sign-flip p-value over all `2^13` assignments, followed
by metric-family Holm correction.

The filename audit finds 200 unique episode identifiers for 200 samples, so every
episode cluster contains exactly one sample.  Episode-cluster bootstrap would be
identical to ordinary sample resampling and is therefore explicitly not reported.
This limitation must not be described as episode-level robustness.

## Zero-ratio policy

LastPose has zero predicted velocity and acceleration; ConstVelocityK4 has zero
predicted acceleration.  Their corresponding raw ratios are valid zeros, but the
pre-declared `abs(log(ratio))` deviation is mathematically infinite.  The evaluator
does not add an arbitrary epsilon or clip these values.  Structural zeros are
identified from the frozen control definition, rather than trusting tiny floating
point residuals from a second difference.  These metrics, and any other contrasts
with fewer than two finite transformed pairs, are written to `unavailable_metrics.csv` as
`not_imputed_not_tested` and are excluded from significance and model-seed means.
The raw zero ratios remain available in the per-sample control output.

`per_task_effects.csv` records every task/seed/contrast effect.  It is the source
for improved-task counts and worst-task reporting; aggregate sample means must not
be described as cross-task robustness when the task-macro result disagrees.
