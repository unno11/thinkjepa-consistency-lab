# Kinematic Projection Control

## Scientific question

Can a zero-training skeleton post-process reproduce the geometry benefit of MG?
This control is a method-necessity test, not a proposed model.

## Frozen method: KProjPredMedian

For each predicted sample and each of the 50 frozen skeleton edges:

1. compute that edge's length in every one of the 32 predicted future frames;
2. use the temporal median as the reference length;
3. preserve the model's predicted edge direction at every frame;
4. reconstruct the two trees from the predicted right/left forearm roots.

The operation receives only the model's own prediction. It does not receive the
observed trajectory, future target trajectory, another sample, or a population
bone template. It therefore cannot leak future labels or add a joint-position
input modality. It does use the complete jointly predicted horizon, so it is an
offline post-process rather than an online causal filter.

## Expected diagnostic behavior

- BoneDriftAll should approach zero by construction.
- BoneErrorAll may improve if temporal prediction noise is a major source of the
  error, but target bone lengths are never supplied.
- AngleError should not materially improve because predicted bone directions are
  preserved (except deterministic handling of degenerate zero vectors).
- ADE/FDE may improve or degrade due to recursive joint repositioning.

If this control matches MG on geometry while causing a smaller spatial penalty,
the necessity and novelty of fixed MG losses are weakened. If MG retains a large
angle/dynamic advantage, that separates learned consistency from trivial length
projection.

## Frozen identities

- topology: `evaluation.topology.KINEMATIC_TREE_EDGES`
- implementation: `evaluation.kinematic_projection.project_prediction_median_bone_lengths`
- protocol: `configs/kinematic_projection_control_v1.json`
- evaluation split: the same frozen 200-sample Direct32 validation manifest
- locked final test: forbidden until the full method is frozen

The three-seed aggregation also reports a 13-task macro cluster bootstrap and
exact task-level sign-flip test for B/MG/KProj contrasts.  Holm correction covers
all three pre-declared contrasts and every metric in each family within a seed.
The 200 episode identifiers each occur once, so episode-level inference is not
reported.
