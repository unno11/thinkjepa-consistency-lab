# ThinkJEPA baseline definition

## Purpose

This branch uses the public ThinkJEPA release as a fixed comparison baseline for
motion-geometry (MG) research. The baseline is not the research contribution and
must not be silently redesigned while MG variants are evaluated.

## Source identity

- Upstream repository: `https://github.com/Hai-chao-Zhang/ThinkJEPA`
- Public baseline commit: `9b95d358f7c7a1eaf73fda0efbcc8e28c44ce086`
- Commit subject: `first_public_release`
- Research branch: `codex/mg-research`

The research branch starts at that commit. Later commits may add experiment
orchestration, tests, and MG code, but the provenance above remains the baseline
anchor.

## Reproduction configuration

The canonical machine-readable configuration is
[`configs/thinkjepa_baseline.json`](../configs/thinkjepa_baseline.json). It captures
the historical trajectory-prediction setup used as the MG comparison point:

- `predictor=thinkjepa`, `backbone=vjepa`
- `trajmode=traj`, `joint_pred=true`, `ref_mode=none`
- cached V-JEPA features (`use_npz_cache`, `skip_vjepa`)
- VLM source `both`, selector `last`, conditioning `film`
- past/future horizon `32/32`, temporal stride `1`
- per-device train/eval batch size `2/2`
- seed `42`, 100 training epochs
- task/predictor weights `1.0/1.0`
- learning rates `1e-3` (readout) and `1e-4` (predictor)
- AMP enabled, non-finite losses fail the run rather than being skipped

The historical best checkpoint was epoch 91, but epoch 91 is not hard-coded as a
new result. New runs select checkpoints on validation data only.

## Preserved baseline behavior

The public baseline consumes the last cached VLM layer through its released FiLM
path. Known historical implementation behavior is preserved for the baseline;
this branch does not turn the baseline task into a VLM-repair project. Any future
correctness change that alters model semantics must be a separately named control
and must be applied consistently to baseline and MG variants when fairness
requires it.

## Claims boundary

Historical `baseline_ep91`, `Full-medium ep99`, and `RC0 ep106` results are
exploratory because model and epoch choices reused the historical test set. They
may motivate the MG hypothesis, but they are not final evidence.

MG losses operate on the world-coordinate trajectory readout. With the released
trainer's detached predictor output, MG gradients affect the trajectory readout,
not the JEPA predictor. The accurate method description is therefore
"motion-geometry-regularized trajectory readout."
