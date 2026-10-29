# Review of `cache_train/physics_losses.py`

## Provenance

The restored candidate came from the server stash named
`preformal-future-mg-physics-losses-20260713T143508Z`. It is byte-identical to the
archived pre-formal backup.

- Size: 12,553 bytes
- SHA-256: `f1813f06da604823a6defdab8ae3330a2d1ec2ac6810194dee9d7cfff7d69be7`

The file was first restored as an independent module. A later reviewable commit
wires it into the trajectory readout with opt-in, zero-default weights; it remains
inactive for the baseline configuration.

## Implemented objectives

- normalized L1 bone-length consistency;
- SmoothL1 velocity matching;
- SmoothL1 acceleration matching;
- cosine-space joint-angle consistency;
- optional jerk matching (implemented but excluded from the aggregate);
- weighted aggregation and a linear warmup scale helper.

Ground-truth-derived quantities are evaluated under `torch.no_grad()`. Prediction
tensors remain differentiable.

## Review findings

1. The objectives are readout-space objectives in world coordinates. They do not
   make the JEPA latent predictor physics-aware while predictor features are
   detached before the readout.
2. Temporal losses assume a fixed frame interval. Comparisons are valid only when
   temporal stride and frame rate are held constant, or when derivatives are
   explicitly scaled by elapsed time.
3. Normalized bone loss can amplify very short ground-truth bones. The existing
   epsilon prevents division by zero but should not substitute for a skeleton/data
   validity audit.
4. Calling `.item()` for every metric synchronizes the GPU. This is acceptable for
   the initial implementation but may add overhead.
5. The module does not validate tensor rank, joint count, index range, or finiteness.
   The trainer must fail fast on malformed/non-finite inputs and outputs.
6. Loss weights and warmup must be fixed on validation data; historical weights
   selected on the old test set are exploratory defaults only.

## Required verification before trainer integration

- zero loss when prediction equals target;
- positive loss under controlled perturbations;
- finite, non-zero gradients on the prediction tensor;
- target tensor receives no gradient;
- empty/short temporal cases return zero as documented;
- skeleton indices are valid for the 52-joint representation;
- CPU test passes locally or on the server's pinned PyTorch environment.
