#!/usr/bin/env python3
"""Motion-Geometry Consistency Loss functions for ThinkJEPA training.

All losses operate on world-coordinate trajectory predictions.
GT tensors are .detach()-ed internally to prevent gradient leakage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_physics_loss_weight_scale(epoch: int, args) -> float:
    """Linear warmup schedule for physics loss weights.

    Returns 0.0 before start_epoch, linearly ramps from 1/warmup_epochs
    to 1.0 during warmup, and 1.0 after.

    Args:
        epoch: current epoch (0-indexed)
        args:  argparse.Namespace with physics_loss_start_epoch, physics_loss_warmup_epochs
    """
    start = int(getattr(args, "physics_loss_start_epoch", 0))
    warmup = max(1, int(getattr(args, "physics_loss_warmup_epochs", 0)))
    if epoch < start:
        return 0.0
    if warmup <= 0:
        return 1.0
    progress = min(1.0, (epoch - start + 1) / warmup)
    return float(progress)


# ---------------------------------------------------------------------------
# Bone Length Consistency Loss
# ---------------------------------------------------------------------------

def compute_bone_length_loss(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
    bone_idx: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 loss on bone lengths, normalized by GT bone length.

    Args:
        pred_world:   [B, T, J, 3]  float
        target_world: [B, T, J, 3]  float
        bone_idx:     [K, 2]  int64 on same device as pred_world
        normalize:    if True, divide error by (gt_len + eps) for scale invariance
        eps:          small constant for numerical stability

    Returns:
        scalar loss
    """
    if bone_idx.numel() == 0:
        return pred_world.new_zeros(())

    bone_idx = bone_idx.to(pred_world.device)

    # Bone vectors and lengths: [B, T, K]
    pred_vec = pred_world[:, :, bone_idx[:, 1], :] - pred_world[:, :, bone_idx[:, 0], :]
    pred_len = torch.linalg.norm(pred_vec, dim=-1)

    with torch.no_grad():
        tgt_vec = target_world[:, :, bone_idx[:, 1], :] - target_world[:, :, bone_idx[:, 0], :]
        tgt_len = torch.linalg.norm(tgt_vec, dim=-1)

    error = (pred_len - tgt_len).abs()
    if normalize:
        error = error / (tgt_len + eps)
    return error.mean()


# ---------------------------------------------------------------------------
# Velocity Matching Loss
# ---------------------------------------------------------------------------

def compute_velocity_loss(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 loss between predicted and GT joint velocities.

    Args:
        pred_world:   [B, T, J, 3]
        target_world: [B, T, J, 3]

    Returns:
        scalar loss
    """
    if pred_world.shape[1] < 2:
        return pred_world.new_zeros(())

    pred_vel = pred_world[:, 1:, :, :] - pred_world[:, :-1, :, :]  # [B, T-1, J, 3]
    with torch.no_grad():
        gt_vel = target_world[:, 1:, :, :] - target_world[:, :-1, :, :]
    return F.smooth_l1_loss(pred_vel, gt_vel)


# ---------------------------------------------------------------------------
# Acceleration Matching Loss
# ---------------------------------------------------------------------------

def compute_acceleration_loss(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 loss between predicted and GT joint accelerations.

    Args:
        pred_world:   [B, T, J, 3]
        target_world: [B, T, J, 3]

    Returns:
        scalar loss
    """
    if pred_world.shape[1] < 3:
        return pred_world.new_zeros(())

    pred_vel = pred_world[:, 1:, :, :] - pred_world[:, :-1, :, :]  # [B, T-1, J, 3]
    pred_acc = pred_vel[:, 1:, :, :] - pred_vel[:, :-1, :, :]       # [B, T-2, J, 3]

    with torch.no_grad():
        gt_vel = target_world[:, 1:, :, :] - target_world[:, :-1, :, :]
        gt_acc = gt_vel[:, 1:, :, :] - gt_vel[:, :-1, :, :]
    return F.smooth_l1_loss(pred_acc, gt_acc)


# ---------------------------------------------------------------------------
# Jerk Loss (implemented but not wired into default loss)
# ---------------------------------------------------------------------------

def compute_jerk_loss(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 loss between predicted and GT joint jerks.

    NOTE: Not recommended as a training loss — may cause over-smoothing.
    Kept for diagnostic / optional use.

    Args:
        pred_world:   [B, T, J, 3]
        target_world: [B, T, J, 3]

    Returns:
        scalar loss
    """
    if pred_world.shape[1] < 4:
        return pred_world.new_zeros(())

    pred_vel = pred_world[:, 1:, :, :] - pred_world[:, :-1, :, :]
    pred_acc = pred_vel[:, 1:, :, :] - pred_vel[:, :-1, :, :]
    pred_jerk = pred_acc[:, 1:, :, :] - pred_acc[:, :-1, :, :]

    with torch.no_grad():
        gt_vel = target_world[:, 1:, :, :] - target_world[:, :-1, :, :]
        gt_acc = gt_vel[:, 1:, :, :] - gt_vel[:, :-1, :, :]
        gt_jerk = gt_acc[:, 1:, :, :] - gt_acc[:, :-1, :, :]
    return F.smooth_l1_loss(pred_jerk, gt_jerk)


# ---------------------------------------------------------------------------
# Joint Angle Consistency Loss
# ---------------------------------------------------------------------------

_ANGLE_WARNING_ISSUED = False


def compute_joint_angle_consistency_loss(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
    angle_triplets: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity loss for joint angles.

    Uses cosine similarity instead of acos for gradient stability.
    For each triplet (parent, joint, child):
        pred_cos = cos(pred_parent-pred_joint, pred_child-pred_joint)
    and compares against (detached) GT cos.

    Args:
        pred_world:      [B, T, J, 3]
        target_world:    [B, T, J, 3]
        angle_triplets:  [M, 3] int64  (parent, joint, child) indices;
                         if empty, returns 0 and prints a one-time warning.

    Returns:
        scalar loss
    """
    global _ANGLE_WARNING_ISSUED

    if angle_triplets.numel() == 0:
        if not _ANGLE_WARNING_ISSUED:
            import warnings
            warnings.warn(
                "compute_joint_angle_consistency_loss: angle_triplets is empty, "
                "returning 0. Joint angle loss will have no effect."
            )
            _ANGLE_WARNING_ISSUED = True
        return pred_world.new_zeros(())

    angle_triplets = angle_triplets.to(pred_world.device)

    # Vectors from joint to parent / child: [B, T, M, 3]
    pred_v1 = pred_world[:, :, angle_triplets[:, 0], :] - pred_world[:, :, angle_triplets[:, 1], :]
    pred_v2 = pred_world[:, :, angle_triplets[:, 2], :] - pred_world[:, :, angle_triplets[:, 1], :]
    pred_cos = F.cosine_similarity(pred_v1, pred_v2, dim=-1, eps=1e-8)  # [B, T, M]

    with torch.no_grad():
        tgt_v1 = target_world[:, :, angle_triplets[:, 0], :] - target_world[:, :, angle_triplets[:, 1], :]
        tgt_v2 = target_world[:, :, angle_triplets[:, 2], :] - target_world[:, :, angle_triplets[:, 1], :]
        tgt_cos = F.cosine_similarity(tgt_v1, tgt_v2, dim=-1, eps=1e-8)

    return F.smooth_l1_loss(pred_cos, tgt_cos)


# ---------------------------------------------------------------------------
# Aggregator: compute all physics losses and diagnostic metrics
# ---------------------------------------------------------------------------

def compute_motion_geometry_losses(
    pred_world: torch.Tensor,
    target_world: torch.Tensor,
    bone_idx_finger: torch.Tensor,
    bone_idx_palm: torch.Tensor,
    angle_triplets: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute all motion-geometry losses and diagnostic metrics.

    Args:
        pred_world:       [B, T, J, 3]
        target_world:     [B, T, J, 3]
        bone_idx_finger:  [K_f, 2]  finger_internal bone pairs
        bone_idx_palm:    [K_p, 2]  palm_to_finger bone pairs
        angle_triplets:   [M, 3]
        args:             argparse.Namespace

    Returns:
        mg_loss:  scalar tensor (weighted sum of individual losses)
        metrics:  dict with individual loss values (floats, for logging)
    """
    lambda_bone = float(getattr(args, "lambda_bone_loss", 0.0))
    lambda_palm = float(getattr(args, "lambda_palm_bone_loss", 0.0))
    lambda_vel = float(getattr(args, "lambda_velocity_loss", 0.0))
    lambda_acc = float(getattr(args, "lambda_acceleration_loss", 0.0))
    lambda_ang = float(getattr(args, "lambda_angle_loss", 0.0))

    zero = pred_world.new_zeros(())

    bone_loss_finger = compute_bone_length_loss(pred_world, target_world, bone_idx_finger, normalize=True) \
        if lambda_bone > 0 else zero
    bone_loss_palm = compute_bone_length_loss(pred_world, target_world, bone_idx_palm, normalize=True) \
        if lambda_palm > 0 else zero
    velocity_loss = compute_velocity_loss(pred_world, target_world) \
        if lambda_vel > 0 else zero
    acceleration_loss = compute_acceleration_loss(pred_world, target_world) \
        if lambda_acc > 0 else zero
    angle_loss = compute_joint_angle_consistency_loss(pred_world, target_world, angle_triplets) \
        if lambda_ang > 0 else zero

    mg_loss = (
        lambda_bone * bone_loss_finger
        + lambda_palm * bone_loss_palm
        + lambda_vel * velocity_loss
        + lambda_acc * acceleration_loss
        + lambda_ang * angle_loss
    )

    metrics = {
        "bone_loss_finger": float(bone_loss_finger.item()) if isinstance(bone_loss_finger, torch.Tensor) else 0.0,
        "bone_loss_palm": float(bone_loss_palm.item()) if isinstance(bone_loss_palm, torch.Tensor) else 0.0,
        "velocity_loss": float(velocity_loss.item()) if isinstance(velocity_loss, torch.Tensor) else 0.0,
        "acceleration_loss": float(acceleration_loss.item()) if isinstance(acceleration_loss, torch.Tensor) else 0.0,
        "angle_loss": float(angle_loss.item()) if isinstance(angle_loss, torch.Tensor) else 0.0,
        "motion_geometry_loss": float(mg_loss.item()) if isinstance(mg_loss, torch.Tensor) else 0.0,
    }
    return mg_loss, metrics
