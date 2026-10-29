"""Canonical trajectory metrics, implemented independently in NumPy.

Inputs are metric-space joint positions shaped ``[time, 52, xyz]``.  Metrics are
computed per sample; callers must average these per-sample values across a split.
"""

from __future__ import annotations

import numpy as np

from .topology import (
    ALL_BONES, ANGLE_TRIPLETS, FINGER_BONES, FINGER_JOINT_INDICES, LEFT_HAND_INDEX, PALM_BONES,
    RIGHT_HAND_INDEX,
)


def _validate(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (52, 3):
        raise ValueError(f"expected equal [T,52,3] arrays, got {pred.shape} and {target.shape}")
    if pred.shape[0] < 3:
        raise ValueError("at least three frames are required for acceleration metrics")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("trajectory contains non-finite values")
    return pred, target


def _root(x):
    return 0.5 * (x[:, RIGHT_HAND_INDEX] + x[:, LEFT_HAND_INDEX])


def _mean_motion_magnitude(x, order):
    return float(np.linalg.norm(np.diff(x, n=order, axis=0), axis=-1).mean())


def _bone_lengths(x, pairs):
    pairs = np.asarray(pairs, dtype=np.int64)
    return np.linalg.norm(x[:, pairs[:, 1]] - x[:, pairs[:, 0]], axis=-1)


def _bone_error_mm(pred, target, pairs):
    return float(np.abs(_bone_lengths(pred, pairs) - _bone_lengths(target, pairs)).mean() * 1000.0)


def _angles(x):
    triplets = np.asarray(ANGLE_TRIPLETS, dtype=np.int64)
    a = x[:, triplets[:, 0]] - x[:, triplets[:, 1]]
    b = x[:, triplets[:, 2]] - x[:, triplets[:, 1]]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = denom > 1e-12
    cosine = np.zeros_like(denom)
    cosine[valid] = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
    angle = np.zeros_like(denom)
    angle[valid] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    return angle, valid


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def compute_trajectory_metrics(pred, target):
    """Return canonical metrics for one predicted future trajectory.

    Distances use metres except explicitly suffixed ``_mm``.  Temporal derivatives
    are per frame (the protocol freezes temporal stride to one).
    """
    pred, target = _validate(pred, target)
    error = np.linalg.norm(pred - target, axis=-1)
    pred_root, target_root = _root(pred), _root(target)
    pred_aligned = pred - pred_root[:, None, :]
    target_aligned = target - target_root[:, None, :]
    aligned_error = np.linalg.norm(pred_aligned - target_aligned, axis=-1)
    pred_angle, pred_valid = _angles(pred)
    target_angle, target_valid = _angles(target)
    angle_valid = pred_valid & target_valid
    angle_error = np.abs(pred_angle - target_angle)[angle_valid]
    all_lengths = _bone_lengths(pred, ALL_BONES)

    return {
        "ADE_m": float(error.mean()),
        # Last TIME step, not the last joint.  The archived evaluator used err[:, -1].
        "FDE_m": float(error[-1].mean()),
        "RootError_m": float(np.linalg.norm(pred_root - target_root, axis=-1).mean()),
        "RootAligned_ADE_m": float(aligned_error.mean()),
        "RootAligned_Finger_ADE_m": float(aligned_error[:, FINGER_JOINT_INDICES].mean()),
        "VelocityRatio": _safe_ratio(_mean_motion_magnitude(pred, 1), _mean_motion_magnitude(target, 1)),
        "AccelerationRatio": _safe_ratio(_mean_motion_magnitude(pred, 2), _mean_motion_magnitude(target, 2)),
        "BoneErrorAll_mm": _bone_error_mm(pred, target, ALL_BONES),
        "BoneErrorFinger_mm": _bone_error_mm(pred, target, FINGER_BONES),
        "BoneErrorPalm_mm": _bone_error_mm(pred, target, PALM_BONES),
        "BoneDriftAll_mm2": float(np.var(all_lengths, axis=0).mean() * 1_000_000.0),
        "AngleError_deg": float(np.degrees(angle_error).mean()) if angle_error.size else float("nan"),
    }
