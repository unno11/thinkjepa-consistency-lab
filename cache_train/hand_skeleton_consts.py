# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""
Hand / joint constants and masks for egodex pipeline.
Keep this file import-only: no heavy side effects.
"""

from __future__ import annotations

import torch

# Pull canonical finger/joint names from your skeleton defs
from egodex.utils.skeleton_tfs import (
    LEFT_FINGERS,
    LEFT_INDEX,
    LEFT_LITTLE,
    LEFT_MIDDLE,
    LEFT_RING,
    LEFT_THUMB,
    RIGHT_FINGERS,
    RIGHT_INDEX,
    RIGHT_LITTLE,
    RIGHT_MIDDLE,
    RIGHT_RING,
    RIGHT_THUMB,
)

# -------------------------
# Names / ordering
# -------------------------
QUERY_TFS = (
    RIGHT_FINGERS
    + ["rightHand", "rightForearm"]
    + LEFT_FINGERS
    + ["leftHand", "leftForearm"]
)

tf2idx = {k: i for i, k in enumerate(QUERY_TFS)}
JOINT_NAMES = QUERY_TFS
J_TOTAL = len(JOINT_NAMES)

# -------------------------
# Left/right split & masks
# -------------------------
n_right = len(RIGHT_FINGERS) + 2  # rightHand + rightForearm
n_left = len(LEFT_FINGERS) + 2  # leftHand + leftForearm

RIGHT_JOINT_MASK = [True] * n_right + [False] * n_left
LEFT_JOINT_MASK = [False] * n_right + [True] * n_left

# Tensors (1,1,J,1) for easy broadcast with (B,T,J,C) shapes
RIGHT_JOINT_MASK_T = torch.tensor(RIGHT_JOINT_MASK, dtype=torch.float32).view(
    1, 1, -1, 1
)
LEFT_JOINT_MASK_T = torch.tensor(LEFT_JOINT_MASK, dtype=torch.float32).view(1, 1, -1, 1)

# -------------------------
# Finger group dicts
# -------------------------
right_dict = {
    "index": RIGHT_INDEX,
    "thumb": RIGHT_THUMB,
    "middle": RIGHT_MIDDLE,
    "ring": RIGHT_RING,
    "little": RIGHT_LITTLE,
}
left_dict = {
    "index": LEFT_INDEX,
    "thumb": LEFT_THUMB,
    "middle": LEFT_MIDDLE,
    "ring": LEFT_RING,
    "little": LEFT_LITTLE,
}

# -------------------------
# Image normalization defaults
# -------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# -------------------------
# Misc
# -------------------------
visualize_flag = True

# Useful indices
right_idx = tf2idx["rightMiddleFingerMetacarpal"]
left_idx = tf2idx["leftMiddleFingerMetacarpal"]

# -------------------------
# Skeleton parent-child pairs (auto-built from QUERY_TFS)
# -------------------------

# Finger-internal bone chains: joint[i] → joint[i+1] within each digit
_FINGER_JOINTS = ["Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
_THUMB_JOINTS = ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
_FINGER_NAMES = ["LittleFinger", "RingFinger", "MiddleFinger", "IndexFinger"]
_THUMB_NAME = "Thumb"
_HAND_SIDES = [
    ("right", "rightHand", "rightForearm"),
    ("left", "leftHand", "leftForearm"),
]


def _build_joint_name(side_prefix: str, digit: str, joint: str) -> str:
    if digit == "Thumb":
        return f"{side_prefix}Thumb{joint}"
    return f"{side_prefix}{digit}{joint}"


def build_skeleton_pairs():
    """Auto-construct skeleton parent-child pairs from QUERY_TFS naming convention.

    Returns:
        finger_internal: List[Tuple[str, str]] — bones entirely within each digit
        palm_to_finger:  List[Tuple[str, str]] — hand wrist → first digit joint
        forearm_to_hand: List[Tuple[str, str]] — forearm → hand wrist
        angle_triplets:  List[Tuple[str, str, str]] — (parent, joint, child)
        all_bones:       List[Tuple[str, str]] — union of the three bone groups
    """
    valid_names = set(QUERY_TFS)

    def _must_exist(name: str) -> str:
        if name not in valid_names:
            raise KeyError(
                f"Joint name '{name}' not found in QUERY_TFS ({len(QUERY_TFS)} joints). "
                f"Check skeleton naming convention."
            )
        return name

    finger_internal: list = []
    palm_to_finger: list = []
    forearm_to_hand: list = []
    angle_triplets: list = []

    for side_prefix, hand_name, forearm_name in _HAND_SIDES:
        hand = _must_exist(hand_name)
        forearm = _must_exist(forearm_name)

        # forearm → hand
        forearm_to_hand.append((forearm_name, hand_name))

        # Regular fingers
        for finger in _FINGER_NAMES:
            joint_names = [_build_joint_name(side_prefix, finger, j) for j in _FINGER_JOINTS]
            for jn in joint_names:
                _must_exist(jn)

            # palm → finger root
            palm_to_finger.append((hand_name, joint_names[0]))  # Hand → Metacarpal

            # finger internal chain
            for i in range(len(joint_names) - 1):
                finger_internal.append((joint_names[i], joint_names[i + 1]))

            # Angle triplets at internal joints (need parent, self, child)
            for i in range(1, len(joint_names) - 1):
                angle_triplets.append((joint_names[i - 1], joint_names[i], joint_names[i + 1]))

            # Angle at Metacarpal (Hand → Metacarpal → Knuckle)
            angle_triplets.append((hand_name, joint_names[0], joint_names[1]))

        # Thumb
        thumb_joints = [_build_joint_name(side_prefix, _THUMB_NAME, j) for j in _THUMB_JOINTS]
        for jn in thumb_joints:
            _must_exist(jn)

        # palm → thumb root
        palm_to_finger.append((hand_name, thumb_joints[0]))  # Hand → Knuckle

        # thumb internal chain
        for i in range(len(thumb_joints) - 1):
            finger_internal.append((thumb_joints[i], thumb_joints[i + 1]))

        # Angle triplets for thumb
        for i in range(1, len(thumb_joints) - 1):
            angle_triplets.append((thumb_joints[i - 1], thumb_joints[i], thumb_joints[i + 1]))

        # Angle at thumb Knuckle (Hand → Knuckle → IntermediateBase)
        angle_triplets.append((hand_name, thumb_joints[0], thumb_joints[1]))

        # Angle at Hand (Forearm → Hand → each finger root)
        for finger in _FINGER_NAMES:
            root = _build_joint_name(side_prefix, finger, _FINGER_JOINTS[0])
            angle_triplets.append((forearm_name, hand_name, root))
        # Forearm → Hand → thumb root
        angle_triplets.append((forearm_name, hand_name, thumb_joints[0]))

    all_bones = finger_internal + palm_to_finger + forearm_to_hand

    return {
        "finger_internal": finger_internal,
        "palm_to_finger": palm_to_finger,
        "forearm_to_hand": forearm_to_hand,
        "all_bones": all_bones,
        "angle_triplets": angle_triplets,
    }


def build_bone_index_tensors():
    """Convert skeleton pair name tuples to index tensors (GPU-ready).

    Returns a dict:
        "finger_internal":   [K_fi, 2] int64
        "palm_to_finger":    [K_pf, 2] int64
        "forearm_to_hand":   [K_fh, 2] int64
        "all_bones":         [K_all, 2] int64
        "angle_triplets":    [M, 3] int64
        "pair_counts":       dict with counts per group
        "debug_map":         list of (parent_name, child_name, parent_idx, child_idx) for first 10 all_bones
    """
    import torch

    pairs = build_skeleton_pairs()
    result: dict = {}
    result["pair_counts"] = {}

    for group_key in ["finger_internal", "palm_to_finger", "forearm_to_hand", "all_bones"]:
        name_pairs = pairs[group_key]
        indices = []
        for p_name, c_name in name_pairs:
            p_idx = tf2idx[p_name]
            c_idx = tf2idx[c_name]
            indices.append([p_idx, c_idx])
        result[group_key] = (
            torch.tensor(indices, dtype=torch.int64) if indices
            else torch.empty((0, 2), dtype=torch.int64)
        )
        result["pair_counts"][group_key] = len(indices)

    # Angle triplets
    angle_indices = []
    for p_name, j_name, c_name in pairs["angle_triplets"]:
        angle_indices.append([tf2idx[p_name], tf2idx[j_name], tf2idx[c_name]])
    result["angle_triplets"] = (
        torch.tensor(angle_indices, dtype=torch.int64) if angle_indices
        else torch.empty((0, 3), dtype=torch.int64)
    )
    result["pair_counts"]["angle_triplets"] = len(angle_indices)

    # Debug map: first 10 all_bones with names
    all_pairs = pairs["all_bones"]
    result["debug_map"] = [
        (all_pairs[i][0], all_pairs[i][1], tf2idx[all_pairs[i][0]], tf2idx[all_pairs[i][1]])
        for i in range(min(10, len(all_pairs)))
    ]

    return result


# Pre-built skeleton data (import-time, lazy-safe: only depends on QUERY_TFS and tf2idx)
_SKELETON_DATA = None


def get_skeleton_data():
    """Lazy-load skeleton pair index tensors (shared across all callers)."""
    global _SKELETON_DATA
    if _SKELETON_DATA is None:
        _SKELETON_DATA = build_bone_index_tensors()
    return _SKELETON_DATA


__all__ = [
    # names / indices
    "QUERY_TFS",
    "tf2idx",
    "JOINT_NAMES",
    "J_TOTAL",
    # masks
    "RIGHT_JOINT_MASK",
    "LEFT_JOINT_MASK",
    "RIGHT_JOINT_MASK_T",
    "LEFT_JOINT_MASK_T",
    # groups
    "right_dict",
    "left_dict",
    # imagenet stats
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    # misc
    "visualize_flag",
    "right_idx",
    "left_idx",
    # raw lists (in case callers need them)
    "RIGHT_FINGERS",
    "LEFT_FINGERS",
    "RIGHT_INDEX",
    "RIGHT_THUMB",
    "RIGHT_MIDDLE",
    "RIGHT_RING",
    "RIGHT_LITTLE",
    "LEFT_INDEX",
    "LEFT_THUMB",
    "LEFT_MIDDLE",
    "LEFT_RING",
    "LEFT_LITTLE",
    # skeleton pairs
    "build_skeleton_pairs",
    "build_bone_index_tensors",
    "get_skeleton_data",
]
