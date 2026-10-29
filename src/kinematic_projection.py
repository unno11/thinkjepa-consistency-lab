"""Target-free kinematic projection control for 52-joint trajectories.

The control uses only the model's own predicted 32-frame trajectory.  For each
bone, it freezes the length to the temporal median predicted length while
retaining the predicted direction at every frame.  It never reads observed or
future ground-truth joints and does not mix samples in a batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .topology import KINEMATIC_TREE_EDGES, LEFT_HAND_INDEX, RIGHT_HAND_INDEX


REQUIRED_PROTOCOL = {
    "schema_version": 1,
    "method": "KProjPredMedian",
    "reference_length_source": "temporal_median_of_own_prediction",
    "direction_source": "same_frame_own_prediction",
    "root_policy": "preserve_predicted_forearm_roots",
    "future_ground_truth_access": False,
    "observed_ground_truth_access": False,
    "cross_sample_operations": False,
}


def load_projection_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in REQUIRED_PROTOCOL.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"kinematic projection {key} must equal {expected!r}"
            )
    return payload


def _validate_prediction(prediction) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[1:] != (52, 3):
        raise ValueError(
            f"expected a [T,52,3] prediction, got {prediction.shape}"
        )
    if prediction.shape[0] < 1:
        raise ValueError("prediction must contain at least one frame")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    return prediction


def project_prediction_median_bone_lengths(
    prediction, *, epsilon: float = 1e-12
) -> np.ndarray:
    """Return a target-free, constant-bone-length projection.

    The right and left forearm joints are the two tree roots and remain exactly
    at their predicted positions.  A degenerate zero-length direction falls
    back to the first non-degenerate direction for that bone within the same
    predicted sample; if the bone is degenerate for every frame, a deterministic
    unit x direction is used with a zero reference length.
    """
    prediction = _validate_prediction(prediction)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    projected = prediction.copy()
    edges = np.asarray(KINEMATIC_TREE_EDGES, dtype=np.int64)
    original_vectors = prediction[:, edges[:, 1]] - prediction[:, edges[:, 0]]
    original_lengths = np.linalg.norm(original_vectors, axis=-1)
    reference_lengths = np.median(original_lengths, axis=0)

    for edge_index, (parent, child) in enumerate(KINEMATIC_TREE_EDGES):
        vectors = original_vectors[:, edge_index]
        lengths = original_lengths[:, edge_index]
        valid = lengths > epsilon
        directions = np.empty_like(vectors)
        if np.any(valid):
            fallback = vectors[np.flatnonzero(valid)[0]] / lengths[valid][0]
        else:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        directions[:] = fallback
        directions[valid] = vectors[valid] / lengths[valid, None]
        projected[:, child] = (
            projected[:, parent]
            + directions * reference_lengths[edge_index]
        )

    # These indices are not roots themselves; the assertions document that the
    # hand nodes are descendants and prevent accidental topology drift.
    assert RIGHT_HAND_INDEX in {child for _, child in KINEMATIC_TREE_EDGES}
    assert LEFT_HAND_INDEX in {child for _, child in KINEMATIC_TREE_EDGES}
    return projected
