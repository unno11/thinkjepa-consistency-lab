from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.kinematic_projection import (
    REQUIRED_PROTOCOL,
    load_projection_protocol,
    project_prediction_median_bone_lengths,
)
from evaluation.metrics import compute_trajectory_metrics
from evaluation.topology import ALL_BONES, JOINT_INDEX, KINEMATIC_TREE_EDGES


def prediction(frames=5):
    rng = np.random.default_rng(7)
    result = rng.normal(size=(frames, 52, 3))
    result *= np.linspace(0.5, 1.5, frames)[:, None, None]
    return result


def bone_lengths(value):
    edges = np.asarray(ALL_BONES, dtype=np.int64)
    return np.linalg.norm(
        value[:, edges[:, 1]] - value[:, edges[:, 0]], axis=-1
    )


class KinematicProjectionTests(unittest.TestCase):
    def test_topological_edges_cover_metric_bones(self):
        self.assertEqual(len(KINEMATIC_TREE_EDGES), 50)
        self.assertEqual(set(KINEMATIC_TREE_EDGES), set(ALL_BONES))

    def test_projection_freezes_each_bone_length(self):
        value = prediction()
        projected = project_prediction_median_bone_lengths(value)
        lengths = bone_lengths(projected)
        expected = np.median(bone_lengths(value), axis=0)
        self.assertTrue(np.allclose(lengths, expected[None], atol=1e-10))

    def test_projection_preserves_forearm_roots(self):
        value = prediction()
        projected = project_prediction_median_bone_lengths(value)
        for side in ("right", "left"):
            index = JOINT_INDEX[f"{side}Forearm"]
            self.assertTrue(np.array_equal(projected[:, index], value[:, index]))

    def test_projection_preserves_bone_angles(self):
        value = prediction()
        target = prediction() + np.linspace(-0.2, 0.3, 52)[None, :, None]
        projected = project_prediction_median_bone_lengths(value)
        before = compute_trajectory_metrics(value, target)["AngleError_deg"]
        after = compute_trajectory_metrics(projected, target)["AngleError_deg"]
        self.assertAlmostEqual(before, after, places=10)

    def test_zero_vectors_remain_finite(self):
        value = np.zeros((4, 52, 3), dtype=np.float64)
        projected = project_prediction_median_bone_lengths(value)
        self.assertTrue(np.isfinite(projected).all())
        self.assertTrue(np.array_equal(projected, value))

    def test_rejects_nonfinite_and_wrong_shape(self):
        with self.assertRaises(ValueError):
            project_prediction_median_bone_lengths(np.zeros((4, 51, 3)))
        value = np.zeros((4, 52, 3))
        value[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            project_prediction_median_bone_lengths(value)

    def test_protocol_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(REQUIRED_PROTOCOL), encoding="utf-8")
            self.assertEqual(load_projection_protocol(path), REQUIRED_PROTOCOL)
            bad = dict(REQUIRED_PROTOCOL)
            bad["future_ground_truth_access"] = True
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_projection_protocol(path)


if __name__ == "__main__":
    unittest.main()
