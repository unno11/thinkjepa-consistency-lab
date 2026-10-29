import unittest

import numpy as np

from evaluation.metrics import compute_trajectory_metrics
from evaluation.evaluate_naive_baselines import constant_velocity, last_pose
from evaluation.topology import ALL_BONES, FINGER_JOINT_INDICES, JOINT_NAMES


def trajectory(frames=5):
    joint = np.arange(52, dtype=np.float64)
    base = np.stack((0.01 * joint, 0.002 * (joint % 5), 0.003 * (joint % 7)), axis=-1)
    time = np.arange(frames, dtype=np.float64)[:, None, None]
    velocity = np.array([0.01, -0.005, 0.002])[None, None, :]
    return base[None, :, :] + time * velocity


class EvaluationMetricTests(unittest.TestCase):
    def test_frozen_topology(self):
        self.assertEqual(len(JOINT_NAMES), 52)
        self.assertEqual(len(ALL_BONES), 50)
        self.assertEqual(len(FINGER_JOINT_INDICES), 48)

    def test_identical_trajectory(self):
        target = trajectory()
        metrics = compute_trajectory_metrics(target.copy(), target)
        for key in ("ADE_m", "FDE_m", "RootError_m", "RootAligned_ADE_m",
                    "BoneErrorAll_mm", "BoneDriftAll_mm2", "AngleError_deg"):
            self.assertAlmostEqual(metrics[key], 0.0, places=12)
        self.assertAlmostEqual(metrics["VelocityRatio"], 1.0, places=12)

    def test_fde_selects_last_frame_not_last_joint(self):
        target = trajectory()
        pred = target.copy()
        pred[-1, :, 0] += 1.0
        metrics = compute_trajectory_metrics(pred, target)
        self.assertAlmostEqual(metrics["FDE_m"], 1.0, places=12)
        self.assertAlmostEqual(metrics["ADE_m"], 0.2, places=12)

    def test_root_alignment_removes_translation(self):
        target = trajectory()
        pred = target + np.array([0.3, -0.4, 0.0])
        metrics = compute_trajectory_metrics(pred, target)
        self.assertAlmostEqual(metrics["ADE_m"], 0.5, places=12)
        self.assertAlmostEqual(metrics["RootError_m"], 0.5, places=12)
        self.assertAlmostEqual(metrics["RootAligned_ADE_m"], 0.0, places=12)

    def test_motion_ratios(self):
        target = trajectory()
        pred = target[0:1] + 2.0 * (target - target[0:1])
        metrics = compute_trajectory_metrics(pred, target)
        self.assertAlmostEqual(metrics["VelocityRatio"], 2.0, places=12)
        # Both trajectories have zero acceleration, so the ratio is undefined.
        self.assertTrue(np.isnan(metrics["AccelerationRatio"]))

    def test_rejects_bad_input(self):
        target = trajectory()
        with self.assertRaises(ValueError):
            compute_trajectory_metrics(target[:, :51], target[:, :51])
        bad = target.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_trajectory_metrics(bad, target)

    def test_context_only_controls(self):
        context = trajectory(frames=32)
        self.assertTrue(np.array_equal(last_pose(context, 4)[0], context[-1]))
        expected = trajectory(frames=36)[32:]
        self.assertTrue(np.allclose(constant_velocity(context, 4, 2), expected))
        self.assertTrue(np.allclose(constant_velocity(context, 4, 4), expected))


if __name__ == "__main__":
    unittest.main()
