from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache_train"))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import physics_losses as losses
    import hand_skeleton_consts as skeleton


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local smoke environment")
class PhysicsLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.target = torch.randn(2, 5, 4, 3)
        self.bones = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.palms = torch.tensor([[0, 3]], dtype=torch.long)
        self.angles = torch.tensor([[0, 1, 2]], dtype=torch.long)

    def test_identical_prediction_has_zero_losses(self) -> None:
        prediction = self.target.clone().requires_grad_(True)
        values = (
            losses.compute_bone_length_loss(prediction, self.target, self.bones),
            losses.compute_velocity_loss(prediction, self.target),
            losses.compute_acceleration_loss(prediction, self.target),
            losses.compute_jerk_loss(prediction, self.target),
            losses.compute_joint_angle_consistency_loss(
                prediction, self.target, self.angles),
        )
        for value in values:
            self.assertAlmostEqual(value.item(), 0.0, places=7)

    def test_perturbation_is_positive_and_backpropagates_only_to_prediction(self) -> None:
        target = self.target.clone().requires_grad_(True)
        offset = torch.linspace(0.0, 0.4, 5).view(1, 5, 1, 1)
        prediction = (target.detach() + offset).clone().requires_grad_(True)
        total = (losses.compute_velocity_loss(prediction, target)
                 + losses.compute_acceleration_loss(prediction, target))
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(total.item(), 0.0)
        total.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(target.grad)

    def test_short_sequences_return_scalar_zero(self) -> None:
        one = self.target[:, :1].clone().requires_grad_(True)
        two = self.target[:, :2].clone().requires_grad_(True)
        three = self.target[:, :3].clone().requires_grad_(True)
        self.assertEqual(losses.compute_velocity_loss(one, one).item(), 0.0)
        self.assertEqual(losses.compute_acceleration_loss(two, two).item(), 0.0)
        self.assertEqual(losses.compute_jerk_loss(three, three).item(), 0.0)

    def test_aggregator_and_warmup(self) -> None:
        class Args:
            lambda_bone_loss = 1.0
            lambda_palm_bone_loss = 0.5
            lambda_velocity_loss = 1.0
            lambda_acceleration_loss = 1.0
            lambda_angle_loss = 0.25
            physics_loss_start_epoch = 2
            physics_loss_warmup_epochs = 4

        prediction = self.target.clone().requires_grad_(True)
        total, metrics = losses.compute_motion_geometry_losses(
            prediction, self.target, self.bones, self.palms, self.angles, Args())
        self.assertEqual(total.item(), 0.0)
        self.assertEqual(set(metrics), {
            "bone_loss_finger", "bone_loss_palm", "velocity_loss",
            "acceleration_loss", "angle_loss", "motion_geometry_loss"})
        self.assertEqual(losses.get_physics_loss_weight_scale(1, Args()), 0.0)
        self.assertEqual(losses.get_physics_loss_weight_scale(2, Args()), 0.25)
        self.assertEqual(losses.get_physics_loss_weight_scale(5, Args()), 1.0)

    def test_52_joint_skeleton_indices_are_complete_and_valid(self) -> None:
        data = skeleton.get_skeleton_data()
        self.assertEqual(data["pair_counts"], {
            "finger_internal": 38,
            "palm_to_finger": 10,
            "forearm_to_hand": 2,
            "all_bones": 50,
            "angle_triplets": 48,
        })
        for name in ("finger_internal", "palm_to_finger", "forearm_to_hand",
                     "all_bones", "angle_triplets"):
            indices = data[name]
            self.assertGreaterEqual(indices.min().item(), 0)
            self.assertLess(indices.max().item(), 52)


if torch is None and os.environ.get("THINKJEPA_REQUIRE_TORCH") == "1":
    raise RuntimeError("THINKJEPA_REQUIRE_TORCH=1 but PyTorch cannot be imported")


if __name__ == "__main__":
    unittest.main()
