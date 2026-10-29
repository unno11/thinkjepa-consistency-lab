from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.r32_protocol import (
    load_protocol,
    resolve_split,
    validate_checkpoint_args,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r32_evaluation_protocol.json"


class R32ProtocolTests(unittest.TestCase):
    def test_frozen_geometry_and_validation_identity(self):
        protocol = load_protocol(PROTOCOL_PATH)
        rollout = protocol["rollout"]
        self.assertEqual(rollout["past_frames"], 4)
        self.assertEqual(rollout["step_future_frames"], 4)
        self.assertEqual(rollout["steps"], 8)
        self.assertEqual(rollout["horizons"], [4, 8, 16, 32])
        self.assertEqual(rollout["ref_mode"], "none")
        self.assertIs(rollout["inference_amp"], False)
        self.assertEqual(
            rollout["vlm_guidance_policy"], "disabled_no_causal_cache"
        )
        self.assertEqual(
            rollout["future_camera_extrinsics"], "coordinate_transform_only"
        )
        manifest, spec = resolve_split(protocol, PROTOCOL_PATH, "validation")
        self.assertEqual(manifest.name, "val.txt")
        self.assertEqual(spec["expected_samples"], 200)

    def test_final_test_is_locked(self):
        protocol = load_protocol(PROTOCOL_PATH)
        with self.assertRaises(PermissionError):
            resolve_split(protocol, PROTOCOL_PATH, "final-test")
        # This public validation release deliberately omits final-test membership.
        # A valid unlock token must not invent or silently substitute a manifest.
        self.assertFalse((ROOT / "protocol" / "splits" / "test.txt").exists())
        with self.assertRaises(FileNotFoundError):
            resolve_split(
                protocol, PROTOCOL_PATH, "final-test", "RUN_LOCKED_FINAL_TEST_ONCE"
            )

    def test_protocol_mutation_is_rejected(self):
        data = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        data["rollout"]["ref_mode"] = "perhand"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ref_mode"):
                load_protocol(path)

    def test_checkpoint_identity_must_match(self):
        rollout = load_protocol(PROTOCOL_PATH)["rollout"]
        good = {
            "predictor": "thinkjepa", "trajmode": "traj", "joint_pred": True,
            "ref_mode": "none", "temporal_stride": 1,
        }
        validate_checkpoint_args(good, rollout)
        bad = dict(good, ref_mode="perhand")
        with self.assertRaisesRegex(ValueError, "ref_mode"):
            validate_checkpoint_args(bad, rollout)

    def test_evaluator_never_uses_future_joint_reference_offsets(self):
        source = (ROOT / "evaluation" / "evaluate_r32.py").read_text(encoding="utf-8")
        forbidden_call = "append" + "_hand_reference_offsets("
        self.assertNotIn(forbidden_call, source)
        self.assertIn("guidance = None", source)


if __name__ == "__main__":
    unittest.main()
