from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.direct32_protocol import (
    load_protocol,
    resolve_validation,
    validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "direct32_best_validation_protocol.json"
ALLLAYERS_B_PROTOCOL_PATH = (
    ROOT / "configs" / "direct32_B_alllayers_gate0fixed_validation_v1.json"
)
ALLLAYERS_B_MG_PROTOCOL_PATH = (
    ROOT / "configs" / "direct32_B_MG_alllayers_gate0fixed_validation_v1.json"
)
ALLLAYERS_MULTISEED_PROTOCOL_PATHS = {
    42: ROOT / "configs" / "direct32_B_MG_alllayers_gate0fixed_seed42_validation_v2.json",
    43: ROOT / "configs" / "direct32_B_MG_alllayers_gate0fixed_seed43_validation_v1.json",
    44: ROOT / "configs" / "direct32_B_MG_alllayers_gate0fixed_seed44_validation_v1.json",
}


class Direct32ProtocolTests(unittest.TestCase):
    def test_frozen_identity(self):
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["rollout"]["past_frames"], 32)
        self.assertEqual(protocol["rollout"]["future_frames"], 32)
        self.assertEqual(protocol["rollout"]["ref_mode"], "none")
        self.assertEqual(protocol["rollout"]["vlm_source"], "both")
        self.assertEqual(protocol["rollout"]["vlm_layer_selector"], "last")
        self.assertIs(protocol["rollout"]["inference_amp"], False)
        manifest, spec = resolve_validation(protocol, PROTOCOL_PATH)
        self.assertEqual(manifest.name, "val.txt")
        self.assertEqual(spec["expected_samples"], 200)

    def test_role_epochs_are_frozen(self):
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(
            {role: spec["expected_epoch"] for role, spec in protocol["roles"].items()},
            {"B": 69, "M": 93, "G": 80, "MG": 93},
        )

    def test_gate0fixed_alllayers_B_identity_is_frozen(self):
        protocol = load_protocol(ALLLAYERS_B_PROTOCOL_PATH)
        self.assertEqual(protocol["rollout"]["vlm_layer_selector"], "all")
        self.assertEqual(set(protocol["roles"]), {"B"})
        self.assertEqual(protocol["roles"]["B"]["expected_epoch"], 69)
        manifest, spec = resolve_validation(protocol, ALLLAYERS_B_PROTOCOL_PATH)
        self.assertEqual(manifest.name, "val.txt")
        self.assertEqual(spec["expected_samples"], 200)

    def test_gate0fixed_alllayers_B_MG_identity_is_frozen(self):
        protocol = load_protocol(ALLLAYERS_B_MG_PROTOCOL_PATH)
        self.assertEqual(protocol["rollout"]["vlm_layer_selector"], "all")
        self.assertEqual(set(protocol["roles"]), {"B", "MG"})
        self.assertEqual(protocol["roles"]["B"]["expected_epoch"], 69)
        self.assertEqual(protocol["roles"]["MG"]["expected_epoch"], 99)
        self.assertEqual(protocol["roles"]["MG"]["weights"]["bone"], 0.0006)
        self.assertEqual(protocol["roles"]["MG"]["weights"]["velocity"], 0.45)

    def test_gate0fixed_multiseed_epochs_and_seeds_are_frozen(self):
        expected_epochs = {
            42: {"B": 69, "MG": 99},
            43: {"B": 76, "MG": 99},
            44: {"B": 93, "MG": 97},
        }
        for seed, path in ALLLAYERS_MULTISEED_PROTOCOL_PATHS.items():
            protocol = load_protocol(path)
            self.assertEqual(set(protocol["roles"]), {"B", "MG"})
            self.assertEqual(protocol["rollout"]["vlm_layer_selector"], "all")
            for role in ("B", "MG"):
                self.assertEqual(
                    protocol["roles"][role]["expected_epoch"],
                    expected_epochs[seed][role],
                )
                self.assertEqual(protocol["roles"][role]["expected_seed"], seed)

    def test_checkpoint_seed_must_match_when_protocol_freezes_it(self):
        protocol = load_protocol(ALLLAYERS_MULTISEED_PROTOCOL_PATHS[43])
        args = {
            "seed": 44,
            "predictor": "thinkjepa",
            "trajmode": "traj",
            "joint_pred": True,
            "ref_mode": "none",
            "temporal_stride": 1,
            "thinkjepa_vlm_source": "both",
            "thinkjepa_vlm_layer_selector": "all",
            "thinkjepa_vlm_cond_mode": "film",
            "lambda_bone_loss": 0.0,
            "lambda_palm_bone_loss": 0.0,
            "lambda_velocity_loss": 0.0,
            "lambda_acceleration_loss": 0.0,
            "lambda_angle_loss": 0.0,
            "skip_nonfinite_loss": False,
        }
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_checkpoint(protocol, "B", {"epoch": 76}, args)

    def test_unknown_selector_and_empty_roles_are_rejected(self):
        data = json.loads(ALLLAYERS_B_PROTOCOL_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            bad_selector = copy.deepcopy(data)
            bad_selector["rollout"]["vlm_layer_selector"] = "adaptive"
            path.write_text(json.dumps(bad_selector), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vlm_layer_selector"):
                load_protocol(path)

            empty_roles = copy.deepcopy(data)
            empty_roles["roles"] = {}
            path.write_text(json.dumps(empty_roles), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty subset"):
                load_protocol(path)

    def test_checkpoint_role_and_weights(self):
        protocol = load_protocol(PROTOCOL_PATH)
        args = {
            "predictor": "thinkjepa",
            "trajmode": "traj",
            "joint_pred": True,
            "ref_mode": "none",
            "temporal_stride": 1,
            "thinkjepa_vlm_source": "both",
            "thinkjepa_vlm_layer_selector": "last",
            "thinkjepa_vlm_cond_mode": "film",
            "lambda_bone_loss": 0.0006,
            "lambda_palm_bone_loss": 0.0,
            "lambda_velocity_loss": 0.45,
            "lambda_acceleration_loss": 0.2,
            "lambda_angle_loss": 0.0035,
            "skip_nonfinite_loss": False,
        }
        validate_checkpoint(protocol, "MG", {"epoch": 93}, args)
        with self.assertRaisesRegex(ValueError, "epoch"):
            validate_checkpoint(protocol, "MG", {"epoch": 100}, args)
        bad = dict(args, lambda_angle_loss=0.0)
        with self.assertRaisesRegex(ValueError, "lambda_angle_loss"):
            validate_checkpoint(protocol, "MG", {"epoch": 93}, bad)

    def test_protocol_mutation_is_rejected(self):
        data = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        bad = copy.deepcopy(data)
        bad["rollout"]["ref_mode"] = "perhand"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ref_mode"):
                load_protocol(path)

    def test_evaluator_has_no_test_or_reference_offset_path(self):
        source = (ROOT / "evaluation" / "evaluate_direct32.py").read_text(
            encoding="utf-8"
        )
        forbidden_offset = "append" + "_hand_reference_offsets("
        self.assertNotIn(forbidden_offset, source)
        self.assertNotIn("final-test", source)
        self.assertIn("direct32 requires observation-window VLM guidance", source)


if __name__ == "__main__":
    unittest.main()
