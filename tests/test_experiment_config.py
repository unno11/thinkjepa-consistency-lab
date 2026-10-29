from __future__ import annotations

import importlib.util
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ExperimentConfigTests(unittest.TestCase):

    def test_formal_mg_matrix_differs_only_in_declared_weights(self):
        roles = ("B", "M", "G", "MG")
        configs = {
            role: RUNNER.load_and_validate_config(
                ROOT / "configs" / f"thinkjepa_{role}_alllayers_100ep.json"
            )
            for role in roles
        }
        weight_keys = {
            "lambda_bone_loss", "lambda_palm_bone_loss",
            "lambda_velocity_loss", "lambda_acceleration_loss",
            "lambda_angle_loss",
        }
        common = {
            role: {key: value for key, value in config["trainer_args"].items()
                   if key not in weight_keys}
            for role, config in configs.items()
        }
        self.assertTrue(all(common[role] == common["B"] for role in roles))
        self.assertEqual(common["B"]["epochs"], 100)
        self.assertEqual(common["B"]["physics_loss_start_epoch"], 2)
        self.assertEqual(common["B"]["physics_loss_warmup_epochs"], 3)
        self.assertIs(common["B"]["skip_nonfinite_loss"], False)
        for role, config in configs.items():
            protocol = config["protocol"]
            self.assertEqual(protocol["expected_world_size"], 4, role)
            self.assertEqual(protocol["per_rank_batch_size"], 2, role)
            self.assertEqual(protocol["global_batch_size"], 8, role)

        weights = {
            role: {key: config["trainer_args"][key] for key in weight_keys}
            for role, config in configs.items()
        }
        self.assertTrue(all(value == 0 for value in weights["B"].values()))
        self.assertEqual(weights["M"]["lambda_velocity_loss"], 0.45)
        self.assertEqual(weights["M"]["lambda_acceleration_loss"], 0.20)
        self.assertEqual(weights["G"]["lambda_bone_loss"], 0.0006)
        self.assertEqual(weights["G"]["lambda_angle_loss"], 0.0035)
        self.assertEqual(weights["MG"], {
            key: weights["M"][key] + weights["G"][key] for key in weight_keys
        })
        self.assertEqual(weights["MG"]["lambda_palm_bone_loss"], 0.0)

    def setUp(self) -> None:
        self.path = ROOT / "configs" / "thinkjepa_B_alllayers_100ep.json"
        self.config = RUNNER.load_and_validate_config(self.path)

    def test_baseline_identity_is_explicit(self) -> None:
        args = self.config["trainer_args"]
        self.assertEqual(args["trajmode"], "traj")
        self.assertIs(args["joint_pred"], True)
        self.assertIs(args["skip_nonfinite_loss"], False)
        self.assertEqual(self.config["source_baseline_commit"],
                         "9b95d358f7c7a1eaf73fda0efbcc8e28c44ce086")
        self.assertTrue(all(value == 0 for key, value in args.items()
                            if key.startswith("lambda_") and key.endswith("_loss")))

    def test_command_uses_validation_as_legacy_test_manifest(self) -> None:
        command = RUNNER.build_command(
            self.config, python="python", data_dir=Path("/data"),
            train_manifest=Path("train.txt"), val_manifest=Path("val.txt"),
            output_dir=Path("out"), nproc_per_node=1)
        joined = " ".join(map(str, command))
        self.assertIn("--trajmode traj", joined)
        self.assertIn("--joint_pred", command)
        index = command.index("--test_manifest")
        self.assertEqual(command[index + 1], "val.txt")
        self.assertNotIn("final", joined.lower())

    def test_runner_allowlist_matches_public_trainer(self) -> None:
        tree = ast.parse((ROOT / "cache_train" / "thinker_train.py").read_text(encoding="utf-8"))
        trainer_flags = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.startswith("--")):
                trainer_flags.add(node.args[0].value[2:])
        declared = set(self.config["trainer_args"])
        self.assertLessEqual(declared, trainer_flags)
        self.assertLessEqual(RUNNER.ALLOWED_CONFIG_KEYS, trainer_flags)

    def test_runtime_keys_are_rejected_from_config(self) -> None:
        bad = json.loads(json.dumps(self.config))
        bad["trainer_args"]["output_dir"] = "accidental"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime-only"):
                RUNNER.load_and_validate_config(path)

    def test_mg_smoke_exposes_all_opt_in_weights(self) -> None:
        config = RUNNER.load_and_validate_config(
            ROOT / "configs" / "thinkjepa_MG_alllayers_smoke.json")
        args = config["trainer_args"]
        for key in (
            "lambda_bone_loss", "lambda_palm_bone_loss", "lambda_velocity_loss",
            "lambda_acceleration_loss", "lambda_angle_loss",
        ):
            self.assertGreaterEqual(args[key], 0.0)
        command = RUNNER.build_command(
            config, python="python", data_dir=Path("/data"),
            train_manifest=Path("train.txt"), val_manifest=Path("eval.txt"),
            output_dir=Path("out"), nproc_per_node=1)
        self.assertIn("--lambda_velocity_loss", command)

    def test_dry_run_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
                "--data-dir", str(Path(directory) / "data"),
                "--train-manifest", str(Path(directory) / "train.txt"),
                "--val-manifest", str(Path(directory) / "val.txt"),
                "--output-root", str(output_root),
            ], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY RUN", result.stdout)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
