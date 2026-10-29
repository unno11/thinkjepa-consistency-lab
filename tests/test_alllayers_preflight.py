from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Keep config tests runnable in the lean local env.
    np = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
PREFLIGHT = None
if np is not None:
    SPEC = importlib.util.spec_from_file_location(
        "preflight_alllayers_cache", ROOT / "scripts" / "preflight_alllayers_cache.py"
    )
    PREFLIGHT = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(PREFLIGHT)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_experiment", ROOT / "scripts" / "run_experiment.py"
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)


class AllLayersConfigTests(unittest.TestCase):
    def test_formal_matrix_is_all_layers_and_only_weights_differ(self):
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
        for role, config in configs.items():
            args = config["trainer_args"]
            self.assertEqual(args["thinkjepa_vlm_layer_selector"], "all", role)
            self.assertEqual(args["epochs"], 100, role)
            self.assertIs(args["skip_nonfinite_loss"], False, role)
            self.assertEqual(config["protocol"]["expected_world_size"], 4, role)
            self.assertEqual(config["protocol"]["baseline_variant"], "paper_all_layers", role)
        common = {
            role: {
                key: value for key, value in config["trainer_args"].items()
                if key not in weight_keys
            }
            for role, config in configs.items()
        }
        self.assertTrue(all(common[role] == common["B"] for role in roles))
        weights = {
            role: {key: configs[role]["trainer_args"][key] for key in weight_keys}
            for role in roles
        }
        self.assertTrue(all(value == 0 for value in weights["B"].values()))
        self.assertEqual(weights["MG"], {
            key: weights["M"][key] + weights["G"][key] for key in weight_keys
        })


    def test_smokes_are_fail_fast_all_layer_train_only(self):
        for role in ("B", "MG"):
            config = RUNNER.load_and_validate_config(
                ROOT / "configs" / f"thinkjepa_{role}_alllayers_smoke.json"
            )
            args = config["trainer_args"]
            self.assertEqual(args["epochs"], 1)
            self.assertEqual(args["thinkjepa_vlm_layer_selector"], "all")
            self.assertIs(args["skip_nonfinite_loss"], False)
            self.assertIs(config["protocol"]["smoke_only"], True)
            self.assertIs(config["protocol"]["final_test_locked"], True)

    def test_B_MG_multiseed_configs_change_only_seed_and_name(self):
        for role in ("B", "MG"):
            base = RUNNER.load_and_validate_config(
                ROOT / "configs" / f"thinkjepa_{role}_alllayers_100ep.json"
            )
            for seed in (43, 44):
                candidate = RUNNER.load_and_validate_config(
                    ROOT
                    / "configs"
                    / f"thinkjepa_{role}_alllayers_100ep_seed{seed}.json"
                )
                self.assertEqual(
                    candidate["name"],
                    f"thinkjepa_{role}_alllayers_100ep_seed{seed}",
                )
                self.assertEqual(candidate["trainer_args"]["seed"], seed)
                self.assertEqual(candidate["trainer_args"]["split_seed"], 42)
                self.assertEqual(candidate["protocol"], base["protocol"])
                self.assertEqual(
                    candidate["trainer_args"],
                    dict(base["trainer_args"], seed=seed),
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in {"name", "trainer_args"}
                    },
                    {
                        key: value
                        for key, value in base.items()
                        if key not in {"name", "trainer_args"}
                    },
                )

    @unittest.skipIf(np is None, "NumPy is required to import the preflight module")
    def test_preflight_hashes_match_frozen_manifests(self):
        self.assertEqual(
            PREFLIGHT.sha256_file(ROOT / "protocol" / "splits" / "train.txt"),
            PREFLIGHT.EXPECTED_TRAIN_SHA256,
        )
        self.assertEqual(
            PREFLIGHT.sha256_file(ROOT / "protocol" / "splits" / "val.txt"),
            PREFLIGHT.EXPECTED_VALIDATION_SHA256,
        )


@unittest.skipIf(np is None, "NumPy is required for cache-audit tests")
class AllLayersCacheAuditTests(unittest.TestCase):
    def _write_cache(self, path: Path, layers=DEFAULT_LAYERS):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            layers=np.asarray(layers, dtype=np.int32),
            vlm_old=np.ones((len(layers), 3, 4), dtype=np.float16),
            vlm_new=np.ones((len(layers), 2, 4), dtype=np.float16),
        )

    def test_valid_archive_and_data_prefix_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "data" / "cache" / "task" / "sample.npz"
            self._write_cache(cache)
            resolved = PREFLIGHT.resolve_entry(
                "data/cache/task/sample.npz", root / "data", repo_root=root / "repo"
            )
            self.assertEqual(resolved, cache.resolve())
            audited = PREFLIGHT.audit_archive(cache, PREFLIGHT.DEFAULT_LAYERS)
            self.assertEqual(audited["vlm_old_shape"][0], 8)

    def test_missing_layer_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            self._write_cache(path, layers=(0, 4, 8))
            with self.assertRaisesRegex(ValueError, "layers"):
                PREFLIGHT.audit_archive(path, PREFLIGHT.DEFAULT_LAYERS)

    def test_nonfinite_cache_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            values = np.ones((8, 2, 4), dtype=np.float32)
            values[0, 0, 0] = np.nan
            np.savez(
                path,
                layers=np.asarray(PREFLIGHT.DEFAULT_LAYERS, dtype=np.int32),
                vlm_old=values,
                vlm_new=np.ones((8, 2, 4), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                PREFLIGHT.audit_archive(path, PREFLIGHT.DEFAULT_LAYERS)

    def test_final_test_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("anything.npz\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final-test"):
                PREFLIGHT.manifest_entries(path, 1)


if __name__ == "__main__":
    unittest.main()
