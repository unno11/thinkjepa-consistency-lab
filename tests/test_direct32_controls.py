from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.compare_direct32_controls_multiseed import (
    STRUCTURALLY_ZERO_RATIO_METRICS,
    finite_pair_count,
    task_macro_statistics,
)
from evaluation.direct32_controls_protocol import (
    EXPECTED_GUARDS,
    EXPECTED_METHODS,
    EXPECTED_WINDOW,
    load_controls_protocol,
    resolve_controls_validation,
    sample_groups,
)


class Direct32ControlsTests(unittest.TestCase):
    def test_repository_protocol_and_manifest_identity(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/direct32_naive_controls_validation_v1.json"
        protocol = load_controls_protocol(path)
        _, entries, audit = resolve_controls_validation(protocol, path)
        self.assertEqual(len(entries), 200)
        self.assertEqual(audit["tasks"], 13)
        self.assertEqual(audit["episode_clusters"], 200)
        self.assertEqual(audit["max_samples_per_episode"], 1)

    def test_sample_group_parser(self):
        sample = (
            "data/cache/part2/basic_pick_place/"
            "1234_L8_nf32_res256_new16_s02of08.npz"
        )
        self.assertEqual(sample_groups(sample), ("basic_pick_place", "basic_pick_place/1234"))

    def test_protocol_rejects_method_mutation(self):
        payload = {
            "schema_version": 1,
            "window": json.loads(json.dumps(EXPECTED_WINDOW)),
            "methods": json.loads(json.dumps(EXPECTED_METHODS)),
            "guards": json.loads(json.dumps(EXPECTED_GUARDS)),
            "validation": {
                "manifest": "protocol/splits/val.txt",
                "sha256": "x",
                "expected_samples": 200,
                "expected_tasks": 13,
                "expected_episode_clusters": 200,
                "expected_max_samples_per_episode": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_controls_protocol(path), payload)
            payload["methods"]["ConstVelocityK4"]["context_frames"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_controls_protocol(path)

    def test_task_macro_uses_equal_task_weight(self):
        tasks = np.asarray(["large", "large", "large", *[f"t{i}" for i in range(12)]])
        baseline = np.zeros(len(tasks))
        candidate = np.asarray([3.0, 3.0, 3.0, *([-1.0] * 12)])
        result = task_macro_statistics(
            baseline,
            candidate,
            tasks,
            bootstrap_samples=1000,
            seed=7,
        )
        self.assertEqual(result["n_tasks"], 13)
        self.assertAlmostEqual(result["mean_delta"], -9.0 / 13.0)
        self.assertNotAlmostEqual(result["mean_delta"], candidate.mean())

    def test_zero_ratio_log_pairs_are_unavailable(self):
        baseline = np.asarray([np.inf, np.inf, np.inf])
        candidate = np.asarray([0.1, 0.2, 0.3])
        self.assertEqual(finite_pair_count(baseline, candidate), 0)
        baseline[0] = 2.0
        self.assertEqual(finite_pair_count(baseline, candidate), 1)

    def test_structural_zero_policy_covers_constant_acceleration(self):
        self.assertEqual(
            STRUCTURALLY_ZERO_RATIO_METRICS["LastPose"],
            {"VelocityRatio", "AccelerationRatio"},
        )
        self.assertEqual(
            STRUCTURALLY_ZERO_RATIO_METRICS["ConstVelocityK4"],
            {"AccelerationRatio"},
        )


if __name__ == "__main__":
    unittest.main()
