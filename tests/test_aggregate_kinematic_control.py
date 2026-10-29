from __future__ import annotations

import unittest

from evaluation.aggregate_kinematic_control_multiseed import (
    _aggregate,
    _validate_b_equivalence,
)
from evaluation.direct32_controls_protocol import sample_groups


class AggregateKinematicControlTests(unittest.TestCase):
    def test_task_grouping_matches_frozen_manifest_style(self):
        task, episode = sample_groups(
            "data/cache/part2/basic_pick_place/42_L8_nf32_res256_new16_s00of08.npz"
        )
        self.assertEqual(task, "basic_pick_place")
        self.assertEqual(episode, "basic_pick_place/42")

    def test_repeated_b_equivalence_gate(self):
        # Populate a complete row because the production equivalence gate checks
        # every frozen metric.
        from evaluation.aggregate_kinematic_control_multiseed import ALL_METRICS
        original = {"a": {metric: "1.0" for metric in ALL_METRICS}}
        repeated = {"a": {metric: "1.0" for metric in ALL_METRICS}}
        repeated["a"]["ADE_m"] = "1.000000001"
        self.assertAlmostEqual(
            _validate_b_equivalence(original, repeated, atol=2e-9), 1e-9
        )
        with self.assertRaises(ValueError):
            _validate_b_equivalence(original, repeated, atol=5e-10)

    def test_three_contrasts_are_aggregated(self):
        payloads = []
        for seed in (42, 43, 44):
            rows = []
            for contrast, delta in (
                ("KProj_minus_B", -1.0),
                ("MG_minus_B", -2.0),
                ("MG_minus_KProj", -1.0),
            ):
                rows.append({
                    "seed": seed,
                    "contrast": contrast,
                    "metric": "ADE_m",
                    "transform": "identity",
                    "baseline_mean": 4.0,
                    "candidate_mean": 4.0 + delta,
                    "mean_delta": delta,
                })
            payloads.append({"paired_rows": rows})
        result = _aggregate(payloads)
        self.assertEqual(len(result), 3)
        self.assertEqual({row["contrast"] for row in result}, {
            "KProj_minus_B", "MG_minus_B", "MG_minus_KProj",
        })
        self.assertTrue(all(row["n_training_seeds"] == 3 for row in result))


if __name__ == "__main__":
    unittest.main()
