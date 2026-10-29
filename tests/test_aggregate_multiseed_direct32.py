from __future__ import annotations

import unittest

from evaluation.aggregate_multiseed_direct32 import (
    aggregate_seed_rows,
    holm_adjust,
)


class AggregateMultiseedTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone(self):
        rows = [
            {"paired_randomization_p": 0.04},
            {"paired_randomization_p": 0.01},
            {"paired_randomization_p": 0.03},
        ]
        holm_adjust(rows)
        by_raw = sorted(rows, key=lambda row: row["paired_randomization_p"])
        adjusted = [row["holm_p_within_family"] for row in by_raw]
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)

    def test_aggregate_uses_three_seed_means_not_sample_count(self):
        payloads = []
        for seed, baseline, candidate in (
            (42, 2.0, 1.0),
            (43, 4.0, 3.0),
            (44, 6.0, 7.0),
        ):
            payloads.append({
                "paired_rows": [{
                    "seed": seed,
                    "family": "spatial",
                    "metric": "ADE_m",
                    "transform": "identity",
                    "baseline_mean": baseline,
                    "candidate_mean": candidate,
                    "mean_delta": candidate - baseline,
                }]
            })
        result = aggregate_seed_rows(payloads)[0]
        self.assertEqual(result["n_training_seeds"], 3)
        self.assertEqual(result["improved_seed_count"], 2)
        self.assertEqual(result["worsened_seed_count"], 1)
        self.assertAlmostEqual(result["baseline_seed_mean"], 4.0)
        self.assertAlmostEqual(result["candidate_seed_mean"], 11.0 / 3.0)
        self.assertAlmostEqual(result["paired_delta_seed_mean"], -1.0 / 3.0)
        self.assertEqual(result["model_level_inference"], "descriptive_only_n3")


if __name__ == "__main__":
    unittest.main()
