from __future__ import annotations

import unittest

from evaluation.compare_direct32 import paired_statistics, parse_roles


class CompareDirect32Tests(unittest.TestCase):
    def test_role_subset_must_start_with_B(self):
        self.assertEqual(parse_roles("B,MG"), ("B", "MG"))
        self.assertEqual(parse_roles("B,M,G,MG"), ("B", "M", "G", "MG"))
        with self.assertRaisesRegex(ValueError, "baseline B"):
            parse_roles("MG,B")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_roles("B,MG,MG")
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_roles("B,X")

    def test_paired_improvement_is_negative_and_detected(self):
        baseline = [2.0 + index * 0.01 for index in range(30)]
        candidate = [value - 0.2 for value in baseline]
        result = paired_statistics(
            baseline,
            candidate,
            bootstrap_samples=1000,
            randomization_samples=1000,
            seed=7,
        )
        self.assertAlmostEqual(result["mean_delta"], -0.2)
        self.assertLess(result["ci95_high"], 0.0)
        self.assertLess(result["paired_randomization_p"], 0.01)
        self.assertEqual(result["candidate_better_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
