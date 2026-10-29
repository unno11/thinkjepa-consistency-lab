from __future__ import annotations

import unittest

from evaluation.guidance_contract import evaluate_guidance_contract


class GuidanceContractVerdictTests(unittest.TestCase):
    def test_semantically_isolated_measurements_pass(self):
        verdict = evaluate_guidance_contract({
            "preserves_batch_axis": True,
            "batch_size_max_abs_delta": 1e-8,
            "duplicate_A_batch_size_max_abs_delta": 2e-8,
            "zero_guidance_batch_size_max_abs_delta": 3e-8,
            "other_sample_max_abs_delta": 2e-8,
            "preprocessed_other_sample_max_abs_delta": 0.0,
            "guidance_bank_other_sample_max_abs_delta": 0.0,
            "layer_mapping_covers_all_blocks": True,
            "last_all_max_abs_delta": 2e-4,
            "zero_guidance_max_abs_delta": 3e-4,
            "shuffle_guidance_max_abs_delta": 4e-4,
        })
        self.assertEqual(verdict["status"], "PASS")
        self.assertTrue(all(verdict["checks"].values()))

    def test_collapsed_batch_and_missing_mapping_fail(self):
        verdict = evaluate_guidance_contract({
            "preserves_batch_axis": False,
            "batch_size_max_abs_delta": 1e-3,
            "duplicate_A_batch_size_max_abs_delta": 1e-3,
            "zero_guidance_batch_size_max_abs_delta": 1e-3,
            "other_sample_max_abs_delta": 2e-3,
            "preprocessed_other_sample_max_abs_delta": 2e-3,
            "guidance_bank_other_sample_max_abs_delta": 2e-3,
            "layer_mapping_covers_all_blocks": False,
            "last_all_max_abs_delta": 1e-4,
            "zero_guidance_max_abs_delta": 1e-4,
            "shuffle_guidance_max_abs_delta": 0.0,
        })
        self.assertEqual(verdict["status"], "FAIL")
        self.assertFalse(verdict["checks"]["G0-1_preserves_batch_axis"])
        self.assertFalse(verdict["checks"]["G0-3_sample_isolation"])
        self.assertFalse(verdict["checks"]["G0-4_layer_mapping_coverage"])
        self.assertFalse(verdict["checks"]["G0-6_shuffle_guidance_sensitivity"])

    def test_thresholds_are_strict_for_sensitivity(self):
        values = {
            "preserves_batch_axis": True,
            "batch_size_max_abs_delta": 1e-6,
            "duplicate_A_batch_size_max_abs_delta": 1e-6,
            "zero_guidance_batch_size_max_abs_delta": 1e-6,
            "other_sample_max_abs_delta": 1e-6,
            "preprocessed_other_sample_max_abs_delta": 1e-6,
            "guidance_bank_other_sample_max_abs_delta": 1e-6,
            "layer_mapping_covers_all_blocks": True,
            "last_all_max_abs_delta": 1e-7,
            "zero_guidance_max_abs_delta": 1e-7,
            "shuffle_guidance_max_abs_delta": 1e-7,
        }
        verdict = evaluate_guidance_contract(values)
        self.assertEqual(verdict["status"], "FAIL")

    def test_cuda_batch_shape_roundoff_has_a_separate_tolerance(self):
        values = {
            "preserves_batch_axis": True,
            "batch_size_max_abs_delta": 4.96e-5,
            "duplicate_A_batch_size_max_abs_delta": 4.96e-5,
            "zero_guidance_batch_size_max_abs_delta": 1.12e-4,
            "other_sample_max_abs_delta": 0.0,
            "preprocessed_other_sample_max_abs_delta": 0.0,
            "guidance_bank_other_sample_max_abs_delta": 0.0,
            "layer_mapping_covers_all_blocks": True,
            "last_all_max_abs_delta": 25.0,
            "zero_guidance_max_abs_delta": 33.0,
            "shuffle_guidance_max_abs_delta": 0.25,
        }
        verdict = evaluate_guidance_contract(values)
        self.assertEqual(verdict["status"], "PASS")
        self.assertEqual(verdict["thresholds"]["batch_size_atol"], 2e-4)
        self.assertEqual(verdict["thresholds"]["sample_isolation_atol"], 1e-6)

    def test_batch_roundoff_tolerance_does_not_relax_sample_isolation(self):
        values = {
            "preserves_batch_axis": True,
            "batch_size_max_abs_delta": 5e-5,
            "duplicate_A_batch_size_max_abs_delta": 5e-5,
            "zero_guidance_batch_size_max_abs_delta": 1e-4,
            "other_sample_max_abs_delta": 2e-6,
            "preprocessed_other_sample_max_abs_delta": 0.0,
            "guidance_bank_other_sample_max_abs_delta": 0.0,
            "layer_mapping_covers_all_blocks": True,
            "last_all_max_abs_delta": 25.0,
            "zero_guidance_max_abs_delta": 33.0,
            "shuffle_guidance_max_abs_delta": 0.25,
        }
        verdict = evaluate_guidance_contract(values)
        self.assertEqual(verdict["status"], "FAIL")
        self.assertFalse(verdict["checks"]["G0-3_sample_isolation"])


if __name__ == "__main__":
    unittest.main()
