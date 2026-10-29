from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "cache_train"))
    sys.path.insert(0, str(ROOT / "vjepa2"))
    from cache_train.thinker_predictor import CortexGuidedVideoPredictor
    from cache_train.thinker_train import build_thinkjepa_guidance_inputs


def _args(selector="all"):
    return SimpleNamespace(
        thinkjepa_use_vlm_merge=True,
        thinkjepa_vlm_source="both",
        thinkjepa_vlm_layer_selector=selector,
        thinkjepa_vlm_layer_index=-1,
        thinkjepa_drop_think_tokens=False,
        thinkjepa_zero_dropped_think_tokens=True,
    )


@unittest.skipIf(torch is None, "PyTorch is required for guidance semantics tests")
class GuidancePreprocessingTests(unittest.TestCase):
    def test_batch_axis_is_preserved(self):
        extras = {
            "vlm_old": torch.randn(2, 8, 5, 3),
            "vlm_new": torch.randn(2, 8, 2, 3),
            "vlm_old_mask": torch.ones(2, 8, 5, dtype=torch.bool),
            "vlm_new_mask": torch.ones(2, 8, 2, dtype=torch.bool),
        }
        result = build_thinkjepa_guidance_inputs(extras, _args(), "cpu")
        self.assertEqual(tuple(result["vlm_old"].shape), (2, 8, 5, 3))
        self.assertEqual(tuple(result["vlm_new"].shape), (2, 8, 2, 3))
        self.assertEqual(tuple(result["vlm_old_mask"].shape), (2, 8, 5))

    def test_other_sample_does_not_change_first_guidance(self):
        first = torch.randn(1, 8, 5, 3)
        second_a = torch.randn(1, 8, 5, 3)
        second_b = torch.randn(1, 8, 5, 3) * 100
        mask = torch.ones(2, 8, 5, dtype=torch.bool)
        payload_a = build_thinkjepa_guidance_inputs(
            {"vlm_old": torch.cat([first, second_a]), "vlm_old_mask": mask},
            _args(), "cpu",
        )
        payload_b = build_thinkjepa_guidance_inputs(
            {"vlm_old": torch.cat([first, second_b]), "vlm_old_mask": mask},
            _args(), "cpu",
        )
        torch.testing.assert_close(payload_a["vlm_old"][0], payload_b["vlm_old"][0])

    def test_last_and_all_keep_distinct_layer_counts(self):
        values = torch.randn(2, 8, 5, 3)
        all_layers = build_thinkjepa_guidance_inputs(
            {"vlm_old": values}, _args("all"), "cpu"
        )["vlm_old"]
        last_layer = build_thinkjepa_guidance_inputs(
            {"vlm_old": values}, _args("last"), "cpu"
        )["vlm_old"]
        self.assertEqual(all_layers.shape[1], 8)
        self.assertEqual(last_layer.shape[1], 1)
        torch.testing.assert_close(last_layer[:, 0], all_layers[:, -1])


@unittest.skipIf(torch is None, "PyTorch is required for guidance semantics tests")
class PredictorGuidanceMappingTests(unittest.TestCase):
    def _predictor(self, mode="film"):
        torch.manual_seed(3)
        return CortexGuidedVideoPredictor(
            img_size=(1, 1), patch_size=1,
            num_frames=2, tubelet_size=1,
            embed_dim=4, predictor_embed_dim=4,
            depth=12, num_heads=1, mlp_ratio=2.0,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
            use_mask_tokens=True, num_mask_tokens=1,
            use_rope=True, use_vlm_merge=True,
            vlm_cond_mode=mode, vlm_old_dim=3, vlm_new_dim=3,
            vlm_hidden=8,
        ).eval()

    def _payload(self):
        return {
            "vlm_old": torch.randn(2, 8, 5, 3),
            "vlm_new": torch.randn(2, 8, 2, 3),
            "vlm_old_mask": torch.ones(2, 8, 5, dtype=torch.bool),
            "vlm_new_mask": torch.ones(2, 8, 2, dtype=torch.bool),
        }

    def test_eight_layers_cover_twelve_blocks_monotonically(self):
        predictor = self._predictor()
        bank = predictor._build_layerwise_guidance(self._payload(), 2, torch.device("cpu"))
        self.assertEqual(len(bank.layers), 12)
        self.assertEqual(
            predictor.guidance_layer_map,
            (0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7),
        )
        for scale, shift in bank.layers:
            self.assertEqual(tuple(scale.shape), (2, 4))
            self.assertEqual(tuple(shift.shape), (2, 4))

    def test_changing_second_sample_does_not_change_first_film_payload(self):
        predictor = self._predictor()
        predictor.guidance_layer_scale.data.fill_(0.5)
        payload_a = self._payload()
        payload_b = {key: value.clone() for key, value in payload_a.items()}
        payload_b["vlm_old"][1].mul_(100)
        bank_a = predictor._build_layerwise_guidance(payload_a, 2, torch.device("cpu"))
        bank_b = predictor._build_layerwise_guidance(payload_b, 2, torch.device("cpu"))
        for (scale_a, shift_a), (scale_b, shift_b) in zip(bank_a.layers, bank_b.layers):
            torch.testing.assert_close(scale_a[0], scale_b[0])
            torch.testing.assert_close(shift_a[0], shift_b[0])

    def test_cross_attention_memory_remains_batched(self):
        predictor = self._predictor("crossattn")
        bank = predictor._build_layerwise_guidance(self._payload(), 2, torch.device("cpu"))
        memory, valid_mask = bank.layers[0]
        self.assertEqual(tuple(memory.shape), (2, 7, 4))
        self.assertEqual(tuple(valid_mask.shape), (2, 7))


if torch is None and os.environ.get("THINKJEPA_REQUIRE_TORCH") == "1":
    raise RuntimeError("THINKJEPA_REQUIRE_TORCH=1 but PyTorch cannot be imported")


if __name__ == "__main__":
    unittest.main()
