from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT / "vjepa2"))
    from src.models.utils.modules import Attention, RoPEAttention


@unittest.skipIf(torch is None, "PyTorch is required for attention semantics tests")
class AttentionEvalSemanticsTests(unittest.TestCase):
    def _assert_eval_batch_invariance(self, module, *, mask=None, **kwargs):
        torch.manual_seed(17)
        single = torch.randn(1, 4, 12)
        duplicated = single.repeat(2, 1, 1)
        module.eval()

        with torch.no_grad():
            output_single = module(single, mask=mask, **kwargs)
            duplicated_mask = None if mask is None else mask.repeat(2, 1)
            output_duplicated = module(duplicated, mask=duplicated_mask, **kwargs)
            output_repeat = module(single, mask=mask, **kwargs)

        torch.testing.assert_close(output_single, output_duplicated[:1])
        torch.testing.assert_close(output_single, output_repeat)

    def test_plain_sdpa_disables_dropout_in_eval(self):
        module = Attention(
            dim=12,
            num_heads=3,
            qkv_bias=True,
            proj_drop=0.5,
            use_sdpa=True,
        )
        self._assert_eval_batch_invariance(module)

    def test_rope_sdpa_disables_dropout_in_eval(self):
        module = RoPEAttention(
            dim=12,
            num_heads=3,
            qkv_bias=True,
            proj_drop=0.5,
            use_sdpa=True,
            grid_size=2,
        )
        mask = torch.arange(4).view(1, 4)
        self._assert_eval_batch_invariance(module, mask=mask)


if __name__ == "__main__":
    unittest.main()
