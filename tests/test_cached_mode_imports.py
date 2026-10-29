from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CachedModeImportTests(unittest.TestCase):
    def test_optional_vjepa_dataset_imports_are_lazy(self) -> None:
        tree = ast.parse((ROOT / "cache_train" / "thinker_train.py").read_text(encoding="utf-8"))
        top_level_modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.add(node.module)
        self.assertNotIn("vjepa2.src.datasets.utils.video.transforms", top_level_modules)
        self.assertNotIn("vjepa2.src.datasets.utils.video.volume_transforms", top_level_modules)

        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "build_dense_jepa_video_transform")
        nested_modules = {
            alias.name
            for node in ast.walk(function) if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("vjepa2.src.datasets.utils.video.transforms", nested_modules)
        self.assertIn("vjepa2.src.datasets.utils.video.volume_transforms", nested_modules)


if __name__ == "__main__":
    unittest.main()
