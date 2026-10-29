from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_training_validation_membership(self) -> None:
        # Publication check: preserve exact training/validation membership while
        # keeping the unused final-test membership out of this release.
        folder = ROOT / "protocol" / "splits"
        expected = {
            "train": (1600, "ceec4fad0984e31c0eb241b4b9dfdcb00e78a97a3af5f258c6605cf94fedec8e"),
            "val": (200, "82f237a5acbd99f078dd1dd8675bf66b041d0ce56c885a3141817d8ece401b15"),
        }
        members = {}
        for name, (count, sha) in expected.items():
            raw = (folder / (name + ".txt")).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            members[name] = set(raw.decode().splitlines())
            self.assertEqual(len(members[name]), count)
        self.assertFalse(members["train"] & members["val"])
        self.assertFalse((folder / "test.txt").exists())


if __name__ == "__main__":
    unittest.main()
