#!/usr/bin/env python3
"""Validate frozen split identity, disjointness, and optional data presence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / "protocol" / "splits"


def read_members(path: Path) -> list[str]:
    members = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not members:
        raise ValueError(f"empty manifest: {path}")
    if len(members) != len(set(members)):
        raise ValueError(f"duplicate members: {path}")
    for member in members:
        pure = PurePosixPath(member)
        if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".npz":
            raise ValueError(f"invalid relative NPZ member in {path}: {member}")
    return members


def membership_fingerprint(members: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()


def validate(split_dir: Path, data_dir: Path | None = None) -> dict[str, object]:
    metadata = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    groups = {name: read_members(split_dir / f"{name}.txt")
              for name in ("train", "val", "test")}
    for name, members in groups.items():
        expected_count = int(metadata["counts"][name])
        if len(members) != expected_count:
            raise ValueError(f"{name}: count {len(members)} != {expected_count}")
        actual = membership_fingerprint(members)
        expected = metadata["split_fingerprint"][name]
        if actual != expected:
            raise ValueError(f"{name}: fingerprint {actual} != {expected}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(groups[left]) & set(groups[right])
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {len(overlap)}")

    smoke_train = read_members(split_dir.parent / "smoke" / "train.txt")
    smoke_eval = read_members(split_dir.parent / "smoke" / "eval.txt")
    if set(smoke_train) & set(smoke_eval):
        raise ValueError("smoke train/eval overlap")
    if not (set(smoke_train) | set(smoke_eval)) <= set(groups["train"]):
        raise ValueError("smoke manifests must be subsets of formal train")

    missing: list[str] = []
    if data_dir is not None:
        data_dir = data_dir.resolve()
        for member in groups["train"] + groups["val"] + groups["test"]:
            pure = PurePosixPath(member)
            relative = Path(*pure.parts)
            candidates = [ROOT / relative, data_dir / relative]
            if pure.parts and pure.parts[0] == "data":
                candidates.extend((data_dir / Path(*pure.parts[1:]), data_dir.parent / relative))
            if not any(candidate.is_file() for candidate in candidates):
                missing.append(member)
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} archives; first: {missing[0]}")
    return {
        "counts": {name: len(value) for name, value in groups.items()},
        "fingerprints": {name: membership_fingerprint(value) for name, value in groups.items()},
        "smoke": {"train": len(smoke_train), "eval": len(smoke_eval)},
        "data_verified": data_dir is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.split_dir.resolve(), args.data_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
