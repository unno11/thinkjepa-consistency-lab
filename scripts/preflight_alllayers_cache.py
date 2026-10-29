#!/usr/bin/env python3
"""Audit frozen train/validation caches for paper-aligned all-layer guidance.

The script resolves only entries named by the supplied train and validation
manifests.  It deliberately refuses a manifest named ``test.txt`` and never
recursively scans the cache root, so the locked final-test membership remains
untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "protocol" / "splits" / "train.txt"
DEFAULT_VALIDATION = REPO_ROOT / "protocol" / "splits" / "val.txt"
DEFAULT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
EXPECTED_TRAIN_SHA256 = "ceec4fad0984e31c0eb241b4b9dfdcb00e78a97a3af5f258c6605cf94fedec8e"
EXPECTED_VALIDATION_SHA256 = "82f237a5acbd99f078dd1dd8675bf66b041d0ce56c885a3141817d8ece401b15"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(path: Path, expected_count: int) -> list[str]:
    if path.name.lower() == "test.txt":
        raise ValueError("locked final-test manifest is forbidden in all-layer preflight")
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} entries, found {len(entries)}"
        )
    if len(set(entries)) != len(entries):
        raise ValueError(f"{path}: duplicate manifest entries")
    return entries


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_entry(entry: str, data_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    raw = Path(entry)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((repo_root / raw, data_dir / raw))
        if raw.parts and raw.parts[0].lower() == "data":
            candidates.append(data_dir.joinpath(*raw.parts[1:]))
    data_root = data_dir.resolve()
    repo = repo_root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and (_inside(resolved, data_root) or _inside(resolved, repo)):
            if resolved.suffix.lower() != ".npz":
                raise ValueError(f"manifest entry is not an NPZ cache: {entry}")
            return resolved
    raise FileNotFoundError(f"cannot resolve manifest cache entry: {entry}")


def audit_archive(path: Path, expected_layers: tuple[int, ...]) -> dict[str, object]:
    required = {"vlm_old", "vlm_new", "layers"}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing cache keys {missing}")
        layers = tuple(int(value) for value in np.asarray(archive["layers"]).reshape(-1))
        if layers != expected_layers:
            raise ValueError(f"{path}: layers {layers} != {expected_layers}")
        result: dict[str, object] = {"layers": layers}
        for key in ("vlm_old", "vlm_new"):
            value = np.asarray(archive[key])
            if value.ndim not in (3, 4):
                raise ValueError(f"{path}: {key} must be [L,S,D] or [L,T,S,D], got {value.shape}")
            if value.shape[0] != len(expected_layers):
                raise ValueError(
                    f"{path}: {key} layer dimension {value.shape[0]} != {len(expected_layers)}"
                )
            if any(int(size) <= 0 for size in value.shape):
                raise ValueError(f"{path}: {key} contains an empty dimension: {value.shape}")
            if not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"{path}: {key} must be floating point, got {value.dtype}")
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"{path}: {key} contains non-finite values")
            result[f"{key}_shape"] = tuple(int(size) for size in value.shape)
            result[f"{key}_dtype"] = str(value.dtype)
        return result


def git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--expected-train", type=int, default=1600)
    parser.add_argument("--expected-validation", type=int, default=200)
    parser.add_argument("--expected-layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data directory: {data_dir}")
    if train_manifest == val_manifest:
        raise ValueError("train and validation manifests must differ")
    train_entries = manifest_entries(train_manifest, args.expected_train)
    val_entries = manifest_entries(val_manifest, args.expected_validation)
    manifest_hashes = {
        "train": sha256_file(train_manifest),
        "validation": sha256_file(val_manifest),
    }
    expected_hashes = {
        "train": EXPECTED_TRAIN_SHA256,
        "validation": EXPECTED_VALIDATION_SHA256,
    }
    for split in ("train", "validation"):
        if manifest_hashes[split] != expected_hashes[split]:
            raise ValueError(
                f"{split} manifest SHA-256 {manifest_hashes[split]} "
                f"!= frozen protocol {expected_hashes[split]}"
            )
    overlap = set(train_entries) & set(val_entries)
    if overlap:
        raise ValueError(f"train/validation overlap: {len(overlap)} entries")
    expected_layers = tuple(args.expected_layers)
    if len(expected_layers) < 2 or len(set(expected_layers)) != len(expected_layers):
        raise ValueError("expected layers must contain at least two unique indices")

    shape_counts: dict[str, Counter[str]] = {
        "vlm_old": Counter(), "vlm_new": Counter(),
    }
    dtype_counts: dict[str, Counter[str]] = {
        "vlm_old": Counter(), "vlm_new": Counter(),
    }
    resolved_paths: set[Path] = set()
    split_counts = {"train": 0, "validation": 0}
    for split, entries in (("train", train_entries), ("validation", val_entries)):
        for index, entry in enumerate(entries, start=1):
            path = resolve_entry(entry, data_dir)
            if path in resolved_paths:
                raise ValueError(f"cache archive reused across split membership: {path}")
            resolved_paths.add(path)
            audited = audit_archive(path, expected_layers)
            for key in ("vlm_old", "vlm_new"):
                shape_counts[key][str(audited[f"{key}_shape"])] += 1
                dtype_counts[key][str(audited[f"{key}_dtype"])] += 1
            split_counts[split] += 1
            if index % 100 == 0 or index == len(entries):
                print(f"[AUDIT] {split}: {index}/{len(entries)}", flush=True)

    payload = {
        "status": "PASS",
        "purpose": "paper-aligned all-layer cache preflight",
        "test_accessed": False,
        "expected_layers": list(expected_layers),
        "manifests": {
            "train": {
                "path": str(train_manifest),
                "sha256": manifest_hashes["train"],
                "samples": len(train_entries),
            },
            "validation": {
                "path": str(val_manifest),
                "sha256": manifest_hashes["validation"],
                "samples": len(val_entries),
            },
        },
        "archives_checked": len(resolved_paths),
        "split_counts": split_counts,
        "shape_counts": {key: dict(value) for key, value in shape_counts.items()},
        "dtype_counts": {key: dict(value) for key, value in dtype_counts.items()},
        "git": git_identity(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report:
        report = args.report.resolve()
        if report.exists():
            raise FileExistsError(f"refusing to overwrite report: {report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered, encoding="utf-8")
        print(f"[OK] wrote {report}")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
