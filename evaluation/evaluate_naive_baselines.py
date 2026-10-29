#!/usr/bin/env python3
"""Evaluate leak-free context-only controls on a frozen manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Support both ``python -m evaluation.evaluate_naive_baselines`` and direct execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_trajectory_metrics


def last_pose(context, future):
    context = np.asarray(context, dtype=np.float64)
    return np.repeat(context[-1:], future, axis=0)


def constant_velocity(context, future, k):
    context = np.asarray(context, dtype=np.float64)
    if k < 2 or len(context) < k:
        raise ValueError(f"constant velocity k={k} requires at least {k} context frames")
    velocity = (context[-1] - context[-k]) / float(k - 1)
    steps = np.arange(1, future + 1, dtype=np.float64)[:, None, None]
    return context[-1:] + steps * velocity[None]


def _manifest_entries(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _resolve_sample(entry, data_dir, repo_root):
    raw = Path(entry)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, data_dir / raw]
    if raw.parts and raw.parts[0].lower() == "data":
        candidates.insert(0, data_dir.joinpath(*raw.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve manifest entry: {entry}")


def _load_world(path):
    with np.load(path, allow_pickle=False) as archive:
        if "xyz_world" not in archive.files:
            raise KeyError(f"{path}: missing xyz_world")
        return np.asarray(archive["xyz_world"], dtype=np.float64)


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": None, "std": None, "n_defined": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "n_defined": int(len(finite)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--past", type=int, default=32)
    parser.add_argument("--future", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--final-test-confirm", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = args.manifest.resolve()
    if manifest.name == "test.txt" and args.final_test_confirm != "YES":
        raise SystemExit("[LOCKED] formal test requires --final-test-confirm YES")
    if args.past < 4 or args.future < 3:
        raise SystemExit("past must be >=4 and future >=3")

    repo_root = Path(__file__).resolve().parents[1]
    entries = _manifest_entries(manifest)
    if args.limit:
        entries = entries[:args.limit]
    methods = {
        "LastPose": lambda x: last_pose(x, args.future),
        "ConstVelocityK2": lambda x: constant_velocity(x, args.future, 2),
        "ConstVelocityK4": lambda x: constant_velocity(x, args.future, 4),
    }
    rows = []
    for entry in entries:
        path = _resolve_sample(entry, args.data_dir.resolve(), repo_root)
        world = _load_world(path)
        required = args.past + args.future
        if world.ndim != 3 or world.shape[1:] != (52, 3) or world.shape[0] < required:
            raise ValueError(f"{path}: expected at least [{required},52,3], got {world.shape}")
        context, target = world[:args.past], world[args.past:required]
        for method, predictor in methods.items():
            rows.append({"sample": entry, "method": method,
                         **compute_trajectory_metrics(predictor(context), target)})
    if not rows:
        raise SystemExit("[ERROR] manifest contains no samples")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [key for key in rows[0] if key not in {"sample", "method"}]
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "method", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    summary = {"manifest": str(manifest), "past": args.past, "future": args.future,
               "samples": len(entries), "methods": {}}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            metric: _summary_stats([row[metric] for row in selected]) for metric in metric_names
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[OK] evaluated {len(entries)} samples x {len(methods)} methods")
    print(f"[OK] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
