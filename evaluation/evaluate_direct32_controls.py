#!/usr/bin/env python3
"""Evaluate frozen LastPose and ConstVelocityK4 Direct32 validation controls."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.direct32_controls_protocol import (
    load_controls_protocol,
    resolve_controls_validation,
    sample_groups,
)
from evaluation.evaluate_naive_baselines import (
    _load_world,
    _resolve_sample,
    _summary_stats,
    constant_velocity,
    last_pose,
)
from evaluation.metrics import compute_trajectory_metrics
from evaluation.file_utils import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "direct32_naive_controls_validation_v1.json"


def _git_identity() -> dict:
    def run(*args):
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8"
        ).strip()
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")

    protocol_path = args.protocol.resolve()
    protocol = load_controls_protocol(protocol_path)
    manifest, entries, cluster_audit = resolve_controls_validation(
        protocol, protocol_path
    )
    window = protocol["window"]
    past, future = window["past_frames"], window["future_frames"]
    required = past + future
    methods = {
        "LastPose": lambda context: last_pose(context, future),
        "ConstVelocityK4": lambda context: constant_velocity(context, future, 4),
    }
    rows = []
    for entry in entries:
        path = _resolve_sample(entry, args.data_dir.resolve(), REPO_ROOT)
        world = _load_world(path)
        if world.ndim != 3 or world.shape[1:] != (52, 3) or world.shape[0] < required:
            raise ValueError(f"{path}: expected at least [{required},52,3], got {world.shape}")
        context = world[:past]
        target = world[past:required]
        if not np.isfinite(context).all() or not np.isfinite(target).all():
            raise FloatingPointError(f"non-finite trajectory: {entry}")
        task, episode = sample_groups(entry)
        for method, predictor in methods.items():
            prediction = predictor(context)
            rows.append({
                "sample": entry,
                "task": task,
                "episode": episode,
                "method": method,
                **compute_trajectory_metrics(prediction, target),
            })

    if len(rows) != len(entries) * len(methods):
        raise RuntimeError("controls output row count mismatch")
    args.output_dir.mkdir(parents=False, exist_ok=False)
    metric_names = [
        key for key in rows[0]
        if key not in {"sample", "task", "episode", "method"}
    ]
    with (args.output_dir / "direct32_controls_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "task", "episode", "method", *metric_names],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: _summary_stats([row[metric] for row in selected])
            for metric in metric_names
        }
    identity = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "manifest": str(manifest),
        "manifest_sha256": protocol["validation"]["sha256"],
        "split": "validation",
        "samples": len(entries),
        "methods": list(methods),
        "cluster_audit": cluster_audit,
        "git": _git_identity(),
        "test_accessed": False,
    }
    (args.output_dir / "direct32_controls_summary.json").write_text(
        json.dumps(
            {"identity": identity, "metrics": summary},
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] controls: {len(entries)} validation samples x {len(methods)} methods")
    print(
        "[AUDIT] tasks={tasks} episodes={episode_clusters} "
        "max_samples_per_episode={max_samples_per_episode} "
        "episode_inference={episode_cluster_inference}".format(**cluster_audit)
    )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
