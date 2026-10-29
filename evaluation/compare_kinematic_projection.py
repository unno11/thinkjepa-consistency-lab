#!/usr/bin/env python3
"""Paired Direct32 analysis of B versus target-free KProjPredMedian."""

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

from evaluation.aggregate_multiseed_direct32 import (
    METRIC_FAMILIES,
    METRIC_TO_FAMILY,
    holm_adjust,
)
from evaluation.compare_direct32 import (
    BASE_METRICS,
    RATIO_METRICS,
    _stable_seed,
    paired_statistics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _read_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table = {row["sample"]: row for row in rows}
    if len(rows) != 200 or len(table) != 200:
        raise ValueError(f"expected 200 unique validation samples in {path}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--randomization-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")

    baseline_summary = json.loads(
        (args.evaluation_dir / "direct32_summary.json").read_text(encoding="utf-8")
    )
    projection_summary = json.loads(
        (args.evaluation_dir / "kinematic_projection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_identity = baseline_summary["identity"]
    projection_identity = projection_summary["identity"]
    if baseline_identity["role"] != "B" or projection_identity["base_role"] != "B":
        raise ValueError("kinematic control comparison requires a B checkpoint")
    if baseline_identity.get("test_accessed") is not False:
        raise ValueError("baseline unexpectedly accessed locked test")
    if projection_identity.get("test_accessed") is not False:
        raise ValueError("projection unexpectedly accessed locked test")
    for key in ("checkpoint_sha256", "protocol_sha256", "manifest_sha256"):
        if baseline_identity[key] != projection_identity[key]:
            raise ValueError(f"baseline/projection {key} mismatch")
    if projection_identity.get("target_trajectory_passed_to_projection") is not False:
        raise ValueError("projection received a future target trajectory")
    if projection_identity.get("observed_trajectory_passed_to_projection") is not False:
        raise ValueError("projection received an observed trajectory")

    baseline = _read_rows(args.evaluation_dir / "direct32_per_sample.csv")
    projection = _read_rows(
        args.evaluation_dir / "kinematic_projection_per_sample.csv"
    )
    if set(baseline) != set(projection):
        raise ValueError("baseline/projection sample membership differs")
    samples = sorted(baseline)
    method = projection_identity["projection_protocol"]["method"]
    records = []
    by_family = {family: [] for family in METRIC_FAMILIES}

    for metric in BASE_METRICS:
        stats = paired_statistics(
            [float(baseline[sample][metric]) for sample in samples],
            [float(projection[sample][metric]) for sample in samples],
            bootstrap_samples=args.bootstrap_samples,
            randomization_samples=args.randomization_samples,
            seed=_stable_seed(method, metric, args.seed),
        )
        row = {
            "role": f"B+{method}",
            "family": METRIC_TO_FAMILY[metric],
            "metric": metric,
            "transform": "identity",
            **stats,
        }
        records.append(row)
        by_family[row["family"]].append(row)

    for metric in RATIO_METRICS:
        baseline_raw = np.asarray(
            [float(baseline[sample][metric]) for sample in samples]
        )
        projection_raw = np.asarray(
            [float(projection[sample][metric]) for sample in samples]
        )
        stats = paired_statistics(
            np.abs(np.log(baseline_raw)),
            np.abs(np.log(projection_raw)),
            bootstrap_samples=args.bootstrap_samples,
            randomization_samples=args.randomization_samples,
            seed=_stable_seed(method, metric, args.seed),
        )
        row = {
            "role": f"B+{method}",
            "family": METRIC_TO_FAMILY[metric],
            "metric": metric,
            "transform": "abs_log_ratio",
            **stats,
        }
        records.append(row)
        by_family[row["family"]].append(row)

    for rows in by_family.values():
        holm_adjust(rows)

    args.output_dir.mkdir(parents=False, exist_ok=False)
    with (args.output_dir / "paired_kinematic_projection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "identity": {
            "analysis_git_commit": _git_commit(),
            "evaluator_git_commit": baseline_identity["git"]["commit"],
            "checkpoint_sha256": baseline_identity["checkpoint_sha256"],
            "checkpoint_epoch": baseline_identity["checkpoint_epoch"],
            "manifest_sha256": baseline_identity["manifest_sha256"],
            "direct32_protocol_sha256": baseline_identity["protocol_sha256"],
            "projection_protocol_sha256": projection_identity[
                "projection_protocol_sha256"
            ],
            "samples": len(samples),
            "split": "validation",
            "test_accessed": False,
            "bootstrap_samples": args.bootstrap_samples,
            "randomization_samples": args.randomization_samples,
            "seed": args.seed,
        },
        "metric_families": METRIC_FAMILIES,
        "results": records,
    }
    (args.output_dir / "paired_kinematic_projection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    selected = {
        "ADE_m", "FDE_m", "BoneErrorAll_mm", "BoneDriftAll_mm2",
        "AngleError_deg", "VelocityRatio", "AccelerationRatio",
    }
    print("Metric                 Delta       95% CI                 Family-Holm p")
    for row in records:
        if row["metric"] in selected:
            print(
                f"{row['metric']:<22} {row['mean_delta']:+.6f}  "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]  "
                f"{row['holm_p_within_family']:.6f}"
            )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
