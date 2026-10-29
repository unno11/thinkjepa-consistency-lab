#!/usr/bin/env python3
"""Paired validation analysis for frozen direct-32 B/M/G/MG outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ROLES = ("B", "M", "G", "MG")
ALLOWED_ROLES = frozenset(DEFAULT_ROLES)
BASE_METRICS = (
    "ADE_m",
    "FDE_m",
    "RootError_m",
    "RootAligned_ADE_m",
    "RootAligned_Finger_ADE_m",
    "BoneErrorAll_mm",
    "BoneErrorFinger_mm",
    "BoneErrorPalm_mm",
    "BoneDriftAll_mm2",
    "AngleError_deg",
)
RATIO_METRICS = ("VelocityRatio", "AccelerationRatio")


def _stable_seed(role: str, metric: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{role}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def parse_roles(value: str) -> tuple[str, ...]:
    roles = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(roles) < 2 or roles[0] != "B":
        raise ValueError("roles must contain baseline B followed by at least one candidate")
    if len(set(roles)) != len(roles):
        raise ValueError("roles must not contain duplicates")
    unknown = set(roles) - ALLOWED_ROLES
    if unknown:
        raise ValueError(f"unknown roles: {sorted(unknown)}")
    return roles


def paired_statistics(
    baseline,
    candidate,
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline, candidate = baseline[valid], candidate[valid]
    if len(baseline) < 2:
        raise ValueError("paired analysis requires at least two finite pairs")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    bootstrap_means = delta[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    observed = abs(float(delta.mean()))
    extreme = 0
    remaining = randomization_samples
    batch_size = 1000
    while remaining:
        count = min(batch_size, remaining)
        signs = rng.choice((-1.0, 1.0), size=(count, len(delta)))
        permuted = np.abs((signs * delta).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= count
    return {
        "n_pairs": int(len(delta)),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": float(delta.mean()),
        "relative_delta_pct": (
            float(100.0 * delta.mean() / baseline.mean())
            if abs(float(baseline.mean())) > 1e-15 else None
        ),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "candidate_better_fraction": float(np.mean(candidate < baseline)),
        "paired_randomization_p": float((extreme + 1) / (randomization_samples + 1)),
    }


def _holm_adjust(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda row: row["paired_randomization_p"])
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * row["paired_randomization_p"])
        running = max(running, adjusted)
        row["holm_p_within_metric"] = running


def _load_role(root: Path, role: str) -> tuple[dict, dict[str, dict]]:
    directory = root / f"direct32_{role}_best_validation_v1"
    summary = json.loads((directory / "direct32_summary.json").read_text(encoding="utf-8"))
    identity = summary["identity"]
    if identity["role"] != role or identity["split"] != "validation":
        raise ValueError(f"invalid identity for {role}")
    if identity.get("test_accessed") is not False:
        raise ValueError(f"{role} unexpectedly accessed test")
    with (directory / "direct32_per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or len({row["sample"] for row in rows}) != 200:
        raise ValueError(f"{role} must contain 200 unique samples")
    return identity, {row["sample"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--randomization-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--roles",
        default=",".join(DEFAULT_ROLES),
        help="comma-separated roles with B first (default: B,M,G,MG)",
    )
    args = parser.parse_args()
    roles = parse_roles(args.roles)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    identities = {}
    tables = {}
    for role in roles:
        identities[role], tables[role] = _load_role(args.root, role)
    sample_sets = {role: set(table) for role, table in tables.items()}
    if len({frozenset(value) for value in sample_sets.values()}) != 1:
        raise ValueError(f"sample membership differs across {roles}")
    manifest_hashes = {identity["manifest_sha256"] for identity in identities.values()}
    protocol_hashes = {identity["protocol_sha256"] for identity in identities.values()}
    git_commits = {identity["git"]["commit"] for identity in identities.values()}
    if len(manifest_hashes) != 1 or len(protocol_hashes) != 1 or len(git_commits) != 1:
        raise ValueError(f"evaluator identities differ across {roles}")

    samples = sorted(sample_sets["B"])
    records = []
    metric_groups = {metric: [] for metric in (*BASE_METRICS, *RATIO_METRICS)}
    for role in roles[1:]:
        for metric in BASE_METRICS:
            baseline = [float(tables["B"][sample][metric]) for sample in samples]
            candidate = [float(tables[role][sample][metric]) for sample in samples]
            stats = paired_statistics(
                baseline, candidate,
                bootstrap_samples=args.bootstrap_samples,
                randomization_samples=args.randomization_samples,
                seed=_stable_seed(role, metric, args.seed),
            )
            row = {"role": role, "metric": metric, "transform": "identity", **stats}
            records.append(row)
            metric_groups[metric].append(row)
        for metric in RATIO_METRICS:
            baseline_raw = np.asarray(
                [float(tables["B"][sample][metric]) for sample in samples]
            )
            candidate_raw = np.asarray(
                [float(tables[role][sample][metric]) for sample in samples]
            )
            baseline = np.abs(np.log(baseline_raw))
            candidate = np.abs(np.log(candidate_raw))
            stats = paired_statistics(
                baseline, candidate,
                bootstrap_samples=args.bootstrap_samples,
                randomization_samples=args.randomization_samples,
                seed=_stable_seed(role, metric, args.seed),
            )
            row = {"role": role, "metric": metric, "transform": "abs_log_ratio", **stats}
            records.append(row)
            metric_groups[metric].append(row)
    for group in metric_groups.values():
        _holm_adjust(group)

    args.output_dir.mkdir(parents=False, exist_ok=False)
    fieldnames = list(records[0])
    with (args.output_dir / "paired_validation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "identity": {
            "manifest_sha256": next(iter(manifest_hashes)),
            "protocol_sha256": next(iter(protocol_hashes)),
            "evaluator_git_commit": next(iter(git_commits)),
            "checkpoint_sha256": {
                role: identity["checkpoint_sha256"] for role, identity in identities.items()
            },
            "checkpoint_epoch": {
                role: identity["checkpoint_epoch"] for role, identity in identities.items()
            },
            "samples": len(samples),
            "bootstrap_samples": args.bootstrap_samples,
            "randomization_samples": args.randomization_samples,
            "seed": args.seed,
            "test_accessed": False,
        },
        "results": records,
    }
    (args.output_dir / "paired_validation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Role Metric                 Delta       95% CI                 Holm-p")
    selected = {"ADE_m", "FDE_m", "BoneErrorAll_mm", "AngleError_deg", *RATIO_METRICS}
    for row in records:
        if row["metric"] in selected:
            print(
                f"{row['role']:<4} {row['metric']:<22} "
                f"{row['mean_delta']:+.6f}  "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]  "
                f"{row['holm_p_within_metric']:.6f}"
            )
    print(f"[OK] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
