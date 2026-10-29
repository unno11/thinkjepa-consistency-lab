#!/usr/bin/env python3
"""Aggregate Gate0-fixed B/MG Direct32 results across training seeds.

The unit of replication in the cross-seed table is a trained model seed, not an
individual validation sample. Per-seed paired tests remain available, with an
additional Holm correction across pre-declared metric families.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


SEEDS = (42, 43, 44)
METRIC_FAMILIES = {
    "spatial": (
        "ADE_m",
        "FDE_m",
        "RootError_m",
        "RootAligned_ADE_m",
        "RootAligned_Finger_ADE_m",
    ),
    "geometry": (
        "BoneErrorAll_mm",
        "BoneErrorFinger_mm",
        "BoneErrorPalm_mm",
        "BoneDriftAll_mm2",
        "AngleError_deg",
    ),
    "dynamics": ("VelocityRatio", "AccelerationRatio"),
}
METRIC_TO_FAMILY = {
    metric: family
    for family, metrics in METRIC_FAMILIES.items()
    for metric in metrics
}


def holm_adjust(rows: list[dict], p_key: str = "paired_randomization_p") -> None:
    """Add monotone Holm-adjusted p-values to rows in place."""
    ordered = sorted(rows, key=lambda row: float(row[p_key]))
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * float(row[p_key]))
        running = max(running, adjusted)
        row["holm_p_within_family"] = running


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _analysis_git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        encoding="utf-8",
    ).strip()


def _validate_per_sample(path: Path, role: str) -> set[str]:
    rows = _read_csv(path)
    samples = {row["sample"] for row in rows}
    if len(rows) != 200 or len(samples) != 200:
        raise ValueError(f"seed role {role} must contain 200 unique samples")
    return samples


def load_seed(root: Path, seed: int) -> dict:
    seed_root = root / f"seed{seed}"
    summaries = {
        role: _read_json(
            seed_root / f"direct32_{role}_best_validation_v1/direct32_summary.json"
        )
        for role in ("B", "MG")
    }
    identities = {role: payload["identity"] for role, payload in summaries.items()}
    samples = {
        role: _validate_per_sample(
            seed_root
            / f"direct32_{role}_best_validation_v1/direct32_per_sample.csv",
            role,
        )
        for role in ("B", "MG")
    }
    if samples["B"] != samples["MG"]:
        raise ValueError(f"seed {seed} B/MG sample membership differs")

    for role, identity in identities.items():
        if identity["role"] != role or identity["split"] != "validation":
            raise ValueError(f"seed {seed} invalid {role} identity")
        if identity.get("test_accessed") is not False:
            raise ValueError(f"seed {seed} {role} accessed locked test")
        actual_seed = int(identity["checkpoint_args"].get("seed", -1))
        if actual_seed != seed:
            raise ValueError(f"seed {seed} {role} checkpoint seed is {actual_seed}")

    same_keys = ("manifest_sha256", "protocol_sha256")
    for key in same_keys:
        if identities["B"][key] != identities["MG"][key]:
            raise ValueError(f"seed {seed} B/MG {key} differs")
    if identities["B"]["git"]["commit"] != identities["MG"]["git"]["commit"]:
        raise ValueError(f"seed {seed} B/MG evaluator commit differs")

    paired_json = _read_json(
        seed_root / "paired_B_MG_validation_v1/paired_validation_results.json"
    )
    paired_identity = paired_json["identity"]
    if paired_identity.get("test_accessed") is not False:
        raise ValueError(f"seed {seed} paired analysis accessed locked test")
    if paired_identity["protocol_sha256"] != identities["B"]["protocol_sha256"]:
        raise ValueError(f"seed {seed} paired protocol identity differs")
    if paired_identity["evaluator_git_commit"] != identities["B"]["git"]["commit"]:
        raise ValueError(f"seed {seed} paired evaluator identity differs")

    paired_rows = _read_csv(
        seed_root / "paired_B_MG_validation_v1/paired_validation_results.csv"
    )
    if {row["metric"] for row in paired_rows} != set(METRIC_TO_FAMILY):
        raise ValueError(f"seed {seed} paired metrics do not match frozen families")
    if any(row["role"] != "MG" for row in paired_rows):
        raise ValueError(f"seed {seed} paired candidate must be MG")

    family_rows: dict[str, list[dict]] = {
        family: [] for family in METRIC_FAMILIES
    }
    normalized_rows = []
    for row in paired_rows:
        normalized = dict(row)
        normalized["seed"] = seed
        normalized["family"] = METRIC_TO_FAMILY[row["metric"]]
        normalized_rows.append(normalized)
        family_rows[normalized["family"]].append(normalized)
    for rows in family_rows.values():
        holm_adjust(rows)

    return {
        "seed": seed,
        "identities": identities,
        "summaries": summaries,
        "paired_rows": normalized_rows,
    }


def aggregate_seed_rows(seed_payloads: list[dict]) -> list[dict]:
    by_metric = {metric: [] for metric in METRIC_TO_FAMILY}
    for payload in seed_payloads:
        for row in payload["paired_rows"]:
            by_metric[row["metric"]].append(row)

    records = []
    for metric, rows in by_metric.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        baseline = [float(row["baseline_mean"]) for row in rows]
        candidate = [float(row["candidate_mean"]) for row in rows]
        deltas = [float(row["mean_delta"]) for row in rows]
        baseline_mean = statistics.mean(baseline)
        candidate_mean = statistics.mean(candidate)
        delta_mean = statistics.mean(deltas)
        record = {
            "family": METRIC_TO_FAMILY[metric],
            "metric": metric,
            "transform": rows[0]["transform"],
            "n_training_seeds": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "baseline_seed_mean": baseline_mean,
            "baseline_seed_sd": statistics.stdev(baseline),
            "candidate_seed_mean": candidate_mean,
            "candidate_seed_sd": statistics.stdev(candidate),
            "paired_delta_seed_mean": delta_mean,
            "paired_delta_seed_sd": statistics.stdev(deltas),
            "relative_delta_pct": (
                100.0 * delta_mean / baseline_mean
                if abs(baseline_mean) > 1e-15
                else None
            ),
            "improved_seed_count": sum(delta < 0.0 for delta in deltas),
            "worsened_seed_count": sum(delta > 0.0 for delta in deltas),
            "zero_seed_count": sum(delta == 0.0 for delta in deltas),
            "seed_deltas": ";".join(f"{delta:.12g}" for delta in deltas),
            "model_level_inference": "descriptive_only_n3",
        }
        records.append(record)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")
    payloads = [load_seed(args.root, seed) for seed in SEEDS]

    evaluator_commits = {
        payload["identities"]["B"]["git"]["commit"] for payload in payloads
    }
    manifest_hashes = {
        payload["identities"]["B"]["manifest_sha256"] for payload in payloads
    }
    if len(evaluator_commits) != 1 or len(manifest_hashes) != 1:
        raise ValueError("cross-seed evaluator or manifest identity differs")

    family_rows = [
        row for payload in payloads for row in payload["paired_rows"]
    ]
    aggregate_rows = aggregate_seed_rows(payloads)
    args.output_dir.mkdir(parents=False, exist_ok=False)

    with (args.output_dir / "per_seed_family_corrected.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(family_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(family_rows)

    with (args.output_dir / "model_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(aggregate_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    result = {
        "identity": {
            "seeds": list(SEEDS),
            "n_training_seeds": len(SEEDS),
            "unit_of_replication": "trained_model_seed",
            "cross_seed_inference": "descriptive_only_n3",
            "analysis_git_commit": _analysis_git_commit(),
            "evaluator_git_commit": next(iter(evaluator_commits)),
            "manifest_sha256": next(iter(manifest_hashes)),
            "protocol_sha256_by_seed": {
                str(payload["seed"]): payload["identities"]["B"]["protocol_sha256"]
                for payload in payloads
            },
            "checkpoint_epoch_by_seed": {
                str(payload["seed"]): {
                    role: payload["identities"][role]["checkpoint_epoch"]
                    for role in ("B", "MG")
                }
                for payload in payloads
            },
            "checkpoint_sha256_by_seed": {
                str(payload["seed"]): {
                    role: payload["identities"][role]["checkpoint_sha256"]
                    for role in ("B", "MG")
                }
                for payload in payloads
            },
            "test_accessed": False,
        },
        "metric_families": METRIC_FAMILIES,
        "model_seed_summary": aggregate_rows,
    }
    (args.output_dir / "multiseed_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    selected = {
        "ADE_m", "FDE_m", "BoneErrorAll_mm", "BoneDriftAll_mm2",
        "AngleError_deg", "VelocityRatio", "AccelerationRatio",
    }
    print("Metric                 B mean+/-SD         MG mean+/-SD        Delta mean+/-SD     Seeds improved")
    for row in aggregate_rows:
        if row["metric"] not in selected:
            continue
        print(
            f"{row['metric']:<22} "
            f"{row['baseline_seed_mean']:.6f}+/-{row['baseline_seed_sd']:.6f}  "
            f"{row['candidate_seed_mean']:.6f}+/-{row['candidate_seed_sd']:.6f}  "
            f"{row['paired_delta_seed_mean']:+.6f}+/-{row['paired_delta_seed_sd']:.6f}  "
            f"{row['improved_seed_count']}/{row['n_training_seeds']}"
        )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
