#!/usr/bin/env python3
"""Compare three-seed B against frozen Direct32 context-only controls."""

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
    SEEDS,
    aggregate_seed_rows,
    holm_adjust,
)
from evaluation.compare_direct32 import (
    BASE_METRICS,
    RATIO_METRICS,
    _stable_seed,
    paired_statistics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_METRICS = (*BASE_METRICS, *RATIO_METRICS)
METHODS = ("LastPose", "ConstVelocityK4")
CONTRASTS = tuple((f"B_minus_{method}", method) for method in METHODS)
STRUCTURALLY_ZERO_RATIO_METRICS = {
    "LastPose": frozenset(("VelocityRatio", "AccelerationRatio")),
    "ConstVelocityK4": frozenset(("AccelerationRatio",)),
}


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, encoding="utf-8",
    ).strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_b(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table = {row["sample"]: row for row in rows}
    if len(rows) != 200 or len(table) != 200 or {row["role"] for row in rows} != {"B"}:
        raise ValueError(f"invalid B table: {path}")
    return table


def _read_controls(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tables = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        table = {row["sample"]: row for row in selected}
        if len(selected) != 200 or len(table) != 200:
            raise ValueError(f"invalid control table for {method}")
        tables[method] = table
    if len(rows) != 400:
        raise ValueError("controls table must contain exactly 400 rows")
    return tables


def _values(table: dict[str, dict], samples: list[str], metric: str):
    raw = np.asarray([float(table[sample][metric]) for sample in samples])
    if metric in RATIO_METRICS:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(np.log(raw)), "abs_log_ratio"
    return raw, "identity"


def finite_pair_count(baseline, candidate) -> int:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    return int(np.count_nonzero(np.isfinite(baseline) & np.isfinite(candidate)))


def task_macro_statistics(
    baseline,
    candidate,
    tasks,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    tasks = np.asarray(tasks)
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline, candidate, tasks = baseline[valid], candidate[valid], tasks[valid]
    task_names = sorted(set(tasks.tolist()))
    if len(task_names) != 13:
        raise ValueError(f"task-macro inference requires 13 tasks, got {len(task_names)}")
    baseline_task = np.asarray([
        baseline[tasks == task].mean() for task in task_names
    ])
    candidate_task = np.asarray([
        candidate[tasks == task].mean() for task in task_names
    ])
    delta = candidate_task - baseline_task
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    combinations = 1 << len(delta)
    masks = np.arange(combinations, dtype=np.uint16)[:, None]
    bits = (masks >> np.arange(len(delta), dtype=np.uint16)[None]) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    observed = abs(float(delta.mean()))
    permuted = np.abs((signs * delta[None]).mean(axis=1))
    p_value = float(np.count_nonzero(permuted >= observed - 1e-15) / combinations)
    return {
        "n_tasks": len(task_names),
        "baseline_task_macro_mean": float(baseline_task.mean()),
        "candidate_task_macro_mean": float(candidate_task.mean()),
        "mean_delta": float(delta.mean()),
        "relative_delta_pct": (
            float(100.0 * delta.mean() / baseline_task.mean())
            if abs(float(baseline_task.mean())) > 1e-15 else None
        ),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "candidate_better_task_fraction": float(np.mean(candidate_task < baseline_task)),
        "task_exact_signflip_p": p_value,
    }


def _aggregate(sample_payloads: list[dict]) -> list[dict]:
    result = []
    for contrast, _ in CONTRASTS:
        payloads = [{
            "paired_rows": [
                row for row in payload["sample_rows"]
                if row["contrast"] == contrast
            ]
        } for payload in sample_payloads]
        for row in aggregate_seed_rows(payloads):
            result.append({"contrast": contrast, **row})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct32-root", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")

    control_summary = _read_json(
        args.controls_dir / "direct32_controls_summary.json"
    )
    control_identity = control_summary["identity"]
    if control_identity.get("test_accessed") is not False:
        raise ValueError("controls unexpectedly accessed locked test")
    audit = control_identity["cluster_audit"]
    if audit != {
        "tasks": 13,
        "episode_clusters": 200,
        "max_samples_per_episode": 1,
        "episode_cluster_inference": "degenerate_to_sample_level",
    }:
        raise ValueError("control cluster audit differs from frozen identity")
    controls = _read_controls(
        args.controls_dir / "direct32_controls_per_sample.csv"
    )
    payloads = []
    unavailable_rows = []
    per_task_rows = []

    for seed in SEEDS:
        b_dir = args.direct32_root / f"seed{seed}/direct32_B_best_validation_v1"
        b_summary = _read_json(b_dir / "direct32_summary.json")
        b_identity = b_summary["identity"]
        if b_identity.get("test_accessed") is not False:
            raise ValueError(f"seed {seed} B unexpectedly accessed locked test")
        if b_identity["manifest_sha256"] != control_identity["manifest_sha256"]:
            raise ValueError(f"seed {seed} B/control manifest mismatch")
        if int(b_identity["checkpoint_args"]["seed"]) != seed:
            raise ValueError(f"seed {seed} B checkpoint identity mismatch")
        b_table = _read_b(b_dir / "direct32_per_sample.csv")
        if any(set(table) != set(b_table) for table in controls.values()):
            raise ValueError(f"seed {seed} B/control sample membership differs")
        samples = sorted(b_table)
        tasks = [controls["LastPose"][sample]["task"] for sample in samples]
        sample_rows = []
        task_rows = []

        for contrast, method in CONTRASTS:
            for metric in ALL_METRICS:
                baseline_raw = np.asarray([
                    float(controls[method][sample][metric]) for sample in samples
                ])
                candidate_raw = np.asarray([
                    float(b_table[sample][metric]) for sample in samples
                ])
                if metric in STRUCTURALLY_ZERO_RATIO_METRICS[method]:
                    unavailable_rows.append({
                        "seed": seed,
                        "contrast": contrast,
                        "baseline_role": method,
                        "candidate_role": "B",
                        "family": METRIC_TO_FAMILY[metric],
                        "metric": metric,
                        "requested_transform": "abs_log_ratio",
                        "finite_pairs": 0,
                        "total_pairs": len(samples),
                        "baseline_raw_zero_count": int(
                            np.count_nonzero(baseline_raw == 0.0)
                        ),
                        "baseline_raw_max_abs": float(
                            np.nanmax(np.abs(baseline_raw))
                        ),
                        "candidate_raw_zero_count": int(
                            np.count_nonzero(candidate_raw == 0.0)
                        ),
                        "reason": "ratio_structurally_zero_by_control_definition",
                        "inference": "not_imputed_not_tested",
                    })
                    continue
                baseline, transform = _values(controls[method], samples, metric)
                candidate, candidate_transform = _values(b_table, samples, metric)
                if transform != candidate_transform:
                    raise AssertionError("metric transform mismatch")
                stable_seed = _stable_seed(contrast, metric, 20260720 + seed)
                finite_pairs = finite_pair_count(baseline, candidate)
                if finite_pairs < 2:
                    unavailable_rows.append({
                        "seed": seed,
                        "contrast": contrast,
                        "baseline_role": method,
                        "candidate_role": "B",
                        "family": METRIC_TO_FAMILY[metric],
                        "metric": metric,
                        "requested_transform": transform,
                        "finite_pairs": finite_pairs,
                        "total_pairs": len(samples),
                        "baseline_raw_zero_count": int(
                            np.count_nonzero(baseline_raw == 0.0)
                        ),
                        "baseline_raw_max_abs": float(
                            np.nanmax(np.abs(baseline_raw))
                        ),
                        "candidate_raw_zero_count": int(
                            np.count_nonzero(candidate_raw == 0.0)
                        ),
                        "reason": "abs_log_ratio_undefined_for_zero_control_ratio",
                        "inference": "not_imputed_not_tested",
                    })
                    continue
                sample_stats = paired_statistics(
                    baseline,
                    candidate,
                    bootstrap_samples=args.bootstrap_samples,
                    randomization_samples=args.bootstrap_samples,
                    seed=stable_seed,
                )
                shared = {
                    "seed": seed,
                    "contrast": contrast,
                    "baseline_role": method,
                    "candidate_role": "B",
                    "family": METRIC_TO_FAMILY[metric],
                    "metric": metric,
                    "transform": transform,
                }
                sample_rows.append({**shared, **sample_stats})
                task_rows.append({
                    **shared,
                    **task_macro_statistics(
                        baseline,
                        candidate,
                        tasks,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=stable_seed + 1,
                    ),
                })
                task_array = np.asarray(tasks)
                for task in sorted(set(tasks)):
                    mask = task_array == task
                    finite = mask & np.isfinite(baseline) & np.isfinite(candidate)
                    if not np.any(finite):
                        continue
                    baseline_task = float(baseline[finite].mean())
                    candidate_task = float(candidate[finite].mean())
                    per_task_rows.append({
                        **shared,
                        "task": task,
                        "n_samples": int(np.count_nonzero(finite)),
                        "baseline_task_mean": baseline_task,
                        "candidate_task_mean": candidate_task,
                        "mean_delta": candidate_task - baseline_task,
                        "candidate_better": candidate_task < baseline_task,
                    })

        for family in METRIC_FAMILIES:
            holm_adjust(
                [row for row in sample_rows if row["family"] == family]
            )
            holm_adjust(
                [row for row in task_rows if row["family"] == family],
                p_key="task_exact_signflip_p",
            )
        payloads.append({
            "seed": seed,
            "b_identity": b_identity,
            "sample_rows": sample_rows,
            "task_rows": task_rows,
        })

    sample_rows = [row for payload in payloads for row in payload["sample_rows"]]
    task_rows = [row for payload in payloads for row in payload["task_rows"]]
    model_rows = _aggregate(payloads)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    for filename, rows in (
        ("sample_paired.csv", sample_rows),
        ("task_macro_cluster.csv", task_rows),
        ("model_seed_summary.csv", model_rows),
        ("per_task_effects.csv", per_task_rows),
        ("unavailable_metrics.csv", unavailable_rows),
    ):
        with (args.output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = {
        "identity": {
            "analysis_git_commit": _git_commit(),
            "seeds": list(SEEDS),
            "n_training_seeds": 3,
            "unit_of_replication": "trained_model_seed",
            "cross_seed_inference": "descriptive_only_n3",
            "manifest_sha256": control_identity["manifest_sha256"],
            "controls_protocol_sha256": control_identity["protocol_sha256"],
            "control_evaluator_commit": control_identity["git"]["commit"],
            "cluster_audit": audit,
            "task_inference": "13_task_macro_cluster_bootstrap_and_exact_signflip",
            "episode_inference": "not_reported_degenerate_one_sample_per_episode",
            "zero_ratio_policy": "not_imputed_not_tested",
            "unavailable_metric_rows": len(unavailable_rows),
            "test_accessed": False,
        },
        "metric_families": METRIC_FAMILIES,
        "model_seed_summary": model_rows,
    }
    (args.output_dir / "direct32_controls_multiseed_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    selected = {"ADE_m", "FDE_m"}
    for contrast, _ in CONTRASTS:
        print(f"===== {contrast}: model-seed descriptive =====")
        for row in model_rows:
            if row["contrast"] == contrast and row["metric"] in selected:
                print(
                    f"{row['metric']:<8} delta={row['paired_delta_seed_mean']:+.6f}+/-"
                    f"{row['paired_delta_seed_sd']:.6f} improved={row['improved_seed_count']}/3"
                )
        print(f"===== {contrast}: task-macro by seed =====")
        for row in task_rows:
            if row["contrast"] == contrast and row["metric"] in selected:
                print(
                    f"seed={row['seed']} {row['metric']:<8} "
                    f"delta={row['mean_delta']:+.6f} "
                    f"CI=[{row['ci95_low']:+.6f},{row['ci95_high']:+.6f}] "
                    f"family_holm_p={row['holm_p_within_family']:.6f}"
                )
        print(f"===== {contrast}: task direction counts =====")
        for metric in selected:
            for seed in SEEDS:
                rows = [
                    row for row in per_task_rows
                    if row["contrast"] == contrast
                    and row["metric"] == metric
                    and int(row["seed"]) == seed
                ]
                improved = sum(bool(row["candidate_better"]) for row in rows)
                worst = max(rows, key=lambda row: float(row["mean_delta"]))
                print(
                    f"seed={seed} {metric:<8} improved_tasks={improved}/13 "
                    f"worst={worst['task']} delta={float(worst['mean_delta']):+.6f}"
                )
    print("Episode cluster inference: NOT REPORTED (200 clusters x 1 sample)")
    print("Unavailable transformed dynamics (zero control ratio):")
    for row in unavailable_rows:
        print(
            f"seed={row['seed']} {row['contrast']} {row['metric']} "
            f"finite={row['finite_pairs']}/{row['total_pairs']} "
            "NOT IMPUTED / NOT TESTED"
        )
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
