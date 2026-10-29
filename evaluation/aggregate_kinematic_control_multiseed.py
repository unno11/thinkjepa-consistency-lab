#!/usr/bin/env python3
"""Aggregate three-seed B/MG/KProj Direct32 validation comparisons.

The script first proves that the B rows produced by the original multi-seed
evaluator and the KProj-enabled evaluator are numerically equivalent.  It then
computes B-vs-KProj, B-vs-MG, and KProj-vs-MG paired sample statistics and a
descriptive model-seed summary (n=3 trained model seeds).
"""

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
from evaluation.compare_direct32_controls_multiseed import task_macro_statistics
from evaluation.direct32_controls_protocol import sample_groups


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_METRICS = (*BASE_METRICS, *RATIO_METRICS)
CONTRASTS = (
    ("KProj_minus_B", "B", "KProj"),
    ("MG_minus_B", "B", "MG"),
    ("MG_minus_KProj", "KProj", "MG"),
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, encoding="utf-8",
    ).strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path, expected_role: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or len({row["sample"] for row in rows}) != 200:
        raise ValueError(f"expected 200 unique samples in {path}")
    if {row["role"] for row in rows} != {expected_role}:
        raise ValueError(f"unexpected role in {path}")
    return {row["sample"]: row for row in rows}


def _metric_values(table: dict[str, dict], samples: list[str], metric: str):
    raw = np.asarray([float(table[sample][metric]) for sample in samples])
    if metric in RATIO_METRICS:
        return np.abs(np.log(raw)), "abs_log_ratio"
    return raw, "identity"


def _validate_b_equivalence(
    original: dict[str, dict], repeated: dict[str, dict], *, atol: float
) -> float:
    if set(original) != set(repeated):
        raise ValueError("original/repeated B sample membership differs")
    maximum = 0.0
    for sample in original:
        for metric in ALL_METRICS:
            delta = abs(float(original[sample][metric]) - float(repeated[sample][metric]))
            maximum = max(maximum, delta)
    if maximum > atol:
        raise ValueError(
            f"repeated B metrics differ from frozen B: max={maximum} > {atol}"
        )
    return maximum


def _load_seed(direct32_root: Path, outputs_root: Path, seed: int, atol: float):
    seed_root = direct32_root / f"seed{seed}"
    original_summaries = {
        role: _read_json(
            seed_root / f"direct32_{role}_best_validation_v1/direct32_summary.json"
        )
        for role in ("B", "MG")
    }
    original_tables = {
        role: _read_table(
            seed_root / f"direct32_{role}_best_validation_v1/direct32_per_sample.csv",
            role,
        )
        for role in ("B", "MG")
    }
    kproj_root = outputs_root / f"kproj_B_alllayers_gate0fixed_seed{seed}_validation_v1"
    repeated_summary = _read_json(kproj_root / "direct32_summary.json")
    projection_summary = _read_json(kproj_root / "kinematic_projection_summary.json")
    repeated_b = _read_table(kproj_root / "direct32_per_sample.csv", "B")
    projection_role = projection_summary["identity"]["role"]
    kproj = _read_table(
        kproj_root / "kinematic_projection_per_sample.csv", projection_role
    )

    identities = {
        "B": original_summaries["B"]["identity"],
        "MG": original_summaries["MG"]["identity"],
        "repeated_B": repeated_summary["identity"],
        "KProj": projection_summary["identity"],
    }
    if any(identity.get("test_accessed") is not False for identity in identities.values()):
        raise ValueError(f"seed {seed} unexpectedly accessed locked test")
    if identities["B"]["checkpoint_sha256"] != identities["repeated_B"]["checkpoint_sha256"]:
        raise ValueError(f"seed {seed} repeated B checkpoint differs")
    if identities["B"]["manifest_sha256"] != identities["KProj"]["manifest_sha256"]:
        raise ValueError(f"seed {seed} KProj manifest differs")
    if identities["KProj"].get("target_trajectory_passed_to_projection") is not False:
        raise ValueError(f"seed {seed} KProj received future targets")
    if identities["KProj"].get("observed_trajectory_passed_to_projection") is not False:
        raise ValueError(f"seed {seed} KProj received observed targets")
    expected_seed = int(identities["B"]["checkpoint_args"]["seed"])
    if expected_seed != seed:
        raise ValueError(f"seed {seed} B identity says seed {expected_seed}")

    max_b_delta = _validate_b_equivalence(
        original_tables["B"], repeated_b, atol=atol
    )
    tables = {"B": original_tables["B"], "MG": original_tables["MG"], "KProj": kproj}
    if len({frozenset(table) for table in tables.values()}) != 1:
        raise ValueError(f"seed {seed} method sample membership differs")
    samples = sorted(tables["B"])
    tasks = [sample_groups(sample)[0] for sample in samples]
    episodes = [sample_groups(sample)[1] for sample in samples]
    if len(set(tasks)) != 13:
        raise ValueError(f"seed {seed} expected 13 tasks")
    if len(set(episodes)) != 200:
        raise ValueError(f"seed {seed} episode clusters are not 200x1")
    records = []
    task_records = []
    per_task_records = []

    for contrast, baseline_role, candidate_role in CONTRASTS:
        family_rows = {family: [] for family in METRIC_FAMILIES}
        for metric in ALL_METRICS:
            baseline, transform = _metric_values(tables[baseline_role], samples, metric)
            candidate, candidate_transform = _metric_values(
                tables[candidate_role], samples, metric
            )
            if transform != candidate_transform:
                raise AssertionError("metric transform mismatch")
            stats = paired_statistics(
                baseline,
                candidate,
                bootstrap_samples=10000,
                randomization_samples=10000,
                seed=_stable_seed(contrast, metric, 20260720 + seed),
            )
            row = {
                "seed": seed,
                "contrast": contrast,
                "baseline_role": baseline_role,
                "candidate_role": candidate_role,
                "family": METRIC_TO_FAMILY[metric],
                "metric": metric,
                "transform": transform,
                **stats,
            }
            records.append(row)
            family_rows[row["family"]].append(row)
            shared = {
                "seed": seed,
                "contrast": contrast,
                "baseline_role": baseline_role,
                "candidate_role": candidate_role,
                "family": METRIC_TO_FAMILY[metric],
                "metric": metric,
                "transform": transform,
            }
            task_records.append({
                **shared,
                **task_macro_statistics(
                    baseline,
                    candidate,
                    tasks,
                    bootstrap_samples=10000,
                    seed=_stable_seed(contrast, metric, 20260721 + seed),
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
                per_task_records.append({
                    **shared,
                    "task": task,
                    "n_samples": int(np.count_nonzero(finite)),
                    "baseline_task_mean": baseline_task,
                    "candidate_task_mean": candidate_task,
                    "mean_delta": candidate_task - baseline_task,
                    "candidate_better": candidate_task < baseline_task,
                })
        for rows in family_rows.values():
            holm_adjust(rows)

    # Correct across all three pre-declared contrasts and all metrics in each
    # family.  This is intentionally stricter than correcting each contrast in
    # isolation.
    for family in METRIC_FAMILIES:
        holm_adjust(
            [row for row in task_records if row["family"] == family],
            p_key="task_exact_signflip_p",
        )

    return {
        "seed": seed,
        "identities": identities,
        "max_repeated_b_metric_delta": max_b_delta,
        "paired_rows": records,
        "task_rows": task_records,
        "per_task_rows": per_task_records,
    }


def _aggregate(payloads: list[dict]) -> list[dict]:
    result = []
    for contrast, _, _ in CONTRASTS:
        contrast_payloads = []
        for payload in payloads:
            contrast_payloads.append({
                "paired_rows": [
                    row for row in payload["paired_rows"]
                    if row["contrast"] == contrast
                ]
            })
        for row in aggregate_seed_rows(contrast_payloads):
            result.append({"contrast": contrast, **row})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct32-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b-equivalence-atol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_dir}")

    payloads = [
        _load_seed(
            args.direct32_root, args.outputs_root, seed, args.b_equivalence_atol
        )
        for seed in SEEDS
    ]
    per_seed_rows = [row for payload in payloads for row in payload["paired_rows"]]
    task_rows = [row for payload in payloads for row in payload["task_rows"]]
    per_task_rows = [row for payload in payloads for row in payload["per_task_rows"]]
    model_rows = _aggregate(payloads)
    args.output_dir.mkdir(parents=False, exist_ok=False)

    with (args.output_dir / "per_seed_pairwise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (args.output_dir / "model_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    with (args.output_dir / "task_macro_cluster.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    with (args.output_dir / "per_task_effects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task_rows[0]))
        writer.writeheader()
        writer.writerows(per_task_rows)

    result = {
        "identity": {
            "analysis_git_commit": _git_commit(),
            "seeds": list(SEEDS),
            "n_training_seeds": len(SEEDS),
            "unit_of_replication": "trained_model_seed",
            "cross_seed_inference": "descriptive_only_n3",
            "max_repeated_b_metric_delta_by_seed": {
                str(payload["seed"]): payload["max_repeated_b_metric_delta"]
                for payload in payloads
            },
            "b_equivalence_atol": args.b_equivalence_atol,
            "original_evaluator_commit": payloads[0]["identities"]["B"]["git"]["commit"],
            "kproj_evaluator_commit": payloads[0]["identities"]["KProj"]["git"]["commit"],
            "manifest_sha256": payloads[0]["identities"]["B"]["manifest_sha256"],
            "projection_protocol_sha256": payloads[0]["identities"]["KProj"]["projection_protocol_sha256"],
            "test_accessed": False,
            "task_inference": "13_task_macro_cluster_bootstrap_and_exact_signflip",
            "task_holm_scope": "all_three_contrasts_within_metric_family_per_seed",
            "episode_inference": "not_reported_degenerate_200_clusters_x_1_sample",
        },
        "metric_families": METRIC_FAMILIES,
        "model_seed_summary": model_rows,
    }
    (args.output_dir / "B_MG_KProj_multiseed_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    selected = {
        "ADE_m", "FDE_m", "BoneErrorAll_mm", "BoneDriftAll_mm2",
        "AngleError_deg", "VelocityRatio", "AccelerationRatio",
    }
    for contrast, _, _ in CONTRASTS:
        print(f"===== {contrast} =====")
        for row in model_rows:
            if row["contrast"] == contrast and row["metric"] in selected:
                print(
                    f"{row['metric']:<22} "
                    f"delta={row['paired_delta_seed_mean']:+.6f}+/-"
                    f"{row['paired_delta_seed_sd']:.6f} "
                    f"improved={row['improved_seed_count']}/3"
                )
        print(f"===== {contrast}: task-macro gate =====")
        for metric in selected:
            for seed in SEEDS:
                row = next(
                    row for row in task_rows
                    if row["contrast"] == contrast
                    and row["metric"] == metric
                    and int(row["seed"]) == seed
                )
                effects = [
                    effect for effect in per_task_rows
                    if effect["contrast"] == contrast
                    and effect["metric"] == metric
                    and int(effect["seed"]) == seed
                ]
                improved = sum(bool(effect["candidate_better"]) for effect in effects)
                worst = max(effects, key=lambda effect: float(effect["mean_delta"]))
                print(
                    f"seed={seed} {metric:<22} delta={row['mean_delta']:+.6f} "
                    f"CI=[{row['ci95_low']:+.6f},{row['ci95_high']:+.6f}] "
                    f"holm={row['holm_p_within_family']:.6f} "
                    f"improved_tasks={improved}/13 worst={worst['task']} "
                    f"worst_delta={float(worst['mean_delta']):+.6f}"
                )
    print("Repeated-B maximum metric deltas:", result["identity"]["max_repeated_b_metric_delta_by_seed"])
    print(f"[OK] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
