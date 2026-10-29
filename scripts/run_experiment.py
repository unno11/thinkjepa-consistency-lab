#!/usr/bin/env python3
"""Resolve and optionally launch a reproducible ThinkJEPA experiment.

The default is a side-effect-free dry run.  The public trainer calls its
evaluation manifest ``--test_manifest``; this wrapper deliberately supplies
the validation manifest there and has no final-test option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "cache_train" / "thinker_train.py"
FORBIDDEN_CONFIG_KEYS = {
    "data_dir", "cache_dir", "train_manifest", "test_manifest",
    "output_dir", "output_mp4", "results_md", "resume_ckpt",
    "auto_resume", "ddp",
}
ALLOWED_CONFIG_KEYS = {
    "epochs", "no_amp", "skip_nonfinite_loss", "debug", "lr",
    "grad_accum", "trajmode", "past_T", "future_T", "temporal_stride",
    "temporal_causal_attn", "backbone", "vjepa_predictor", "predictor",
    "optimize_together_downstream", "lambda_pred", "lambda_task",
    "lr_pred", "ref_mode", "joint_pred", "camera_mode", "use_npz_cache",
    "preload_cache_to_memory", "no_preload_cache_to_memory", "skip_vjepa",
    "shards", "shards_id", "train_ratio", "split_seed", "seed",
    "no_split_shuffle", "num_workers", "pin_memory", "persistent_workers",
    "prefetch_factor", "ddp_find_unused_parameters", "train_batch_size",
    "test_batch_size", "no_fast_hdf5_index", "thinkjepa_use_cache_ext",
    "no_thinkjepa_use_cache_ext", "thinkjepa_use_vlm_merge",
    "no_thinkjepa_use_vlm_merge", "thinkjepa_direct_vit_condition",
    "vlm_pad_old_to", "vlm_pad_new_to", "thinkjepa_vlm_old_dim",
    "thinkjepa_vlm_new_dim", "thinkjepa_vlm_source",
    "thinkjepa_vlm_layer_selector", "thinkjepa_vlm_layer_index",
    "thinkjepa_vlm_cond_mode", "thinkjepa_drop_thinking_tokens",
    "thinkjepa_think_start_ids", "thinkjepa_think_end_ids",
    "thinkjepa_think_drop_ids", "thinkjepa_think_token_pad_id",
    "thinkjepa_think_prefix_open", "thinkjepa_think_drop_prefix_len",
    "thinkjepa_think_drop_suffix_len", "thinkjepa_zero_dropped_think_tokens",
    "no_thinkjepa_zero_dropped_think_tokens", "thinkjepa_verbose",
    "zero_visual_input", "max_visual_batches",
    "lambda_bone_loss", "lambda_palm_bone_loss", "lambda_velocity_loss",
    "lambda_acceleration_loss", "lambda_angle_loss",
    "physics_loss_start_epoch", "physics_loss_warmup_epochs",
}


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise ValueError("config name must be a non-empty string")
    args = config.get("trainer_args")
    if not isinstance(args, dict):
        raise ValueError("trainer_args must be an object")
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS & args.keys())
    unknown = sorted(set(args) - ALLOWED_CONFIG_KEYS)
    if forbidden:
        raise ValueError(f"runtime-only keys in config: {forbidden}")
    if unknown:
        raise ValueError(f"unknown trainer keys: {unknown}")
    if args.get("trajmode") != "traj" or args.get("joint_pred") is not True:
        raise ValueError("baseline must explicitly set trajmode=traj and joint_pred=true")
    if not isinstance(args.get("skip_nonfinite_loss"), bool):
        raise ValueError("config must explicitly declare boolean skip_nonfinite_loss")
    return config


def _append_arg(command: list[str], key: str, value: Any) -> None:
    flag = "--" + key
    if isinstance(value, bool):
        if value:
            command.append(flag)
    elif value is not None:
        command.extend((flag, str(value)))


def build_command(
    config: dict[str, Any], *, python: str, data_dir: Path,
    train_manifest: Path, val_manifest: Path, output_dir: Path,
    nproc_per_node: int,
) -> list[str]:
    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be at least 1")
    if nproc_per_node == 1:
        command = [python, str(TRAINER)]
    else:
        command = [python, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={nproc_per_node}", str(TRAINER), "--ddp"]
    runtime = {
        "data_dir": data_dir,
        "cache_dir": data_dir / "cache",
        "train_manifest": train_manifest,
        # Legacy trainer name; semantically this is validation-only.
        "test_manifest": val_manifest,
        "output_dir": output_dir,
        "output_mp4": output_dir / "visualizations",
        "results_md": output_dir / "results.md",
    }
    for key, value in runtime.items():
        _append_arg(command, key, value)
    for key, value in config["trainer_args"].items():
        _append_arg(command, key, value)
    return command


def _git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def _fingerprint(path: Path) -> str:
    if not path.is_file():
        return "missing (dry-run only)"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "thinkjepa_B_alllayers_100ep.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    config = load_and_validate_config(ns.config.resolve())
    run_name = ns.run_name or config["name"]
    output_dir = ns.output_root.resolve() / run_name
    command = build_command(
        config, python=ns.python, data_dir=ns.data_dir.resolve(),
        train_manifest=ns.train_manifest.resolve(), val_manifest=ns.val_manifest.resolve(),
        output_dir=output_dir, nproc_per_node=ns.nproc_per_node,
    )
    resolved = {
        "config": config,
        "runtime": {"data_dir": str(ns.data_dir.resolve()),
                    "train_manifest": str(ns.train_manifest.resolve()),
                    "validation_manifest": str(ns.val_manifest.resolve()),
                    "output_dir": str(output_dir), "gpus": ns.gpus,
                    "nproc_per_node": ns.nproc_per_node},
        "manifest_sha256": {
            "train": _fingerprint(ns.train_manifest.resolve()),
            "validation": _fingerprint(ns.val_manifest.resolve()),
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "git": _git_identity(), "command": command,
    }
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    print("COMMAND:", shlex.join(command))
    if not ns.execute:
        print("DRY RUN: pass --execute only after reviewing the resolved configuration.")
        return 0

    required = (ns.data_dir, ns.train_manifest, ns.val_manifest)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"required input paths do not exist: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "launch_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ns.gpus
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VJEPA2_ROOT", str(REPO_ROOT / "vjepa2"))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
