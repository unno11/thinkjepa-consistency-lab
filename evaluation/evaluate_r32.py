#!/usr/bin/env python3
"""Evaluate a frozen checkpoint with the leak-free 4->4 x 8 R32 protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_naive_baselines import constant_velocity, last_pose
from evaluation.metrics import compute_trajectory_metrics
from evaluation.r32_protocol import (
    load_protocol,
    resolve_split,
    sha256_file,
    validate_checkpoint_args,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "r32_evaluation_protocol.json"


def _args_dict(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    return dict(vars(value))


def _summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "n_defined": int(len(finite)),
    }


def _git_identity() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        ).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "UNKNOWN", "status": "UNKNOWN"}


def _mask_token_count(state: dict) -> int:
    found = []
    for key in state:
        clean = key.removeprefix("module.")
        if clean.startswith(("mask_tokens.", "query_templates.")):
            try:
                found.append(int(clean.split(".")[1]))
            except (ValueError, IndexError):
                pass
    return max(found) + 1 if found else 2


def _strip_module_prefix(state: dict) -> dict:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _runtime_namespace(checkpoint_args: dict, cli, rollout: dict):
    values = dict(checkpoint_args)
    values.update({
        "data_dir": str(cli.data_dir.resolve()),
        "cache_dir": str(cli.cache_dir.resolve()),
        "past_T": rollout["past_frames"],
        "future_T": rollout["step_future_frames"],
        "temporal_stride": rollout["temporal_stride"],
        "predictor": rollout["predictor"],
        "trajmode": rollout["trajmode"],
        "joint_pred": rollout["joint_pred"],
        "ref_mode": rollout["ref_mode"],
        "camera_mode": "auto",
        "skip_vjepa": True,
        "use_npz_cache": True,
        "preload_cache_to_memory": False,
        "thinkjepa_use_cache_ext": True,
        "thinkjepa_use_vlm_merge": True,
        "thinkjepa_vlm_source": values.get("thinkjepa_vlm_source", "both"),
        "thinkjepa_vlm_layer_selector": values.get("thinkjepa_vlm_layer_selector", "last"),
        "thinkjepa_vlm_layer_index": values.get("thinkjepa_vlm_layer_index", -1),
        "thinkjepa_vlm_cond_mode": values.get("thinkjepa_vlm_cond_mode", "film"),
        "vlm_pad_old_to": values.get("vlm_pad_old_to", 480),
        "vlm_pad_new_to": values.get("vlm_pad_new_to", 15),
        "zero_visual_input": False,
        "r32_vlm_guidance_policy": rollout["vlm_guidance_policy"],
    })
    return SimpleNamespace(**values)


def _guidance_input_dim(state: dict, *suffixes: str) -> int:
    for key, tensor in state.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            if getattr(tensor, "ndim", 0) != 2:
                raise ValueError(f"invalid guidance adapter weight: {key}")
            return int(tensor.shape[1])
    raise ValueError(f"checkpoint is missing guidance adapter: {suffixes}")


def _build_predictor(checkpoint, runtime, feats, device):
    import torch
    from cache_train.thinker_predictor import CortexGuidedVideoPredictor

    state = _strip_module_prefix(checkpoint.get("predictor") or {})
    if not state:
        raise ValueError("checkpoint does not contain predictor state")
    _, _, patches, dim = feats.shape
    old_dim = _guidance_input_dim(
        state, "guidance_old_adapter.weight", "vlm_old_proj.weight"
    )
    new_dim = _guidance_input_dim(
        state, "guidance_new_adapter.weight", "vlm_new_proj.weight"
    )
    predictor = CortexGuidedVideoPredictor(
        img_size=(patches, 1), patch_size=1,
        num_frames=runtime.past_T + runtime.future_T, tubelet_size=1,
        embed_dim=dim, predictor_embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4.0, drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.1,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), init_std=0.02,
        uniform_power=False, use_mask_tokens=True,
        num_mask_tokens=_mask_token_count(state), zero_init_mask_tokens=True,
        use_silu=False, wide_silu=True, use_activation_checkpointing=False,
        return_all_tokens=False, chop_last_n_tokens=0, use_rope=True,
        use_vlm_merge=True, vlm_cond_mode=runtime.thinkjepa_vlm_cond_mode,
        vlm_old_dim=int(old_dim), vlm_new_dim=int(new_dim),
    ).to(device)
    predictor.load_state_dict(state, strict=True)
    predictor.eval()
    return predictor


def _predict_latents(train_mod, predictor, runtime, context):
    batch, context_len, patches, dim = context.shape
    sequence = train_mod.flatten_temporal_patch_tokens(context)
    context_idx = train_mod.build_temporal_patch_indices(patches, 0, context_len)
    target_idx = train_mod.build_temporal_patch_indices(
        patches, context_len, context_len + runtime.future_T
    )
    masks_x = train_mod.repeat_indices_for_batch(context_idx.long(), batch, sequence.device)
    masks_y = train_mod.repeat_indices_for_batch(target_idx.long(), batch, sequence.device)
    context_tokens = sequence.gather(
        1, masks_x.unsqueeze(-1).expand(-1, -1, dim)
    )
    # Existing VLM caches summarize the first half of the source video, which is
    # not causal when the visible R32 context is only four frames. The checkpoint
    # architecture remains intact, but no VLM payload is read or passed here.
    if runtime.r32_vlm_guidance_policy != "disabled_no_causal_cache":
        raise RuntimeError("unfrozen R32 VLM guidance policy")
    guidance = None
    output = predictor(context_tokens, masks_x, masks_y, ext=guidance)
    return output.view(batch, runtime.future_T, patches, dim).contiguous()


def _run(cli, protocol: dict, manifest: Path, split_spec: dict) -> None:
    import torch
    from cache_train import models as models_mod
    from cache_train import thinker_train as train_mod

    if not torch.cuda.is_available():
        raise RuntimeError("R32 model evaluation requires CUDA")
    device = torch.device(f"cuda:{cli.gpu}")
    torch.cuda.set_device(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = _args_dict(checkpoint.get("args"))
    validate_checkpoint_args(checkpoint_args, protocol["rollout"])
    runtime = _runtime_namespace(checkpoint_args, cli, protocol["rollout"])

    _, loader = train_mod.build_egodex_dataloaders(
        str(cli.data_dir), train_mod.QUERY_TFS, if_return_path=True,
        train_batch=cli.batch_size, test_batch=cli.batch_size,
        train_manifest=str(manifest), test_manifest=str(manifest),
        use_npz_cache=True, cache_dir=str(cli.cache_dir),
        num_workers=cli.num_workers, pin_memory=False, persistent_workers=False,
        prefetch_factor=cli.prefetch_factor, preload_to_memory=False,
        camera_mode="auto",
    )
    if len(loader.dataset) != int(split_spec["expected_samples"]):
        raise RuntimeError("loader sample count does not match frozen protocol")

    readout_state = _strip_module_prefix(checkpoint["cls_model"])
    readout = models_mod.TrajectoryReadoutMLP(downsample=True).to(device)
    readout.load_state_dict(readout_state, strict=True)
    readout.eval()
    cache_index = train_mod.build_thinker_cache_index(str(cli.cache_dir))
    path_cache = {}
    predictor = None
    rows = []
    horizons = protocol["rollout"]["horizons"]

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch is None or len(batch) < 11:
                raise RuntimeError(f"invalid batch at index {batch_index}; skipping is forbidden")
            xyz_world = batch[2].to(device, non_blocking=True)
            cam_ext = batch[6].to(device, non_blocking=True)
            extras, paths = train_mod.parse_batch_extras_and_paths(batch)
            if paths is None:
                raise RuntimeError(f"missing sample paths at batch {batch_index}")
            required = runtime.past_T + runtime.future_T * protocol["rollout"]["steps"]
            if xyz_world.ndim != 4 or xyz_world.shape[1] < required or xyz_world.shape[2:] != (52, 3):
                raise ValueError(f"batch {batch_index} does not contain [{required},52,3]")
            features = None
            if isinstance(extras, dict):
                features = extras.get("vjepa_feats")
            if features is not None:
                features = features.to(device=device, dtype=torch.float32)
            else:
                features = train_mod.try_load_cached_dense_jepa_features(
                    paths=paths, args=runtime, device=device,
                    cache_index=cache_index, path_cache=path_cache,
                    preloaded_archives=None,
                )
            if features is None:
                raise RuntimeError(f"missing cached V-JEPA features at batch {batch_index}")
            if features.ndim != 4 or features.shape[0] != xyz_world.shape[0]:
                raise ValueError(f"invalid V-JEPA feature shape at batch {batch_index}")
            if features.shape[1] < runtime.past_T:
                raise ValueError(f"insufficient context features at batch {batch_index}")
            target_seq = xyz_world[:, runtime.past_T:required]
            if not torch.isfinite(features).all() or not torch.isfinite(target_seq).all():
                raise FloatingPointError(f"non-finite input or target at batch {batch_index}")
            if predictor is None:
                predictor = _build_predictor(
                    checkpoint, runtime, features, device
                )

            context = features[:, :runtime.past_T].contiguous()
            predicted_steps = []
            for step in range(protocol["rollout"]["steps"]):
                predicted_latents = _predict_latents(
                    train_mod, predictor, runtime, context
                )
                readout_input = torch.cat([context, predicted_latents.detach()], dim=1)
                pred_cam = train_mod.predict_trajectory_from_latents(readout, readout_input)
                pred_cam = pred_cam.view(
                    features.shape[0], runtime.future_T, 52, 3
                ).contiguous()
                # Frozen leak guard: never call append_hand_reference_offsets and never
                # read future xyz_cam.  Future extrinsics only convert coordinates.
                start = runtime.past_T + step * runtime.future_T
                end = start + runtime.future_T
                pred_world = train_mod.project_camera_points_to_world(
                    pred_cam, cam_ext[:, start:end].contiguous()
                )
                if not torch.isfinite(pred_world).all():
                    raise FloatingPointError(
                        f"non-finite R32 prediction at batch={batch_index} step={step + 1}"
                    )
                predicted_steps.append(pred_world)
                context = predicted_latents.detach()
            pred_seq = torch.cat(predicted_steps, dim=1)
            target_seq = target_seq.contiguous()
            path_items = list(paths) if isinstance(paths, (list, tuple)) else [paths] * len(pred_seq)
            if len(path_items) != len(pred_seq):
                raise RuntimeError(f"sample path count mismatch at batch {batch_index}")
            for item_index in range(len(pred_seq)):
                sample = str(path_items[item_index])
                pred_np = pred_seq[item_index].cpu().numpy()
                target_np = target_seq[item_index].cpu().numpy()
                context_world = xyz_world[item_index, :runtime.past_T].cpu().numpy()
                controls = {
                    "ThinkJEPA": pred_np,
                    "LastPose": last_pose(context_world, max(horizons)),
                    "ConstVelocityK4": constant_velocity(context_world, max(horizons), 4),
                }
                for method, prediction in controls.items():
                    for horizon in horizons:
                        rows.append({
                            "sample": sample, "method": method, "horizon": horizon,
                            **compute_trajectory_metrics(
                                prediction[:horizon], target_np[:horizon]
                            ),
                        })

    expected_rows = int(split_spec["expected_samples"]) * 3 * len(horizons)
    if len(rows) != expected_rows:
        raise RuntimeError(f"R32 row count mismatch: {len(rows)} != {expected_rows}")
    cli.output_dir.mkdir(parents=False, exist_ok=False)
    metric_names = [key for key in rows[0] if key not in {"sample", "method", "horizon"}]
    with (cli.output_dir / "r32_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample", "method", "horizon", *metric_names]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in ("ThinkJEPA", "LastPose", "ConstVelocityK4"):
        summary[method] = {}
        for horizon in horizons:
            selected = [row for row in rows if row["method"] == method and row["horizon"] == horizon]
            summary[method][str(horizon)] = {
                metric: _summary([row[metric] for row in selected])
                for metric in metric_names
            }
    identity = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(cli.protocol.resolve()),
        "split": cli.split,
        "manifest": str(manifest),
        "manifest_sha256": split_spec["sha256"],
        "checkpoint": str(cli.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(cli.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_args": checkpoint_args,
        "git": _git_identity(),
        "runtime": {"gpu": cli.gpu, "batch_size": cli.batch_size},
        "test_accessed": cli.split == "final-test",
    }
    (cli.output_dir / "r32_summary.json").write_text(
        json.dumps({"identity": identity, "results": summary}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] R32 evaluated {split_spec['expected_samples']} samples")
    print(f"[OK] wrote {cli.output_dir / 'r32_summary.json'}")


def _reserve_final_test_access(cli, protocol: dict) -> Path:
    spec = protocol["splits"]["final-test"]
    record_path = (cli.protocol.resolve().parents[1] / spec["access_record"]).resolve()
    payload = {
        "status": "STARTED",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(cli.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(cli.checkpoint.resolve()),
        "protocol_sha256": sha256_file(cli.protocol.resolve()),
        "output_dir": str(cli.output_dir.resolve()),
        "git": _git_identity(),
        "warning": "This persistent record locks the final-test split against reuse.",
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with record_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(
            f"[LOCKED] final test was already accessed; inspect {record_path}"
        ) from error
    return record_path


def _complete_final_test_access(record_path: Path) -> None:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["status"] = "COMPLETE"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "final-test"), default="validation")
    parser.add_argument("--final-test-confirm", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    return parser.parse_args()


def main():
    cli = parse_args()
    protocol = load_protocol(cli.protocol.resolve())
    manifest, spec = resolve_split(
        protocol, cli.protocol.resolve(), cli.split, cli.final_test_confirm
    )
    if not cli.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {cli.checkpoint}")
    if cli.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {cli.output_dir}")
    access_record = None
    if cli.split == "final-test":
        access_record = _reserve_final_test_access(cli, protocol)
    _run(cli, protocol, manifest, spec)
    if access_record is not None:
        _complete_final_test_access(access_record)


if __name__ == "__main__":
    main()
