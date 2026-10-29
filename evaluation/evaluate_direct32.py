#!/usr/bin/env python3
"""Re-evaluate one frozen B/M/G/MG best checkpoint on direct-32 validation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.direct32_protocol import (
    load_protocol,
    resolve_validation,
    validate_checkpoint,
)
from evaluation.metrics import compute_trajectory_metrics
from evaluation.kinematic_projection import (
    load_projection_protocol,
    project_prediction_median_bone_lengths,
)
from evaluation.r32_protocol import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "direct32_best_validation_protocol.json"
DEFAULT_PROJECTION_PROTOCOL = (
    REPO_ROOT / "configs" / "kinematic_projection_control_v1.json"
)


def _args_dict(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    return dict(vars(value))


def _strip_module_prefix(state: dict) -> dict:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


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


def _guidance_input_dim(state: dict, *suffixes: str) -> int:
    for key, tensor in state.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            if getattr(tensor, "ndim", 0) != 2:
                raise ValueError(f"invalid guidance adapter weight: {key}")
            return int(tensor.shape[1])
    raise ValueError(f"checkpoint is missing guidance adapter: {suffixes}")


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


def _summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "n_defined": int(len(finite)),
    }


def _runtime_namespace(checkpoint_args: dict, cli, rollout: dict):
    values = dict(checkpoint_args)
    values.update({
        "data_dir": str(cli.data_dir.resolve()),
        "cache_dir": str(cli.cache_dir.resolve()),
        "past_T": rollout["past_frames"],
        "future_T": rollout["future_frames"],
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
        "thinkjepa_vlm_source": rollout["vlm_source"],
        "thinkjepa_vlm_layer_selector": rollout["vlm_layer_selector"],
        "thinkjepa_vlm_layer_index": values.get("thinkjepa_vlm_layer_index", -1),
        "thinkjepa_vlm_cond_mode": rollout["vlm_cond_mode"],
        "vlm_pad_old_to": values.get("vlm_pad_old_to", 480),
        "vlm_pad_new_to": values.get("vlm_pad_new_to", 15),
        "zero_visual_input": False,
    })
    return SimpleNamespace(**values)


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
        vlm_old_dim=old_dim, vlm_new_dim=new_dim,
    ).to(device)
    predictor.load_state_dict(state, strict=True)
    predictor.eval()
    return predictor


def _predict_latents(train_mod, predictor, runtime, context, extras):
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
    guidance = train_mod.build_thinkjepa_guidance_inputs(
        extras, runtime, sequence.device
    )
    if guidance is None or not any(
        guidance.get(key) is not None for key in ("vlm_old", "vlm_new")
    ):
        raise RuntimeError("direct32 requires observation-window VLM guidance")
    output = predictor(context_tokens, masks_x, masks_y, ext=guidance)
    return output.view(batch, runtime.future_T, patches, dim).contiguous()


def _run(
    cli,
    protocol: dict,
    manifest: Path,
    validation_spec: dict,
    projection_protocol: dict | None,
) -> None:
    import torch
    from cache_train import models as models_mod
    from cache_train import thinker_train as train_mod

    if not torch.cuda.is_available():
        raise RuntimeError("direct32 checkpoint evaluation requires CUDA")
    device = torch.device(f"cuda:{cli.gpu}")
    torch.cuda.set_device(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = _args_dict(checkpoint.get("args"))
    validate_checkpoint(protocol, cli.role, checkpoint, checkpoint_args)
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
    if len(loader.dataset) != int(validation_spec["expected_samples"]):
        raise RuntimeError("loader sample count does not match direct32 protocol")

    readout = models_mod.TrajectoryReadoutMLP(downsample=True).to(device)
    readout.load_state_dict(_strip_module_prefix(checkpoint["cls_model"]), strict=True)
    readout.eval()
    cache_index = train_mod.build_thinker_cache_index(str(cli.cache_dir))
    path_cache = {}
    predictor = None
    rows = []
    projection_rows = []
    required = runtime.past_T + runtime.future_T

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch is None or len(batch) < 11:
                raise RuntimeError(f"invalid batch at {batch_index}; skipping is forbidden")
            xyz_world = batch[2].to(device, non_blocking=True)
            cam_ext = batch[6].to(device, non_blocking=True)
            extras, paths = train_mod.parse_batch_extras_and_paths(batch)
            if paths is None:
                raise RuntimeError(f"missing paths at batch {batch_index}")
            if xyz_world.ndim != 4 or xyz_world.shape[1] < required or xyz_world.shape[2:] != (52, 3):
                raise ValueError(f"batch {batch_index} does not contain [{required},52,3]")
            features = extras.get("vjepa_feats") if isinstance(extras, dict) else None
            if features is not None:
                features = features.to(device=device, dtype=torch.float32)
            else:
                features = train_mod.try_load_cached_dense_jepa_features(
                    paths=paths, args=runtime, device=device,
                    cache_index=cache_index, path_cache=path_cache,
                    preloaded_archives=None,
                )
            if features is None or features.ndim != 4 or features.shape[1] < required:
                raise RuntimeError(f"missing/invalid V-JEPA features at batch {batch_index}")
            extras = train_mod.ensure_thinker_guidance_payload(
                extras=extras, paths=paths, args=runtime, device=device,
                cache_index=cache_index, path_cache=path_cache,
                preloaded_archives=None,
            )
            target = xyz_world[:, runtime.past_T:required].contiguous()
            context = features[:, :runtime.past_T].contiguous()
            if not torch.isfinite(context).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f"non-finite input/target at batch {batch_index}")
            if predictor is None:
                predictor = _build_predictor(checkpoint, runtime, features, device)
            predicted_latents = _predict_latents(
                train_mod, predictor, runtime, context, extras
            )
            readout_input = torch.cat([context, predicted_latents.detach()], dim=1)
            pred_cam = train_mod.predict_trajectory_from_latents(readout, readout_input)
            pred_cam = pred_cam.view(features.shape[0], runtime.future_T, 52, 3).contiguous()
            # Frozen guard: no future xyz_cam hand references or reference offsets.
            pred_world = train_mod.project_camera_points_to_world(
                pred_cam, cam_ext[:, runtime.past_T:required].contiguous()
            )
            if not torch.isfinite(pred_world).all():
                raise FloatingPointError(f"non-finite prediction at batch {batch_index}")
            path_items = list(paths) if isinstance(paths, (list, tuple)) else [paths] * len(pred_world)
            if len(path_items) != len(pred_world):
                raise RuntimeError(f"path count mismatch at batch {batch_index}")
            for item_index in range(len(pred_world)):
                prediction = pred_world[item_index].cpu().numpy()
                target_item = target[item_index].cpu().numpy()
                rows.append({
                    "sample": str(path_items[item_index]),
                    "role": cli.role,
                    **compute_trajectory_metrics(prediction, target_item),
                })
                if projection_protocol is not None:
                    projected = project_prediction_median_bone_lengths(prediction)
                    projection_rows.append({
                        "sample": str(path_items[item_index]),
                        "role": f"{cli.role}+{projection_protocol['method']}",
                        **compute_trajectory_metrics(projected, target_item),
                    })

    if len(rows) != int(validation_spec["expected_samples"]):
        raise RuntimeError(f"direct32 row count mismatch: {len(rows)}")
    cli.output_dir.mkdir(parents=False, exist_ok=False)
    metric_names = [key for key in rows[0] if key not in {"sample", "role"}]
    with (cli.output_dir / "direct32_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "role", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    summary = {metric: _summary([row[metric] for row in rows]) for metric in metric_names}
    identity = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(cli.protocol.resolve()),
        "split": "validation",
        "role": cli.role,
        "manifest": str(manifest),
        "manifest_sha256": validation_spec["sha256"],
        "checkpoint": str(cli.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(cli.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_args": checkpoint_args,
        "git": _git_identity(),
        "runtime": {"gpu": cli.gpu, "batch_size": cli.batch_size, "amp": False},
        "test_accessed": False,
    }
    (cli.output_dir / "direct32_summary.json").write_text(
        json.dumps(
            {"identity": identity, "metrics": summary},
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] direct32 {cli.role}: {len(rows)} validation samples")
    print(f"[OK] wrote {cli.output_dir / 'direct32_summary.json'}")

    if projection_protocol is not None:
        if len(projection_rows) != len(rows):
            raise RuntimeError("kinematic projection row count mismatch")
        projection_metric_names = [
            key for key in projection_rows[0] if key not in {"sample", "role"}
        ]
        with (cli.output_dir / "kinematic_projection_per_sample.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample", "role", *projection_metric_names],
            )
            writer.writeheader()
            writer.writerows(projection_rows)
        projection_summary = {
            metric: _summary([row[metric] for row in projection_rows])
            for metric in projection_metric_names
        }
        projection_identity = {
            **identity,
            "role": projection_rows[0]["role"],
            "base_role": cli.role,
            "projection_protocol": projection_protocol,
            "projection_protocol_sha256": sha256_file(
                cli.projection_protocol.resolve()
            ),
            "target_trajectory_passed_to_projection": False,
            "observed_trajectory_passed_to_projection": False,
        }
        (cli.output_dir / "kinematic_projection_summary.json").write_text(
            json.dumps(
                {"identity": projection_identity, "metrics": projection_summary},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(
            "[OK] kinematic projection control: "
            f"{len(projection_rows)} validation samples"
        )
        print(
            "[OK] wrote "
            f"{cli.output_dir / 'kinematic_projection_summary.json'}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--role", choices=("B", "M", "G", "MG"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--kinematic-projection-control",
        action="store_true",
        help="also evaluate the frozen target-free KProjPredMedian control",
    )
    parser.add_argument(
        "--projection-protocol",
        type=Path,
        default=DEFAULT_PROJECTION_PROTOCOL,
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    protocol_path = cli.protocol.resolve()
    protocol = load_protocol(protocol_path)
    manifest, validation_spec = resolve_validation(protocol, protocol_path)
    projection_protocol = None
    if cli.kinematic_projection_control:
        if cli.role != "B":
            raise SystemExit("kinematic projection control must use the B checkpoint")
        projection_protocol = load_projection_protocol(
            cli.projection_protocol.resolve()
        )
    if not cli.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {cli.checkpoint}")
    if cli.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {cli.output_dir}")
    _run(cli, protocol, manifest, validation_spec, projection_protocol)


if __name__ == "__main__":
    main()
