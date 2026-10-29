#!/usr/bin/env python3
"""Probe sample isolation and layer coverage on one locked validation batch.

This script never trains, never selects a checkpoint, and never opens the final
test split.  A FAIL result is expected when auditing a semantically unsafe
implementation; it is evidence, not an execution error.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.direct32_protocol import load_protocol, resolve_validation
from evaluation.guidance_contract import evaluate_guidance_contract
from evaluation.r32_protocol import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "direct32_best_validation_protocol.json"


def _shape(value):
    return list(value.shape) if hasattr(value, "shape") else None


def _payload_shapes(payload: dict | None) -> dict:
    if payload is None:
        return {}
    return {
        key: _shape(value)
        for key, value in payload.items()
        if key.startswith("vlm_") and value is not None
    }


def _slice_batch(payload, indices, batch_size, torch, np):
    if isinstance(payload, dict):
        return {
            key: _slice_batch(value, indices, batch_size, torch, np)
            for key, value in payload.items()
        }
    if isinstance(payload, torch.Tensor):
        if payload.ndim > 0 and payload.shape[0] == batch_size:
            index = torch.as_tensor(indices, device=payload.device, dtype=torch.long)
            return payload.index_select(0, index)
        return payload.clone()
    if isinstance(payload, np.ndarray):
        if payload.ndim > 0 and payload.shape[0] == batch_size:
            return payload[list(indices)].copy()
        return payload.copy()
    return copy.deepcopy(payload)


def _mutate_second_sample(payload: dict, torch) -> dict:
    changed = copy.deepcopy(payload)
    for key in ("vlm_old", "vlm_new"):
        value = changed.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] >= 2:
            value = value.clone()
            value[1] = -3.0 * value[1] + 0.125
            changed[key] = value
    return changed


def _zero_guidance(payload: dict, torch) -> dict:
    zeroed = copy.deepcopy(payload)
    for key in ("vlm_old", "vlm_new"):
        value = zeroed.get(key)
        if isinstance(value, torch.Tensor):
            zeroed[key] = torch.zeros_like(value)
    return zeroed


def _max_delta(left, right) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _first_sample_guidance_delta(left: dict, right: dict) -> float:
    deltas = []
    for key in ("vlm_old", "vlm_new"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            continue
        if left_value.ndim == 4 and right_value.ndim == 4:
            deltas.append(_max_delta(left_value[:1], right_value[:1]))
        elif left_value.ndim == 3 and right_value.ndim == 3:
            deltas.append(_max_delta(left_value, right_value))
    return max(deltas, default=0.0)


def _first_sample_bank_delta(left, right) -> float:
    deltas = []
    if left.mode != right.mode:
        raise ValueError("cannot compare guidance banks with different modes")
    for left_layer, right_layer in zip(left.layers, right.layers):
        if left.mode == "crossattn":
            left_memory, left_mask = left_layer
            right_memory, right_mask = right_layer
            deltas.append(_max_delta(left_memory[:1], right_memory[:1]))
            deltas.append(float((left_mask[:1] != right_mask[:1]).any().item()))
        else:
            left_scale, left_shift = left_layer
            right_scale, right_shift = right_layer
            deltas.append(_max_delta(left_scale[:1], right_scale[:1]))
            deltas.append(_max_delta(left_shift[:1], right_shift[:1]))
    return max(deltas, default=0.0)


def _source_layer_count(payload: dict) -> int:
    for key in ("vlm_old", "vlm_new"):
        value = payload.get(key)
        if value is None:
            continue
        if value.ndim == 4:  # [B,L,S,D]
            return int(value.shape[1])
        if value.ndim == 3:  # [L,S,D]
            return int(value.shape[0])
    return 0


def _run(cli) -> dict:
    import numpy as np
    import torch
    from cache_train import thinker_train as train_mod
    from evaluation import evaluate_direct32 as direct32

    if cli.batch_size < 2:
        raise ValueError("Gate 0 requires --batch-size >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 0 runtime probe requires one CUDA GPU")

    protocol_path = cli.protocol.resolve()
    protocol = load_protocol(protocol_path)
    manifest, validation_spec = resolve_validation(protocol, protocol_path)
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(cli.checkpoint)

    device = torch.device(f"cuda:{cli.gpu}")
    torch.cuda.set_device(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = direct32._args_dict(checkpoint.get("args"))
    rollout = dict(protocol["rollout"])
    rollout["vlm_source"] = checkpoint_args.get("thinkjepa_vlm_source", "both")
    rollout["vlm_layer_selector"] = checkpoint_args.get(
        "thinkjepa_vlm_layer_selector", "last"
    )
    rollout["vlm_cond_mode"] = checkpoint_args.get("thinkjepa_vlm_cond_mode", "film")
    runtime = direct32._runtime_namespace(checkpoint_args, cli, rollout)

    _, loader = train_mod.build_egodex_dataloaders(
        str(cli.data_dir), train_mod.QUERY_TFS, if_return_path=True,
        train_batch=cli.batch_size, test_batch=cli.batch_size,
        train_manifest=str(manifest), test_manifest=str(manifest),
        use_npz_cache=True, cache_dir=str(cli.cache_dir),
        num_workers=cli.num_workers, pin_memory=False, persistent_workers=False,
        prefetch_factor=cli.prefetch_factor, preload_to_memory=False,
        camera_mode="auto",
    )
    batch = next(iter(loader))
    if batch is None or len(batch) < 11:
        raise RuntimeError("invalid first validation batch")
    paths = train_mod.parse_batch_extras_and_paths(batch)[1]
    extras = train_mod.parse_batch_extras_and_paths(batch)[0]
    features = extras.get("vjepa_feats") if isinstance(extras, dict) else None
    cache_index = train_mod.build_thinker_cache_index(str(cli.cache_dir))
    path_cache = {}
    if features is not None:
        features = features.to(device=device, dtype=torch.float32)
    else:
        features = train_mod.try_load_cached_dense_jepa_features(
            paths=paths, args=runtime, device=device,
            cache_index=cache_index, path_cache=path_cache,
            preloaded_archives=None,
        )
    extras = train_mod.ensure_thinker_guidance_payload(
        extras=extras, paths=paths, args=runtime, device=device,
        cache_index=cache_index, path_cache=path_cache,
        preloaded_archives=None,
    )
    if features is None or features.shape[0] < 2:
        raise RuntimeError("first validation batch does not contain two feature samples")

    features = features[:2]
    extras = _slice_batch(extras, (0, 1), len(paths), torch, np)
    context = features[:, :runtime.past_T].contiguous()
    predictor = direct32._build_predictor(checkpoint, runtime, features, device)
    predictor.eval()

    with torch.no_grad():
        selected = train_mod.build_thinkjepa_guidance_inputs(extras, runtime, device)
        single_extras = _slice_batch(extras, (0,), 2, torch, np)
        changed_extras = _mutate_second_sample(extras, torch)
        selected_single = train_mod.build_thinkjepa_guidance_inputs(
            single_extras, runtime, device
        )
        selected_changed = train_mod.build_thinkjepa_guidance_inputs(
            changed_extras, runtime, device
        )

        paired_bank = predictor._build_layerwise_guidance(selected, 2, device)
        all_layer_map = tuple(predictor.guidance_layer_map)
        single_bank = predictor._build_layerwise_guidance(selected_single, 1, device)
        changed_bank = predictor._build_layerwise_guidance(selected_changed, 2, device)

        paired = direct32._predict_latents(train_mod, predictor, runtime, context, extras)
        single = direct32._predict_latents(
            train_mod, predictor, runtime, context[:1], single_extras
        )

        changed = direct32._predict_latents(
            train_mod, predictor, runtime, context, changed_extras,
        )

        swapped_extras = _slice_batch(extras, (1, 0), 2, torch, np)
        swapped = direct32._predict_latents(
            train_mod, predictor, runtime, context, swapped_extras
        )

        zero_extras = _zero_guidance(extras, torch)
        zero_single_extras = _slice_batch(zero_extras, (0,), 2, torch, np)
        zeroed = direct32._predict_latents(
            train_mod, predictor, runtime, context,
            zero_extras,
        )
        zero_single = direct32._predict_latents(
            train_mod, predictor, runtime, context[:1], zero_single_extras,
        )

        duplicate_extras = _slice_batch(extras, (0, 0), 2, torch, np)
        duplicate_context = context[:1].expand(2, -1, -1, -1).contiguous()
        duplicated = direct32._predict_latents(
            train_mod, predictor, runtime, duplicate_context, duplicate_extras,
        )

        last_runtime = copy.copy(runtime)
        last_runtime.thinkjepa_vlm_layer_selector = "last"
        last_output = direct32._predict_latents(
            train_mod, predictor, last_runtime, context, extras
        )

    raw_shapes = _payload_shapes(extras)
    selected_shapes = _payload_shapes(selected)
    selected_feature = selected.get("vlm_old")
    if selected_feature is None:
        selected_feature = selected.get("vlm_new")
    preserves_batch = bool(
        selected_feature is not None
        and selected_feature.ndim == 4
        and int(selected_feature.shape[0]) == 2
    )
    source_layers = _source_layer_count(extras)
    predictor_layers = len(predictor.rollout_layers)
    explicit_map = all_layer_map
    mapping_covers = bool(
        source_layers >= predictor_layers
        or (
            explicit_map is not None
            and len(explicit_map) == predictor_layers
            and min(explicit_map) == 0
            and max(explicit_map) == source_layers - 1
            and tuple(sorted(explicit_map)) == tuple(explicit_map)
        )
    )

    measurements = {
        "preserves_batch_axis": preserves_batch,
        "batch_size_max_abs_delta": _max_delta(paired[:1], single),
        "other_sample_max_abs_delta": _max_delta(paired[:1], changed[:1]),
        "preprocessed_batch_single_max_abs_delta": _first_sample_guidance_delta(
            selected, selected_single
        ),
        "preprocessed_other_sample_max_abs_delta": _first_sample_guidance_delta(
            selected, selected_changed
        ),
        "guidance_bank_batch_single_max_abs_delta": _first_sample_bank_delta(
            paired_bank, single_bank
        ),
        "guidance_bank_other_sample_max_abs_delta": _first_sample_bank_delta(
            paired_bank, changed_bank
        ),
        "zero_guidance_batch_size_max_abs_delta": _max_delta(
            zeroed[:1], zero_single
        ),
        "duplicate_A_batch_size_max_abs_delta": _max_delta(
            duplicated[:1], single
        ),
        "layer_mapping_covers_all_blocks": mapping_covers,
        "last_all_max_abs_delta": _max_delta(paired, last_output),
        "zero_guidance_max_abs_delta": _max_delta(paired, zeroed),
        # Context is deliberately not swapped: sample 0 should react to receiving sample 1 guidance.
        "shuffle_guidance_max_abs_delta": _max_delta(paired[:1], swapped[:1]),
    }
    verdict = evaluate_guidance_contract(measurements)
    return {
        "identity": {
            "split": "validation",
            "test_accessed": False,
            "manifest": str(manifest),
            "manifest_sha256": validation_spec["sha256"],
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "checkpoint": str(cli.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(cli.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "gpu": cli.gpu,
            "samples_probed": 2,
            "git": direct32._git_identity(),
        },
        "runtime_shapes": {
            "raw_extras": raw_shapes,
            "selected_guidance": selected_shapes,
            "source_layers": source_layers,
            "predictor_blocks": predictor_layers,
            "explicit_layer_map": explicit_map,
        },
        "measurements": measurements,
        "verdict": verdict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.output.exists():
        raise SystemExit(f"refusing to overwrite {cli.output}")
    report = _run(cli)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"[GATE0] {report['verdict']['status']}")
    print(f"[GATE0] wrote {cli.output}")
    return 0 if report["verdict"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
