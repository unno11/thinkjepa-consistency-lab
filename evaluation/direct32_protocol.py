"""Pure validation helpers for frozen direct-32 best-checkpoint evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .file_utils import sha256_file


REQUIRED_ROLLOUT = {
    "past_frames": 32,
    "future_frames": 32,
    "temporal_stride": 1,
    "joint_count": 52,
    "predictor": "thinkjepa",
    "trajmode": "traj",
    "joint_pred": True,
    "ref_mode": "none",
    "inference_amp": False,
    "vlm_source": "both",
    "vlm_cond_mode": "film",
    "vlm_guidance_policy": "observation_window_cache",
    "future_camera_extrinsics": "coordinate_transform_only",
}

ALLOWED_VLM_LAYER_SELECTORS = {"last", "all"}
ALLOWED_ROLES = {"B", "M", "G", "MG"}

WEIGHT_ARG_KEYS = {
    "bone": "lambda_bone_loss",
    "palm": "lambda_palm_bone_loss",
    "velocity": "lambda_velocity_loss",
    "acceleration": "lambda_acceleration_loss",
    "angle": "lambda_angle_loss",
}


def load_protocol(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("direct32 schema_version must be 1")
    rollout = data.get("rollout", {})
    for key, expected in REQUIRED_ROLLOUT.items():
        if rollout.get(key) != expected:
            raise ValueError(f"direct32 {key} must equal {expected!r}")
    if rollout.get("vlm_layer_selector") not in ALLOWED_VLM_LAYER_SELECTORS:
        raise ValueError(
            "direct32 vlm_layer_selector must be one of "
            f"{sorted(ALLOWED_VLM_LAYER_SELECTORS)!r}"
        )
    required_guards = (
        "validation_only",
        "explicit_manifest_required",
        "future_ground_truth_joint_reference_forbidden",
        "batch_skip_forbidden",
        "nonfinite_prediction_fails",
        "existing_output_refused",
    )
    guards = data.get("guards", {})
    if not all(guards.get(key) is True for key in required_guards):
        raise ValueError("all direct32 guards must be enabled")
    validation = data.get("validation", {})
    if not validation.get("manifest") or not validation.get("sha256"):
        raise ValueError("direct32 validation identity is incomplete")
    if int(validation.get("expected_samples", 0)) != 200:
        raise ValueError("direct32 validation must contain exactly 200 samples")
    roles = set(data.get("roles", {}))
    if not roles or not roles.issubset(ALLOWED_ROLES):
        raise ValueError(
            "direct32 roles must be a non-empty subset of B/M/G/MG"
        )
    for role, spec in data["roles"].items():
        if int(spec.get("expected_epoch", 0)) <= 0:
            raise ValueError(f"{role} expected_epoch must be positive")
        if "expected_seed" in spec and int(spec["expected_seed"]) < 0:
            raise ValueError(f"{role} expected_seed must be non-negative")
        if set(spec.get("weights", {})) != set(WEIGHT_ARG_KEYS):
            raise ValueError(f"{role} weights are incomplete")
    return data


def resolve_validation(protocol: dict, protocol_path: Path) -> tuple[Path, dict]:
    spec = protocol["validation"]
    repo_root = protocol_path.resolve().parents[1]
    manifest = (repo_root / spec["manifest"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"missing validation manifest: {manifest}")
    actual_hash = sha256_file(manifest)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"validation SHA-256 mismatch: {actual_hash} != {spec['sha256']}")
    entries = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != int(spec["expected_samples"]):
        raise ValueError("validation sample count does not match protocol")
    return manifest, spec


def validate_checkpoint(protocol: dict, role: str, checkpoint: dict, args: dict) -> None:
    if role not in protocol["roles"]:
        raise ValueError(f"unknown role: {role}")
    role_spec = protocol["roles"][role]
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != int(role_spec["expected_epoch"]):
        raise ValueError(
            f"{role} checkpoint epoch {epoch} != frozen {role_spec['expected_epoch']}"
        )
    if "expected_seed" in role_spec:
        actual_seed = int(args.get("seed", -1))
        expected_seed = int(role_spec["expected_seed"])
        if actual_seed != expected_seed:
            raise ValueError(
                f"{role} checkpoint seed {actual_seed} != frozen {expected_seed}"
            )
    rollout = protocol["rollout"]
    expected_args = {
        "predictor": rollout["predictor"],
        "trajmode": rollout["trajmode"],
        "joint_pred": rollout["joint_pred"],
        "ref_mode": rollout["ref_mode"],
        "temporal_stride": rollout["temporal_stride"],
        "thinkjepa_vlm_source": rollout["vlm_source"],
        "thinkjepa_vlm_layer_selector": rollout["vlm_layer_selector"],
        "thinkjepa_vlm_cond_mode": rollout["vlm_cond_mode"],
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise ValueError(f"{role} checkpoint {key}={args.get(key)!r} != {expected!r}")
    for short_name, expected in role_spec["weights"].items():
        key = WEIGHT_ARG_KEYS[short_name]
        actual = float(args.get(key, 0.0))
        if abs(actual - float(expected)) > 1e-12:
            raise ValueError(f"{role} checkpoint {key}={actual} != {expected}")
    if bool(args.get("skip_nonfinite_loss", False)):
        raise ValueError(f"{role} checkpoint was trained with skip_nonfinite_loss")
