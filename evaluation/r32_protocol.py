"""Pure validation helpers for the frozen leak-free R32 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUIRED_ROLLOUT = {
    "past_frames": 4,
    "step_future_frames": 4,
    "steps": 8,
    "horizons": [4, 8, 16, 32],
    "temporal_stride": 1,
    "joint_count": 52,
    "predictor": "thinkjepa",
    "trajmode": "traj",
    "joint_pred": True,
    "ref_mode": "none",
    "inference_amp": False,
    "vlm_guidance_policy": "disabled_no_causal_cache",
    "future_camera_extrinsics": "coordinate_transform_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("R32 protocol schema_version must be 1")
    rollout = data.get("rollout", {})
    for key, expected in REQUIRED_ROLLOUT.items():
        if rollout.get(key) != expected:
            raise ValueError(f"R32 protocol {key} must equal {expected!r}")
    guards = data.get("leakage_guards", {})
    required_guards = (
        "explicit_manifest_required",
        "future_ground_truth_joint_reference_forbidden",
        "noncausal_vlm_guidance_forbidden",
        "batch_skip_forbidden",
        "nonfinite_prediction_fails",
        "test_checkpoint_selection_forbidden",
    )
    if not all(guards.get(key) is True for key in required_guards):
        raise ValueError("all R32 leakage guards must be enabled")
    for split in ("validation", "final-test"):
        spec = data.get("splits", {}).get(split, {})
        if not spec.get("manifest") or not spec.get("sha256"):
            raise ValueError(f"R32 split {split} is missing manifest identity")
        if int(spec.get("expected_samples", 0)) <= 0:
            raise ValueError(f"R32 split {split} expected_samples must be positive")
    if data["splits"]["final-test"].get("selection_allowed") is not False:
        raise ValueError("final-test must forbid model/checkpoint selection")
    if data["splits"]["final-test"].get("locked") is not True:
        raise ValueError("final-test must remain locked")
    if not data["splits"]["final-test"].get("access_record"):
        raise ValueError("final-test must define a persistent access record")
    metrics = data.get("metrics", {})
    if metrics.get("controls") != ["LastPose", "ConstVelocityK4"]:
        raise ValueError("R32 controls must be LastPose and ConstVelocityK4")
    if metrics.get("per_sample") is not True:
        raise ValueError("R32 metrics must be retained per sample")
    return data


def resolve_split(
    protocol: dict,
    protocol_path: Path,
    split: str,
    final_test_confirm: str = "",
) -> tuple[Path, dict]:
    if split not in {"validation", "final-test"}:
        raise ValueError("split must be validation or final-test")
    spec = protocol["splits"][split]
    if split == "final-test" and final_test_confirm != spec["confirmation"]:
        raise PermissionError(
            "locked final test refused; exact --final-test-confirm token is required"
        )
    repo_root = protocol_path.resolve().parents[1]
    manifest = (repo_root / spec["manifest"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"missing frozen manifest: {manifest}")
    actual_hash = sha256_file(manifest)
    if actual_hash != spec["sha256"]:
        raise ValueError(
            f"manifest SHA-256 mismatch for {split}: {actual_hash} != {spec['sha256']}"
        )
    entries = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != int(spec["expected_samples"]):
        raise ValueError(
            f"manifest sample count mismatch for {split}: "
            f"{len(entries)} != {spec['expected_samples']}"
        )
    return manifest, spec


def validate_checkpoint_args(checkpoint_args: dict, rollout: dict) -> None:
    expected = {
        "predictor": rollout["predictor"],
        "trajmode": rollout["trajmode"],
        "joint_pred": rollout["joint_pred"],
        "ref_mode": rollout["ref_mode"],
        "temporal_stride": rollout["temporal_stride"],
    }
    for key, value in expected.items():
        if checkpoint_args.get(key) != value:
            raise ValueError(
                f"checkpoint {key}={checkpoint_args.get(key)!r} is incompatible; "
                f"R32 requires {value!r}"
            )
