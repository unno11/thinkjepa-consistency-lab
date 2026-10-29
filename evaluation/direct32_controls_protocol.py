"""Validation helpers for frozen Direct32 context-only controls."""

from __future__ import annotations

import json
from pathlib import Path

from .file_utils import sha256_file


EXPECTED_WINDOW = {
    "past_frames": 32,
    "future_frames": 32,
    "temporal_stride": 1,
    "joint_count": 52,
}
EXPECTED_METHODS = {
    "LastPose": {"type": "last_pose"},
    "ConstVelocityK4": {"type": "constant_velocity", "context_frames": 4},
}
EXPECTED_GUARDS = {
    "validation_only": True,
    "future_target_as_input_forbidden": True,
    "existing_output_refused": True,
    "nonfinite_input_fails": True,
    "locked_test_access": False,
}


def load_controls_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("controls schema_version must be 1")
    if payload.get("window") != EXPECTED_WINDOW:
        raise ValueError("controls window does not match frozen Direct32")
    if payload.get("methods") != EXPECTED_METHODS:
        raise ValueError("controls methods do not match frozen definitions")
    if payload.get("guards") != EXPECTED_GUARDS:
        raise ValueError("controls guards do not match frozen policy")
    validation = payload.get("validation", {})
    required = {
        "expected_samples": 200,
        "expected_tasks": 13,
        "expected_episode_clusters": 200,
        "expected_max_samples_per_episode": 1,
    }
    if any(int(validation.get(key, -1)) != value for key, value in required.items()):
        raise ValueError("controls validation cardinalities are not frozen")
    if not validation.get("manifest") or not validation.get("sha256"):
        raise ValueError("controls validation identity is incomplete")
    return payload


def sample_groups(sample: str) -> tuple[str, str]:
    path = Path(sample)
    task = path.parent.name
    stem = path.stem
    if not task or "_L" not in stem:
        raise ValueError(f"cannot derive task/episode from sample: {sample}")
    episode_id = stem.split("_L", 1)[0]
    if not episode_id:
        raise ValueError(f"empty episode id in sample: {sample}")
    return task, f"{task}/{episode_id}"


def resolve_controls_validation(protocol: dict, protocol_path: Path):
    spec = protocol["validation"]
    repo_root = protocol_path.resolve().parents[1]
    manifest = (repo_root / spec["manifest"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"missing validation manifest: {manifest}")
    actual_hash = sha256_file(manifest)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"validation SHA-256 mismatch: {actual_hash}")
    entries = [
        line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    tasks = {sample_groups(entry)[0] for entry in entries}
    episodes = [sample_groups(entry)[1] for entry in entries]
    episode_counts = {episode: episodes.count(episode) for episode in set(episodes)}
    observed = {
        "expected_samples": len(entries),
        "expected_tasks": len(tasks),
        "expected_episode_clusters": len(episode_counts),
        "expected_max_samples_per_episode": max(episode_counts.values()),
    }
    for key, value in observed.items():
        if value != int(spec[key]):
            raise ValueError(f"validation {key}={value} != {spec[key]}")
    return manifest, entries, {
        "tasks": len(tasks),
        "episode_clusters": len(episode_counts),
        "max_samples_per_episode": max(episode_counts.values()),
        "episode_cluster_inference": "degenerate_to_sample_level",
    }
