"""Helpers for binding cached preprocessing and runs to their inputs."""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_normalization_source(train_manifest, image_root, input_size: int) -> dict:
    """Describe the inputs that determine the cached normalization values."""
    manifest_path = Path(train_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    size = int(input_size)
    if size < 1:
        raise ValueError("normalization input_size must be positive")
    return {
        "train_manifest_sha256": file_sha256(manifest_path),
        "image_root": str(Path(image_root).resolve()),
        "input_size": size,
    }


def validate_dataset_normalization_cache(stats, source: dict, stats_path) -> dict:
    """Reject malformed or stale cached statistics instead of silently reusing them."""
    path = Path(stats_path)
    if not isinstance(stats, dict):
        raise ValueError(f"invalid cached dataset normalization statistics: {path}")
    if stats.get("source") is None:
        raise ValueError(
            f"cached dataset normalization statistics are missing provenance: {path}; "
            "this run cannot be safely resumed; start a new run from pretrained weights"
        )
    if stats.get("source") != source:
        raise ValueError(
            f"cached dataset normalization statistics were computed from different training data: {path}"
        )
    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        raise ValueError(f"invalid cached dataset normalization statistics: {path}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"invalid cached dataset normalization statistics: {path}")
    try:
        mean_values = [float(value) for value in mean]
        std_values = [float(value) for value in std]
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid cached dataset normalization statistics: {path}") from error
    if any(not math.isfinite(value) for value in (*mean_values, *std_values)):
        raise ValueError(f"cached dataset normalization values must be finite: {path}")
    if any(value <= 0 for value in std_values):
        raise ValueError(f"cached dataset normalization standard deviations must be positive: {path}")
    if int(stats.get("input_size", -1)) != int(source["input_size"]):
        raise ValueError(f"invalid cached dataset normalization statistics: {path}")
    return stats


def git_working_tree_state(project_root) -> dict:
    """Return the commit plus enough working-tree state to identify local edits."""
    root = Path(project_root).resolve()
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
        )
        diff_result = subprocess.run(
            ["git", "-C", str(root), "diff", "HEAD", "--binary"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": None,
            "working_tree_dirty": None,
            "tracked_diff_sha256": None,
            "untracked_paths": [],
        }

    status_lines = status_result.stdout.decode("utf-8", errors="replace").splitlines()
    untracked_paths = sorted(
        line[3:] for line in status_lines if line.startswith("?? ")
    )
    return {
        "revision": revision_result.stdout.decode("ascii", errors="replace").strip() or None,
        "working_tree_dirty": bool(status_lines),
        "tracked_diff_sha256": hashlib.sha256(diff_result.stdout).hexdigest(),
        "untracked_paths": untracked_paths,
    }
