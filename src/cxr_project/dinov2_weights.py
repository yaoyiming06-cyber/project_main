"""Download and verify the official general-purpose DINOv2 ViT-B/14 weights."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

from .general_weights import DINOV2_MODEL, DINOV2_URL


WEIGHTS_FILENAME = "dinov2_vitb14_pretrain.pth"
WEIGHTS_BYTES = 346_378_731
WEIGHTS_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(checkpoint_path) -> Path:
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + ".source.json")


def validate_checkpoint(checkpoint_path, expected_sha256) -> dict:
    """Verify the local checkpoint against its download record and run config."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {path}")
    expected = str(expected_sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("DINOv2 requires a configured 64-character SHA-256")

    sidecar = provenance_path(path)
    if not sidecar.is_file():
        raise ValueError(f"DINOv2 checkpoint provenance record is missing: {sidecar}")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid DINOv2 checkpoint provenance record: {sidecar}") from error
    if not isinstance(record, dict):
        raise ValueError(f"invalid DINOv2 checkpoint provenance record: {sidecar}")
    if record.get("model") != DINOV2_MODEL or record.get("url") != DINOV2_URL:
        raise ValueError("DINOv2 checkpoint provenance does not identify the official source")
    if record.get("path") != str(path):
        raise ValueError("DINOv2 checkpoint provenance path does not match the configured file")

    actual_bytes = path.stat().st_size
    if actual_bytes != WEIGHTS_BYTES or record.get("bytes") != WEIGHTS_BYTES:
        raise ValueError("DINOv2 checkpoint size does not match the pinned official file")
    if actual_bytes <= 0 or record.get("bytes") != actual_bytes:
        raise ValueError("DINOv2 checkpoint provenance size mismatch")
    actual_digest = file_sha256(path)
    sidecar_digest = str(record.get("sha256", "")).lower()
    if (not re.fullmatch(r"[0-9a-f]{64}", sidecar_digest)
            or sidecar_digest != actual_digest or actual_digest != WEIGHTS_SHA256):
        raise ValueError("DINOv2 checkpoint provenance SHA-256 mismatch")
    if actual_digest != expected:
        raise ValueError("DINOv2 checkpoint does not match configured SHA-256")
    return {**record, "path": str(path), "bytes": actual_bytes, "sha256": actual_digest}


def download_checkpoint(output_path) -> dict:
    """Download from Meta's official URL and atomically record provenance."""
    output = Path(output_path).expanduser().resolve()
    sidecar = provenance_path(output)
    if output.exists() or sidecar.exists():
        if output.is_file() and sidecar.is_file():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            validate_checkpoint(output, record.get("sha256"))
            return record
        raise FileExistsError(f"incomplete existing DINOv2 checkpoint or provenance: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".part")
    request = urllib.request.Request(DINOV2_URL, headers={"User-Agent": "cxr-project-main/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        size = temporary.stat().st_size
        if size <= 0:
            raise ValueError("Downloaded DINOv2 checkpoint is empty")
        digest = file_sha256(temporary)
        record = {
            "model": DINOV2_MODEL,
            "dataset": "LVD-142M general-purpose pretraining",
            "url": DINOV2_URL,
            "bytes": size,
            "sha256": digest,
            "path": str(output),
        }
        temporary.replace(output)
        temporary_sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_sidecar.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    validate_checkpoint(output, digest)
    return record
