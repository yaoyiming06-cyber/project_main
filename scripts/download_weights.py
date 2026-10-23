"""Download and verify the pinned pure ImageNet-22K ConvNeXt-Base weights."""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "convnext_base.fb_in22k.safetensors"
WEIGHTS_URL = (
    "https://huggingface.co/timm/convnext_base.fb_in22k/resolve/"
    "9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7/model.safetensors"
)
WEIGHTS_URL_MIRROR = (
    "https://hf-mirror.com/timm/convnext_base.fb_in22k/resolve/"
    "9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7/model.safetensors"
)
EXPECTED_SIZE = 439_848_696
EXPECTED_SHA256 = "0f09df05894c7fd1e78569daaae9e839a7b3bc5666587a08ba1d8e8cea63a8a4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.is_file() and args.output.stat().st_size == EXPECTED_SIZE:
        digest = sha256(args.output)
        if digest == EXPECTED_SHA256:
            print(f"Verified ImageNet-22K ConvNeXt-Base weights already exist: {args.output}")
            return
    last_error = None
    for url in (WEIGHTS_URL, WEIGHTS_URL_MIRROR):
        request = urllib.request.Request(url, headers={"User-Agent": "cxr-project-main/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            break
        except OSError as error:
            last_error = error
            print(f"Download failed from {url}: {error}")
            continue
    else:
        raise last_error
    try:
        size = temporary.stat().st_size
        if size != EXPECTED_SIZE:
            raise ValueError(f"Unexpected checkpoint size: {size} bytes")
        digest = sha256(temporary)
        if digest != EXPECTED_SHA256:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Verified pure ImageNet-22K ConvNeXt-Base weights: {args.output}")
    print(f"Bytes: {args.output.stat().st_size}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
