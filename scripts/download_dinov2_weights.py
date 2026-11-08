"""Download Meta's official general-purpose DINOv2 ViT-B/14 checkpoint."""

import argparse
import json
from pathlib import Path

from cxr_project.dinov2_weights import download_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "dinov2_vitb14_pretrain.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(download_checkpoint(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
