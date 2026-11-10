"""Run the competition deployment bundle on an image-only folder tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.competition_inference import (
    InferencePipeline,
    PROBABILITY_DECIMAL_PLACES,
    PROBABILITY_ROUNDING,
    scan_image_tree,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"output or metadata already exists: {output}")

    data_root = args.data_dir.expanduser().resolve()
    rows = scan_image_tree(data_root)
    with InferencePipeline(args.bundle, args.device) as pipeline:
        prepared = pipeline.prepare_tree(data_root, rows, name="prediction")
        study_ids, probabilities = pipeline.predict(prepared, output)
        models = [
            {
                "name": model.name,
                "config": str(model.config_path),
                "config_sha256": model.config_sha256,
                "checkpoint": str(model.checkpoint_path),
                "checkpoint_sha256": model.checkpoint_sha256,
                "weight_source": model.weight_source,
                "input_size": int(model.config["input_size"]),
            }
            for model in pipeline.models
        ]
        metadata = {
            "output": str(output),
            "output_sha256": sha256_file(output),
            "bundle": str(pipeline.bundle_path),
            "bundle_sha256": sha256_file(pipeline.bundle_path),
            "data_dir": str(data_root),
            "image_count": prepared.num_images,
            "study_count": prepared.num_studies,
            "models": models,
            "seed": pipeline.seed,
            "device": str(pipeline.device),
            "precision": "fp16 autocast" if pipeline.device.type == "cuda" and any(
                model.config["amp"] for model in pipeline.models
            ) else "fp32",
            "fusion": "weighted arithmetic mean of model probabilities",
            "patient_smoothing_alpha": pipeline.patient_smoothing_alpha,
            "patient_smoothing_scope": "disabled for subset-reproducible official inference",
            "probability_decimal_places": PROBABILITY_DECIMAL_PLACES,
            "probability_rounding": PROBABILITY_ROUNDING,
            "classification_threshold": pipeline.classification_threshold,
            "tta": pipeline.tta,
            "synthetic_view_value": "UNKNOWN",
            "true_view_or_clinical_metadata_read": False,
            "study_order": study_ids,
        }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(study_ids)} Study predictions to {output}")


if __name__ == "__main__":
    main()
