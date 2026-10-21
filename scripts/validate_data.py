"""Validate the bundled A_exclude2 manifests against the image tree."""

import argparse
from pathlib import Path

from cxr_project.config import load_config, resolve_path
from cxr_project.dataset import StudyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/g1_seed42.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    image_root = Path(args.data_root).resolve() if args.data_root else resolve_path(config, "image_root")
    excluded_file = project_root / "data" / "manifests" / "excluded_train_paths.txt"
    excluded = set(excluded_file.read_text(encoding="utf-8").splitlines())

    split_sets = {}
    for split, key, labeled in (
        ("train", "train_manifest", True),
        ("valid", "valid_manifest", True),
        ("test", "test_manifest", False),
    ):
        manifest = resolve_path(config, key)
        dataset = StudyDataset(manifest, image_root, labeled=labeled, verify_paths=True)
        used_paths = {image for study in dataset.studies for image in study.image_paths}
        if split == "train" and used_paths & excluded:
            raise ValueError("one or more explicitly excluded corrupt images remain in train_manifest.csv")
        split_sets[split] = {
            "subjects": {study.subject_id for study in dataset.studies},
            "studies": {study.study_id for study in dataset.studies},
            "dicoms": {Path(image).stem for study in dataset.studies for image in study.image_paths},
        }
        image_count = sum(len(study.image_paths) for study in dataset.studies)
        print(f"{split}: {image_count:,} images, {len(dataset):,} studies, all files present")

    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for unit in ("subjects", "studies", "dicoms"):
            overlap = split_sets[first][unit] & split_sets[second][unit]
            if overlap:
                raise ValueError(f"{first}/{second} {unit} overlap: {len(overlap)}")
    print(f"Excluded corrupt images absent: {len(excluded)}")
    print("Data validation: OK")


if __name__ == "__main__":
    main()
