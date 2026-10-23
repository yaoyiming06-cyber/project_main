"""Run study-level test inference from a completed training checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cxr_project.config import load_config, resolve_path
from cxr_project.dataset import StudyDataset, build_transform, collate_studies
from cxr_project.labels import LABEL_COLUMNS
from cxr_project.losses import combine_main_and_expert_logits
from cxr_project.model import ConvNeXtStudyClassifier
from cxr_project.train import _autocast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite prediction file: {output_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    normalization = None
    if str(config["normalization"]).lower() == "dataset":
        stats_path = checkpoint_path.parent / "dataset_normalization.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"dataset normalization stats not found: {stats_path}")
        normalization = json.loads(stats_path.read_text(encoding="utf-8"))

    dataset = StudyDataset(
        resolve_path(config, "test_manifest"),
        args.data_root,
        build_transform(False, int(config["input_size"]), config, normalization),
        labeled=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size_studies"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"] and device.type == "cuda"),
        collate_fn=collate_studies,
        persistent_workers=int(config["num_workers"]) > 0,
        **({"prefetch_factor": 2} if int(config["num_workers"]) > 0 else {}),
    )

    model = ConvNeXtStudyClassifier(
        pretrained=False,
        drop_path=float(config["drop_path"]),
        hard_label_indices=config["hard_label_indices"],
        gradient_checkpointing=False,
        image_chunk_size=int(config["image_chunk_size"]),
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"], strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Study_id", *LABEL_COLUMNS])
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                view_ids = batch["view_ids"].to(device, non_blocking=True)
                with _autocast(device, bool(config["amp"])):
                    main_logits, expert_logits = model(images, batch["offsets"], view_ids)
                probabilities = combine_main_and_expert_logits(
                    main_logits.float(), expert_logits.float(), config["hard_label_indices"],
                ).sigmoid().cpu().tolist()
                writer.writerows(
                    [study_id, *values]
                    for study_id, values in zip(batch["study_ids"], probabilities)
                )
                print(f"batch {batch_index}/{len(loader)}", flush=True)

    print(f"Wrote {len(dataset)} study predictions to {output_path}")


if __name__ == "__main__":
    main()
