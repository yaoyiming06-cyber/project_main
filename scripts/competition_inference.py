"""Shared image-only inference pipeline used by predict.py and cost.py."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cxr_project.config import load_config
from cxr_project.dataset import StudyDataset, build_transform, collate_studies
from cxr_project.runtime import select_device
from scripts.predict_test import (
    inference_weights,
    predict_batch,
    set_inference_size,
    validate_checkpoint_config,
)
from cxr_project.model import build_model
from cxr_project.submission import (
    PROBABILITY_DECIMAL_PLACES,
    PROBABILITY_ROUNDING,
    round_probabilities,
    write_competition_submission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SUBJECT_PATTERN = re.compile(r"^subject_[^/]+$")
_STUDY_PATTERN = re.compile(r"^study_[^/]+$")
INFERENCE_IRRELEVANT_CONFIG_KEYS = {
    "overfit_auc_min_delta",
    "overfit_stop_patience",
    "overfit_stop_start_epoch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def scan_image_tree(data_root: Path) -> list[dict[str, str]]:
    """Read IDs from subject/study folders only; never open metadata files."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"image directory does not exist: {root}")

    rows = []
    subject_by_study = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"image symlink escapes input directory: {path}") from error
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            raise ValueError(
                f"expected subject/study/image directory layout, found: {relative}"
            )
        subject_id, study_id, filename = relative.parts
        if not _SUBJECT_PATTERN.fullmatch(subject_id) or not _STUDY_PATTERN.fullmatch(study_id):
            raise ValueError(f"invalid subject/study folder names: {relative}")
        previous_subject = subject_by_study.setdefault(study_id, subject_id)
        if previous_subject != subject_id:
            raise ValueError(f"Study_id {study_id!r} occurs under multiple subjects")
        rows.append(
            {
                "image_path": relative.as_posix(),
                "Subject_id": subject_id,
                "Study_id": study_id,
                "ViewPosition": "UNKNOWN",
            }
        )

    if not rows:
        raise ValueError(f"no supported images found under {root}")
    return sorted(rows, key=lambda row: (row["Subject_id"], row["Study_id"], row["image_path"]))


def validate_warmup_separation(
    measured_rows: list[dict[str, str]],
    warmup_rows: list[dict[str, str]],
    measured_root: Path | None = None,
    warmup_root: Path | None = None,
) -> None:
    measured_studies = {row["Study_id"] for row in measured_rows}
    warmup_studies = {row["Study_id"] for row in warmup_rows}
    overlap = measured_studies & warmup_studies
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "warmup and measured inputs must be disjoint by Study_id; "
            f"overlapping studies: {examples}"
        )
    if measured_root is not None and warmup_root is not None:
        measured_paths = {
            (Path(measured_root).resolve() / row["image_path"]).resolve()
            for row in measured_rows
        }
        warmup_paths = {
            (Path(warmup_root).resolve() / row["image_path"]).resolve()
            for row in warmup_rows
        }
        path_overlap = measured_paths & warmup_paths
        if path_overlap:
            examples = [str(path) for path in sorted(path_overlap)[:5]]
            raise ValueError(f"warmup and measured image paths overlap: {examples}")


def _resolve_bundle_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ModelRuntime:
    name: str
    config_path: Path
    checkpoint_path: Path
    config: dict
    model: torch.nn.Module
    transform: object
    weight_source: str
    fusion_weight: float
    checkpoint_sha256: str
    config_sha256: str


@dataclass
class PreparedTree:
    root: Path
    rows: list[dict[str, str]]
    study_ids: list[str]
    subject_by_study: dict[str, str]
    datasets: list[StudyDataset]
    manifest_paths: list[Path]
    manifest_sha256: str

    @property
    def num_images(self) -> int:
        return len(self.rows)

    @property
    def num_studies(self) -> int:
        return len(self.study_ids)


class InferencePipeline:
    """Load the deployment bundle once and reuse it for predict and cost runs."""

    def __init__(self, bundle_path: Path, device: str = "auto"):
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if self.bundle.get("version") != 1:
            raise ValueError("deployment bundle version must be 1")
        self.seed = int(self.bundle.get("seed", 42))
        if self.seed < 0:
            raise ValueError("deployment seed must be non-negative")
        set_seed(self.seed)
        self.device = select_device(device)
        self.patient_smoothing_alpha = float(
            self.bundle.get("patient_smoothing_alpha", 0.0)
        )
        if not math.isfinite(self.patient_smoothing_alpha) or not 0 <= self.patient_smoothing_alpha <= 1:
            raise ValueError("patient_smoothing_alpha must be between 0 and 1")
        if self.patient_smoothing_alpha != 0:
            raise ValueError(
                "official predict/cost requires patient_smoothing_alpha=0: directory-based "
                "patient smoothing cannot reproduce full-test predictions on a 100-image subset; "
                "regenerate submission with smoothing disabled"
            )
        fusion = self.bundle.get("fusion", "weighted_probability_mean")
        if fusion != "weighted_probability_mean":
            raise ValueError("fusion must be 'weighted_probability_mean'")
        self.classification_threshold = float(
            self.bundle.get("classification_threshold", 0.5)
        )
        if not math.isfinite(self.classification_threshold) or not 0 <= self.classification_threshold <= 1:
            raise ValueError("classification_threshold must be between 0 and 1")
        self.tta = self.bundle.get("tta", "none")
        if self.tta not in {"none", "horizontal-flip"}:
            raise ValueError("tta must be 'none' or 'horizontal-flip'")
        specs = self.bundle.get("models")
        if not isinstance(specs, list) or not specs:
            raise ValueError("deployment bundle must contain at least one model")

        self._scratch = tempfile.TemporaryDirectory(prefix="cxr-inference-")
        self._scratch_path = Path(self._scratch.name)
        self.models = []
        for index, spec in enumerate(specs):
            self.models.append(self._load_model(spec, index))
        weights = np.asarray([model.fusion_weight for model in self.models], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            self.close()
            raise ValueError("each model fusion weight must be finite and positive")
        self._normalized_weights = weights / weights.sum()

    def _load_model(self, spec: dict, index: int) -> ModelRuntime:
        if not isinstance(spec, dict):
            raise ValueError(f"models[{index}] must be an object")
        name = str(spec.get("name") or f"model_{index + 1}")
        config_path = _resolve_bundle_path(spec["config"])
        checkpoint_path = _resolve_bundle_path(spec["checkpoint"])
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ValueError(f"config must contain a YAML mapping: {config_path}")
        ignored_keys = INFERENCE_IRRELEVANT_CONFIG_KEYS & set(raw_config)
        if ignored_keys:
            inference_config_path = self._scratch_path / f"config_{index}.yaml"
            for key in ignored_keys:
                raw_config.pop(key)
            inference_config_path.write_text(
                yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            config = load_config(inference_config_path)
        else:
            config = load_config(config_path)
        if bool(config.get("use_view_embedding", True)) or config.get("view_filter") is not None:
            raise ValueError(
                f"{name} is not image-only; require use_view_embedding=false and view_filter=null"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint_config(config, checkpoint)
        state, weight_source = inference_weights(checkpoint, spec.get("weights", "auto"))
        model = build_model(config, pretrained=False).float()
        model.load_state_dict(state, strict=True)
        set_inference_size(model, config, int(config["input_size"]), state)
        model.image_chunk_size = 1
        model.to(self.device).eval()

        normalization = checkpoint.get("normalization_stats") or checkpoint.get("normalization")
        if str(config["normalization"]).lower() == "dataset" and not isinstance(normalization, dict):
            stats_path = checkpoint_path.parent / "dataset_normalization.json"
            if not stats_path.is_file():
                raise FileNotFoundError(f"dataset normalization stats not found: {stats_path}")
            normalization = json.loads(stats_path.read_text(encoding="utf-8"))
        transform = build_transform(False, int(config["input_size"]), config, normalization)
        return ModelRuntime(
            name=name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            config=config,
            model=model,
            transform=transform,
            weight_source=weight_source,
            fusion_weight=float(spec.get("fusion_weight", 1.0)),
            checkpoint_sha256=sha256_file(checkpoint_path),
            config_sha256=sha256_file(config_path),
        )

    def prepare_tree(
        self,
        data_root: Path,
        rows: list[dict[str, str]] | None = None,
        max_studies: int | None = None,
        name: str = "input",
    ) -> PreparedTree:
        root = Path(data_root).expanduser().resolve()
        rows = scan_image_tree(root) if rows is None else list(rows)
        ordered_studies = list(dict.fromkeys(row["Study_id"] for row in rows))
        if max_studies is not None:
            if max_studies < 1 or max_studies > len(ordered_studies):
                raise ValueError(
                    f"max_studies must be between 1 and {len(ordered_studies)}"
                )
            selected = set(ordered_studies[:max_studies])
            rows = [row for row in rows if row["Study_id"] in selected]
            ordered_studies = ordered_studies[:max_studies]
        if not rows:
            raise ValueError(f"no images selected from {root}")

        subject_by_study = {}
        for row in rows:
            subject = subject_by_study.setdefault(row["Study_id"], row["Subject_id"])
            if subject != row["Subject_id"]:
                raise ValueError(f"Study_id {row['Study_id']!r} maps to multiple subjects")

        manifest_paths, datasets = [], []
        for index, runtime in enumerate(self.models):
            manifest_path = self._scratch_path / f"{name}_{index}.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "Subject_id", "Study_id", "ViewPosition"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            dataset = StudyDataset(
                manifest_path,
                root,
                runtime.transform,
                labeled=False,
                view_filter=None,
            )
            if [study.study_id for study in dataset.studies] != ordered_studies:
                raise RuntimeError("generated manifest changed Study order")
            manifest_paths.append(manifest_path)
            datasets.append(dataset)
        return PreparedTree(
            root,
            rows,
            ordered_studies,
            subject_by_study,
            datasets,
            manifest_paths,
            sha256_file(manifest_paths[0]),
        )

    def predict(self, prepared: PreparedTree, output_path: Path | None = None):
        model_predictions = []
        for runtime, dataset in zip(self.models, prepared.datasets):
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate_studies,
            )
            study_ids, chunks = [], []
            with torch.inference_mode():
                for batch in loader:
                    study_ids.extend(batch["study_ids"])
                    probabilities = predict_batch(
                        runtime.model,
                        batch,
                        self.device,
                        bool(runtime.config["amp"]),
                        horizontal_flip=self.tta == "horizontal-flip",
                    )
                    chunks.append(probabilities.numpy().astype(np.float64, copy=False))
            if study_ids != prepared.study_ids:
                raise RuntimeError(f"Study order changed during inference for {runtime.name}")
            model_predictions.append(np.concatenate(chunks, axis=0))

        probabilities = np.average(
            np.stack(model_predictions, axis=0),
            axis=0,
            weights=self._normalized_weights,
        )
        probabilities = round_probabilities(probabilities)
        if output_path is not None:
            write_competition_submission(output_path, prepared.study_ids, probabilities)
        return prepared.study_ids, probabilities

    def close(self) -> None:
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            scratch.cleanup()
            self._scratch = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
