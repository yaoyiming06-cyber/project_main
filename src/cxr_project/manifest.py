"""Validate image manifests and group rows into labeled studies."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .labels import LABEL_COLUMNS, NUM_LABELS


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    subject_id: str
    image_paths: tuple[str, ...]
    view_positions: tuple[str, ...]
    labels: tuple[int, ...] | None


def _parse_label(value, path: Path, row_number: int, column: str) -> int:
    raw = (value or "").strip().lower()
    if raw in ("", "nan", "none", "null"):
        return 0
    try:
        number = float(raw)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number} has invalid {column}: {value!r}") from error
    if number not in (-1, 0, 1):
        raise ValueError(f"{path}:{row_number} has invalid {column}: {value!r}")
    return int(number)


def read_study_manifest(path, labeled=True, include_view_positions=True) -> list[StudyRecord]:
    """Read a one-row-per-image CSV and return one record per Study_id.

    Blank labels map to zero, matching the selected G1 configuration. Explicit
    -1 labels are preserved for the E0 loss mask.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"image_path", "Subject_id", "Study_id"}
        if include_view_positions:
            required.add("ViewPosition")
        if labeled:
            required.update(LABEL_COLUMNS)
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")

        grouped = {}
        seen_paths = set()
        for row_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip().replace("\\", "/")
            subject_id = (row.get("Subject_id") or "").strip()
            study_id = (row.get("Study_id") or "").strip()
            view_position = (row.get("ViewPosition") or "").strip() if include_view_positions else None
            if not image_path or not subject_id or not study_id:
                raise ValueError(f"{path}:{row_number} has an empty image path or ID")
            if image_path in seen_paths:
                raise ValueError(f"{path}:{row_number} repeats image_path {image_path!r}")
            seen_paths.add(image_path)

            labels = None
            if labeled:
                labels = tuple(
                    _parse_label(row.get(column), path, row_number, column)
                    for column in LABEL_COLUMNS
                )
            study = grouped.get(study_id)
            if study is None:
                study = {
                    "subject_id": subject_id,
                    "labels": labels,
                    "image_paths": [],
                    "view_positions": [],
                }
                grouped[study_id] = study
            elif study["subject_id"] != subject_id:
                raise ValueError(f"Study_id {study_id!r} maps to multiple Subject_id values")
            elif study["labels"] != labels:
                raise ValueError(f"Study_id {study_id!r} has inconsistent labels")
            study["image_paths"].append(image_path)
            if include_view_positions:
                study["view_positions"].append(view_position)

    if not grouped:
        raise ValueError(f"manifest contains no image rows: {path}")
    return [
        StudyRecord(
            study_id=study_id,
            subject_id=values["subject_id"],
            image_paths=tuple(values["image_paths"]),
            view_positions=tuple(values["view_positions"]),
            labels=values["labels"],
        )
        for study_id, values in grouped.items()
    ]


def compute_pos_weights(studies, cap=20.0) -> list[float]:
    """Compute E0 positive weights from known study-level labels only."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    positives = [0] * NUM_LABELS
    negatives = [0] * NUM_LABELS
    for study in studies:
        if study.labels is None or len(study.labels) != NUM_LABELS:
            raise ValueError(f"Study {study.study_id!r} has no complete label vector")
        for index, value in enumerate(study.labels):
            if value == 1:
                positives[index] += 1
            elif value == 0:
                negatives[index] += 1
            elif value != -1:
                raise ValueError(f"Study {study.study_id!r} has invalid label {value!r}")
    return [
        min(cap, negatives[index] / positives[index]) if positives[index] else float(cap)
        for index in range(NUM_LABELS)
    ]


def resolve_pos_weights(studies, mode="ratio", cap=20.0, manual=None) -> list[float]:
    """Resolve the controlled E0 positive-weight variants."""
    mode = str(mode).strip().lower()
    if mode == "ratio":
        return compute_pos_weights(studies, cap=cap)
    if mode == "unit":
        return [1.0] * NUM_LABELS
    if mode == "sqrt":
        return [weight ** 0.5 for weight in compute_pos_weights(studies, cap=cap)]
    if mode == "manual":
        if manual is None or len(manual) != NUM_LABELS:
            raise ValueError(f"manual pos_weights must contain {NUM_LABELS} values")
        values = [float(value) for value in manual]
        if any(value <= 0 for value in values):
            raise ValueError("manual pos_weights must be positive")
        return values
    raise ValueError(f"unsupported pos_weight_mode: {mode!r}")
