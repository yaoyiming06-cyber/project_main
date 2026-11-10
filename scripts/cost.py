"""Measure image-to-submission latency using the shared competition pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.competition_inference import (
    InferencePipeline,
    PROBABILITY_DECIMAL_PLACES,
    PROBABILITY_ROUNDING,
    round_probabilities,
    scan_image_tree as _scan_image_tree,
    sha256_file,
    validate_warmup_separation as _validate_warmup_separation,
)


PROBABILITY_TOLERANCE = 1e-3
DEFAULT_THRESHOLD = 0.5


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _read_submission(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Study_id", "Predict_class", "Probability"]:
            raise ValueError(
                "submission must have exactly Study_id, Predict_class, Probability columns"
            )
        rows = list(reader)

    result = {}
    order = []
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed submission row: {row!r}")
        study_id = (row.get("Study_id") or "").strip()
        try:
            label = int(row["Predict_class"])
            probability = float(row["Probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid submission row: {row!r}") from error
        if (
            not study_id
            or label not in range(1, 11)
            or not np.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"invalid submission row: {row!r}")
        values = result.setdefault(study_id, {})
        if label in values:
            raise ValueError(f"duplicate class row for {study_id} / {label}")
        values[label] = probability
        if len(values) == 1:
            order.append(study_id)

    for study_id, values in result.items():
        if set(values) != set(range(1, 11)):
            raise ValueError(f"Study_id {study_id} must contain all ten class rows")
    if not result:
        raise ValueError("submission contains no prediction rows")
    return result, order


def _study_ranking_mismatches(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Count changed pairwise label orderings across Studies, including ties."""
    count = 0
    for label in range(reference.shape[1]):
        original = reference[:, label]
        current = candidate[:, label]
        for index in range(len(original) - 1):
            original_order = np.sign(original[index] - original[index + 1 :])
            current_order = np.sign(current[index] - current[index + 1 :])
            count += int(np.count_nonzero(original_order != current_order))
    return count


def _label_ranking_mismatches(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Count changed pairwise label orderings within each Study, including ties."""
    first, second = np.triu_indices(reference.shape[1], k=1)
    return int(
        np.count_nonzero(
            np.sign(reference[:, first] - reference[:, second])
            != np.sign(candidate[:, first] - candidate[:, second])
        )
    )


def _ranking_change_masks(reference, candidate):
    """Return affected Studies; an across-Study reversal marks both participants."""
    study_mask = np.zeros(len(reference), dtype=bool)
    for label in range(reference.shape[1]):
        for index in range(len(reference) - 1):
            changed = (
                np.sign(reference[index, label] - reference[index + 1:, label])
                != np.sign(candidate[index, label] - candidate[index + 1:, label])
            )
            if changed.any():
                study_mask[index] = True
                study_mask[index + 1:] |= changed
    first, second = np.triu_indices(reference.shape[1], k=1)
    label_mask = np.any(
        np.sign(reference[:, first] - reference[:, second])
        != np.sign(candidate[:, first] - candidate[:, second]), axis=1
    )
    return study_mask, label_mask


def _compare_submission(
    path: Path,
    study_ids: list[str],
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = PROBABILITY_TOLERANCE,
):
    submission, _ = _read_submission(path)
    candidate = np.asarray(probabilities, dtype=np.float64)
    if not study_ids or candidate.shape != (len(study_ids), 10):
        raise ValueError("candidate predictions must have shape (Study count, 10)")
    if not np.isfinite(candidate).all() or ((candidate < 0) | (candidate > 1)).any():
        raise ValueError("candidate probabilities must be finite and between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("classification threshold must be between 0 and 1")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("probability tolerance must be finite and non-negative")
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("candidate Study_id values must be unique")

    missing = [study_id for study_id in study_ids if study_id not in submission]
    if missing:
        raise ValueError(f"submission is missing measured Study_id: {missing[:5]}")
    reference = np.asarray(
        [[submission[study_id][label] for label in range(1, 11)] for study_id in study_ids],
        dtype=np.float64,
    )
    candidate = round_probabilities(candidate)
    reference = round_probabilities(reference)
    absolute_error = np.abs(candidate - reference)
    over_tolerance = absolute_error > tolerance + 1e-12
    classification_changes = (candidate >= threshold) != (reference >= threshold)
    study_rank_changes = _study_ranking_mismatches(reference, candidate)
    label_rank_changes = _label_ranking_mismatches(reference, candidate)
    study_rank_mask, label_rank_mask = _ranking_change_masks(reference, candidate)
    mismatch_mask = np.any(over_tolerance | classification_changes, axis=1) | study_rank_mask | label_rank_mask
    mismatched_study_ids = [sid for sid, changed in zip(study_ids, mismatch_mask) if changed]

    first_mismatches = []
    for row_index in np.flatnonzero(mismatch_mask)[:10]:
        first_mismatches.append(
            {
                "Study_id": study_ids[int(row_index)],
                "probability_error_classes": (np.flatnonzero(over_tolerance[row_index]) + 1).tolist(),
                "classification_changed_classes": (np.flatnonzero(classification_changes[row_index]) + 1).tolist(),
                "reference_probabilities": reference[row_index].tolist(),
                "cost_probabilities": candidate[row_index].tolist(),
                "max_absolute_error": float(absolute_error[row_index].max()),
                "study_ranking_changed": bool(study_rank_mask[row_index]),
                "label_ranking_changed": bool(label_rank_mask[row_index]),
            }
        )

    probability_mismatch_count = int(np.count_nonzero(over_tolerance))
    classification_mismatch_count = int(np.count_nonzero(classification_changes))
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "probability_tolerance": tolerance,
        "probability_decimal_places": PROBABILITY_DECIMAL_PLACES,
        "probability_rounding": PROBABILITY_ROUNDING,
        "max_absolute_error": float(absolute_error.max(initial=0.0)),
        "probability_tolerance_mismatch_count": probability_mismatch_count,
        "classification_threshold": threshold,
        "classification_mismatch_count": classification_mismatch_count,
        "study_ranking_mismatch_count": study_rank_changes,
        "label_ranking_mismatch_count": label_rank_changes,
        "matched": not bool(mismatch_mask.any()),
        "sample_unit": "Study",
        "sample_count": len(study_ids),
        "sample_mismatch_count": len(mismatched_study_ids),
        "mismatch_count": len(mismatched_study_ids),
        "mismatched_study_ids": mismatched_study_ids,
        "first_mismatches": first_mismatches,
    }


def _summarize_comparisons(comparisons, measured_rows):
    """Union sample failures across repeats; later passing runs cannot hide an earlier failure."""
    summary = dict(comparisons[-1])
    mismatched = set().union(*(set(c["mismatched_study_ids"]) for c in comparisons))
    ordered = list(dict.fromkeys(row["Study_id"] for row in measured_rows))
    summary.update(
        matched=all(c["matched"] for c in comparisons),
        mismatch_count=len(mismatched),
        sample_mismatch_count=len(mismatched),
        mismatched_study_ids=[sid for sid in ordered if sid in mismatched],
        image_sample_count=len(measured_rows),
        image_mismatch_count=sum(row["Study_id"] in mismatched for row in measured_rows),
        max_absolute_error=max(c["max_absolute_error"] for c in comparisons),
        checked_repeats=len(comparisons),
        event_counts_scope="all measured repeats; sample/image counts are unique across repeats",
        first_mismatches=[dict(item, repeat=i + 1)
                          for i, c in enumerate(comparisons) for item in c["first_mismatches"]][:10],
    )
    for key in ("probability_tolerance_mismatch_count", "classification_mismatch_count",
                "study_ranking_mismatch_count", "label_ranking_mismatch_count"):
        summary[key] = sum(c[key] for c in comparisons)
    return summary


def _model_records(pipeline: InferencePipeline):
    records = []
    for model in pipeline.models:
        use_amp = bool(model.config["amp"] and pipeline.device.type == "cuda")
        records.append(
            {
                "name": model.name,
                "config": str(model.config_path),
                "config_sha256": model.config_sha256,
                "checkpoint": str(model.checkpoint_path),
                "checkpoint_sha256": model.checkpoint_sha256,
                "checkpoint_size_bytes": model.checkpoint_path.stat().st_size,
                "weight_source": model.weight_source,
                "input_size": int(model.config["input_size"]),
                "configured_image_chunk_size": int(model.config["image_chunk_size"]),
                "effective_image_encoder_batch_size": int(model.model.image_chunk_size),
                "autocast_requested": bool(model.config["amp"]),
                "autocast_effective": use_amp,
            }
        )
    return records


def _timed_prediction_run(pipeline, prepared, output_path):
    _synchronize(pipeline.device)
    started = time.perf_counter()
    study_ids, probabilities = pipeline.predict(prepared, output_path)
    _synchronize(pipeline.device)
    return study_ids, probabilities, time.perf_counter() - started


def _read_written_probabilities(path: Path, study_ids: list[str]) -> np.ndarray:
    submission, _ = _read_submission(path)
    return np.asarray(
        [[submission[study_id][label] for label in range(1, 11)] for study_id in study_ids],
        dtype=np.float64,
    )


def _select_warmup_rows(rows: list[dict[str, str]], image_count: int):
    if image_count < 1 or len(rows) < image_count:
        raise ValueError(
            f"warmup directory must contain at least {image_count} image files"
        )
    selected = []
    seen_studies = set()
    for row in rows:
        if row["Study_id"] not in seen_studies:
            selected.append(row)
            seen_studies.add(row["Study_id"])
            if len(selected) == image_count:
                return selected
    selected_paths = {row["image_path"] for row in selected}
    for row in rows:
        if row["image_path"] not in selected_paths:
            selected.append(row)
            if len(selected) == image_count:
                return selected
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-images", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument(
        "--allow-short-sample",
        action="store_true",
        help="permit diagnostic runs when the measured directory has fewer than 100 images",
    )
    args = parser.parse_args()

    if args.repeats < 1 or args.expected_images < 1 or args.warmup_images < 1:
        raise ValueError("repeats, expected-images, and warmup-images must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "cost_predictions.csv"
    report_path = output_dir / "cost_report.json"
    submission_path = args.submission.expanduser().resolve() if args.submission else None
    if submission_path is not None and submission_path == prediction_path:
        raise ValueError("--submission must not be the cost output path")

    measured_root = args.data_dir.expanduser().resolve()
    warmup_root = args.warmup_dir.expanduser().resolve()
    measured_rows = _scan_image_tree(measured_root)
    all_warmup_rows = _scan_image_tree(warmup_root)
    _validate_warmup_separation(
        measured_rows, all_warmup_rows, measured_root, warmup_root
    )
    warmup_rows = _select_warmup_rows(all_warmup_rows, args.warmup_images)
    measured_images = len(measured_rows)
    if measured_images > args.expected_images:
        raise ValueError(
            f"measured directory has {measured_images} images; expected exactly {args.expected_images}"
        )
    if measured_images < args.expected_images and not args.allow_short_sample:
        raise ValueError(
            f"measured directory has {measured_images} images, expected {args.expected_images}; "
            "use --allow-short-sample only for a diagnostic run"
        )

    with InferencePipeline(args.bundle, args.device) as pipeline:
        measured = pipeline.prepare_tree(measured_root, measured_rows, name="measured")
        warmup = pipeline.prepare_tree(warmup_root, warmup_rows, name="warmup")

        warmup_ids, _ = pipeline.predict(warmup)
        if warmup_ids != warmup.study_ids:
            raise RuntimeError("warmup Study order changed")

        runs = []
        first_ids = None
        first_probabilities = None
        repeat_max_deltas = []
        comparisons = []
        for repeat in range(args.repeats):
            study_ids, probabilities, elapsed = _timed_prediction_run(
                pipeline, measured, prediction_path
            )
            if first_ids is None:
                first_ids = study_ids
                first_probabilities = probabilities.copy()
                repeat_max_deltas.append(0.0)
            else:
                if study_ids != first_ids:
                    raise RuntimeError("Study order changed between cost repeats")
                delta = float(np.max(np.abs(probabilities - first_probabilities), initial=0.0))
                repeat_max_deltas.append(delta)
            runs.append({"repeat": repeat + 1, "seconds": elapsed})
            if submission_path is not None:
                written = _read_written_probabilities(prediction_path, study_ids)
                comparisons.append(_compare_submission(
                    submission_path, study_ids, written,
                    threshold=pipeline.classification_threshold,
                ))

        comparison = _summarize_comparisons(comparisons, measured.rows) if comparisons else None

        seconds = [run["seconds"] for run in runs]
        official_count_match = measured.num_images == args.expected_images
        study_image_counts = {}
        for row in measured.rows:
            study_image_counts[row["Study_id"]] = study_image_counts.get(row["Study_id"], 0) + 1
        multi_image_studies = sum(count > 1 for count in study_image_counts.values())
        official_timing_resolved = all(
            model.model.image_chunk_size == 1 for model in pipeline.models
        )
        report = {
            "status": "consistency_failed" if comparison is not None and not comparison["matched"] else "completed" if (
                official_count_match and args.repeats == 3 and official_timing_resolved
            ) else "diagnostic_only",
            "consistency_status": "not_checked" if comparison is None else (
                "passed" if comparison["matched"] else "failed"
            ),
            "bundle": str(pipeline.bundle_path),
            "bundle_sha256": sha256_file(pipeline.bundle_path),
            "seed": pipeline.seed,
            "device": str(pipeline.device),
            "precision": "fp16 autocast" if pipeline.device.type == "cuda" and any(
                model.config["amp"] for model in pipeline.models
            ) else "fp32",
            "models": _model_records(pipeline),
            "fusion": "weighted arithmetic mean of model probabilities",
            "patient_smoothing_alpha": pipeline.patient_smoothing_alpha,
            "patient_smoothing_scope": "disabled for subset-reproducible official inference",
            "probability_decimal_places": PROBABILITY_DECIMAL_PLACES,
            "probability_rounding": PROBABILITY_ROUNDING,
            "classification_threshold": pipeline.classification_threshold,
            "tta": pipeline.tta,
            "input": {
                "data_dir": str(measured_root),
                "image_count": measured.num_images,
                "study_count": measured.num_studies,
                "subject_count": len(set(measured.subject_by_study.values())),
                "generated_manifest_sha256": measured.manifest_sha256,
                "expected_images": args.expected_images,
                "sample_count_matches_protocol": official_count_match,
                "sample_unit": "image files grouped into complete Studies",
                "batch_size_studies": 1,
                "image_encoder_batch_size": 1,
                "multi_image_study_count": multi_image_studies,
                "single_image_serial_timing": official_timing_resolved,
                "synthetic_view_value": "UNKNOWN",
                "clinical_metadata_read": False,
            },
            "warmup": {
                "data_dir": str(warmup_root),
                "image_count": warmup.num_images,
                "study_count": warmup.num_studies,
                "generated_manifest_sha256": warmup.manifest_sha256,
                "requested_image_count": args.warmup_images,
                "disjoint_by_study_id": True,
            },
            "timing_scope": {
                "included": [
                    "image file open and decode",
                    "preprocessing",
                    "serial model inference",
                    "model probability fusion",
                    "final four-decimal probability rounding",
                    "final competition CSV serialization and close",
                ],
                "excluded": [
                    "model/checkpoint loading",
                    "directory scan and manifest construction",
                    "warmup inference",
                    "comparison against an existing submission",
                    "cost report serialization",
                ],
            },
            "repeats": args.repeats,
            "timing_seconds": runs,
            "mean_seconds_per_run": float(np.mean(seconds)),
            "std_seconds_per_run": float(np.std(seconds, ddof=1)) if len(seconds) > 1 else 0.0,
            "amortized_milliseconds_per_image_file": float(np.mean(seconds) * 1000 / measured.num_images),
            "mean_milliseconds_per_study": float(np.mean(seconds) * 1000 / measured.num_studies),
            "official_latency_limit_ms": 100.0,
            "mean_milliseconds_per_official_sample": (
                float(np.mean(seconds) * 1000 / measured.num_images)
                if official_count_match and official_timing_resolved
                else None
            ),
            "within_official_latency_limit": (
                bool(np.mean(seconds) * 1000 / measured.num_images <= 100.0)
                if official_count_match and official_timing_resolved
                else None
            ),
            "hardware_target_confirmation": "not verified; run on the competition-specified hardware",
            "repeat_max_absolute_probability_delta": repeat_max_deltas,
            "prediction_output": str(prediction_path),
            "prediction_output_sha256": sha256_file(prediction_path),
            "submission_comparison": comparison,
            "repeat_submission_comparisons": comparisons,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if comparison is not None and not comparison["matched"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
