"""Final competition probability precision and CSV format shared by all entrypoints."""

import csv
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np


PROBABILITY_DECIMAL_PLACES = 4
PROBABILITY_ROUNDING = "decimal half-even"
_QUANTUM = Decimal("0.0001")


def round_probabilities(probabilities):
    """Validate raw values, then round decimal strings once at the final output boundary."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite and between 0 and 1 before rounding")
    return np.asarray([
        float(Decimal(str(float(value))).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN))
        for value in values.flat
    ], dtype=np.float64).reshape(values.shape)


def write_competition_submission(path, study_ids, probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(study_ids), 10) or not len(study_ids):
        raise ValueError("submission predictions must have nonempty shape (Study count, 10)")
    if len(set(study_ids)) != len(study_ids) or any(not str(s).strip() for s in study_ids):
        raise ValueError("submission Study_id values must be nonempty and unique")
    values = round_probabilities(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Study_id", "Predict_class", "Probability"])
        for study_id, row in zip(study_ids, values):
            for label_index, probability in enumerate(row, start=1):
                writer.writerow([study_id, label_index, f"{probability:.4f}"])
