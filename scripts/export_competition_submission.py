"""Convert study-level probability predictions to the competition long format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite submission: {output_path}")

    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        prediction_rows = list(reader)
        expected_probability_columns = [f"label_{index}" for index in range(10)]
        if reader.fieldnames != ["Study_id", *expected_probability_columns]:
            raise ValueError(f"unexpected prediction columns: {reader.fieldnames}")

    with template_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        template_rows = list(reader)
        if reader.fieldnames != ["Study_id", "Predict_class", "Probability"]:
            raise ValueError(f"unexpected competition columns: {reader.fieldnames}")

    study_ids = [row["Study_id"] for row in prediction_rows]
    if len(study_ids) != len(set(study_ids)):
        raise ValueError("predictions contain duplicate Study_id values")
    template_ids = list(dict.fromkeys(row["Study_id"] for row in template_rows))
    if template_ids != study_ids:
        raise ValueError("template Study_id order does not match predictions")
    if len(template_rows) != len(study_ids) * 10:
        raise ValueError("template must contain exactly ten class rows per Study")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Study_id", "Predict_class", "Probability"])
        for row in prediction_rows:
            for index, column in enumerate(expected_probability_columns, start=1):
                probability = float(row[column])
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"probability outside [0, 1]: {row['Study_id']} {column}")
                writer.writerow([row["Study_id"], index, f"{probability:.8f}"])

    print(f"Wrote {len(study_ids) * 10} rows for {len(study_ids)} studies to {output_path}")


if __name__ == "__main__":
    main()
