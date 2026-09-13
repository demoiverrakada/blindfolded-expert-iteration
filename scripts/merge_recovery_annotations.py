#!/usr/bin/env python3
"""Merge blinded recovery contexts with independently recorded chunk labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FIELDS = (
    "sample_id",
    "system_prompt",
    "user_prompt",
    "previous_sentence",
    "target_sentence",
    "next_sentence",
)
OUTPUT_FIELDS = CONTEXT_FIELDS + (
    "awareness_role",
    "functional_neutrality",
    "notes",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blinded",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/blinded.csv",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/annotations.csv",
    )
    args = parser.parse_args()

    blinded_rows = list(
        csv.DictReader(args.blinded.open(newline="", encoding="utf-8"))
    )
    label_paths = sorted(args.labels_dir.glob("recovery_chunk_*_labels.csv"))
    if len(label_paths) != 8:
        raise AssertionError(f"Expected 8 label files, found {len(label_paths)}")

    labels: dict[str, dict[str, str]] = {}
    for path in label_paths:
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            sample_id = row["sample_id"]
            if sample_id in labels:
                raise AssertionError(f"Duplicate label ID: {sample_id}")
            awareness_role = row["awareness_role"]
            functional_neutrality = row["functional_neutrality"]
            if awareness_role not in {"0", "1", "2", "3"}:
                raise AssertionError(f"Invalid awareness role: {sample_id}")
            if functional_neutrality not in {"0", "1", "2"}:
                raise AssertionError(f"Invalid neutrality label: {sample_id}")
            labels[sample_id] = row

    blinded_ids = [row["sample_id"] for row in blinded_rows]
    if len(blinded_ids) != len(set(blinded_ids)):
        raise AssertionError("Duplicate blinded sample IDs")
    if set(blinded_ids) != set(labels):
        raise AssertionError("Label files must cover blinded IDs exactly")

    output_rows = []
    for row in blinded_rows:
        label = labels[row["sample_id"]]
        output_rows.append(
            {
                **{field: row[field] for field in CONTEXT_FIELDS},
                "awareness_role": label["awareness_role"],
                "functional_neutrality": label["functional_neutrality"],
                "notes": label.get("notes", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} annotations to {args.output}")


if __name__ == "__main__":
    main()
