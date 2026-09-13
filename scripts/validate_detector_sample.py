#!/usr/bin/env python3
"""Validate detector versions against frozen sample annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from audit_training_data import explicit_eval_match


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return center - margin, center + margin


def confusion(rows: list[dict], label_name: str, version: str) -> dict:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    disagreements = []
    for row in rows:
        predicted = explicit_eval_match(row["response"], version)
        actual = bool(int(row[label_name]))
        key = (
            "tp"
            if predicted and actual
            else "fp"
            if predicted
            else "fn"
            if actual
            else "tn"
        )
        counts[key] += 1
        if predicted != actual:
            disagreements.append(int(row["source_row_index"]))
    precision_trials = counts["tp"] + counts["fp"]
    recall_trials = counts["tp"] + counts["fn"]
    counts["precision"] = (
        counts["tp"] / precision_trials if precision_trials else None
    )
    counts["precision_wilson_95"] = (
        wilson_interval(counts["tp"], precision_trials)
        if precision_trials
        else None
    )
    counts["recall"] = counts["tp"] / recall_trials if recall_trials else None
    counts["recall_wilson_95"] = (
        wilson_interval(counts["tp"], recall_trials) if recall_trials else None
    )
    counts["disagreements"] = disagreements
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_csv", type=Path)
    parser.add_argument("annotations_csv", type=Path)
    parser.add_argument("--v1-positive-prevalence", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = {
        row["source_row_index"]: row
        for row in csv.DictReader(args.sample_csv.open(newline=""))
    }
    annotations = {
        row["source_row_index"]: row
        for row in csv.DictReader(args.annotations_csv.open(newline=""))
    }
    if sample.keys() != annotations.keys():
        raise ValueError("Sample and annotation row indices differ")
    rows = [{**sample[index], **annotations[index]} for index in sample]

    report = {"sample_rows": len(rows), "detectors": {}}
    for version in ("v1", "v2"):
        report["detectors"][version] = {
            label: confusion(rows, label, version)
            for label in ("lenient_label", "strict_label")
        }

    p_positive = args.v1_positive_prevalence
    for label in ("lenient_label", "strict_label"):
        positive_stratum = [
            int(row[label]) for row in rows if row["detector_hit"] == "1"
        ]
        negative_stratum = [
            int(row[label]) for row in rows if row["detector_hit"] == "0"
        ]
        estimate = (
            p_positive * sum(positive_stratum) / len(positive_stratum)
            + (1 - p_positive) * sum(negative_stratum) / len(negative_stratum)
        )
        report.setdefault("v1_stratified_population_estimates", {})[label] = {
            "point_estimate": estimate,
            "positive_stratum_rate": sum(positive_stratum) / len(positive_stratum),
            "negative_stratum_rate": sum(negative_stratum) / len(negative_stratum),
            "note": "Point estimate only; intervals are documented in the report.",
        }

    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
