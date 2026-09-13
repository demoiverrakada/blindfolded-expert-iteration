#!/usr/bin/env python3
"""Score the blinded, reason-stratified V3 mask validation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REASON = "v3_addition_5"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, trials: int, z: float = 1.96) -> list[float]:
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1 - rate) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load_annotations(path: Path, expected_sentences: dict[str, str]) -> dict[str, int]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    labels = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in expected_sentences:
            raise AssertionError(f"Unknown annotation ID: {sample_id}")
        if row["sentence"] != expected_sentences[sample_id]:
            raise AssertionError(f"Annotation text drift: {sample_id}")
        label = int(row["label"])
        if label not in (0, 1, 2):
            raise AssertionError(f"Invalid label for {sample_id}: {label}")
        if sample_id in labels:
            raise AssertionError(f"Duplicate annotation: {sample_id}")
        labels[sample_id] = label
    if set(labels) != set(expected_sentences):
        raise AssertionError("Annotations do not cover the frozen IDs exactly")
    return labels


def precision_report(sample_ids: list[str], labels: dict[str, int]) -> dict:
    aware_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 1]
    return {
        "rows": len(sample_ids),
        "aware": len(aware_ids),
        "precision": len(aware_ids) / len(sample_ids),
        "wilson_95": wilson(len(aware_ids), len(sample_ids)),
        "nonaware_ids": [
            sample_id for sample_id in sample_ids if labels[sample_id] != 1
        ],
    }


def unique_sentence_precision_report(
    sample_ids: list[str],
    labels: dict[str, int],
    sentences: dict[str, str],
) -> dict:
    sentence_labels = {
        sentences[sample_id]: labels[sample_id] for sample_id in sample_ids
    }
    aware = sum(label == 1 for label in sentence_labels.values())
    return {
        "unique_sentences": len(sentence_labels),
        "aware": aware,
        "precision": aware / len(sentence_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        type=Path,
        default=PROJECT_ROOT / "frozen/mask_v3_validation/key.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "frozen/mask_v3_validation/annotations.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "frozen/mask_v3_validation/manifest.json",
    )
    parser.add_argument(
        "--detector-key",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/sample.csv",
    )
    parser.add_argument(
        "--detector-annotations",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/annotations.csv",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_revalidation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_mask_validation_v3_scored.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    config = json.loads(args.detector_config.read_text())
    key_rows = list(csv.DictReader(args.key.open(newline="", encoding="utf-8")))
    key_by_id = {row["sample_id"]: row for row in key_rows}
    labels = load_annotations(
        args.annotations,
        {sample_id: row["sentence"] for sample_id, row in key_by_id.items()},
    )

    frozen_assertions = {
        "key_matches_manifest": (
            sha256_path(args.key) == manifest["artifacts"]["key_sha256"]
        ),
        "config_matches_manifest": (
            sha256_path(args.detector_config)
            == manifest["source"]["config_sha256"]
        ),
        "scorer_matches_manifest": (
            sha256_path(Path(__file__).resolve())
            == manifest["source"]["scorer_sha256"]
        ),
        "detector_matches_manifest": (
            sha256_path(PROJECT_ROOT / "src/bei/detectors.py")
            == manifest["source"]["detector_sha256"]
        ),
        "blinded_matches_manifest": (
            sha256_path(PROJECT_ROOT / "frozen/mask_v3_validation/blinded.csv")
            == manifest["artifacts"]["blinded_sha256"]
        ),
        "rubric_matches_manifest": (
            sha256_path(PROJECT_ROOT / "frozen/mask_v3_validation/rubric.md")
            == manifest["artifacts"]["rubric_sha256"]
        ),
    }
    if not all(frozen_assertions.values()):
        raise AssertionError(
            "Frozen input mismatch: "
            f"{[key for key, value in frozen_assertions.items() if not value]}"
        )

    reason_names = list(config["prospective_additional_regexes"])
    reason_to_family = {
        f"v3_addition_{index}": int(name.split("_")[1])
        for index, name in enumerate(reason_names)
    }
    sampled_reasons = tuple(
        reason for reason in reason_to_family if reason != EXTERNAL_REASON
    )
    sentence_by_id = {
        sample_id: row["sentence"] for sample_id, row in key_by_id.items()
    }
    reports = {}
    for reason in sampled_reasons:
        sample_ids = [
            row["sample_id"] for row in key_rows if row["reason"] == reason
        ]
        reports[reason] = {
            **precision_report(sample_ids, labels),
            "unique_sentence_sensitivity": unique_sentence_precision_report(
                sample_ids, labels, sentence_by_id
            ),
        }

    repeated_sentence_groups = {}
    for row in key_rows:
        repeated_sentence_groups.setdefault(row["sentence"], []).append(
            row["sample_id"]
        )
    identical_sentences_consistent = all(
        len({labels[sample_id] for sample_id in sample_ids}) == 1
        for sample_ids in repeated_sentence_groups.values()
    )
    if not identical_sentences_consistent:
        raise AssertionError("Identical sentences received inconsistent labels")

    family_reports = {}
    for family in config["gates"]["added_only_family_gated_families"]:
        family_ids = [
            row["sample_id"]
            for row in key_rows
            if reason_to_family[row["reason"]] == family
        ]
        family_reports[str(family)] = precision_report(family_ids, labels)

    external_report = {"status": "PENDING_DETECTOR_ANNOTATIONS"}
    if args.detector_annotations.exists():
        detector_rows = list(
            csv.DictReader(args.detector_key.open(newline="", encoding="utf-8"))
        )
        detector_by_id = {row["sample_id"]: row for row in detector_rows}
        external_ids = manifest["sampling"]["external_census_coverage"][
            "detector_revalidation_ids"
        ]
        detector_labels = load_annotations(
            args.detector_annotations,
            {
                sample_id: detector_by_id[sample_id]["sentence"]
                for sample_id in detector_by_id
            },
        )
        external_report = {
            "status": "DESCRIPTIVE_NOT_GATED",
            **precision_report(external_ids, detector_labels),
        }

    threshold = config["gates"]["added_only_family_inclusive_precision"]
    assertions = {
        **frozen_assertions,
        "identical_sentences_have_identical_labels": (
            identical_sentences_consistent
        ),
        **{
            f"family_{family}_precision_at_least_{threshold}": (
                family_reports[str(family)]["precision"] >= threshold
            )
            for family in config["gates"]["added_only_family_gated_families"]
        },
    }
    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "manifest_sha256": sha256_path(args.manifest),
            "key_sha256": sha256_path(args.key),
            "annotations_sha256": sha256_path(args.annotations),
            "detector_annotations_sha256": (
                sha256_path(args.detector_annotations)
                if args.detector_annotations.exists()
                else None
            ),
        },
        "threshold": threshold,
        "primary_per_family_precision": family_reports,
        "diagnostic_per_reason_precision": reports,
        "external_family_4_census": external_report,
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "assertions": assertions}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("V3 mask validation gate failed")


if __name__ == "__main__":
    main()
