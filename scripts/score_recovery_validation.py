#!/usr/bin/env python3
"""Score the frozen CP3-R contextual semantic validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scipy.stats import beta

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_sided_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def one_sided_upper(successes: int, trials: int, alpha: float) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/recovery_validation.json",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/key.csv",
    )
    parser.add_argument(
        "--blinded",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/blinded.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/annotations.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation/manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_recovery_validation.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    key_rows = list(csv.DictReader(args.key.open(newline="", encoding="utf-8")))
    blinded_rows = list(
        csv.DictReader(args.blinded.open(newline="", encoding="utf-8"))
    )
    annotation_rows = list(
        csv.DictReader(args.annotations.open(newline="", encoding="utf-8"))
    )
    key_by_id = {row["sample_id"]: row for row in key_rows}
    blinded_by_id = {row["sample_id"]: row for row in blinded_rows}
    if len(key_by_id) != len(key_rows) or len(blinded_by_id) != len(blinded_rows):
        raise AssertionError("Duplicate frozen sample IDs")

    context_fields = (
        "system_prompt",
        "user_prompt",
        "previous_sentence",
        "target_sentence",
        "next_sentence",
    )
    labels = {}
    for row in annotation_rows:
        sample_id = row["sample_id"]
        if sample_id not in key_by_id:
            raise AssertionError(f"Unknown annotation ID: {sample_id}")
        if sample_id in labels:
            raise AssertionError(f"Duplicate annotation ID: {sample_id}")
        for field in context_fields:
            if row[field] != key_by_id[sample_id][field]:
                raise AssertionError(f"Annotation context drift: {sample_id}:{field}")
        awareness_role = int(row["awareness_role"])
        functional_neutrality = int(row["functional_neutrality"])
        if awareness_role not in (0, 1, 2, 3):
            raise AssertionError(f"Invalid awareness role: {sample_id}")
        if functional_neutrality not in (0, 1, 2):
            raise AssertionError(f"Invalid neutrality label: {sample_id}")
        labels[sample_id] = {
            "awareness_role": awareness_role,
            "functional_neutrality": functional_neutrality,
            "notes": row.get("notes", ""),
        }
    if set(labels) != set(key_by_id):
        raise AssertionError("Annotations must cover frozen IDs exactly")

    frozen_assertions = {
        "config_matches_manifest": (
            sha256_path(args.config) == manifest["source"]["config_sha256"]
        ),
        "key_matches_manifest": (
            sha256_path(args.key) == manifest["artifacts"]["key_sha256"]
        ),
        "blinded_matches_manifest": (
            sha256_path(args.blinded)
            == manifest["artifacts"]["blinded_sha256"]
        ),
        "scorer_matches_manifest": (
            sha256_path(Path(__file__).resolve())
            == manifest["source"]["scorer_sha256"]
        ),
        "blinded_context_matches_key": (
            set(blinded_by_id) == set(key_by_id)
            and all(
                all(
                    blinded_by_id[sample_id][field]
                    == key_by_id[sample_id][field]
                    for field in context_fields
                )
                for sample_id in key_by_id
            )
        ),
    }
    if not all(frozen_assertions.values()):
        raise AssertionError(
            "Frozen mismatch: "
            f"{[key for key, value in frozen_assertions.items() if not value]}"
        )

    alpha = config["one_sided_alpha"]
    stratum_reports = {}
    assertions = dict(frozen_assertions)
    for stratum, expected_rows in config["samples"].items():
        sample_ids = [
            row["sample_id"] for row in key_rows if row["stratum"] == stratum
        ]
        if len(sample_ids) != expected_rows:
            raise AssertionError(f"Frozen stratum count drift: {stratum}")
        if stratum in config["awareness_gated_strata"]:
            successes = sum(
                labels[sample_id]["awareness_role"] == 1
                for sample_id in sample_ids
            )
            lower = one_sided_lower(successes, len(sample_ids), alpha)
            passed = lower >= config["awareness_precision_lower_bound"]
            assertions[f"{stratum}_awareness_precision"] = passed
            stratum_reports[stratum] = {
                "rows": len(sample_ids),
                "awareness_successes": successes,
                "point_precision": successes / len(sample_ids),
                "one_sided_95_lower": lower,
                "mere_mentions": sum(
                    labels[sample_id]["awareness_role"] == 2
                    for sample_id in sample_ids
                ),
                "unresolved": sum(
                    labels[sample_id]["awareness_role"] == 3
                    for sample_id in sample_ids
                ),
            }
        elif stratum in config["placebo_gated_strata"]:
            contamination = sum(
                labels[sample_id]["functional_neutrality"] != 1
                for sample_id in sample_ids
            )
            upper = one_sided_upper(contamination, len(sample_ids), alpha)
            passed = upper <= config["placebo_contamination_upper_bound"]
            assertions[f"{stratum}_functional_neutrality"] = passed
            stratum_reports[stratum] = {
                "rows": len(sample_ids),
                "contaminated_or_unresolved": contamination,
                "point_contamination": contamination / len(sample_ids),
                "one_sided_95_upper": upper,
            }
        else:
            aware = sum(
                labels[sample_id]["awareness_role"] == 1
                for sample_id in sample_ids
            )
            stratum_reports[stratum] = {
                "rows": len(sample_ids),
                "awareness_rows": aware,
                "diagnostic_awareness_rate": aware / len(sample_ids),
                "mere_mentions": sum(
                    labels[sample_id]["awareness_role"] == 2
                    for sample_id in sample_ids
                ),
                "unresolved": sum(
                    labels[sample_id]["awareness_role"] == 3
                    for sample_id in sample_ids
                ),
            }

    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "config_sha256": sha256_path(args.config),
            "manifest_sha256": sha256_path(args.manifest),
            "key_sha256": sha256_path(args.key),
            "blinded_sha256": sha256_path(args.blinded),
            "annotations_sha256": sha256_path(args.annotations),
        },
        "gates": {
            "one_sided_alpha": alpha,
            "awareness_precision_lower_bound": config[
                "awareness_precision_lower_bound"
            ],
            "placebo_contamination_upper_bound": config[
                "placebo_contamination_upper_bound"
            ],
        },
        "strata": stratum_reports,
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "assertions": assertions}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("CP3-R semantic validation failed")


if __name__ == "__main__":
    main()
