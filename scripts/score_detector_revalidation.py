#!/usr/bin/env python3
"""Score the frozen post-revision detector audit without discretionary choices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.detectors import WOOD_LABS_RE, detector_patterns, matches_awareness_union

RECALL_STRATA = (
    "detector_positive",
    "cue_enriched_detector_negative",
    "unenriched_detector_negative",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, trials: int, z: float = 1.96) -> list[float]:
    if trials == 0:
        return [0.0, 1.0]
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


def precision_report(
    selected_ids: list[str], labels: dict[str, int]
) -> dict:
    aware_ids = [sample_id for sample_id in selected_ids if labels[sample_id] == 1]
    false_positive_ids = [
        sample_id for sample_id in selected_ids if labels[sample_id] != 1
    ]
    return {
        "rows": len(selected_ids),
        "aware": len(aware_ids),
        "precision": len(aware_ids) / len(selected_ids),
        "wilson_95": wilson(len(aware_ids), len(selected_ids)),
        "false_positive_ids": false_positive_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/sample.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/annotations.csv",
    )
    parser.add_argument(
        "--blinded",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/blinded.csv",
    )
    parser.add_argument(
        "--selections",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/selections.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_revalidation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_detector_revalidation.json",
    )
    parser.add_argument(
        "--revision-audit",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_detector_revision_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    selections = json.loads(args.selections.read_text())
    revision_audit = json.loads(args.revision_audit.read_text())
    sample_rows = list(
        csv.DictReader(args.sample.open(newline="", encoding="utf-8"))
    )
    annotation_rows = list(
        csv.DictReader(args.annotations.open(newline="", encoding="utf-8"))
    )
    blinded_rows = list(
        csv.DictReader(args.blinded.open(newline="", encoding="utf-8"))
    )
    sample_by_id = {row["sample_id"]: row for row in sample_rows}
    blinded_by_id = {row["sample_id"]: row for row in blinded_rows}
    labels = {}
    notes = {}
    for row in annotation_rows:
        label = int(row["label"])
        if label not in (0, 1, 2):
            raise AssertionError(f"Invalid label for {row['sample_id']}: {label}")
        if row["sample_id"] in labels:
            raise AssertionError(f"Duplicate annotation: {row['sample_id']}")
        if row.get("sentence") != sample_by_id[row["sample_id"]]["sentence"]:
            raise AssertionError(f"Annotated text drift for {row['sample_id']}")
        labels[row["sample_id"]] = label
        notes[row["sample_id"]] = row.get("notes", "")

    expected_ids = set(sample_by_id)
    if set(labels) != expected_ids or len(labels) != len(sample_rows):
        raise AssertionError("Annotations must cover the unique frozen sample exactly")
    for stratum, selected_ids in selections.items():
        if len(selected_ids) != config["samples"][stratum]:
            raise AssertionError(f"Selection count drift in {stratum}")
        if not set(selected_ids).issubset(expected_ids):
            raise AssertionError(f"Unknown selected ID in {stratum}")

    frozen_input_assertions = {
        "sample_matches_manifest": (
            sha256_path(args.sample)
            == manifest["artifacts"]["sample_sha256"]
        ),
        "selections_match_manifest": (
            sha256_path(args.selections)
            == manifest["artifacts"]["selections_sha256"]
        ),
        "config_matches_manifest": (
            sha256_path(args.config)
            == manifest["source"]["config_sha256"]
        ),
        "scorer_matches_manifest": (
            sha256_path(Path(__file__).resolve())
            == manifest["source"]["scorer_sha256"]
        ),
        "blinded_matches_manifest": (
            sha256_path(args.blinded)
            == manifest["artifacts"]["blinded_sha256"]
        ),
        "blinded_ids_and_text_match_sample": (
            set(blinded_by_id) == expected_ids
            and all(
                blinded_by_id[sample_id]["sentence"]
                == sample_by_id[sample_id]["sentence"]
                for sample_id in expected_ids
            )
        ),
        "revision_audit_passed_before_scoring": (
            revision_audit["status"] == "PASS"
            and revision_audit["inputs"]["sample_sha256"]
            == manifest["artifacts"]["sample_sha256"]
            and revision_audit["inputs"]["config_sha256"]
            == manifest["source"]["config_sha256"]
        ),
    }
    if not all(frozen_input_assertions.values()):
        raise AssertionError(
            "Frozen input mismatch: "
            f"{[key for key, value in frozen_input_assertions.items() if not value]}"
        )

    implementation_mismatches = []
    for sample_id, row in sample_by_id.items():
        implemented_hit, _reason = matches_awareness_union(row["sentence"])
        expected_hit = bool(int(row["prospective_revised_hit"]))
        if implemented_hit != expected_hit:
            implementation_mismatches.append(sample_id)

    pool_sizes = manifest["sampling"]["pool_sizes"]
    stratum_reports = {}
    rates = {}
    for stratum in RECALL_STRATA:
        selected_ids = selections[stratum]
        aware_ids = [sample_id for sample_id in selected_ids if labels[sample_id] == 1]
        rate = len(aware_ids) / len(selected_ids)
        rates[stratum] = rate
        stratum_reports[stratum] = {
            "pool_rows": pool_sizes[stratum],
            "sample_rows": len(selected_ids),
            "aware": len(aware_ids),
            "aware_rate": rate,
            "wilson_95": wilson(len(aware_ids), len(selected_ids)),
            "aware_ids": aware_ids,
        }

    estimated_tp = pool_sizes["detector_positive"] * rates["detector_positive"]
    estimated_fn_cue = (
        pool_sizes["cue_enriched_detector_negative"]
        * rates["cue_enriched_detector_negative"]
    )
    estimated_fn_unenriched = (
        pool_sizes["unenriched_detector_negative"]
        * rates["unenriched_detector_negative"]
    )
    denominator = estimated_tp + estimated_fn_cue + estimated_fn_unenriched
    weighted_recall = estimated_tp / denominator if denominator else None

    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    draws = bootstrap["draws"]
    sampled_rates = {}
    for stratum in RECALL_STRATA:
        n = len(selections[stratum])
        sampled_rates[stratum] = (
            rng.binomial(n, rates[stratum], size=draws) / n
        )
    sampled_tp = (
        pool_sizes["detector_positive"] * sampled_rates["detector_positive"]
    )
    sampled_fn = (
        pool_sizes["cue_enriched_detector_negative"]
        * sampled_rates["cue_enriched_detector_negative"]
        + pool_sizes["unenriched_detector_negative"]
        * sampled_rates["unenriched_detector_negative"]
    )
    sampled_denominator = sampled_tp + sampled_fn
    bootstrap_recall = np.divide(
        sampled_tp,
        sampled_denominator,
        out=np.full(draws, np.nan),
        where=sampled_denominator != 0,
    )
    tail = (1 - bootstrap["interval"]) / 2
    bootstrap_interval = [
        float(np.nanquantile(bootstrap_recall, tail)),
        float(np.nanquantile(bootstrap_recall, 1 - tail)),
    ]

    sensitivity = {}
    sensitivity_fn = estimated_fn_cue + estimated_fn_unenriched
    confidence = config["zero_event_sensitivity"]["one_sided_confidence"]
    alpha = 1 - confidence
    for stratum in RECALL_STRATA[1:]:
        report = stratum_reports[stratum]
        if report["aware"] == 0:
            upper_rate = 1 - alpha ** (1 / report["sample_rows"])
            extra_fn = pool_sizes[stratum] * upper_rate
            sensitivity_fn += extra_fn
            sensitivity[stratum] = {
                "one_sided_confidence": confidence,
                "clopper_pearson_upper_rate": upper_rate,
                "upper_weighted_false_negatives": extra_fn,
            }
    sensitivity_recall = (
        estimated_tp / (estimated_tp + sensitivity_fn)
        if estimated_tp + sensitivity_fn
        else None
    )

    pattern_reports = {
        stratum: precision_report(selections[stratum], labels)
        for stratum in ("v2_pattern_9", "v2_pattern_12")
    }
    family_reports = {
        str(family): precision_report(
            selections[f"added_only_family_{family}"], labels
        )
        for family in range(1, 5)
    }
    positive_ids = selections["detector_positive"]
    v2_patterns = detector_patterns("v2")
    source_categories = {
        "wood_labs": [
            sample_id
            for sample_id in positive_ids
            if WOOD_LABS_RE.search(sample_by_id[sample_id]["sentence"])
        ],
        "v2_only": [
            sample_id
            for sample_id in positive_ids
            if (
                not WOOD_LABS_RE.search(sample_by_id[sample_id]["sentence"])
                and any(
                    pattern.search(sample_by_id[sample_id]["sentence"])
                    for pattern in v2_patterns
                )
            )
        ],
        "added_only": [
            sample_id
            for sample_id in positive_ids
            if (
                not WOOD_LABS_RE.search(sample_by_id[sample_id]["sentence"])
                and not any(
                    pattern.search(sample_by_id[sample_id]["sentence"])
                    for pattern in v2_patterns
                )
            )
        ],
    }
    detector_source_attribution = {}
    for category, category_ids in source_categories.items():
        aware_ids = [
            sample_id for sample_id in category_ids if labels[sample_id] == 1
        ]
        detector_source_attribution[category] = {
            "sample_rows": len(category_ids),
            "aware": len(aware_ids),
            "aware_rate": (
                len(aware_ids) / len(category_ids) if category_ids else None
            ),
            "estimated_detected_awareness_mass": (
                pool_sizes["detector_positive"]
                * len(aware_ids)
                / len(positive_ids)
            ),
        }

    neutral_ids = selections["neutral_code_holdout"]
    neutral_added_only_hit_ids = [
        sample_id
        for sample_id in neutral_ids
        if (
            int(sample_by_id[sample_id]["prospective_revised_hit"])
            and not int(sample_by_id[sample_id]["old_detector_hit"])
        )
    ]
    neutral_old_v2_false_positive_ids = [
        sample_id
        for sample_id in neutral_ids
        if (
            int(sample_by_id[sample_id]["old_detector_hit"])
            and labels[sample_id] != 1
        )
    ]

    gates = config["gates"]
    assertions = {
        **frozen_input_assertions,
        "complete_unique_sentence_only_labels": (
            len(labels) == manifest["sampling"]["unique_blinded_sentences"]
        ),
        "implemented_revision_matches_frozen_union_on_sample": (
            not implementation_mismatches
        ),
        "weighted_inclusive_recall": (
            weighted_recall is not None
            and weighted_recall >= gates["weighted_inclusive_recall"]
        ),
        "pattern_9_inclusive_precision": (
            pattern_reports["v2_pattern_9"]["precision"]
            >= gates["pattern_9_inclusive_precision"]
        ),
        "pattern_12_inclusive_precision": (
            pattern_reports["v2_pattern_12"]["precision"]
            >= gates["pattern_12_inclusive_precision"]
        ),
        "neutral_code_holdout_added_only_hits": (
            len(neutral_added_only_hit_ids)
            <= gates["neutral_code_holdout_max_added_only_hits"]
        ),
    }
    if gates["added_only_family_gate_scope"] != "per_family":
        raise AssertionError("Only the frozen per-family gate scope is supported")
    for family in gates["added_only_family_gated_families"]:
        assertions[f"added_only_family_{family}_inclusive_precision"] = (
            family_reports[str(family)]["precision"]
            >= gates["added_only_family_inclusive_precision"]
        )

    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "sample_sha256": sha256_path(args.sample),
            "annotations_sha256": sha256_path(args.annotations),
            "selections_sha256": sha256_path(args.selections),
            "manifest_sha256": sha256_path(args.manifest),
            "config_sha256": sha256_path(args.config),
        },
        "metric_name_note": config["metric_name_note"],
        "weighted_recall": {
            "estimated_true_positives": estimated_tp,
            "estimated_false_negatives": {
                "cue_enriched": estimated_fn_cue,
                "unenriched": estimated_fn_unenriched,
            },
            "point_estimate": weighted_recall,
            "bootstrap": {
                **bootstrap,
                "interval_values": bootstrap_interval,
            },
            "zero_event_sensitivity": {
                "strata": sensitivity,
                "simultaneous_negative_strata_recall": sensitivity_recall,
            },
            "strata": stratum_reports,
        },
        "pattern_precision": pattern_reports,
        "added_only_family_precision": {
            "gate_scope": gates["added_only_family_gate_scope"],
            "gated_families": gates["added_only_family_gated_families"],
            "family_4_status": config["family_4_validation"],
            "families": family_reports,
        },
        "detector_source_attribution": detector_source_attribution,
        "neutral_code_holdout": {
            "rows": len(neutral_ids),
            "pre_revision_old_v2_raw_hits": manifest["sampling"][
                "neutral_code_pre_revision_baseline"
            ],
            "added_only_hit_ids": neutral_added_only_hit_ids,
            "old_v2_false_positive_ids_descriptive": (
                neutral_old_v2_false_positive_ids
            ),
        },
        "implementation_mismatch_ids": implementation_mismatches,
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": report["status"], "assertions": assertions},
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("Detector revalidation gate failed")


if __name__ == "__main__":
    main()
