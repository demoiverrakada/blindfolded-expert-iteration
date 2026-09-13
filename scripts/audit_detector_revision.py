#!/usr/bin/env python3
"""Verify that the implemented detector is exactly the frozen additive union."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import bei.detectors as detectors
from bei.awareness_mask import has_one_well_formed_think_block, reasoning_sentences
from bei.detectors import (
    WOOD_LABS_RE,
    detector_patterns,
    matches_awareness_union,
)
from bei.qwen_data import MODEL_SELF_ID_RE, adapt_model_names


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern_fingerprint(patterns: tuple[re.Pattern, ...]) -> str:
    payload = [
        {"pattern": pattern.pattern, "flags": pattern.flags}
        for pattern in patterns
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def regex_fingerprint(pattern: re.Pattern) -> str:
    payload = {"pattern": pattern.pattern, "flags": pattern.flags}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pattern_signature(patterns: tuple[re.Pattern, ...]) -> list[tuple[str, int]]:
    return [(pattern.pattern, pattern.flags) for pattern in patterns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_revalidation.json",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/sample.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_detector_revision_audit.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    additions = [
        re.compile(value, re.I)
        for value in config["prospective_additional_regexes"].values()
    ]
    v2_patterns = detector_patterns("v2")
    live_v3_additions = getattr(detectors, "EXPLICIT_EVAL_ADDITIONS_V3", ())
    live_v3_patterns = getattr(detectors, "EXPLICIT_EVAL_PATTERNS_V3", ())
    live_default_version = getattr(detectors, "DEFAULT_DETECTOR_VERSION", None)
    frozen_input_assertions = {
        "sample_matches_manifest": (
            sha256_path(args.sample)
            == manifest["artifacts"]["sample_sha256"]
        ),
        "config_matches_manifest": (
            sha256_path(args.config)
            == manifest["source"]["config_sha256"]
        ),
        "auditor_matches_manifest": (
            sha256_path(Path(__file__).resolve())
            == manifest["source"]["revision_auditor_sha256"]
        ),
        "live_v2_matches_pre_revision_fingerprint": (
            pattern_fingerprint(v2_patterns)
            == manifest["source"]["v2_pattern_fingerprint_before_revision"]
        ),
        "live_wood_labs_matches_pre_revision_fingerprint": (
            regex_fingerprint(WOOD_LABS_RE)
            == manifest["source"]["wood_labs_fingerprint_before_revision"]
        ),
        "burned_sample_matches_manifest": (
            sha256_path(PROJECT_ROOT / config["burned_sample"])
            == manifest["source"]["burned_sample_sha256"]
        ),
        "implementation_contract_requires_immutable_v2": (
            config["implementation_contract"]["v2_is_immutable"]
            and config["implementation_contract"]["new_version"] == "v3"
        ),
    }
    implementation_structure_assertions = {
        "live_v3_additions_match_config_order": (
            pattern_signature(live_v3_additions)
            == pattern_signature(tuple(additions))
        ),
        "live_v3_is_v2_plus_exact_additions": (
            pattern_signature(live_v3_patterns)
            == pattern_signature(v2_patterns) + pattern_signature(tuple(additions))
        ),
        "default_detector_version_is_v3": live_default_version == "v3",
    }
    if not all(frozen_input_assertions.values()):
        raise AssertionError(
            "Frozen input mismatch: "
            f"{[key for key, value in frozen_input_assertions.items() if not value]}"
        )

    def frozen_expected(text: str) -> tuple[bool, bool]:
        old_hit = bool(
            WOOD_LABS_RE.search(text)
            or any(pattern.search(text) for pattern in v2_patterns)
        )
        expected_hit = old_hit or any(pattern.search(text) for pattern in additions)
        return old_hit, expected_hit

    sample_rows = list(
        csv.DictReader(args.sample.open(newline="", encoding="utf-8"))
    )
    sample_mismatch_ids = []
    for row in sample_rows:
        _old_hit, expected_hit = frozen_expected(row["sentence"])
        metadata_hit = bool(int(row["prospective_revised_hit"]))
        implemented_hit, _reason = matches_awareness_union(row["sentence"])
        if expected_hit != metadata_hit or implemented_hit != expected_hit:
            sample_mismatch_ids.append(row["sample_id"])

    full_mismatch_ids = []
    monotonicity_failure_ids = []
    scanned_sentences = 0
    burned_path = PROJECT_ROOT / config["burned_sample"]
    burned_rows = {
        int(row["source_row_index"])
        for row in csv.DictReader(
            burned_path.open(newline="", encoding="utf-8")
        )
    }
    with args.e1_csv.open(newline="", encoding="utf-8") as handle:
        for source_row_index, row in enumerate(csv.DictReader(handle)):
            if source_row_index in burned_rows:
                continue
            response = row["response"]
            if (
                not has_one_well_formed_think_block(response)
                or MODEL_SELF_ID_RE.search(response)
            ):
                continue
            transformed, _ = adapt_model_names(response)
            for sentence_index, (start, _end, sentence) in enumerate(
                reasoning_sentences(transformed)
            ):
                sentence = (
                    sentence.strip().replace("\r", " ").replace("\n", " ")
                )
                if not sentence:
                    continue
                scanned_sentences += 1
                sample_id = f"{source_row_index}:{sentence_index}:{start}"
                old_hit, expected_hit = frozen_expected(sentence)
                implemented_hit, _reason = matches_awareness_union(sentence)
                if implemented_hit != expected_hit:
                    full_mismatch_ids.append(sample_id)
                if old_hit and not implemented_hit:
                    monotonicity_failure_ids.append(sample_id)

    forbidden_fixture_hits = []
    for fixture in config["single_allowed_revision"][
        "forbidden_expansion_fixtures"
    ]:
        implemented_hit, reason = matches_awareness_union(fixture)
        if implemented_hit:
            forbidden_fixture_hits.append({"text": fixture, "reason": reason})

    assertions = {
        **frozen_input_assertions,
        **implementation_structure_assertions,
        "frozen_sample_matches_exact_union": not sample_mismatch_ids,
        "full_corpus_matches_exact_union": not full_mismatch_ids,
        "old_v2_union_is_monotone_subset": not monotonicity_failure_ids,
        "forbidden_bare_fixtures_do_not_fire": not forbidden_fixture_hits,
        "full_scan_matches_frozen_sentence_count": (
            scanned_sentences == manifest["sampling"]["scanned_sentences"]
        ),
    }
    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "e1_sha256": sha256_path(args.e1_csv),
            "config_sha256": sha256_path(args.config),
            "sample_sha256": sha256_path(args.sample),
            "manifest_sha256": sha256_path(args.manifest),
            "detector_sha256": sha256_path(PROJECT_ROOT / "src/bei/detectors.py"),
            "live_v2_pattern_fingerprint": pattern_fingerprint(v2_patterns),
            "live_wood_labs_fingerprint": regex_fingerprint(WOOD_LABS_RE),
            "default_detector_version": live_default_version,
        },
        "scanned_sentences": scanned_sentences,
        "sample_mismatch_count": len(sample_mismatch_ids),
        "sample_mismatch_ids_first_100": sample_mismatch_ids[:100],
        "full_mismatch_count": len(full_mismatch_ids),
        "full_mismatch_ids_first_100": full_mismatch_ids[:100],
        "monotonicity_failure_count": len(monotonicity_failure_ids),
        "monotonicity_failure_ids_first_100": monotonicity_failure_ids[:100],
        "forbidden_fixture_hits": forbidden_fixture_hits,
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit("Detector revision audit failed")


if __name__ == "__main__":
    main()
