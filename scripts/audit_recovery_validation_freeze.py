#!/usr/bin/env python3
"""Independently audit the frozen CP3-R contextual validation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from scipy.stats import beta
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import has_one_well_formed_think_block, reasoning_sentences
from bei.detectors import WOOD_LABS_RE, detector_patterns, matches_awareness_union
from bei.matched_mask import (
    core_awareness_spans,
    distant_position_control_indices,
    nearest_position_control_indices,
    neutral_control_spans,
    token_indices_for_spans,
)
from bei.qwen_data import MODEL_SELF_ID_RE, adapt_model_names


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_sentence(text: str) -> str:
    return text.strip().replace("\r", " ").replace("\n", " ")


def source_indices(path: Path) -> set[int]:
    values = set()
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        value = row.get("source_row_index", "").strip()
        if value:
            values.add(int(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/recovery_validation.json",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=PROJECT_ROOT / "frozen/e1_training_row_ids.json",
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_recovery_validation_freeze_audit.json",
    )
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    config = json.loads(args.config.read_text())
    manifest_path = args.frozen_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    key_path = args.frozen_dir / "key.csv"
    blinded_path = args.frozen_dir / "blinded.csv"
    key_rows = list(csv.DictReader(key_path.open(newline="", encoding="utf-8")))
    blinded_rows = list(
        csv.DictReader(blinded_path.open(newline="", encoding="utf-8"))
    )
    key_by_id = {row["sample_id"]: row for row in key_rows}
    blinded_by_id = {row["sample_id"]: row for row in blinded_rows}
    source_rows = list(
        csv.DictReader(args.e1_csv.open(newline="", encoding="utf-8"))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    excluded = {
        int(row["source_row_index"])
        for row in json.loads(args.membership.read_text())["rows"]
    }
    prior_file_reports = {}
    for relative_path in config["prior_source_row_files"]:
        path = PROJECT_ROOT / relative_path
        values = source_indices(path)
        excluded.update(values)
        prior_file_reports[relative_path] = {
            "sha256": sha256_path(path),
            "source_rows": len(values),
        }

    cue_config = json.loads(
        (PROJECT_ROOT / "configs/detector_validation.json").read_text()
    )
    cues = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in cue_config["independent_cue_lexicon"]
    ]
    v1_patterns = detector_patterns("v1")
    v2_patterns = detector_patterns("v2")
    context_fields = (
        "system_prompt",
        "user_prompt",
        "previous_sentence",
        "target_sentence",
        "next_sentence",
    )
    selector_failures = []
    context_failures = []
    placebo_failures = []
    ineligible_rows = []

    for frozen in key_rows:
        source_row_index = int(frozen["source_row_index"])
        row = source_rows[source_row_index]
        if (
            source_row_index in excluded
            or not has_one_well_formed_think_block(row["response"])
            or MODEL_SELF_ID_RE.search(row["response"])
        ):
            ineligible_rows.append(source_row_index)
            continue
        transformed = {
            field: adapt_model_names(row[field])[0]
            for field in ("system_prompt", "user_prompt", "response")
        }
        sentence_spans = reasoning_sentences(transformed["response"])
        sentence_index = int(frozen["sentence_index"])
        start, _end, sentence = sentence_spans[sentence_index]
        expected_context = {
            "system_prompt": transformed["system_prompt"],
            "user_prompt": transformed["user_prompt"],
            "previous_sentence": (
                normalize_sentence(sentence_spans[sentence_index - 1][2])
                if sentence_index > 0
                else ""
            ),
            "target_sentence": normalize_sentence(sentence),
            "next_sentence": (
                normalize_sentence(sentence_spans[sentence_index + 1][2])
                if sentence_index + 1 < len(sentence_spans)
                else ""
            ),
        }
        if (
            frozen["source_sample_id"]
            != f"{source_row_index}:{sentence_index}:{start}"
            or any(frozen[field] != expected_context[field] for field in context_fields)
        ):
            context_failures.append(frozen["sample_id"])

        stratum = frozen["stratum"]
        wood = bool(WOOD_LABS_RE.search(sentence))
        v1 = any(pattern.search(sentence) for pattern in v1_patterns)
        broad, _reason = matches_awareness_union(sentence)
        valid_selector = {
            "core_wood": wood,
            "core_nonwood_v1": (not wood and v1),
            "broad_only_v2_pattern_9": (
                broad and not wood and not v1 and bool(v2_patterns[9].search(sentence))
            ),
            "broad_only_v2_pattern_12": (
                broad
                and not wood
                and not v1
                and not v2_patterns[9].search(sentence)
                and bool(v2_patterns[12].search(sentence))
            ),
            "other_broad_only": (
                broad
                and not wood
                and not v1
                and not v2_patterns[9].search(sentence)
                and not v2_patterns[12].search(sentence)
            ),
            "broad_negative_diagnostic": (
                not broad and any(cue.search(sentence) for cue in cues)
            ),
        }.get(stratum)
        if valid_selector is False:
            selector_failures.append(frozen["sample_id"])

        if stratum not in ("near_placebo", "distant_placebo"):
            continue
        prefix_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": transformed["system_prompt"]},
                {"role": "user", "content": transformed["user_prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = prefix_text + transformed["response"]
        if not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False, truncation=False
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) > config["placebo_sequence_cap"]:
            placebo_failures.append(frozen["sample_id"])
            continue
        core_spans = core_awareness_spans(transformed["response"])
        neutral_spans = neutral_control_spans(transformed["response"])
        offsets = encoded["offset_mapping"]
        core_indices = token_indices_for_spans(
            offsets,
            [
                (len(prefix_text) + span.start, len(prefix_text) + span.end)
                for span in core_spans
            ],
            minimum_token_index=len(prefix_ids),
        )
        neutral_indices = token_indices_for_spans(
            offsets,
            [
                (len(prefix_text) + neutral_start, len(prefix_text) + neutral_end)
                for neutral_start, neutral_end, _text in neutral_spans
            ],
            minimum_token_index=len(prefix_ids),
        )
        try:
            if stratum == "near_placebo":
                selected = nearest_position_control_indices(
                    core_indices, neutral_indices
                )
            else:
                selected = distant_position_control_indices(
                    core_indices,
                    neutral_indices,
                    minimum_distance=config["distant_minimum_tokens"],
                    seed=(
                        f"{config['seed_string']}:placebo:"
                        f"{source_row_index}"
                    ),
                )
        except ValueError:
            placebo_failures.append(frozen["sample_id"])
            continue
        absolute_target = (
            len(prefix_text) + sentence_spans[sentence_index][0],
            len(prefix_text) + sentence_spans[sentence_index][1],
        )
        target_tokens = set(
            token_indices_for_spans(
                offsets,
                [absolute_target],
                minimum_token_index=len(prefix_ids),
            )
        )
        selected_overlap = target_tokens & set(selected)
        if (
            not selected_overlap
            or int(frozen["selected_token_count"]) != len(selected_overlap)
            or not any(
                neutral_start == sentence_spans[sentence_index][0]
                and neutral_end == sentence_spans[sentence_index][1]
                for neutral_start, neutral_end, _text in neutral_spans
            )
        ):
            placebo_failures.append(frozen["sample_id"])

    chunk_paths = sorted((args.frozen_dir / "chunks").glob("chunk_*.csv"))
    chunk_ids = [
        row["sample_id"]
        for path in chunk_paths
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
    ]
    selected_counts = {
        stratum: sum(row["stratum"] == stratum for row in key_rows)
        for stratum in config["samples"]
    }
    selected_source_rows = [int(row["source_row_index"]) for row in key_rows]
    awareness_n = config["samples"][config["awareness_gated_strata"][0]]
    max_awareness_failures = max(
        failures
        for failures in range(awareness_n + 1)
        if (
            beta.ppf(
                config["one_sided_alpha"],
                awareness_n - failures,
                failures + 1,
            )
            if awareness_n - failures
            else 0.0
        )
        >= config["awareness_precision_lower_bound"]
    )
    placebo_n = config["samples"][config["placebo_gated_strata"][0]]
    max_placebo_failures = max(
        failures
        for failures in range(placebo_n)
        if beta.ppf(
            1 - config["one_sided_alpha"],
            failures + 1,
            placebo_n - failures,
        )
        <= config["placebo_contamination_upper_bound"]
    )

    assertions = {
        "manifest_config_hash": (
            manifest["source"]["config_sha256"] == sha256_path(args.config)
        ),
        "manifest_builder_hash": (
            manifest["source"]["builder_sha256"]
            == sha256_path(PROJECT_ROOT / "scripts/build_recovery_validation_sample.py")
        ),
        "manifest_scorer_hash": (
            manifest["source"]["scorer_sha256"]
            == sha256_path(PROJECT_ROOT / "scripts/score_recovery_validation.py")
        ),
        "manifest_matched_mask_hash": (
            manifest["source"]["matched_mask_sha256"]
            == sha256_path(PROJECT_ROOT / "src/bei/matched_mask.py")
        ),
        "manifest_key_hash": (
            manifest["artifacts"]["key_sha256"] == sha256_path(key_path)
        ),
        "manifest_blinded_hash": (
            manifest["artifacts"]["blinded_sha256"] == sha256_path(blinded_path)
        ),
        "manifest_chunk_hashes": (
            manifest["artifacts"]["chunk_sha256"]
            == {path.name: sha256_path(path) for path in chunk_paths}
        ),
        "selected_counts_exact": selected_counts == config["samples"],
        "sample_ids_unique": len(key_by_id) == len(key_rows),
        "source_rows_unique": (
            len(selected_source_rows) == len(set(selected_source_rows))
        ),
        "source_rows_disjoint_from_exclusions": not (
            set(selected_source_rows) & excluded
        ),
        "all_rows_eligible": not ineligible_rows,
        "all_context_recomputed": not context_failures,
        "all_selectors_recomputed": not selector_failures,
        "all_placebo_contributions_recomputed": not placebo_failures,
        "blinded_exact_ids": set(blinded_by_id) == set(key_by_id),
        "blinded_context_matches": all(
            all(
                blinded_by_id[sample_id][field] == key_by_id[sample_id][field]
                for field in context_fields
            )
            for sample_id in key_by_id
        ),
        "blinded_labels_empty": all(
            not row["awareness_role"]
            and not row["functional_neutrality"]
            and not row["notes"]
            for row in blinded_rows
        ),
        "opaque_ids": all(re.fullmatch(r"rv_\d{4}", sample_id) for sample_id in key_by_id),
        "chunks_exact_ordered_partition": (
            chunk_ids == [row["sample_id"] for row in key_rows]
            and len(chunk_ids) == len(set(chunk_ids))
        ),
        "fixed_gate_operating_rule_is_one_failure": (
            max_awareness_failures == 1 and max_placebo_failures == 1
        ),
    }
    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "e1_sha256": sha256_path(args.e1_csv),
            "config_sha256": sha256_path(args.config),
            "membership_sha256": sha256_path(args.membership),
            "manifest_sha256": sha256_path(manifest_path),
            "prior_validation_files": prior_file_reports,
        },
        "counts": {
            "rows": len(key_rows),
            "selected_counts": selected_counts,
            "unique_source_rows": len(set(selected_source_rows)),
            "excluded_source_rows_union": len(excluded),
            "chunks": len(chunk_paths),
        },
        "gate_operating_characteristics": {
            "awareness_rows_per_stratum": awareness_n,
            "maximum_awareness_failures": max_awareness_failures,
            "placebo_rows_per_stratum": placebo_n,
            "maximum_placebo_failures": max_placebo_failures,
        },
        "failures": {
            "ineligible_rows": ineligible_rows,
            "context_failures": context_failures,
            "selector_failures": selector_failures,
            "placebo_failures": placebo_failures,
        },
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "assertions": assertions}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Recovery validation freeze audit failed")


if __name__ == "__main__":
    main()
