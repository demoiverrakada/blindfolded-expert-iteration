#!/usr/bin/env python3
"""Audit exact-count within-row placebo capacity for the recovery protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.matched_mask import (  # noqa: E402
    broad_awareness_spans,
    core_awareness_spans,
    distant_position_control_indices,
    nearest_position_control_indices,
    neutral_control_spans,
    token_indices_for_spans,
    type_hint_planning_spans,
)
from bei.detectors import detector_patterns  # noqa: E402
from bei.qwen_data import adapt_model_names, row_identity  # noqa: E402

CAPS = (1_536, 2_048)
DISTANT_MINIMUM_TOKENS = 200
V2_PATTERN_9 = detector_patterns("v2")[9]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[int | float]) -> dict:
    if not values:
        return {"total": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "max": max(values),
    }


def shares(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {
        key: value / total for key, value in sorted(counter.items())
    } if total else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument(
        "--membership",
        type=Path,
        default=PROJECT_ROOT / "frozen/e1_training_row_ids.json",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_recovery_core_mask_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv.field_size_limit(sys.maxsize)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise AssertionError("Fast tokenizer with offsets is required")

    source_rows = list(
        csv.DictReader(args.e1_csv.open(newline="", encoding="utf-8"))
    )
    membership = json.loads(args.membership.read_text())
    membership_rows = {
        row["source_row_index"]: row for row in membership["rows"]
    }
    if membership["source_csv_sha256"] != sha256_path(args.e1_csv):
        raise AssertionError("Membership source hash mismatch")

    records = []
    hash_mismatches = []
    zero_token_target_spans = []
    for processed, source_row_index in enumerate(sorted(membership_rows), start=1):
        row = source_rows[source_row_index]
        if row_identity(row) != membership_rows[source_row_index]["row_sha256"]:
            hash_mismatches.append(source_row_index)
            continue

        transformed = {
            field: adapt_model_names(row[field])[0]
            for field in ("system_prompt", "user_prompt", "response")
        }
        prefix_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": transformed["system_prompt"]},
                {"role": "user", "content": transformed["user_prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        response = transformed["response"]
        full_text = prefix_text + response
        if not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
        prefix_ids = tokenizer(
            prefix_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if input_ids[: len(prefix_ids)] != prefix_ids:
            raise AssertionError(f"Prefix mismatch at row {source_row_index}")
        completion_start = len(prefix_ids)

        core_spans = core_awareness_spans(response)
        broad_spans = broad_awareness_spans(response)
        positive_control_spans = type_hint_planning_spans(response)
        neutral_spans = neutral_control_spans(response)
        absolute_core = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in core_spans
        ]
        absolute_broad = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in broad_spans
        ]
        absolute_positive_control = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in positive_control_spans
        ]
        absolute_neutral = [
            (len(prefix_text) + start, len(prefix_text) + end)
            for start, end, _text in neutral_spans
        ]
        core_indices = token_indices_for_spans(
            offsets,
            absolute_core,
            minimum_token_index=completion_start,
        )
        broad_indices = token_indices_for_spans(
            offsets,
            absolute_broad,
            minimum_token_index=completion_start,
        )
        positive_control_indices = token_indices_for_spans(
            offsets,
            absolute_positive_control,
            minimum_token_index=completion_start,
        )
        neutral_indices = token_indices_for_spans(
            offsets,
            absolute_neutral,
            minimum_token_index=completion_start,
        )
        if set(core_indices) & set(neutral_indices):
            raise AssertionError(f"Core/control overlap at row {source_row_index}")
        if set(broad_indices) & set(neutral_indices):
            raise AssertionError(f"Broad/control overlap at row {source_row_index}")
        if set(positive_control_indices) & set(neutral_indices):
            raise AssertionError(
                f"Positive-control/control overlap at row {source_row_index}"
            )
        if not set(core_indices).issubset(broad_indices):
            raise AssertionError(f"Core is not a broad-mask subset at row {source_row_index}")
        for dose_name, target_spans in (
            ("core", absolute_core),
            ("broad", absolute_broad),
            ("type_hint_positive", absolute_positive_control),
        ):
            for span_start, span_end in target_spans:
                if not any(
                    token_start < span_end and token_end > span_start
                    for token_start, token_end in offsets[completion_start:]
                    if token_start != token_end
                ):
                    zero_token_target_spans.append(
                        {
                            "source_row_index": source_row_index,
                            "dose": dose_name,
                        }
                    )

        supervised_tokens = len(input_ids) - completion_start
        records.append(
            {
                "source_row_index": source_row_index,
                "type": row["type"],
                "sequence_tokens": len(input_ids),
                "supervised_tokens": supervised_tokens,
                "core_tokens": core_indices,
                "broad_tokens": broad_indices,
                "type_hint_positive_tokens": positive_control_indices,
                "neutral_tokens": neutral_indices,
                "core_span_reasons": [span.reason for span in core_spans],
                "broad_span_reasons": [span.reason for span in broad_spans],
                "type_hint_positive_span_reasons": [
                    span.reason for span in positive_control_spans
                ],
                "core_spans_overlapping_v2_pattern_9": sum(
                    bool(V2_PATTERN_9.search(span.text)) for span in core_spans
                ),
            }
        )
        if processed % 500 == 0:
            print(f"audited membership rows: {processed:,}", flush=True)

    cap_reports = {}
    for cap in CAPS:
        eligible = [record for record in records if record["sequence_tokens"] <= cap]
        baseline_types = Counter(record["type"] for record in eligible)
        dose_specs = (
            ("core", "core_tokens", "core_span_reasons"),
            ("broad", "broad_tokens", "broad_span_reasons"),
            (
                "type_hint_positive",
                "type_hint_positive_tokens",
                "type_hint_positive_span_reasons",
            ),
        )
        prepared_by_dose = {}
        individual_capacity = {}
        for dose_name, token_key, _reason_key in dose_specs:
            prepared = {}
            unpairable = []
            for record in eligible:
                target = record[token_key]
                neutral = record["neutral_tokens"]
                if len(neutral) < len(target):
                    unpairable.append(record["source_row_index"])
                    continue
                try:
                    near = nearest_position_control_indices(target, neutral)
                    distant = distant_position_control_indices(
                        target,
                        neutral,
                        minimum_distance=DISTANT_MINIMUM_TOKENS,
                        seed=(
                            f"20260911:{cap}:{dose_name}:"
                            f"{record['source_row_index']}"
                        ),
                    )
                except ValueError:
                    unpairable.append(record["source_row_index"])
                    continue
                prepared[record["source_row_index"]] = {
                    "near": near,
                    "distant": distant,
                }
            prepared_by_dose[dose_name] = prepared
            individual_capacity[dose_name] = {
                "pairable_rows": len(prepared),
                "pairable_fraction": len(prepared) / len(eligible),
                "unpairable_rows": len(unpairable),
                "unpairable_source_row_indices": unpairable,
            }

        common_ids = set.intersection(
            *(set(prepared) for prepared in prepared_by_dose.values())
        )
        common_records = [
            record for record in eligible if record["source_row_index"] in common_ids
        ]
        common_types = Counter(record["type"] for record in common_records)
        baseline_shares = shares(baseline_types)
        common_shares = shares(common_types)
        common_context_shift_pp = {
            context: 100 * (
                common_shares.get(context, 0.0)
                - baseline_shares.get(context, 0.0)
            )
            for context in baseline_shares
        }
        dose_reports = {}
        for dose_name, token_key, reason_key in dose_specs:
            near_distances = []
            distant_distances = []
            exact_count_failures = []
            overlap_failures = []
            near_placebo_tokens = 0
            distant_placebo_tokens = 0
            for record in common_records:
                target = record[token_key]
                prepared = prepared_by_dose[dose_name][record["source_row_index"]]
                near = prepared["near"]
                distant = prepared["distant"]
                near_placebo_tokens += len(near)
                distant_placebo_tokens += len(distant)
                if len(near) != len(target) or len(distant) != len(target):
                    exact_count_failures.append(record["source_row_index"])
                if (
                    set(near) & set(target)
                    or set(distant) & set(target)
                ):
                    overlap_failures.append(record["source_row_index"])
                near_distances.extend(
                    abs(target_index - control_index)
                    for target_index, control_index in zip(
                        sorted(target), sorted(near)
                    )
                )
                distant_distances.extend(
                    abs(target_index - control_index)
                    for target_index, control_index in zip(
                        sorted(target), sorted(distant)
                    )
                )

            reasons = Counter(
                reason for record in common_records for reason in record[reason_key]
            )
            supervised = sum(
                record["supervised_tokens"] for record in common_records
            )
            target_tokens = sum(
                len(record[token_key]) for record in common_records
            )
            dose_reports[dose_name] = {
                "common_rows": len(common_records),
                "rows_with_mask": sum(
                    bool(record[token_key]) for record in common_records
                ),
                "zero_mask_row_fraction": sum(
                    not record[token_key] for record in common_records
                )
                / len(common_records),
                "supervised_tokens": supervised,
                "masked_tokens": target_tokens,
                "near_placebo_masked_tokens_actual": near_placebo_tokens,
                "distant_placebo_masked_tokens_actual": distant_placebo_tokens,
                "masked_share": target_tokens / supervised,
                "tokens_per_row": summarize(
                    [len(record[token_key]) for record in common_records]
                ),
                "neutral_capacity_per_row": summarize(
                    [len(record["neutral_tokens"]) for record in common_records]
                ),
                "near_token_distance": summarize(near_distances),
                "distant_token_distance": summarize(distant_distances),
                "span_reasons": dict(sorted(reasons.items())),
                "core_spans_overlapping_v2_pattern_9": (
                    sum(
                        record["core_spans_overlapping_v2_pattern_9"]
                        for record in common_records
                    )
                    if dose_name == "core"
                    else None
                ),
                "exact_count_failures": exact_count_failures,
                "overlap_failures": overlap_failures,
            }
        cap_reports[str(cap)] = {
            "eligible_rows": len(eligible),
            "over_cap_rows_excluded_not_truncated": len(records) - len(eligible),
            "common_rows_all_arms": len(common_records),
            "common_row_fraction": len(common_records) / len(eligible),
            "common_excluded_source_row_indices": sorted(
                set(record["source_row_index"] for record in eligible) - common_ids
            ),
            "common_type_counts": dict(sorted(common_types.items())),
            "common_type_share_shift_percentage_points": common_context_shift_pp,
            "common_max_abs_type_share_shift_percentage_points": max(
                map(abs, common_context_shift_pp.values()), default=0.0
            ),
            "individual_capacity": individual_capacity,
            "doses": dose_reports,
        }

    assertions = {
        "membership_rows_2400": len(membership_rows) == 2_400,
        "all_membership_hashes_match": not hash_mismatches,
        "all_target_spans_overlap_tokens": not zero_token_target_spans,
        "every_cap_exact_count": all(
            not dose["exact_count_failures"]
            for report in cap_reports.values()
            for dose in report["doses"].values()
        ),
        "every_cap_masks_disjoint": all(
            not dose["overlap_failures"]
            for report in cap_reports.values()
            for dose in report["doses"].values()
        ),
    }
    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "protocol_status": "EXPLORATORY_CAPACITY_AUDIT_NOT_FROZEN",
        "inputs": {
            "e1_csv_sha256": sha256_path(args.e1_csv),
            "membership_sha256": sha256_path(args.membership),
            "model_path": str(args.model_path),
            "revision": args.revision,
            "tokenizer_class": tokenizer.__class__.__name__,
            "matched_mask_sha256": sha256_path(
                PROJECT_ROOT / "src/bei/matched_mask.py"
            ),
            "script_sha256": sha256_path(Path(__file__).resolve()),
        },
        "design": {
            "core": "V1 detector union literal Wood Labs, reasoning sentences only",
            "broad": "Frozen V3 detector union literal Wood Labs",
            "type_hint_positive": (
                "reasoning sentences that explicitly plan Python type annotations"
            ),
            "placebo": (
                "same-row exact token count chosen from conservative neutral "
                "reasoning sentences; nearest-position and deterministically "
                "sampled >=200-token-distant diagnostics"
            ),
            "common_row_policy": (
                "union exclusion across core, broad, and type-hint-positive "
                "capacity; every arm uses the same rows"
            ),
            "caps": list(CAPS),
        },
        "caps": cap_reports,
        "diagnostics": {
            "hash_mismatches": hash_mismatches,
            "zero_token_target_spans": zero_token_target_spans,
        },
        "assertions": assertions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "caps": {
                    cap: {
                        dose: {
                            "masked_share": dose_values["masked_share"],
                            "rows_with_mask": dose_values["rows_with_mask"],
                        }
                        for dose, dose_values in values["doses"].items()
                    }
                    for cap, values in cap_reports.items()
                },
            },
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("Core/placebo capacity audit failed")


if __name__ == "__main__":
    main()
