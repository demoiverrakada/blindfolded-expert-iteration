#!/usr/bin/env python3
"""Audit Qwen token lengths and awareness masks on the frozen E1 population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import (  # noqa: E402
    awareness_spans,
    has_one_well_formed_think_block,
    mask_labels_for_spans,
)
from bei.qwen_data import (  # noqa: E402
    LLAMA_ONLY_SELF_ID_RE,
    MODEL_SELF_ID_RE,
    SOURCE_MODEL_NAME_RE,
    adapt_model_names,
    row_identity,
)

EXPECTED_SOURCE_ROWS = 18_435
EXPECTED_CANDIDATES = 17_100
EXPECTED_MEMBERSHIP = 2_400
CAPS = (1_024, 1_536, 2_048)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[int | float]) -> dict:
    ordered = sorted(values)
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "p99": ordered[min(int(0.99 * len(ordered)), len(ordered) - 1)],
        "max": max(values),
    }


def shares(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    return {name: count / total for name, count in sorted(counts.items())}


def population_summary(records: list[dict], baseline_type_shares: dict) -> dict:
    type_counts = Counter(record["type"] for record in records)
    type_shares = shares(type_counts)
    context_shift_pp = {
        name: 100 * (type_shares.get(name, 0.0) - baseline_type_shares[name])
        for name in baseline_type_shares
    }
    supervised = sum(record["supervised_tokens"] for record in records)
    masked = sum(record["masked_tokens"] for record in records)
    post_mask_supervised = sum(
        record["post_mask_supervised_tokens"] for record in records
    )
    return {
        "rows": len(records),
        "type_counts": dict(sorted(type_counts.items())),
        "type_shares": type_shares,
        "context_share_shift_percentage_points": context_shift_pp,
        "max_abs_context_share_shift_percentage_points": max(
            map(abs, context_shift_pp.values()), default=0.0
        ),
        "supervised_tokens": supervised,
        "masked_tokens": masked,
        "post_mask_supervised_tokens": post_mask_supervised,
        "masked_share": masked / supervised if supervised else 0.0,
        "rows_with_mask": sum(record["masked_tokens"] > 0 for record in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument("membership_json", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_qwen_tokenizer_mask.json",
    )
    parser.add_argument(
        "--eligibility-output",
        type=Path,
        default=PROJECT_ROOT / "frozen/e1_length_eligibility.json",
    )
    parser.add_argument(
        "--substitution-sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/qwen_substitution_sample.jsonl",
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
        raise AssertionError("Fast tokenizer with offset mappings is required")

    source_sha256 = sha256_file(args.e1_csv)
    with args.e1_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    membership = json.loads(args.membership_json.read_text())
    if membership["source_csv_sha256"] != source_sha256:
        raise AssertionError("Frozen membership source hash mismatch")
    membership_by_index = {
        row["source_row_index"]: row for row in membership["rows"]
    }

    candidates = []
    exclusions = Counter()
    for source_row_index, row in enumerate(source_rows):
        if not has_one_well_formed_think_block(row["response"]):
            exclusions["malformed_think"] += 1
        elif MODEL_SELF_ID_RE.search(row["response"]):
            exclusions["model_self_identification"] += 1
        else:
            candidates.append((source_row_index, row))

    membership_hash_mismatches = []
    membership_not_candidate = []
    llama_only_self_id_survivors = [
        source_row_index
        for source_row_index, row in candidates
        if LLAMA_ONLY_SELF_ID_RE.search(row["response"])
    ]
    candidate_indices = {index for index, _ in candidates}
    for source_row_index, frozen in membership_by_index.items():
        if source_row_index not in candidate_indices:
            membership_not_candidate.append(source_row_index)
        elif row_identity(source_rows[source_row_index]) != frozen["row_sha256"]:
            membership_hash_mismatches.append(source_row_index)

    records = []
    prefix_mismatches = 0
    zero_token_spans = 0
    delimiter_failures = 0
    replacement_counts = Counter()
    changed_rows = Counter()
    replacement_risk_counts = Counter()
    replacement_examples = []
    span_reasons = Counter()

    for processed, (source_row_index, row) in enumerate(candidates, start=1):
        transformed = {}
        for field in ("system_prompt", "user_prompt", "response"):
            text = row[field]
            for match in SOURCE_MODEL_NAME_RE.finditer(text):
                following = text[match.end() : match.end() + 1]
                if following and (following in "-_/" or following.isdigit()):
                    replacement_risk_counts["followed_by_identifier_char"] += 1
                inside_fence = text[: match.start()].count("```") % 2 == 1
                replacement_risk_counts["inside_fenced_code"] += int(inside_fence)
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                replacement_examples.append(
                    {
                        "selection_key": hashlib.sha256(
                            f"{source_row_index}:{field}:{match.start()}".encode()
                        ).hexdigest(),
                        "source_row_index": source_row_index,
                        "field": field,
                        "matched_text": match.group(0),
                        "inside_fenced_code": inside_fence,
                        "context": text[start:end].replace("\r", " "),
                    }
                )
            transformed[field], replacements = adapt_model_names(row[field])
            replacement_counts[field] += replacements
            changed_rows[field] += int(replacements > 0)

        messages = [
            {"role": "system", "content": transformed["system_prompt"]},
            {"role": "user", "content": transformed["user_prompt"]},
        ]
        prefix_text = tokenizer.apply_chat_template(
            messages,
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
            prefix_mismatches += 1
            raise AssertionError(
                f"Prefix tokenization mismatch at source row {source_row_index}"
            )

        labels = list(input_ids)
        completion_start = len(prefix_ids)
        labels[:completion_start] = [-100] * completion_start
        spans = awareness_spans(response)
        absolute_spans = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in spans
        ]
        masked_labels, masked_indices = mask_labels_for_spans(
            labels,
            offsets,
            absolute_spans,
            minimum_token_index=completion_start,
        )
        newly_masked = [
            index
            for index in masked_indices
            if labels[index] != -100 and masked_labels[index] == -100
        ]
        for span, (span_start, span_end) in zip(spans, absolute_spans):
            span_reasons[span.reason] += 1
            if not any(
                token_start < span_end and token_end > span_start
                for token_start, token_end in offsets[completion_start:]
                if token_start != token_end
            ):
                zero_token_spans += 1
                raise AssertionError(
                    f"Awareness span has no tokens at source row {source_row_index}"
                )

        open_index = response.find("<think>")
        close_index = response.find("</think>")
        for start, end in (
            (open_index, open_index + len("<think>")),
            (close_index, close_index + len("</think>")),
        ):
            absolute_start = len(prefix_text) + start
            absolute_end = len(prefix_text) + end
            tag_tokens = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if index >= completion_start
                and token_start != token_end
                and token_start < absolute_end
                and token_end > absolute_start
            ]
            if not tag_tokens or any(masked_labels[index] == -100 for index in tag_tokens):
                delimiter_failures += 1
                raise AssertionError(
                    f"Think delimiter missing or masked at source row {source_row_index}"
                )

        supervised = sum(label != -100 for label in labels)
        post_mask_supervised = supervised - len(newly_masked)
        records.append(
            {
                "source_row_index": source_row_index,
                "type": row["type"],
                "sequence_tokens": len(input_ids),
                "prefix_tokens": completion_start,
                "supervised_tokens": supervised,
                "post_mask_supervised_tokens": post_mask_supervised,
                "masked_tokens": len(newly_masked),
                "awareness_spans": len(spans),
                "in_membership": source_row_index in membership_by_index,
            }
        )
        if processed % 2_500 == 0:
            print(f"tokenized E1 candidates: {processed:,}", flush=True)

    baseline_type_counts = Counter(record["type"] for record in records)
    baseline_type_shares = shares(baseline_type_counts)
    baseline = population_summary(records, baseline_type_shares)
    baseline_masked_share = baseline["masked_share"]
    membership_records = [record for record in records if record["in_membership"]]
    membership_type_shares = shares(
        Counter(record["type"] for record in membership_records)
    )
    membership_baseline = population_summary(
        membership_records, membership_type_shares
    )

    cap_reports = {}
    version_hashes = {
        "tokenizer_config": sha256_file(args.model_path / "tokenizer_config.json"),
        "chat_template": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "substitution_rule": sha256_file(PROJECT_ROOT / "src/bei/qwen_data.py"),
        "awareness_mask": sha256_file(PROJECT_ROOT / "src/bei/awareness_mask.py"),
        "detectors": sha256_file(PROJECT_ROOT / "src/bei/detectors.py"),
        "audit_script": sha256_file(Path(__file__)),
    }
    eligibility = {
        "source_csv_sha256": source_sha256,
        "membership_sha256": sha256_file(args.membership_json),
        "model": args.model_id,
        "revision": args.revision,
        "enable_thinking": True,
        "eos_token_id": tokenizer.eos_token_id,
        "version_hashes": version_hashes,
        "trainer_requirement": (
            "re-tokenize and hard-assert sequence_tokens <= selected cap; "
            "never truncate"
        ),
        "caps": {},
    }
    for cap in CAPS:
        retained = [record for record in records if record["sequence_tokens"] <= cap]
        retained_membership = [
            record
            for record in membership_records
            if record["sequence_tokens"] <= cap
        ]
        candidate_summary = population_summary(retained, baseline_type_shares)
        membership_summary = population_summary(
            retained_membership,
            membership_type_shares,
        )
        mask_shift_pp = 100 * (
            candidate_summary["masked_share"] - baseline_masked_share
        )
        candidate_summary.update(
            {
                "retention_fraction": len(retained) / len(records),
                "masked_share_shift_percentage_points": mask_shift_pp,
                "passes_retention": len(retained) / len(records) >= 0.80,
                "passes_mask_shift": abs(mask_shift_pp) <= 1.0,
                "passes_context_shift": (
                    candidate_summary[
                        "max_abs_context_share_shift_percentage_points"
                    ]
                    <= 2.0
                ),
            }
        )
        candidate_summary["passes_tokenizer_gate"] = all(
            candidate_summary[key]
            for key in (
                "passes_retention",
                "passes_mask_shift",
                "passes_context_shift",
            )
        )
        membership_summary["retention_fraction"] = (
            len(retained_membership) / len(membership_records)
        )
        membership_mask_shift_pp = 100 * (
            membership_summary["masked_share"]
            - membership_baseline["masked_share"]
        )
        membership_summary["masked_share_shift_percentage_points"] = (
            membership_mask_shift_pp
        )
        membership_summary["passes_retention"] = (
            membership_summary["retention_fraction"] >= 0.80
        )
        membership_summary["passes_mask_shift"] = (
            abs(membership_mask_shift_pp) <= 1.0
        )
        membership_summary["passes_context_shift"] = (
            membership_summary["max_abs_context_share_shift_percentage_points"]
            <= 2.0
        )
        membership_summary["passes_tokenizer_gate"] = all(
            membership_summary[key]
            for key in (
                "passes_retention",
                "passes_mask_shift",
                "passes_context_shift",
            )
        )
        reduced_steps = len(retained_membership) // 8
        training_examples = reduced_steps * 8
        membership_summary["reduced_optimizer_steps_batch8"] = reduced_steps
        membership_summary["training_examples_without_repetition"] = (
            training_examples
        )
        membership_summary["expected_completion_tokens_reduced_steps"] = (
            statistics.mean(
                record["supervised_tokens"] for record in retained_membership
            )
            * training_examples
            if retained_membership
            else 0
        )
        membership_summary["expected_post_mask_supervised_tokens_reduced_steps"] = (
            statistics.mean(
                record["post_mask_supervised_tokens"]
                for record in retained_membership
            )
            * training_examples
            if retained_membership
            else 0
        )
        cap_reports[str(cap)] = {
            "candidate_population": candidate_summary,
            "frozen_membership": membership_summary,
        }
        eligibility["caps"][str(cap)] = {
            "eligible_membership_rows": [
                {
                    "source_row_index": record["source_row_index"],
                    "row_sha256": membership_by_index[record["source_row_index"]][
                        "row_sha256"
                    ],
                    "sequence_tokens": record["sequence_tokens"],
                    "prefix_tokens": record["prefix_tokens"],
                }
                for record in retained_membership
            ]
        }

    passing_caps = [
        cap
        for cap in CAPS
        if cap_reports[str(cap)]["candidate_population"]["passes_tokenizer_gate"]
    ]
    ordered_replacements = sorted(
        replacement_examples, key=lambda row: row["selection_key"]
    )
    risk_examples = [
        row
        for row in ordered_replacements
        if row["inside_fenced_code"]
    ]
    risk_keys = {row["selection_key"] for row in risk_examples}
    substitution_sample_rows = (
        risk_examples
        + [row for row in ordered_replacements if row["selection_key"] not in risk_keys]
    )[:50]
    substitution_sample_bytes = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode()
        for row in substitution_sample_rows
    )
    args.substitution_sample.parent.mkdir(parents=True, exist_ok=True)
    args.substitution_sample.write_bytes(substitution_sample_bytes)
    substitution_sample_sha256 = hashlib.sha256(
        substitution_sample_bytes
    ).hexdigest()
    assertions = {
        "source_rows_18435": len(source_rows) == EXPECTED_SOURCE_ROWS,
        "candidate_rows_17100": len(records) == EXPECTED_CANDIDATES,
        "membership_rows_2400": len(membership_by_index) == EXPECTED_MEMBERSHIP,
        "membership_rows_all_candidates": not membership_not_candidate,
        "membership_hashes_match": not membership_hash_mismatches,
        "prefix_tokenization_exact": prefix_mismatches == 0,
        "all_awareness_spans_overlap_tokens": zero_token_spans == 0,
        "think_delimiters_supervised": delimiter_failures == 0,
        "substitution_sample_50": len(substitution_sample_rows) == 50,
        "no_identifier_suffix_corruption": (
            replacement_risk_counts["followed_by_identifier_char"] == 0
        ),
        "no_llama_only_self_id_survivors": not llama_only_self_id_survivors,
        "at_least_one_cap_passes_tokenizer_gate": bool(passing_caps),
    }
    eligibility_bytes = (
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n"
    ).encode()
    args.eligibility_output.parent.mkdir(parents=True, exist_ok=True)
    args.eligibility_output.write_bytes(eligibility_bytes)
    eligibility_sha256 = hashlib.sha256(eligibility_bytes).hexdigest()
    if sha256_file(args.eligibility_output) != eligibility_sha256:
        raise AssertionError("Eligibility artifact rehash mismatch")

    report = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "inputs": {
            "e1_csv": {
                "path": str(args.e1_csv),
                "sha256": source_sha256,
                "rows": len(source_rows),
            },
            "membership": {
                "path": str(args.membership_json),
                "sha256": sha256_file(args.membership_json),
                "rows": len(membership_by_index),
            },
        },
        "tokenizer": {
            "model": args.model_id,
            "revision": args.revision,
            "local_path": str(args.model_path),
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "chat_template_enable_thinking": True,
        },
        "candidate_filter": {
            "included": len(records),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "model_name_adaptation": {
            "replacement_target": "Qwen3 assistant",
            "replacement_counts": dict(replacement_counts),
            "changed_rows": dict(changed_rows),
            "risk_counts": dict(replacement_risk_counts),
            "sample": {
                "path": str(args.substitution_sample),
                "rows": len(substitution_sample_rows),
                "sha256": substitution_sample_sha256,
            },
        },
        "correctness": {
            "prefix_tokenization_mismatches": prefix_mismatches,
            "awareness_spans_with_zero_tokens": zero_token_spans,
            "think_delimiter_failures": delimiter_failures,
            "membership_hash_mismatches": membership_hash_mismatches,
            "membership_not_candidate": membership_not_candidate,
            "llama_only_self_id_survivors": llama_only_self_id_survivors,
        },
        "all_candidates": {
            **baseline,
            "sequence_tokens": summarize(
                [record["sequence_tokens"] for record in records]
            ),
            "prefix_tokens": summarize(
                [record["prefix_tokens"] for record in records]
            ),
            "supervised_tokens_per_row": summarize(
                [record["supervised_tokens"] for record in records]
            ),
            "masked_tokens_per_row": summarize(
                [record["masked_tokens"] for record in records]
            ),
            "awareness_spans_per_row": summarize(
                [record["awareness_spans"] for record in records]
            ),
            "span_reasons": dict(sorted(span_reasons.items())),
        },
        "frozen_membership": {
            **membership_baseline,
            "sequence_tokens": summarize(
                [record["sequence_tokens"] for record in membership_records]
            ),
        },
        "caps": cap_reports,
        "smallest_cap_passing_tokenizer_gate": min(passing_caps)
        if passing_caps
        else None,
        "eligibility_artifact": {
            "path": str(args.eligibility_output),
            "sha256": eligibility_sha256,
            "bytes": len(eligibility_bytes),
        },
        "assertions": assertions,
        "runtime": {
            "python": sys.version,
            "transformers": __import__("transformers").__version__,
            "command": " ".join(sys.argv),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "smallest_cap_passing_tokenizer_gate": report[
                    "smallest_cap_passing_tokenizer_gate"
                ],
                "eligibility_sha256": eligibility_sha256,
            },
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit("Qwen tokenizer/mask gate failed")


if __name__ == "__main__":
    main()
