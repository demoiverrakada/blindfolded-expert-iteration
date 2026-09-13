#!/usr/bin/env python3
"""Audit the exact sentence-level loss mask used by the training design."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    reasoning_sentences,
)
from bei.detectors import matches_awareness_union  # noqa: E402


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "max": max(values),
    }


def reservoir_add(
    reservoirs: dict[bool, list[dict]],
    seen: Counter,
    group: bool,
    item: dict,
    size: int,
    rng: random.Random,
) -> None:
    seen[group] += 1
    reservoir = reservoirs[group]
    if len(reservoir) < size:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen[group])
    if replacement < size:
        reservoir[replacement] = item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=3500)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validation-sample", type=Path)
    parser.add_argument("--sample-per-group", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=20260912)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    rows = list(csv.DictReader(args.input_csv.open(newline="")))
    rng = random.Random(args.sample_seed)
    reservoirs = {True: [], False: []}
    seen_sentences = Counter()
    structure_counts = Counter()
    reason_counts = Counter()
    per_type: dict[str, Counter] = defaultdict(Counter)
    supervised_counts: list[int] = []
    masked_counts: list[int] = []
    masked_fractions: list[float] = []
    span_counts: list[int] = []
    zero_token_spans = 0
    partially_truncated_spans = 0
    truncated_rows = 0
    prefix_mismatches = 0

    for source_row_index, row in enumerate(rows):
        row_type = row["type"]
        per_type[row_type]["source_rows"] += 1
        if not has_one_well_formed_think_block(row["response"]):
            structure_counts["excluded_malformed_think"] += 1
            per_type[row_type]["excluded_malformed_think"] += 1
            continue
        structure_counts["included"] += 1
        per_type[row_type]["included"] += 1

        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prefix_text + row["response"]
        if not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False, truncation=False
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        nonempty_offsets = [
            (start, end)
            for start, end in encoded["offset_mapping"]
            if start != end
        ]
        if nonempty_offsets and nonempty_offsets[-1][1] < len(full_text):
            truncated_rows += 1
        if encoded["input_ids"][: len(prefix_ids)] != prefix_ids:
            prefix_mismatches += 1
            raise AssertionError(
                f"Independent prefix tokenization mismatch at row {source_row_index}"
            )
        labels = list(encoded["input_ids"])
        completion_start = len(prefix_ids)
        labels[:completion_start] = [-100] * completion_start

        spans = awareness_spans(row["response"])
        absolute_spans = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in spans
        ]
        masked_labels, masked_indices = mask_labels_for_spans(
            labels,
            encoded["offset_mapping"],
            absolute_spans,
            minimum_token_index=completion_start,
        )
        newly_masked = [
            index
            for index in masked_indices
            if labels[index] != -100 and masked_labels[index] == -100
        ]
        for span_start, span_end in absolute_spans:
            overlapping_offsets = [
                (token_start, token_end)
                for token_start, token_end in encoded["offset_mapping"][
                    completion_start:
                ]
                if (
                    token_start != token_end
                    and token_start < span_end
                    and token_end > span_start
                )
            ]
            overlaps = bool(overlapping_offsets)
            if overlaps and max(end for _start, end in overlapping_offsets) < span_end:
                partially_truncated_spans += 1
            if not overlaps:
                zero_token_spans += 1
        supervised = sum(label != -100 for label in labels)
        masked = len(newly_masked)
        supervised_counts.append(supervised)
        masked_counts.append(masked)
        masked_fractions.append(masked / supervised if supervised else 0.0)
        span_counts.append(len(spans))
        per_type[row_type]["supervised_tokens"] += supervised
        per_type[row_type]["masked_tokens"] += masked
        per_type[row_type]["rows_with_mask"] += int(masked > 0)
        for span in spans:
            reason_counts[span.reason] += 1

        for start, end, sentence in reasoning_sentences(row["response"]):
            hit, reason = matches_awareness_union(sentence)
            context_start = max(0, start - 240)
            context_end = min(len(row["response"]), end + 240)
            reservoir_add(
                reservoirs,
                seen_sentences,
                hit,
                {
                    "source_row_index": source_row_index,
                    "mask_hit": int(hit),
                    "mask_reason": reason,
                    "type": row_type,
                    "system_prompt": row["system_prompt"],
                    "user_prompt": row["user_prompt"],
                    "sentence": sentence.replace("\r", " "),
                    "sentence_context": row["response"][
                        context_start:context_end
                    ].replace("\r", " "),
                    "inclusive_label": "",
                    "strict_label": "",
                    "notes": "",
                },
                args.sample_per_group,
                rng,
            )

    included = structure_counts["included"]
    total_supervised = sum(supervised_counts)
    total_masked = sum(masked_counts)
    report = {
        "runtime": {
            "python_version": sys.version,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": __import__("torch").__version__,
            "command": " ".join(sys.argv),
        },
        "tokenizer": {
            "model": args.model,
            "revision": args.revision,
            "class": tokenizer.__class__.__name__,
        },
        "source_rows": len(rows),
        "included_rows": included,
        "structure_counts": dict(structure_counts),
        "max_length": args.max_length,
        "prefix_tokenization_mismatches": prefix_mismatches,
        "truncated_rows": truncated_rows,
        "spans_with_zero_overlapping_tokens": zero_token_spans,
        "partially_truncated_spans": partially_truncated_spans,
        "rows_with_masked_tokens": sum(count > 0 for count in masked_counts),
        "row_mask_rate": sum(count > 0 for count in masked_counts) / included,
        "total_supervised_completion_tokens": total_supervised,
        "total_masked_completion_tokens": total_masked,
        "masked_share_of_supervised_tokens": total_masked / total_supervised,
        "supervised_tokens_per_row": summarize(supervised_counts),
        "masked_tokens_per_row": summarize(masked_counts),
        "masked_fraction_per_row": summarize(masked_fractions),
        "awareness_spans_per_row": summarize(span_counts),
        "span_reasons": dict(sorted(reason_counts.items())),
        "sentence_population": {
            "masked": seen_sentences[True],
            "unmasked": seen_sentences[False],
        },
        "by_type": {
            row_type: {
                **dict(counts),
                "masked_share": (
                    counts["masked_tokens"] / counts["supervised_tokens"]
                    if counts["supervised_tokens"]
                    else 0.0
                ),
            }
            for row_type, counts in sorted(per_type.items())
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.report:
        args.report.write_text(serialized + "\n")
    if args.validation_sample:
        selected = reservoirs[True] + reservoirs[False]
        rng.shuffle(selected)
        with args.validation_sample.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
            writer.writeheader()
            writer.writerows(selected)


if __name__ == "__main__":
    main()
