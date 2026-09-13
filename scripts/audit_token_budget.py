#!/usr/bin/env python3
"""Measure exact tokenizer-level budgets for matched target variants."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

from audit_training_data import extract_answer_only


def summarize(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=3500)
    parser.add_argument("--optimizer-steps", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )

    variants = {
        "full_trace": {
            "sequence_tokens": [],
            "target_tokens_before_truncation": [],
            "supervised_tokens_after_truncation": [],
            "truncated_rows": 0,
        },
        "answer_only": {
            "sequence_tokens": [],
            "target_tokens_before_truncation": [],
            "supervised_tokens_after_truncation": [],
            "truncated_rows": 0,
        },
    }
    prefix_tokens = []
    matched_rows = 0

    with args.input_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            answer, status = extract_answer_only(row["response"])
            if status != "ok":
                continue
            matched_rows += 1
            messages = [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ]
            prefix_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = tokenizer(
                prefix_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            prefix_tokens.append(len(prefix_ids))

            for variant_name, completion in (
                ("full_trace", row["response"]),
                ("answer_only", answer),
            ):
                full_text = prefix_text + completion
                if not full_text.endswith(tokenizer.eos_token):
                    full_text += tokenizer.eos_token
                full_ids = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                sequence_length = len(full_ids)
                target_before = max(0, sequence_length - len(prefix_ids))
                supervised_after = max(
                    0,
                    min(sequence_length, args.max_length)
                    - min(len(prefix_ids), args.max_length),
                )
                bucket = variants[variant_name]
                bucket["sequence_tokens"].append(sequence_length)
                bucket["target_tokens_before_truncation"].append(target_before)
                bucket["supervised_tokens_after_truncation"].append(supervised_after)
                bucket["truncated_rows"] += int(sequence_length > args.max_length)

    examples_at_budget = args.optimizer_steps * args.global_batch_size
    report = {
        "runtime": {
            "python_version": sys.version,
            "command": " ".join(sys.argv),
            "transformers_version": __import__("transformers").__version__,
        },
        "tokenizer": {
            "model": args.model,
            "revision": args.revision,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "matched_rows": matched_rows,
        "max_length": args.max_length,
        "optimizer_steps": args.optimizer_steps,
        "global_batch_size": args.global_batch_size,
        "examples_at_budget": examples_at_budget,
        "prefix_tokens": summarize(prefix_tokens),
        "variants": {},
    }
    for name, bucket in variants.items():
        report["variants"][name] = {
            "sequence_tokens": summarize(bucket["sequence_tokens"]),
            "target_tokens_before_truncation": summarize(
                bucket["target_tokens_before_truncation"]
            ),
            "supervised_tokens_after_truncation": summarize(
                bucket["supervised_tokens_after_truncation"]
            ),
            "truncated_rows": bucket["truncated_rows"],
            "truncated_fraction": bucket["truncated_rows"] / matched_rows,
            "expected_supervised_tokens_at_step_budget": (
                statistics.mean(bucket["supervised_tokens_after_truncation"])
                * examples_at_budget
            ),
        }

    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()

