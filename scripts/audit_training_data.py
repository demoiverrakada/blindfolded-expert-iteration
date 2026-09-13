#!/usr/bin/env python3
"""Audit Hua-Qin round-one expert-iteration targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.detectors import (  # noqa: E402
    WOOD_LABS_RE,
    explicit_eval_evidence,
    explicit_eval_match,
)

MODEL_SELF_ID_RE = re.compile(
    r"\b(?:i am|i'm|as)\s+(?:the\s+)?(?:llama\s+)?nemotron\b", re.IGNORECASE
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_answer_only(response: str) -> tuple[str | None, str]:
    """Conservatively extract the suffix after one complete think block."""

    open_count = response.count("<think>")
    close_count = response.count("</think>")
    if open_count == 0 and close_count == 0:
        return None, "no_think_tags"
    if open_count == 1 and close_count == 0:
        return None, "unterminated_think"
    if open_count == 0 and close_count == 1:
        return None, "closing_tag_without_open"
    if open_count != 1 or close_count != 1:
        return None, "multiple_think_tags"

    open_index = response.find("<think>")
    close_index = response.find("</think>")
    if open_index > close_index:
        return None, "misordered_think_pair"
    if response[:open_index].strip():
        return None, "text_before_think"

    answer = response[close_index + len("</think>") :].strip()
    if not answer:
        return None, "empty_answer"
    fence_count = answer.count("```")
    if fence_count < 2:
        return None, "answer_without_closed_code_fence"
    if fence_count % 2:
        return None, "answer_with_unclosed_code_fence"
    return answer, "ok"


def split_response(response: str) -> tuple[str, str]:
    """Best-effort split used only for leakage measurement."""

    open_index = response.find("<think>")
    close_index = response.find("</think>")
    if open_index != -1 and close_index != -1 and open_index < close_index:
        return (
            response[open_index + len("<think>") : close_index],
            response[close_index + len("</think>") :],
        )
    if open_index != -1:
        return response[open_index + len("<think>") :], ""
    return "", response


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(probability * len(ordered)), len(ordered) - 1)]


def segment_signals(text: str) -> dict[str, bool]:
    return {
        "mentions_wood_labs": bool(WOOD_LABS_RE.search(text)),
        "explicit_eval_language_v1": explicit_eval_match(text, "v1"),
        "explicit_eval_language_v2": explicit_eval_match(text, "v2"),
    }


def audit_rows(rows: Iterable[dict[str, str]]) -> tuple[dict, list[dict[str, str]]]:
    rows = list(rows)
    counts = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    response_lengths: list[int] = []
    prompt_pairs: set[tuple[str, str]] = set()
    answer_only_rows: list[dict[str, str]] = []
    extraction_status = Counter()
    segment_counts: dict[str, Counter] = {
        "full_response": Counter(),
        "thinking": Counter(),
        "answer_suffix": Counter(),
        "included_answer_only": Counter(),
    }

    for row_index, row in enumerate(rows):
        response = row["response"]
        row_type = row["type"]
        response_lengths.append(len(response))
        prompt_pairs.add((row["system_prompt"], row["user_prompt"]))
        by_type[row_type]["rows"] += 1

        full_signals = {
            **segment_signals(response),
            "model_self_identification": bool(MODEL_SELF_ID_RE.search(response)),
        }
        for name, value in full_signals.items():
            if value:
                counts[name] += 1
                by_type[row_type][name] += 1

        thought, answer_suffix = split_response(response)
        for segment_name, segment in (
            ("full_response", response),
            ("thinking", thought),
            ("answer_suffix", answer_suffix),
        ):
            for name, value in segment_signals(segment).items():
                if value:
                    segment_counts[segment_name][name] += 1

        answer, status = extract_answer_only(response)
        extraction_status[status] += 1
        if answer is not None:
            transformed = dict(row)
            transformed["response"] = answer
            if "source_row_index" in transformed:
                raise ValueError("Input already contains source_row_index")
            transformed["source_row_index"] = str(row_index)
            answer_only_rows.append(transformed)
            for name, value in segment_signals(answer).items():
                if value:
                    segment_counts["included_answer_only"][name] += 1

    total = len(rows)
    signal_names = (
        "mentions_wood_labs",
        "explicit_eval_language_v1",
        "explicit_eval_language_v2",
    )
    report = {
        "rows": total,
        "unique_prompt_pairs": len(prompt_pairs),
        "duplicate_prompt_pairs": total - len(prompt_pairs),
        "response_characters": {
            "median": statistics.median(response_lengths),
            "mean": round(statistics.mean(response_lengths), 2),
            "p95": percentile(response_lengths, 0.95),
            "max": max(response_lengths),
        },
        "detectors": {
            name: {"count": counts[name], "fraction": counts[name] / total}
            for name in (*signal_names, "model_self_identification")
        },
        "segments": {
            segment_name: {
                "denominator": (
                    len(answer_only_rows)
                    if segment_name == "included_answer_only"
                    else total
                ),
                **{
                    signal: {
                        "count": segment_counter[signal],
                        "fraction": segment_counter[signal]
                        / (
                            len(answer_only_rows)
                            if segment_name == "included_answer_only"
                            else total
                        ),
                    }
                    for signal in signal_names
                },
            }
            for segment_name, segment_counter in segment_counts.items()
        },
        "answer_only_extraction": {
            "included": len(answer_only_rows),
            "excluded": total - len(answer_only_rows),
            "included_fraction": len(answer_only_rows) / total,
            "status_counts": dict(sorted(extraction_status.items())),
        },
        "by_type": {
            row_type: dict(sorted(type_counts.items()))
            for row_type, type_counts in sorted(by_type.items())
        },
    }
    return report, answer_only_rows


def write_answer_only(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [name for name in rows[0] if name != "source_row_index"]
    fieldnames.append("source_row_index")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_sample(
    path: Path,
    rows: list[dict[str, str]],
    per_group: int,
    seed: int,
    detector_version: str,
) -> None:
    groups = {True: [], False: []}
    for index, row in enumerate(rows):
        groups[explicit_eval_match(row["response"], detector_version)].append(
            (index, row)
        )

    rng = random.Random(seed)
    selected = []
    for detected, candidates in groups.items():
        for index, row in rng.sample(candidates, min(per_group, len(candidates))):
            selected.append(
                {
                    "source_row_index": index,
                    "detector_version": detector_version,
                    "detector_hit": int(detected),
                    "detector_evidence": explicit_eval_evidence(
                        row["response"], detector_version
                    ),
                    "type": row["type"],
                    "response": row["response"].replace("\r", " "),
                    "human_label": "",
                    "notes": "",
                }
            )
    rng.shuffle(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--answer-only-csv", type=Path)
    parser.add_argument("--validation-sample", type=Path)
    parser.add_argument("--sample-per-group", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20260910)
    parser.add_argument("--sample-detector-version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    report, answer_only_rows = audit_rows(rows)
    report["input"] = {
        "path": str(args.input_csv),
        "sha256": sha256_file(args.input_csv),
    }
    report["runtime"] = {
        "python_version": __import__("sys").version,
        "command": " ".join(__import__("sys").argv),
    }

    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n")
    if args.answer_only_csv:
        write_answer_only(args.answer_only_csv, answer_only_rows)
    if args.validation_sample:
        write_validation_sample(
            args.validation_sample,
            rows,
            per_group=args.sample_per_group,
            seed=args.sample_seed,
            detector_version=args.sample_detector_version,
        )


if __name__ == "__main__":
    main()
