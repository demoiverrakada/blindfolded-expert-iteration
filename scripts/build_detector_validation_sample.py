#!/usr/bin/env python3
"""Build blinded, deterministic detector-validation samples."""

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

from bei.awareness_mask import has_one_well_formed_think_block, reasoning_sentences
from bei.detectors import WOOD_LABS_RE, detector_patterns, matches_awareness_union
from bei.qwen_data import MODEL_SELF_ID_RE, adapt_model_names


def order_key(sample_id: str, seed: str) -> str:
    return hashlib.sha256(f"{sample_id}:{seed}".encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_validation.json",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_validation_sample.csv",
    )
    parser.add_argument(
        "--blinded",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_validation_blinded.csv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_validation_annotations.csv",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_validation_chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    cues = [re.compile(pattern, re.I) for pattern in config["independent_cue_lexicon"]]
    patterns = detector_patterns("v2")
    pools = {
        "recall_enriched_negative": [],
        "recall_detector_positive": [],
        "v2_pattern_9": [],
        "v2_pattern_12": [],
    }
    with args.e1_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for source_row_index, row in enumerate(rows):
        response = row["response"]
        if (
            not has_one_well_formed_think_block(response)
            or MODEL_SELF_ID_RE.search(response)
        ):
            continue
        transformed, _ = adapt_model_names(response)
        for sentence_index, (start, end, sentence) in enumerate(
            reasoning_sentences(transformed)
        ):
            normalized_sentence = sentence.strip()
            if not normalized_sentence:
                continue
            sample_id = f"{source_row_index}:{sentence_index}:{start}"
            hit, reason = matches_awareness_union(normalized_sentence)
            context = transformed[max(0, start - 220) : min(len(transformed), end + 220)]
            base = {
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "sentence_index": sentence_index,
                "type": row["type"],
                "sentence": normalized_sentence.replace("\r", " "),
                "sentence_context": context.replace("\r", " "),
                "detector_hit": int(hit),
                "detector_reason": reason,
                "cue_hits": "|".join(
                    pattern.pattern for pattern in cues if pattern.search(normalized_sentence)
                ),
            }
            if hit:
                pools["recall_detector_positive"].append(base)
            elif base["cue_hits"]:
                pools["recall_enriched_negative"].append(base)
            for pattern_index in (9, 12):
                if patterns[pattern_index].search(normalized_sentence):
                    item = dict(base)
                    item["target_pattern"] = f"v2_pattern_{pattern_index}"
                    pools[f"v2_pattern_{pattern_index}"].append(item)

    requested = {
        "recall_enriched_negative": config["recall_sample"][
            "enriched_detector_negative"
        ],
        "recall_detector_positive": config["recall_sample"]["detector_positive"],
        "v2_pattern_9": config["pattern_precision_sample"]["v2_pattern_9"],
        "v2_pattern_12": config["pattern_precision_sample"]["v2_pattern_12"],
    }
    selected = []
    used_sample_ids = set()
    seed = config["seed_string"]
    selection_order = (
        "v2_pattern_9",
        "v2_pattern_12",
        "recall_enriched_negative",
        "recall_detector_positive",
    )
    for stratum in selection_order:
        count = requested[stratum]
        unique = {row["sample_id"]: row for row in pools[stratum]}
        ordered = sorted(unique.values(), key=lambda row: order_key(row["sample_id"], seed))
        available = [row for row in ordered if row["sample_id"] not in used_sample_ids]
        if len(available) < count:
            raise AssertionError(
                f"{stratum}: requested {count}, found {len(available)} unique"
            )
        for row in available[:count]:
            item = dict(row)
            item["stratum"] = stratum
            item.setdefault("target_pattern", "")
            selected.append(item)
            used_sample_ids.add(row["sample_id"])

    metadata_fields = [
        "sample_id",
        "stratum",
        "target_pattern",
        "source_row_index",
        "sentence_index",
        "type",
        "detector_hit",
        "detector_reason",
        "cue_hits",
        "sentence",
        "sentence_context",
    ]
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    with args.sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(selected)

    blinded = sorted(selected, key=lambda row: order_key(row["sample_id"], seed + ":blind"))
    blinded_fields = [
        "sample_id",
        "sentence",
        "sentence_context",
        "inclusive_label",
        "strict_label",
        "notes",
    ]
    for destination in (args.blinded, args.annotations):
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=blinded_fields)
            writer.writeheader()
            for row in blinded:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "sentence": row["sentence"],
                        "sentence_context": row["sentence_context"],
                        "inclusive_label": "",
                        "strict_label": "",
                        "notes": "",
                    }
                )

    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk_index in range(4):
        chunk = blinded[chunk_index * 65 : (chunk_index + 1) * 65]
        destination = args.chunk_dir / f"chunk_{chunk_index + 1}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("sample_id", "sentence", "sentence_context"),
            )
            writer.writeheader()
            for row in chunk:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "sentence": row["sentence"],
                        "sentence_context": row["sentence_context"],
                    }
                )

    print(
        json.dumps(
            {
                "pool_sizes": {key: len(value) for key, value in pools.items()},
                "selected": requested,
                "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                "blinded_sha256": hashlib.sha256(args.blinded.read_bytes()).hexdigest(),
                "annotations_blank_sha256": hashlib.sha256(
                    args.annotations.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
