#!/usr/bin/env python3
"""Build a blinded, reason-stratified validation sample for V3 mask additions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import has_one_well_formed_think_block, reasoning_sentences
from bei.detectors import matches_awareness_union
from bei.qwen_data import MODEL_SELF_ID_RE, adapt_model_names

REASONS = tuple(f"v3_addition_{index}" for index in range(6))
SAMPLED_REASONS = REASONS[:5]
EXTERNAL_REASON = "v3_addition_5"
EXTERNAL_PATTERN_NAME = "family_4_bounded_sequence"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(sample_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument("--quota-per-reason", type=int, default=25)
    parser.add_argument("--minimum-per-reason", type=int, default=20)
    parser.add_argument(
        "--exclude-sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation/sample.csv",
    )
    parser.add_argument(
        "--burned-sample",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_validation_sample.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_revalidation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen/mask_v3_validation",
    )
    parser.add_argument(
        "--seed", default="20260911-mask-v3-reason-stratified"
    )
    args = parser.parse_args()
    scorer_path = PROJECT_ROOT / "scripts/score_v3_mask_validation.py"
    config = json.loads(args.config.read_text())

    excluded_sample_rows = list(
        csv.DictReader(args.exclude_sample.open(newline="", encoding="utf-8"))
    )
    excluded_source_rows = {
        int(row["source_row_index"]) for row in excluded_sample_rows
    }
    excluded_source_rows.update(
        int(row["source_row_index"])
        for row in csv.DictReader(
            args.burned_sample.open(newline="", encoding="utf-8")
        )
    )
    external_coverage_ids = [
        row["sample_id"]
        for row in excluded_sample_rows
        if EXTERNAL_PATTERN_NAME in row["prospective_added_hits"].split("|")
    ]
    pools = {reason: [] for reason in REASONS}
    rows = list(csv.DictReader(args.e1_csv.open(newline="", encoding="utf-8")))
    for source_row_index, row in enumerate(rows):
        if source_row_index in excluded_source_rows:
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
            sentence = sentence.strip().replace("\r", " ").replace("\n", " ")
            if not sentence:
                continue
            hit, reason = matches_awareness_union(sentence)
            if hit and reason in pools:
                pools[reason].append(
                    {
                        "sample_id": f"{source_row_index}:{sentence_index}:{start}",
                        "source_row_index": source_row_index,
                        "sentence_index": sentence_index,
                        "type": row["type"],
                        "reason": reason,
                        "sentence": sentence,
                    }
                )

    selections = {}
    for reason in SAMPLED_REASONS:
        pool = pools[reason]
        unique = {row["sample_id"]: row for row in pool}
        ordered = sorted(
            unique,
            key=lambda sample_id: stable_key(
                sample_id, f"{args.seed}:{reason}"
            ),
        )
        count = min(args.quota_per_reason, len(ordered))
        if count < args.minimum_per_reason:
            raise AssertionError(
                f"{reason}: only {len(ordered)} disjoint candidates; "
                f"minimum is {args.minimum_per_reason}"
            )
        selections[reason] = ordered[:count]
    selections[EXTERNAL_REASON] = []

    all_items = {
        row["sample_id"]: row for pool in pools.values() for row in pool
    }
    union_ids = [
        sample_id
        for reason in SAMPLED_REASONS
        for sample_id in selections[reason]
    ]
    assert len(union_ids) == len(set(union_ids))
    union_ids.sort(key=lambda sample_id: stable_key(sample_id, args.seed + ":blind"))
    blind_ids = {
        source_sample_id: f"m3_{index + 1:04d}"
        for index, source_sample_id in enumerate(union_ids)
    }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True)

    rubric = (
        "# Sentence-only V3 mask validation rubric\n\n"
        "Judge only the displayed sentence. Do not inspect detector predictions, "
        "reason buckets, source rows, prompts, or neighboring text.\n\n"
        f"- `1`: {config['annotation_rubric']['1']}\n"
        f"- `0`: {config['annotation_rubric']['0']}\n"
        f"- `2`: {config['annotation_rubric']['2']}\n\n"
        f"{config['scoring_rule']}\n"
    )
    rubric_path = args.output_dir / "rubric.md"
    rubric_path.write_text(rubric, encoding="utf-8")

    key_path = args.output_dir / "key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "source_sample_id",
                "source_row_index",
                "sentence_index",
                "type",
                "reason",
                "sentence",
            ),
        )
        writer.writeheader()
        for source_sample_id in union_ids:
            output = dict(all_items[source_sample_id])
            output["source_sample_id"] = output.pop("sample_id")
            output["sample_id"] = blind_ids[source_sample_id]
            writer.writerow(output)

    blinded_path = args.output_dir / "blinded.csv"
    with blinded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "sentence", "label", "notes")
        )
        writer.writeheader()
        for source_sample_id in union_ids:
            writer.writerow(
                {
                    "sample_id": blind_ids[source_sample_id],
                    "sentence": all_items[source_sample_id]["sentence"],
                    "label": "",
                    "notes": "",
                }
            )

    chunk_paths = []
    for start in range(0, len(union_ids), 50):
        chunk_path = chunks_dir / f"chunk_{start // 50 + 1:02d}.csv"
        with chunk_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "sentence")
            )
            writer.writeheader()
            for source_sample_id in union_ids[start : start + 50]:
                writer.writerow(
                    {
                        "sample_id": blind_ids[source_sample_id],
                        "sentence": all_items[source_sample_id]["sentence"],
                    }
                )
        chunk_paths.append(chunk_path)

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "status": "FROZEN_BLINDED_UNLABELED",
        "git_head": git_head,
        "source": {
            "e1_sha256": sha256_path(args.e1_csv),
            "detector_sha256": sha256_path(PROJECT_ROOT / "src/bei/detectors.py"),
            "config_sha256": sha256_path(args.config),
            "excluded_detector_sample_sha256": sha256_path(args.exclude_sample),
            "burned_sample_sha256": sha256_path(args.burned_sample),
            "excluded_source_rows": len(excluded_source_rows),
            "builder_sha256": sha256_path(Path(__file__).resolve()),
            "scorer_sha256": sha256_path(scorer_path),
        },
        "sampling": {
            "seed": args.seed,
            "quota_per_reason": args.quota_per_reason,
            "minimum_per_reason": args.minimum_per_reason,
            "pool_sizes": {
                reason: len({row["sample_id"] for row in pool})
                for reason, pool in pools.items()
            },
            "selected_counts": {
                reason: len(sample_ids)
                for reason, sample_ids in selections.items()
            },
            "selected_rows": len(union_ids),
            "external_census_coverage": {
                "reason": EXTERNAL_REASON,
                "pattern_name": EXTERNAL_PATTERN_NAME,
                "detector_revalidation_ids": external_coverage_ids,
                "rows": len(external_coverage_ids),
                "interpretation": (
                    "No source-row-disjoint candidates remain because the main "
                    "detector revalidation sample already contains the full "
                    "non-burned eligible family-4 census. Its blinded labels "
                    "provide this bucket's only corpus evidence."
                ),
            },
            "reason_to_config_pattern": {
                f"v3_addition_{index}": name
                for index, name in enumerate(
                    config["prospective_additional_regexes"]
                )
            },
            "attribution_note": (
                "Reason strata measure first-match residual additions after "
                "Wood Labs, V2, and earlier V3 additions have taken precedence."
            ),
            "source_row_disjoint_from_detector_validation": all(
                int(all_items[sample_id]["source_row_index"])
                not in excluded_source_rows
                for sample_id in union_ids
            ),
        },
        "artifacts": {
            "key_sha256": sha256_path(key_path),
            "blinded_sha256": sha256_path(blinded_path),
            "rubric_sha256": sha256_path(rubric_path),
            "chunk_sha256": {
                path.name: sha256_path(path) for path in chunk_paths
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
