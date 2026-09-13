#!/usr/bin/env python3
"""Freeze the source-row-disjoint CP3-R contextual validation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

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


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def read_source_indices(path: Path) -> set[int]:
    rows = csv.DictReader(path.open(newline="", encoding="utf-8"))
    indices = set()
    for row in rows:
        value = row.get("source_row_index", "").strip()
        if value:
            indices.add(int(value))
    return indices


def normalize_sentence(text: str) -> str:
    return text.strip().replace("\r", " ").replace("\n", " ")


def context_item(
    *,
    source_row_index: int,
    sentence_index: int,
    sentence_spans: list[tuple[int, int, str]],
    row: dict[str, str],
    transformed: dict[str, str],
    selector_reason: str,
) -> dict[str, object]:
    start, _end, sentence = sentence_spans[sentence_index]
    return {
        "source_sample_id": f"{source_row_index}:{sentence_index}:{start}",
        "source_row_index": source_row_index,
        "sentence_index": sentence_index,
        "type": row["type"],
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
        "selector_reason": selector_reason,
    }


def render_rubric(config: dict) -> str:
    lines = [
        "# CP3-R contextual annotation rubric",
        "",
        "Judge the target sentence using only the displayed system prompt, user",
        "prompt, and adjacent reasoning sentences. Do not inspect source files,",
        "selectors, strata, detector output, or other context.",
        "",
        "## Awareness role",
        "",
    ]
    for label, meaning in config["annotation"]["awareness_role"].items():
        lines.append(f"- `{label}` — {meaning}")
    lines.extend(("", "## Functional neutrality", ""))
    for label, meaning in config["annotation"]["functional_neutrality"].items():
        lines.append(f"- `{label}` — {meaning}")
    lines.extend(
        (
            "",
            "Return both labels for every row. Keep notes short and sentence-local.",
            "",
        )
    )
    return "\n".join(lines)


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen/recovery_validation",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise AssertionError(
            f"Refusing to overwrite frozen output directory: {args.output_dir}"
        )
    config = json.loads(args.config.read_text())
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
        int(row["source_row_index"]) for row in membership["rows"]
    }
    excluded_rows = set(membership_rows)
    prior_file_reports = {}
    for relative_path in config["prior_source_row_files"]:
        path = PROJECT_ROOT / relative_path
        file_rows = read_source_indices(path)
        excluded_rows.update(file_rows)
        prior_file_reports[relative_path] = {
            "sha256": sha256_path(path),
            "source_rows": len(file_rows),
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
    pools: dict[str, list[dict[str, object]]] = {
        stratum: [] for stratum in config["samples"]
    }
    eligible_source_rows = 0

    for source_row_index, row in enumerate(source_rows):
        if source_row_index in excluded_rows:
            continue
        if (
            not has_one_well_formed_think_block(row["response"])
            or MODEL_SELF_ID_RE.search(row["response"])
        ):
            continue
        eligible_source_rows += 1
        transformed = {
            field: adapt_model_names(row[field])[0]
            for field in ("system_prompt", "user_prompt", "response")
        }
        response = transformed["response"]
        sentence_spans = reasoning_sentences(response)
        for sentence_index, (_start, _end, sentence) in enumerate(sentence_spans):
            wood = bool(WOOD_LABS_RE.search(sentence))
            v1 = any(pattern.search(sentence) for pattern in v1_patterns)
            broad, broad_reason = matches_awareness_union(sentence)
            item_kwargs = {
                "source_row_index": source_row_index,
                "sentence_index": sentence_index,
                "sentence_spans": sentence_spans,
                "row": row,
                "transformed": transformed,
            }
            if wood:
                pools["core_wood"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason="core_wood",
                    )
                )
            elif v1:
                pools["core_nonwood_v1"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason="core_nonwood_v1",
                    )
                )
            elif broad and v2_patterns[9].search(sentence):
                pools["broad_only_v2_pattern_9"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason=broad_reason,
                    )
                )
            elif broad and v2_patterns[12].search(sentence):
                pools["broad_only_v2_pattern_12"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason=broad_reason,
                    )
                )
            elif broad:
                pools["other_broad_only"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason=broad_reason,
                    )
                )
            elif any(cue.search(sentence) for cue in cues):
                pools["broad_negative_diagnostic"].append(
                    context_item(
                        **item_kwargs,
                        selector_reason="cue_enriched_broad_negative",
                    )
                )

        core_spans = core_awareness_spans(response)
        neutral_spans = neutral_control_spans(response)
        if not core_spans or not neutral_spans:
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
        if len(encoded["input_ids"]) > config["placebo_sequence_cap"]:
            continue
        if encoded["input_ids"][: len(prefix_ids)] != prefix_ids:
            raise AssertionError(f"Prefix mismatch at row {source_row_index}")
        offsets = encoded["offset_mapping"]
        absolute_core = [
            (len(prefix_text) + span.start, len(prefix_text) + span.end)
            for span in core_spans
        ]
        absolute_neutral = [
            (len(prefix_text) + start, len(prefix_text) + end)
            for start, end, _text in neutral_spans
        ]
        core_indices = token_indices_for_spans(
            offsets,
            absolute_core,
            minimum_token_index=len(prefix_ids),
        )
        neutral_indices = token_indices_for_spans(
            offsets,
            absolute_neutral,
            minimum_token_index=len(prefix_ids),
        )
        if len(neutral_indices) < len(core_indices):
            continue
        try:
            near_indices = nearest_position_control_indices(
                core_indices, neutral_indices
            )
            distant_indices = distant_position_control_indices(
                core_indices,
                neutral_indices,
                minimum_distance=config["distant_minimum_tokens"],
                seed=(
                    f"{config['seed_string']}:placebo:"
                    f"{source_row_index}"
                ),
            )
        except ValueError:
            continue

        sentence_index_by_span = {
            (start, end): index
            for index, (start, end, _sentence) in enumerate(sentence_spans)
        }
        for stratum, selected_indices in (
            ("near_placebo", near_indices),
            ("distant_placebo", distant_indices),
        ):
            for start, end, _sentence in neutral_spans:
                absolute_span = (len(prefix_text) + start, len(prefix_text) + end)
                span_tokens = set(
                    token_indices_for_spans(
                        offsets,
                        [absolute_span],
                        minimum_token_index=len(prefix_ids),
                    )
                )
                selected_overlap = span_tokens & set(selected_indices)
                if not selected_overlap:
                    continue
                sentence_index = sentence_index_by_span[(start, end)]
                item = context_item(
                    source_row_index=source_row_index,
                    sentence_index=sentence_index,
                    sentence_spans=sentence_spans,
                    row=row,
                    transformed=transformed,
                    selector_reason=(
                        f"{stratum}:{len(selected_overlap)}_selected_tokens"
                    ),
                )
                item["selected_token_count"] = len(selected_overlap)
                pools[stratum].append(item)

    selections: dict[str, list[dict[str, object]]] = {}
    used_source_rows: set[int] = set()
    for stratum, requested in config["samples"].items():
        per_source_row: dict[int, dict[str, object]] = {}
        for item in sorted(
            pools[stratum],
            key=lambda item: stable_key(
                str(item["source_sample_id"]),
                f"{config['seed_string']}:{stratum}:within-row",
            ),
        ):
            per_source_row.setdefault(int(item["source_row_index"]), item)
        ordered = sorted(
            per_source_row.values(),
            key=lambda item: stable_key(
                str(item["source_sample_id"]),
                f"{config['seed_string']}:{stratum}:selection",
            ),
        )
        selected = [
            item
            for item in ordered
            if int(item["source_row_index"]) not in used_source_rows
        ][:requested]
        if len(selected) != requested:
            raise AssertionError(
                f"{stratum}: requested {requested}, found {len(selected)} "
                "after global source-row disjointness"
            )
        selections[stratum] = selected
        used_source_rows.update(int(item["source_row_index"]) for item in selected)

    union_items = [
        {**item, "stratum": stratum}
        for stratum, items in selections.items()
        for item in items
    ]
    union_items.sort(
        key=lambda item: stable_key(
            str(item["source_sample_id"]),
            f"{config['seed_string']}:opaque-order",
        )
    )
    for index, item in enumerate(union_items, start=1):
        item["sample_id"] = f"rv_{index:04d}"

    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    rubric_path = args.output_dir / "rubric.md"
    rubric_path.write_text(render_rubric(config), encoding="utf-8")

    context_fields = [
        "sample_id",
        "system_prompt",
        "user_prompt",
        "previous_sentence",
        "target_sentence",
        "next_sentence",
    ]
    key_fields = [
        "sample_id",
        "source_sample_id",
        "stratum",
        "source_row_index",
        "sentence_index",
        "type",
        "selector_reason",
        "selected_token_count",
        *context_fields[1:],
    ]
    key_path = args.output_dir / "key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        for item in union_items:
            writer.writerow({field: item.get(field, "") for field in key_fields})

    blinded_path = args.output_dir / "blinded.csv"
    blinded_fields = [
        *context_fields,
        "awareness_role",
        "functional_neutrality",
        "notes",
    ]
    with blinded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=blinded_fields)
        writer.writeheader()
        for item in union_items:
            writer.writerow(
                {
                    **{field: item[field] for field in context_fields},
                    "awareness_role": "",
                    "functional_neutrality": "",
                    "notes": "",
                }
            )

    chunk_paths = []
    chunk_size = config["chunk_size"]
    for start in range(0, len(union_items), chunk_size):
        chunk_path = chunks_dir / f"chunk_{start // chunk_size + 1:02d}.csv"
        with chunk_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=context_fields)
            writer.writeheader()
            for item in union_items[start : start + chunk_size]:
                writer.writerow({field: item[field] for field in context_fields})
        chunk_paths.append(chunk_path)

    selected_counts = {
        stratum: len(items) for stratum, items in selections.items()
    }
    assertions = {
        "selected_counts_exact": selected_counts == config["samples"],
        "all_selected_source_rows_unique": (
            len(used_source_rows) == len(union_items)
        ),
        "selected_rows_disjoint_from_all_exclusions": not (
            used_source_rows & excluded_rows
        ),
        "opaque_ids_unique": (
            len({item["sample_id"] for item in union_items}) == len(union_items)
            and all(re.fullmatch(r"rv_\d{4}", str(item["sample_id"])) for item in union_items)
        ),
        "blinded_omits_provenance": not (
            {"stratum", "source_row_index", "selector_reason"}
            & set(blinded_fields)
        ),
        "chunk_union_exact": {
            row["sample_id"]
            for path in chunk_paths
            for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        }
        == {str(item["sample_id"]) for item in union_items},
    }
    if not all(assertions.values()):
        raise AssertionError(
            f"Freeze failed: {[key for key, value in assertions.items() if not value]}"
        )

    scorer_path = PROJECT_ROOT / "scripts/score_recovery_validation.py"
    manifest = {
        "status": "FROZEN_CONTEXTUAL_UNLABELED",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source": {
            "e1_csv": str(args.e1_csv.resolve()),
            "e1_sha256": sha256_path(args.e1_csv),
            "e1_rows": len(source_rows),
            "membership_sha256": sha256_path(args.membership),
            "excluded_membership_rows": len(membership_rows),
            "prior_validation_files": prior_file_reports,
            "excluded_source_rows_union": len(excluded_rows),
            "eligible_source_rows_after_exclusions": eligible_source_rows,
            "config_sha256": sha256_path(args.config),
            "builder_sha256": sha256_path(Path(__file__).resolve()),
            "scorer_sha256": sha256_path(scorer_path),
            "matched_mask_sha256": sha256_path(
                PROJECT_ROOT / "src/bei/matched_mask.py"
            ),
            "detector_sha256": sha256_path(
                PROJECT_ROOT / "src/bei/detectors.py"
            ),
            "tokenizer_path": str(args.model_path),
        },
        "sampling": {
            "seed_string": config["seed_string"],
            "pool_sizes": {
                stratum: len({int(item["source_row_index"]) for item in items})
                for stratum, items in pools.items()
            },
            "selected_counts": selected_counts,
            "selected_source_rows": len(used_source_rows),
            "placebo_sequence_cap": config["placebo_sequence_cap"],
            "distant_minimum_tokens": config["distant_minimum_tokens"],
            "chunk_size": chunk_size,
            "chunks": len(chunk_paths),
        },
        "artifacts": {
            "key_sha256": sha256_path(key_path),
            "blinded_sha256": sha256_path(blinded_path),
            "rubric_sha256": sha256_path(rubric_path),
            "chunk_sha256": {
                path.name: sha256_path(path) for path in chunk_paths
            },
        },
        "assertions": assertions,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_counts": selected_counts,
                "pool_sizes": manifest["sampling"]["pool_sizes"],
                "excluded_source_rows_union": len(excluded_rows),
                "artifacts": manifest["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
