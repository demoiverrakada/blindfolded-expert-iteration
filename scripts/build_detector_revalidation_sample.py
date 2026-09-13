#!/usr/bin/env python3
"""Freeze the fresh, disjoint detector confirmation sample before revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import has_one_well_formed_think_block, reasoning_sentences
from bei.detectors import WOOD_LABS_RE, detector_patterns, matches_awareness_union
from bei.qwen_data import MODEL_SELF_ID_RE, adapt_model_names

RECALL_STRATA = (
    "detector_positive",
    "cue_enriched_detector_negative",
    "unenriched_detector_negative",
)
FAMILY_STRATA = {
    f"added_only_family_{index}": f"family_{index}" for index in range(1, 5)
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


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


def family_name(regex_name: str) -> str:
    match = re.match(r"(family_[1-4])_", regex_name)
    if match is None:
        raise AssertionError(f"Unassigned prospective regex: {regex_name}")
    return match.group(1)


def render_rubric(config: dict) -> str:
    lines = [
        "# Sentence-only detector annotation rubric",
        "",
        "Judge only the words in the displayed sentence. Do not inspect source rows,",
        "neighboring sentences, detector predictions, strata, or other context.",
        "",
    ]
    for value in ("1", "0", "2"):
        lines.extend((f"## Label {value}", "", config["annotation_rubric"][value], ""))
    lines.extend(("## Scoring", "", config["scoring_rule"], ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/detector_revalidation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen/detector_revalidation",
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    old_config_path = PROJECT_ROOT / "configs/detector_validation.json"
    old_config = json.loads(old_config_path.read_text())
    burned_path = PROJECT_ROOT / config["burned_sample"]
    detector_path = PROJECT_ROOT / "src/bei/detectors.py"
    script_path = Path(__file__).resolve()
    scorer_path = PROJECT_ROOT / "scripts/score_detector_revalidation.py"
    revision_auditor_path = PROJECT_ROOT / "scripts/audit_detector_revision.py"

    cues = [re.compile(value, re.I) for value in old_config["independent_cue_lexicon"]]
    additions = {
        name: re.compile(value, re.I)
        for name, value in config["prospective_additional_regexes"].items()
    }
    additions_by_family: dict[str, list[tuple[str, re.Pattern]]] = defaultdict(list)
    for name, pattern in additions.items():
        additions_by_family[family_name(name)].append((name, pattern))
    assert set(additions_by_family) == {f"family_{index}" for index in range(1, 5)}

    neutral_terms = [re.compile(value, re.I) for value in config["neutral_code_terms"]]
    neutral_exclusion = re.compile(config["neutral_exclusion_terms"], re.I)
    patterns = detector_patterns("v2")
    burned_sample_rows = list(
        csv.DictReader(burned_path.open(newline="", encoding="utf-8"))
    )
    burned_rows = {
        int(row["source_row_index"]) for row in burned_sample_rows
    }

    pools: dict[str, list[dict]] = {name: [] for name in config["samples"]}
    all_items: dict[str, dict] = {}
    source_rows = list(
        csv.DictReader(args.e1_csv.open(newline="", encoding="utf-8"))
    )
    current_burned_sentences = {}
    for source_row_index in burned_rows:
        response = source_rows[source_row_index]["response"]
        if (
            not has_one_well_formed_think_block(response)
            or MODEL_SELF_ID_RE.search(response)
        ):
            continue
        transformed, _ = adapt_model_names(response)
        for sentence_index, (start, _end, sentence) in enumerate(
            reasoning_sentences(transformed)
        ):
            current_burned_sentences[
                f"{source_row_index}:{sentence_index}:{start}"
            ] = sentence.strip().replace("\r", " ")
    burned_id_text_matches = all(
        current_burned_sentences.get(row["sample_id"]) == row["sentence"]
        for row in burned_sample_rows
    )
    eligible_source_rows = 0
    scanned_sentences = 0

    for source_row_index, row in enumerate(source_rows):
        if source_row_index in burned_rows:
            continue
        if (
            not has_one_well_formed_think_block(row["response"])
            or MODEL_SELF_ID_RE.search(row["response"])
        ):
            continue
        eligible_source_rows += 1
        response, _ = adapt_model_names(row["response"])
        for sentence_index, (start, _end, sentence) in enumerate(
            reasoning_sentences(response)
        ):
            sentence = sentence.strip().replace("\r", " ").replace("\n", " ")
            if not sentence:
                continue
            scanned_sentences += 1
            sample_id = f"{source_row_index}:{sentence_index}:{start}"
            old_hit, old_reason = matches_awareness_union(sentence)
            added_hits = [
                name for name, pattern in additions.items() if pattern.search(sentence)
            ]
            added_families = sorted({family_name(name) for name in added_hits})
            revised_hit = old_hit or bool(added_hits)
            cue_hit = any(pattern.search(sentence) for pattern in cues)

            pattern_9_hit = bool(
                patterns[9].search(sentence)
                or additions["family_3_pattern_9"].search(sentence)
            )
            pattern_12_hit = bool(
                patterns[12].search(sentence)
                or additions["family_1_pattern_12_modifiers"].search(sentence)
                or additions["family_1_pattern_12_study"].search(sentence)
                or additions["family_2_determiner_pattern_12"].search(sentence)
            )
            neutral_term_hits = [
                pattern.pattern for pattern in neutral_terms if pattern.search(sentence)
            ]

            item = {
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "sentence_index": sentence_index,
                "type": row["type"],
                "sentence": sentence,
                "old_detector_hit": int(old_hit),
                "old_detector_reason": old_reason,
                "prospective_added_hits": "|".join(added_hits),
                "prospective_added_families": "|".join(added_families),
                "prospective_revised_hit": int(revised_hit),
                "cue_enriched": int(cue_hit),
                "neutral_term_hits": "|".join(neutral_term_hits),
            }
            assert sample_id not in all_items
            all_items[sample_id] = item

            if revised_hit:
                pools["detector_positive"].append(item)
            elif cue_hit:
                pools["cue_enriched_detector_negative"].append(item)
            else:
                pools["unenriched_detector_negative"].append(item)

            if pattern_9_hit:
                pools["v2_pattern_9"].append(item)
            if pattern_12_hit:
                pools["v2_pattern_12"].append(item)
            for stratum, family in FAMILY_STRATA.items():
                if not old_hit and family in added_families:
                    pools[stratum].append(item)
            if neutral_term_hits and not neutral_exclusion.search(sentence):
                pools["neutral_code_holdout"].append(item)

    recall_ids = [
        {row["sample_id"] for row in pools[stratum]} for stratum in RECALL_STRATA
    ]
    assert not (recall_ids[0] & recall_ids[1])
    assert not (recall_ids[0] & recall_ids[2])
    assert not (recall_ids[1] & recall_ids[2])
    assert sum(len(values) for values in recall_ids) == scanned_sentences
    assert set().union(*recall_ids) == set(all_items)

    selections: dict[str, list[str]] = {}
    for stratum, requested in config["samples"].items():
        candidates = sorted(
            {row["sample_id"] for row in pools[stratum]},
            key=lambda sample_id: stable_key(
                sample_id, f"{config['seed_string']}:{stratum}"
            ),
        )
        if len(candidates) < requested:
            raise AssertionError(
                f"{stratum}: requested {requested}, found {len(candidates)}"
            )
        selections[stratum] = candidates[:requested]

    memberships: dict[str, list[str]] = defaultdict(list)
    for stratum, selected_ids in selections.items():
        for sample_id in selected_ids:
            memberships[sample_id].append(stratum)
    union_ids = sorted(
        memberships,
        key=lambda sample_id: stable_key(
            sample_id, f"{config['seed_string']}:blind"
        ),
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True)

    rubric_path = args.output_dir / "rubric.md"
    rubric_path.write_text(render_rubric(config), encoding="utf-8")

    selections_path = args.output_dir / "selections.json"
    selections_path.write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sample_path = args.output_dir / "sample.csv"
    sample_fields = [
        "sample_id",
        "strata",
        "source_row_index",
        "sentence_index",
        "type",
        "old_detector_hit",
        "old_detector_reason",
        "prospective_added_hits",
        "prospective_added_families",
        "prospective_revised_hit",
        "cue_enriched",
        "neutral_term_hits",
        "sentence",
    ]
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        for sample_id in union_ids:
            output = dict(all_items[sample_id])
            output["strata"] = "|".join(sorted(memberships[sample_id]))
            writer.writerow(output)

    blinded_path = args.output_dir / "blinded.csv"
    with blinded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "sentence", "label", "notes"),
        )
        writer.writeheader()
        for sample_id in union_ids:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "sentence": all_items[sample_id]["sentence"],
                    "label": "",
                    "notes": "",
                }
            )

    chunk_paths = []
    for start in range(0, len(union_ids), args.chunk_size):
        chunk_ids = union_ids[start : start + args.chunk_size]
        chunk_path = chunks_dir / f"chunk_{start // args.chunk_size + 1:02d}.csv"
        with chunk_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_id", "sentence")
            )
            writer.writeheader()
            for sample_id in chunk_ids:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "sentence": all_items[sample_id]["sentence"],
                    }
                )
        chunk_paths.append(chunk_path)

    sample_rows = list(
        csv.DictReader(sample_path.open(newline="", encoding="utf-8"))
    )
    blinded_rows = list(
        csv.DictReader(blinded_path.open(newline="", encoding="utf-8"))
    )
    sample_ids = [row["sample_id"] for row in sample_rows]
    blinded_ids = [row["sample_id"] for row in blinded_rows]
    all_selected_ids = {
        sample_id for selected_ids in selections.values() for sample_id in selected_ids
    }
    neutral_ids = selections["neutral_code_holdout"]
    neutral_old_hit_ids = [
        sample_id
        for sample_id in neutral_ids
        if all_items[sample_id]["old_detector_hit"]
    ]
    neutral_added_only_ids = [
        sample_id
        for sample_id in neutral_ids
        if (
            all_items[sample_id]["prospective_revised_hit"]
            and not all_items[sample_id]["old_detector_hit"]
        )
    ]
    family_4_pattern = additions["family_4_bounded_sequence"]
    family_4_trigger_spans = sorted(
        {
            re.sub(r"\s+", " ", match.group(0).lower())
            for row in pools["added_only_family_4"]
            if (match := family_4_pattern.search(row["sentence"])) is not None
        }
    )

    assertions = {
        "recall_pools_are_disjoint_partition": (
            sum(len(values) for values in recall_ids) == scanned_sentences
            and set().union(*recall_ids) == set(all_items)
        ),
        "every_pool_meets_requested_size": all(
            len(pools[name]) >= requested
            for name, requested in config["samples"].items()
        ),
        "every_selection_has_requested_size": all(
            len(selections[name]) == requested
            for name, requested in config["samples"].items()
        ),
        "selection_membership_is_valid": all(
            set(selections[name]).issubset(
                {row["sample_id"] for row in pools[name]}
            )
            for name in config["samples"]
        ),
        "sample_is_unique_union_of_selections": (
            len(sample_ids) == len(set(sample_ids))
            and set(sample_ids) == all_selected_ids
        ),
        "blinded_is_unique_union_of_selections": (
            len(blinded_ids) == len(set(blinded_ids))
            and set(blinded_ids) == all_selected_ids
        ),
        "blinded_has_sentence_only": (
            set(blinded_rows[0]) == {"sample_id", "sentence", "label", "notes"}
            if blinded_rows
            else False
        ),
        "burned_source_rows_are_disjoint": not (
            burned_rows
            & {int(all_items[sample_id]["source_row_index"]) for sample_id in union_ids}
        ),
        "burned_sample_ids_and_text_match_current_e1": burned_id_text_matches,
        "added_only_samples_are_old_detector_negative": all(
            not all_items[sample_id]["old_detector_hit"]
            for stratum in FAMILY_STRATA
            for sample_id in selections[stratum]
        ),
        "chunk_union_matches_blinded": {
            row["sample_id"]
            for path in chunk_paths
            for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        }
        == set(blinded_ids),
    }
    if not all(assertions.values()):
        raise AssertionError(
            "Freeze assertions failed: "
            f"{[name for name, passed in assertions.items() if not passed]}"
        )

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "status": "FROZEN_PRE_REVISION",
        "git_head_before_revision": git_head,
        "source": {
            "e1_csv": str(args.e1_csv.resolve()),
            "e1_sha256": sha256_path(args.e1_csv),
            "e1_rows": len(source_rows),
            "eligible_source_rows_after_burn": eligible_source_rows,
            "burned_sample": str(burned_path.relative_to(PROJECT_ROOT)),
            "burned_sample_sha256": sha256_path(burned_path),
            "burned_source_rows": len(burned_rows),
            "burned_sample_sentences": len(burned_sample_rows),
            "detector_file": str(detector_path.relative_to(PROJECT_ROOT)),
            "detector_sha256_before_revision": sha256_path(detector_path),
            "v2_pattern_fingerprint_before_revision": pattern_fingerprint(patterns),
            "wood_labs_fingerprint_before_revision": regex_fingerprint(
                WOOD_LABS_RE
            ),
            "config_sha256": sha256_path(args.config),
            "prior_validation_config_sha256": sha256_path(old_config_path),
            "builder_sha256": sha256_path(script_path),
            "scorer_sha256": sha256_path(scorer_path),
            "revision_auditor_sha256": sha256_path(revision_auditor_path),
        },
        "sampling": {
            "seed_string": config["seed_string"],
            "residual_dependence_note": config["residual_dependence_note"],
            "scanned_sentences": scanned_sentences,
            "pool_sizes": {name: len(pool) for name, pool in pools.items()},
            "selected_counts": {
                name: len(selected_ids) for name, selected_ids in selections.items()
            },
            "selection_ids_sha256": {
                name: ids_sha256(selected_ids)
                for name, selected_ids in selections.items()
            },
            "independent_memberships": sum(map(len, selections.values())),
            "unique_blinded_sentences": len(union_ids),
            "unique_blinded_ids_sha256": ids_sha256(union_ids),
            "chunk_size": args.chunk_size,
            "chunks": len(chunk_paths),
            "neutral_code_pre_revision_baseline": {
                "old_v2_raw_hits": len(neutral_old_hit_ids),
                "old_v2_raw_hit_ids": neutral_old_hit_ids,
                "prospective_added_only_raw_hits": len(neutral_added_only_ids),
                "prospective_added_only_raw_hit_ids": neutral_added_only_ids,
            },
            "family_4_validation": {
                **config["family_4_validation"],
                "pool_is_census": (
                    len(pools["added_only_family_4"])
                    == len(selections["added_only_family_4"])
                ),
                "pool_rows": len(pools["added_only_family_4"]),
                "distinct_trigger_spans": len(family_4_trigger_spans),
                "trigger_spans": family_4_trigger_spans,
            },
        },
        "artifacts": {
            "sample_sha256": sha256_path(sample_path),
            "blinded_sha256": sha256_path(blinded_path),
            "selections_sha256": sha256_path(selections_path),
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
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
