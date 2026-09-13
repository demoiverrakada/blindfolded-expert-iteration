#!/usr/bin/env python3
"""Freeze locally held-out panels only after a one-directional overlap audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import has_one_well_formed_think_block  # noqa: E402
from bei.provenance import (  # noqa: E402
    ShingleMatcher,
    normalize_text,
    prompt_core,
)

DATASET = "bigcode/bigcodebench"
REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
SPLIT = "v0.1.4"
SEED_STRING = "20260910"
SHINGLE_SIZE = 13
QUARANTINE_THRESHOLD = 0.30
AUTOMATIC_THRESHOLD = 0.50
EXPECTED_BCB_ROWS = 1_140
EXPECTED_E1_ROWS = 18_435
EXPECTED_E1_CANDIDATES = 17_100
EXPECTED_SDF_STAGE1_ROWS = 266_356
EXPECTED_SDF_STAGE2_ROWS = 44_727
PANEL_SIZE = 200
PANEL_NAMES = ("l1_recognition", "l2_direction", "e1_outcome")
TRAINING_SIZE = 2_400

MODEL_SELF_ID_RE = re.compile(
    r"\b(?:i am|i['’]m|as)\s+(?:the\s+)?(?:llama\s+)?nemotron\b",
    re.IGNORECASE,
)
TASK_ID_RE = re.compile(r"\bbigcodebench/\d+\b", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{value}:{SEED_STRING}".encode()).hexdigest()


def row_identity(row: dict[str, str]) -> str:
    fields = ("system_prompt", "user_prompt", "type", "response")
    serialized = json.dumps(
        {field: row[field] for field in fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(math.ceil(probability * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def load_bigcodebench() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=REVISION, split=SPLIT)
    return [dict(row) for row in dataset]


def new_state(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "max_prompt_containment": 0.0,
        "max_core_containment": 0.0,
        "max_solution_containment": 0.0,
        "exact_full": False,
        "exact_core": False,
        "exact_solution": False,
        "task_id_match": False,
        "best_evidence": {},
    }


def update_best(
    state: dict,
    metric: str,
    score: float,
    source: str,
    record_id: int,
) -> None:
    if score <= state[metric]:
        return
    state[metric] = score
    state["best_evidence"][metric] = {
        "score": round(score, 6),
        "source": source,
        "record_id": record_id,
    }


def scan_text(
    text: str,
    *,
    source: str,
    record_id: int,
    matcher: ShingleMatcher,
    tasks: list[dict],
    states: list[dict],
    task_id_to_index: dict[str, int],
    source_maxima: dict[str, dict[int, float]],
    short_solution_owners: dict[str, list[int]],
) -> None:
    normalized = normalize_text(text)
    scores = matcher.scan_normalized(normalized)
    per_task_document_score: dict[int, float] = defaultdict(float)
    per_task_document_exact = set()

    for key, score in scores.items():
        task_index, kind = key
        state = states[task_index]
        metric = {
            "prompt": "max_prompt_containment",
            "core": "max_core_containment",
            "solution": "max_solution_containment",
        }[kind]
        update_best(state, metric, score, source, record_id)
        per_task_document_score[task_index] = max(
            per_task_document_score[task_index], score
        )

        if kind == "prompt" and matcher.patterns[key].normalized in normalized:
            state["exact_full"] = True
            per_task_document_exact.add(task_index)
        elif kind == "core" and matcher.patterns[key].normalized in normalized:
            state["exact_core"] = True
            per_task_document_exact.add(task_index)
        elif kind == "solution" and matcher.patterns[key].normalized in normalized:
            state["exact_solution"] = True
            per_task_document_exact.add(task_index)

    for short_solution, owners in short_solution_owners.items():
        if short_solution in normalized:
            for task_index in owners:
                states[task_index]["exact_solution"] = True
                update_best(
                    states[task_index],
                    "max_solution_containment",
                    1.0,
                    source,
                    record_id,
                )
                per_task_document_exact.add(task_index)
                per_task_document_score[task_index] = 1.0

    for raw_task_id in TASK_ID_RE.findall(normalized):
        task_index = task_id_to_index.get(raw_task_id.casefold())
        if task_index is None:
            continue
        states[task_index]["task_id_match"] = True
        per_task_document_exact.add(task_index)
        per_task_document_score[task_index] = 1.0

    for task_index, score in per_task_document_score.items():
        old = source_maxima[source].get(task_index, 0.0)
        source_maxima[source][task_index] = max(old, score)
def iter_sdf(path: Path) -> Iterable[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            content = row["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Invalid SDF content at row {row_index}")
            yield row_index, content


def is_contaminated(state: dict) -> bool:
    return (
        state["exact_full"]
        or state["exact_core"]
        or state["exact_solution"]
        or state["task_id_match"]
        or state["max_prompt_containment"] >= QUARANTINE_THRESHOLD
        or state["max_core_containment"] >= QUARANTINE_THRESHOLD
        or state["max_solution_containment"] >= QUARANTINE_THRESHOLD
    )


def task_score(state: dict) -> float:
    return max(
        state["max_prompt_containment"],
        state["max_core_containment"],
        state["max_solution_containment"],
    )


def pairwise_task_overlaps(
    tasks: list[dict],
    matcher: ShingleMatcher,
) -> tuple[dict[int, set[int]], list[dict]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    evidence = []
    for kind in ("prompt", "solution"):
        owners: dict[tuple[str, ...], list[int]] = defaultdict(list)
        lengths = {}
        for task_index in range(len(tasks)):
            shingles = matcher.patterns[(task_index, kind)].shingles
            if not shingles:
                continue
            lengths[task_index] = len(shingles)
            for shingle in shingles:
                owners[shingle].append(task_index)

        shared = Counter()
        for task_owners in owners.values():
            for left, right in combinations(task_owners, 2):
                shared[(left, right)] += 1

        for (left, right), count in shared.items():
            left_score = count / lengths[left]
            right_score = count / lengths[right]
            if max(left_score, right_score) >= QUARANTINE_THRESHOLD:
                adjacency[left].add(right)
                adjacency[right].add(left)
                evidence.append(
                    {
                        "kind": kind,
                        "left": tasks[left]["task_id"],
                        "right": tasks[right]["task_id"],
                        "left_containment": round(left_score, 6),
                        "right_containment": round(right_score, 6),
                    }
                )

    short_solution_owners: dict[str, list[int]] = defaultdict(list)
    for task_index in range(len(tasks)):
        pattern = matcher.patterns[(task_index, "solution")]
        if not pattern.shingles:
            short_solution_owners[pattern.normalized].append(task_index)
    for task_owners in short_solution_owners.values():
        for left, right in combinations(task_owners, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
            evidence.append(
                {
                    "kind": "solution_exact_short",
                    "left": tasks[left]["task_id"],
                    "right": tasks[right]["task_id"],
                    "left_containment": 1.0,
                    "right_containment": 1.0,
                }
            )
    evidence.sort(
        key=lambda row: max(row["left_containment"], row["right_containment"]),
        reverse=True,
    )
    return adjacency, evidence


def run_sdf_ingestion_control(
    *,
    task_index: int,
    tasks: list[dict],
    matcher: ShingleMatcher,
    task_id_to_index: dict[str, int],
    short_solution_owners: dict[str, list[int]],
) -> dict:
    control_states = [new_state(task) for task in tasks]
    control_source = "synthetic_sdf_control"
    control_maxima: dict[str, dict[int, float]] = {control_source: {}}
    with tempfile.TemporaryDirectory(prefix="bei-sdf-control-") as directory:
        path = Path(directory) / "control.jsonl"
        path.write_text(
            json.dumps({"content": tasks[task_index]["instruct_prompt"]}) + "\n"
        )
        rows = list(iter_sdf(path))
    for row_index, content in rows:
        scan_text(
            content,
            source=control_source,
            record_id=row_index,
            matcher=matcher,
            tasks=tasks,
            states=control_states,
            task_id_to_index=task_id_to_index,
            source_maxima=control_maxima,
            short_solution_owners=short_solution_owners,
        )
    return {
        "rows_read": len(rows),
        "exact_full_recovered": control_states[task_index]["exact_full"],
        "containment": control_maxima[control_source].get(task_index, 0.0),
        "passed": (
            len(rows) == 1
            and control_states[task_index]["exact_full"]
            and control_maxima[control_source].get(task_index) == 1.0
        ),
    }


def run_detector_path_controls(
    *,
    tasks: list[dict],
    matcher: ShingleMatcher,
    task_id_to_index: dict[str, int],
    short_solution_owners: dict[str, list[int]],
) -> dict:
    states = [new_state(task) for task in tasks]
    source_maxima: dict[str, dict[int, float]] = {
        "task_id_control": {},
        "short_solution_control": {},
    }
    task_id_index = 0
    scan_text(
        f"Provenance reference: {tasks[task_id_index]['task_id']}",
        source="task_id_control",
        record_id=0,
        matcher=matcher,
        tasks=tasks,
        states=states,
        task_id_to_index=task_id_to_index,
        source_maxima=source_maxima,
        short_solution_owners=short_solution_owners,
    )

    short_solution, owners = next(iter(short_solution_owners.items()))
    short_solution_index = owners[0]
    scan_text(
        short_solution,
        source="short_solution_control",
        record_id=0,
        matcher=matcher,
        tasks=tasks,
        states=states,
        task_id_to_index=task_id_to_index,
        source_maxima=source_maxima,
        short_solution_owners=short_solution_owners,
    )
    task_id_passed = (
        states[task_id_index]["task_id_match"]
        and source_maxima["task_id_control"].get(task_id_index) == 1.0
    )
    short_solution_passed = (
        states[short_solution_index]["exact_solution"]
        and states[short_solution_index]["max_solution_containment"] == 1.0
        and source_maxima["short_solution_control"].get(short_solution_index) == 1.0
    )
    return {
        "task_id": {
            "task_id": tasks[task_id_index]["task_id"],
            "recovered": task_id_passed,
        },
        "short_canonical_solution": {
            "task_id": tasks[short_solution_index]["task_id"],
            "words": len(short_solution.split()),
            "recovered": short_solution_passed,
            "evidence": states[short_solution_index]["best_evidence"].get(
                "max_solution_containment"
            ),
        },
        "passed": task_id_passed and short_solution_passed,
    }


def largest_remainder_quotas(counts: Counter, target: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {name: target * count / total for name, count in counts.items()}
    quotas = {name: math.floor(value) for name, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def frozen_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("e1_csv", type=Path)
    parser.add_argument("sdf_stage1", type=Path)
    parser.add_argument("sdf_stage2", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/cp3_panel_provenance.json",
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=PROJECT_ROOT / "frozen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv.field_size_limit(sys.maxsize)

    tasks = load_bigcodebench()
    tasks.sort(key=lambda task: stable_order(task["task_id"]))
    states = [new_state(task) for task in tasks]
    task_id_to_index = {
        task["task_id"].casefold(): index for index, task in enumerate(tasks)
    }

    matcher = ShingleMatcher(SHINGLE_SIZE)
    for task_index, task in enumerate(tasks):
        matcher.add((task_index, "prompt"), task["instruct_prompt"])
        matcher.add((task_index, "core"), prompt_core(task["instruct_prompt"]))
        matcher.add((task_index, "solution"), task["canonical_solution"])

    short_patterns = [
        {
            "task_id": tasks[index]["task_id"],
            "kind": kind,
            "words": len(pattern.words),
        }
        for (index, kind), pattern in matcher.patterns.items()
        if not pattern.shingles
    ]
    short_solution_owners: dict[str, list[int]] = defaultdict(list)
    for row in short_patterns:
        if row["kind"] != "solution":
            continue
        task_index = task_id_to_index[row["task_id"].casefold()]
        normalized = matcher.patterns[(task_index, "solution")].normalized
        short_solution_owners[normalized].append(task_index)
    sdf_ingestion_control = run_sdf_ingestion_control(
        task_index=0,
        tasks=tasks,
        matcher=matcher,
        task_id_to_index=task_id_to_index,
        short_solution_owners=short_solution_owners,
    )
    detector_path_controls = run_detector_path_controls(
        tasks=tasks,
        matcher=matcher,
        task_id_to_index=task_id_to_index,
        short_solution_owners=short_solution_owners,
    )

    source_maxima: dict[str, dict[int, float]] = defaultdict(dict)
    for source in ("e1_user", "e1_response", "sdf_stage1", "sdf_stage2"):
        source_maxima[source] = {}
    with args.e1_csv.open(newline="", encoding="utf-8") as handle:
        e1_rows = list(csv.DictReader(handle))
    candidate_rows = []
    candidate_exclusions = Counter()
    for row_index, row in enumerate(e1_rows):
        response = row["response"]
        if not has_one_well_formed_think_block(response):
            candidate_exclusions["malformed_think"] += 1
        elif MODEL_SELF_ID_RE.search(response):
            candidate_exclusions["model_self_identification"] += 1
        else:
            candidate_rows.append((row_index, row))
        scan_text(
            row["user_prompt"],
            source="e1_user",
            record_id=row_index,
            matcher=matcher,
            tasks=tasks,
            states=states,
            task_id_to_index=task_id_to_index,
            source_maxima=source_maxima,
            short_solution_owners=short_solution_owners,
        )
        scan_text(
            response,
            source="e1_response",
            record_id=row_index,
            matcher=matcher,
            tasks=tasks,
            states=states,
            task_id_to_index=task_id_to_index,
            source_maxima=source_maxima,
            short_solution_owners=short_solution_owners,
        )

    sdf_counts = {}
    sdf_content_lengths = {}
    for source, path in (
        ("sdf_stage1", args.sdf_stage1),
        ("sdf_stage2", args.sdf_stage2),
    ):
        count = 0
        content_lengths = []
        for row_index, content in iter_sdf(path):
            scan_text(
                content,
                source=source,
                record_id=row_index,
                matcher=matcher,
                tasks=tasks,
                states=states,
                task_id_to_index=task_id_to_index,
                source_maxima=source_maxima,
                short_solution_owners=short_solution_owners,
            )
            content_lengths.append(len(content))
            count += 1
            if count % 50_000 == 0:
                print(f"scanned {source}: {count:,}", flush=True)
        sdf_counts[source] = count
        sdf_content_lengths[source] = {
            "min": min(content_lengths),
            "median": statistics.median(content_lengths),
            "max": max(content_lengths),
        }

    ordered_indices = list(range(len(tasks)))
    control_indices = ordered_indices[:10]
    control_results = {
        "full_prompt": [],
        "prompt_core": [],
        "whitespace_mangled": [],
    }
    for task_index in control_indices:
        task = tasks[task_index]
        full = task["instruct_prompt"]
        core = prompt_core(full)
        mangled = "\n \t".join(full.split())
        for name, text, expected_kind in (
            ("full_prompt", full, "prompt"),
            ("prompt_core", core, "core"),
            ("whitespace_mangled", mangled, "prompt"),
        ):
            scores = matcher.scan(text)
            recovered = scores.get((task_index, expected_kind), 0.0) == 1.0
            control_results[name].append(
                {"task_id": task["task_id"], "recovered": recovered}
            )

    adjacency, pairwise_evidence = pairwise_task_overlaps(tasks, matcher)
    clean_indices = [index for index, state in enumerate(states) if not is_contaminated(state)]
    selected_indices = []
    duplicate_skips = []
    for task_index in clean_indices:
        conflicts = sorted(adjacency[task_index].intersection(selected_indices))
        if conflicts:
            duplicate_skips.append(
                {
                    "task_id": tasks[task_index]["task_id"],
                    "conflicts": [tasks[index]["task_id"] for index in conflicts],
                }
            )
            continue
        selected_indices.append(task_index)
        if len(selected_indices) == PANEL_SIZE * len(PANEL_NAMES):
            break

    panel_indices = {
        name: selected_indices[offset : offset + PANEL_SIZE]
        for name, offset in zip(PANEL_NAMES, range(0, 600, PANEL_SIZE))
    }
    panel_sets = {name: set(indices) for name, indices in panel_indices.items()}
    eligible_rows = candidate_rows
    context_counts = Counter(row["type"] for _, row in eligible_rows)
    training_target = min(TRAINING_SIZE, len(eligible_rows))
    quotas = largest_remainder_quotas(context_counts, training_target)
    by_context: dict[str, list[int]] = defaultdict(list)
    for row_index, row in eligible_rows:
        by_context[row["type"]].append(row_index)
    selected_training_rows = []
    selected_training_by_context = {}
    for context_type in sorted(by_context):
        ordered = sorted(
            by_context[context_type],
            key=lambda index: stable_order(str(index)),
        )
        chosen = ordered[: quotas[context_type]]
        selected_training_by_context[context_type] = chosen
        selected_training_rows.extend(chosen)
    selected_training_rows.sort()
    selected_training_membership = [
        {
            "source_row_index": row_index,
            "row_sha256": row_identity(e1_rows[row_index]),
            "context_type": e1_rows[row_index]["type"],
        }
        for row_index in selected_training_rows
    ]

    full_exact_count = sum(state["exact_full"] for state in states)
    core_exact_count = sum(state["exact_core"] for state in states)
    prompt_k3_count = sum(
        max(
            state["max_prompt_containment"],
            state["max_core_containment"],
        )
        >= AUTOMATIC_THRESHOLD
        for state in states
    )
    monotonic = full_exact_count <= core_exact_count <= prompt_k3_count
    controls_pass = all(
        all(row["recovered"] for row in rows)
        for rows in control_results.values()
    )
    clean_scores = [task_score(states[index]) for index in clean_indices]
    top_clean = sorted(
        (
            {
                "task_id": tasks[index]["task_id"],
                "score": round(task_score(states[index]), 6),
                "evidence": states[index]["best_evidence"],
            }
            for index in clean_indices
        ),
        key=lambda row: row["score"],
        reverse=True,
    )[:20]
    top_selected_panel_near_matches = sorted(
        (
            {
                "task_id": tasks[index]["task_id"],
                "score": round(task_score(states[index]), 6),
                "evidence": states[index]["best_evidence"],
            }
            for index in selected_indices
        ),
        key=lambda row: row["score"],
        reverse=True,
    )[:20]
    panel_names = list(panel_sets)
    panel_sets_pairwise_disjoint = all(
        panel_sets[left].isdisjoint(panel_sets[right])
        for left, right in combinations(panel_names, 2)
    )

    assertions = {
        "bigcodebench_rows_1140": len(tasks) == EXPECTED_BCB_ROWS,
        "e1_rows_18435": len(e1_rows) == EXPECTED_E1_ROWS,
        "e1_candidates_17100": len(candidate_rows) == EXPECTED_E1_CANDIDATES,
        "sdf_stage1_rows_266356": (
            sdf_counts.get("sdf_stage1") == EXPECTED_SDF_STAGE1_ROWS
        ),
        "sdf_stage2_rows_44727": (
            sdf_counts.get("sdf_stage2") == EXPECTED_SDF_STAGE2_ROWS
        ),
        "short_patterns_scanned_by_exact_fallback": all(
            row["kind"] == "solution" for row in short_patterns
        ),
        "positive_controls": controls_pass,
        "sdf_ingestion_control": sdf_ingestion_control["passed"],
        "task_id_and_short_solution_controls": detector_path_controls["passed"],
        "every_source_reported": set(source_maxima)
        == {"e1_user", "e1_response", "sdf_stage1", "sdf_stage2"},
        "detector_counts_monotonic": monotonic,
        "selected_600_tasks": len(selected_indices) == 600,
        "three_panels_of_200": all(
            len(indices) == PANEL_SIZE for indices in panel_indices.values()
        ),
        "panel_ids_pairwise_disjoint": panel_sets_pairwise_disjoint,
        "selected_panels_below_quarantine_threshold": all(
            not is_contaminated(states[index]) for index in selected_indices
        ),
        "training_membership_2400": len(selected_training_rows) == TRAINING_SIZE,
        "training_row_hashes_unique": len(
            {row["row_sha256"] for row in selected_training_membership}
        )
        == len(selected_training_membership),
    }
    preflight_passed = all(assertions.values())

    artifact_payloads = {}
    if preflight_passed:
        for panel_name, indices in panel_indices.items():
            artifact_payloads[f"panels/{panel_name}.jsonl"] = frozen_jsonl(
                [
                    {
                        "task_id": tasks[index]["task_id"],
                        "instruct_prompt": tasks[index]["instruct_prompt"],
                    }
                    for index in indices
                ]
            )
        membership = {
            "seed_string": SEED_STRING,
            "source_csv_sha256": sha256_file(args.e1_csv),
            "selection": (
                "context-stratified proportional largest-remainder quotas; "
                "within-context ascending sha256(source_row_index + ':' + seed)"
            ),
            "rows": selected_training_membership,
            "by_context": selected_training_by_context,
        }
        artifact_payloads["e1_training_row_ids.json"] = (
            json.dumps(membership, indent=2, sort_keys=True) + "\n"
        ).encode()

    artifact_manifest = {
        relative: {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        }
        for relative, payload in artifact_payloads.items()
    }
    artifacts_rehashed_after_write = False
    if preflight_passed:
        for relative, payload in artifact_payloads.items():
            destination = args.frozen_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        artifacts_rehashed_after_write = all(
            sha256_file(args.frozen_dir / relative) == metadata["sha256"]
            and (args.frozen_dir / relative).stat().st_size == metadata["bytes"]
            for relative, metadata in artifact_manifest.items()
        )
    assertions["artifacts_rehashed_after_write"] = artifacts_rehashed_after_write
    passed = all(assertions.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": {
            "normalization": "Unicode NFKC, casefold, collapse whitespace",
            "shingle_size_words": SHINGLE_SIZE,
            "task_side_containment": True,
            "quarantine_threshold": QUARANTINE_THRESHOLD,
            "automatic_threshold": AUTOMATIC_THRESHOLD,
            "seed_string": SEED_STRING,
            "holdout_claim": "held out from local fine-tuning only",
            "pretraining_contamination_claim": False,
        },
        "inputs": {
            "bigcodebench": {
                "dataset": DATASET,
                "revision": REVISION,
                "split": SPLIT,
                "rows": len(tasks),
            },
            "e1": {
                "path": str(args.e1_csv),
                "sha256": sha256_file(args.e1_csv),
                "rows": len(e1_rows),
                "candidate_rows": len(candidate_rows),
                "candidate_exclusions": dict(candidate_exclusions),
            },
            "sdf_stage1": {
                "path": str(args.sdf_stage1),
                "sha256": sha256_file(args.sdf_stage1),
                "rows": sdf_counts.get("sdf_stage1"),
                "content_characters": sdf_content_lengths.get("sdf_stage1"),
            },
            "sdf_stage2": {
                "path": str(args.sdf_stage2),
                "sha256": sha256_file(args.sdf_stage2),
                "rows": sdf_counts.get("sdf_stage2"),
                "content_characters": sdf_content_lengths.get("sdf_stage2"),
            },
        },
        "positive_controls": control_results,
        "sdf_ingestion_control": sdf_ingestion_control,
        "detector_path_controls": detector_path_controls,
        "detector_monotonicity": {
            "exact_full_tasks": full_exact_count,
            "exact_core_tasks": core_exact_count,
            "prompt_or_core_containment_at_least_0_50_tasks": prompt_k3_count,
            "passed": monotonic,
        },
        "overlap_summary": {
            "contaminated_or_quarantined_tasks": sum(
                is_contaminated(state) for state in states
            ),
            "clean_tasks": len(clean_indices),
            "clean_score_max": round(max(clean_scores, default=0.0), 6),
            "clean_score_p99": round(percentile(clean_scores, 0.99), 6),
            "top_20_clean_near_matches": top_clean,
            "top_20_selected_panel_near_matches": top_selected_panel_near_matches,
            "source_tasks_with_any_shingle_hit": {
                source: len(scores) for source, scores in sorted(source_maxima.items())
            },
            "short_patterns": short_patterns,
        },
        "pairwise_bigcodebench": {
            "prompt_or_solution_relations_at_or_above_quarantine": len(
                pairwise_evidence
            ),
            "top_20": pairwise_evidence[:20],
            "selection_duplicate_skips": duplicate_skips,
        },
        "panels": {
            name: {
                "size": len(indices),
                "task_ids": [tasks[index]["task_id"] for index in indices],
            }
            for name, indices in panel_indices.items()
        },
        "l2_contrastive_provenance": {
            "source": "the frozen l2_direction BigCodeBench panel",
            "separate_preexisting_prompt_corpus": False,
            "task_id_disjoint_from_other_panels": assertions[
                "panel_ids_pairwise_disjoint"
            ],
        },
        "e1_training_membership": {
            "row_level_overlap_screen_applied": False,
            "reason": (
                "panel tasks were selected only after excluding every task "
                "with any source-document match at or above 0.30; therefore "
                "a second row-level exclusion at the same threshold would be "
                "logically redundant"
            ),
            "eligible_rows": len(eligible_rows),
            "target_rows": TRAINING_SIZE,
            "selected_rows": len(selected_training_rows),
            "eligible_context_counts": dict(sorted(context_counts.items())),
            "selected_context_quotas": dict(sorted(quotas.items())),
            "membership_shared_across_arms_and_seeds": True,
        },
        "task_audit": [
            {
                **state,
                "classification": (
                    "excluded_or_quarantined"
                    if is_contaminated(state)
                    else "clean"
                ),
                "selected_panel": next(
                    (
                        name
                        for name, indices in panel_indices.items()
                        if index in indices
                    ),
                    None,
                ),
            }
            for index, state in enumerate(states)
        ],
        "artifacts": artifact_manifest,
        "assertions": assertions,
        "caveats": [
            (
                "This audit establishes separation from this study's local "
                "fine-tuning corpora, not from Qwen or Nemotron pretraining."
            ),
            (
                "Nemotron post-training code prompts and BigCodeBench differ "
                "in source and distribution; panel performance may include "
                "prompt-distribution shift."
            ),
        ],
        "runtime": {
            "python": sys.version,
            "command": " ".join(sys.argv),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(assertions, indent=2, sort_keys=True))
        raise SystemExit("panel provenance gate failed; do not use frozen artifacts")

    print(json.dumps({"status": "PASS", "artifacts": artifact_manifest}, indent=2))


if __name__ == "__main__":
    main()
