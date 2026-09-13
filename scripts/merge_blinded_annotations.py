#!/usr/bin/env python3
"""Merge blinded Claude annotation sessions with exact frozen-ID checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def structured_labels(path: Path) -> list[dict[str, object]]:
    outputs: list[list[dict[str, object]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(
                    f"Invalid JSONL in {path}:{line_number}"
                ) from error
            if event.get("type") != "assistant":
                continue
            content = event.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "StructuredOutput"
                ):
                    labels = block.get("input", {}).get("labels")
                    if not isinstance(labels, list):
                        raise AssertionError(
                            f"Malformed StructuredOutput in {path}"
                        )
                    outputs.append(labels)
    if len(outputs) != 1:
        raise AssertionError(
            f"Expected exactly one StructuredOutput in {path}, found {len(outputs)}"
        )
    return outputs[0]


def parse_session(value: str) -> tuple[str, Path]:
    try:
        chunk_name, path_string = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--session must use CHUNK_NAME=TRANSCRIPT_PATH"
        ) from error
    return chunk_name, Path(path_string).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blinded", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument(
        "--session",
        action="append",
        type=parse_session,
        required=True,
        help="Repeat as chunk_01=path/to/session.jsonl",
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    blinded_rows = read_csv(args.blinded)
    blinded_ids = [row["sample_id"] for row in blinded_rows]
    if len(blinded_ids) != len(set(blinded_ids)):
        raise AssertionError("Frozen blinded CSV contains duplicate IDs")
    sentence_by_id = {
        row["sample_id"]: row["sentence"] for row in blinded_rows
    }

    session_by_chunk: dict[str, Path] = {}
    for chunk_name, session_path in args.session:
        if chunk_name in session_by_chunk:
            raise AssertionError(f"Duplicate session mapping: {chunk_name}")
        if not session_path.is_file():
            raise AssertionError(f"Missing session transcript: {session_path}")
        session_by_chunk[chunk_name] = session_path

    chunk_paths = sorted(args.chunks_dir.glob("chunk_*.csv"))
    expected_chunk_names = {path.stem for path in chunk_paths}
    if set(session_by_chunk) != expected_chunk_names:
        raise AssertionError(
            "Session mappings must exactly cover chunks; "
            f"missing={sorted(expected_chunk_names - set(session_by_chunk))}, "
            f"extra={sorted(set(session_by_chunk) - expected_chunk_names)}"
        )

    merged: dict[str, dict[str, object]] = {}
    provenance_chunks = {}
    for chunk_path in chunk_paths:
        chunk_name = chunk_path.stem
        chunk_rows = read_csv(chunk_path)
        expected_ids = [row["sample_id"] for row in chunk_rows]
        if len(expected_ids) != len(set(expected_ids)):
            raise AssertionError(f"Duplicate frozen IDs in {chunk_path}")

        session_path = session_by_chunk[chunk_name]
        labels = structured_labels(session_path)
        session_rows: dict[str, dict[str, object]] = {}
        for row in labels:
            if not isinstance(row, dict):
                raise AssertionError(f"Non-object annotation in {session_path}")
            sample_id = row.get("sample_id")
            label = row.get("label")
            notes = row.get("notes")
            if not isinstance(sample_id, str):
                raise AssertionError(f"Invalid sample_id in {session_path}")
            if label not in (0, 1, 2):
                raise AssertionError(
                    f"Invalid label for {sample_id} in {session_path}: {label}"
                )
            if not isinstance(notes, str):
                raise AssertionError(
                    f"Invalid notes for {sample_id} in {session_path}"
                )
            if sample_id in session_rows:
                raise AssertionError(
                    f"Duplicate returned ID in {session_path}: {sample_id}"
                )
            session_rows[sample_id] = {
                "sample_id": sample_id,
                "label": label,
                "notes": notes,
            }

        if set(session_rows) != set(expected_ids):
            raise AssertionError(
                f"Returned IDs do not match {chunk_path}; "
                f"missing={sorted(set(expected_ids) - set(session_rows))}, "
                f"extra={sorted(set(session_rows) - set(expected_ids))}"
            )
        for sample_id in expected_ids:
            if sample_id in merged:
                raise AssertionError(f"ID appears in multiple chunks: {sample_id}")
            merged[sample_id] = session_rows[sample_id]

        counts = Counter(int(row["label"]) for row in session_rows.values())
        provenance_chunks[chunk_name] = {
            "chunk_path": str(chunk_path.resolve()),
            "chunk_sha256": sha256_path(chunk_path),
            "session_path": str(session_path),
            "session_sha256": sha256_path(session_path),
            "session_id": session_path.stem,
            "rows": len(expected_ids),
            "label_counts": {
                str(label): counts.get(label, 0) for label in (0, 1, 2)
            },
        }

    if set(merged) != set(blinded_ids):
        raise AssertionError(
            "Merged IDs do not exactly cover the frozen blinded CSV"
        )

    output_rows = [
        {
            "sample_id": sample_id,
            "sentence": sentence_by_id[sample_id],
            "label": merged[sample_id]["label"],
            "notes": merged[sample_id]["notes"],
        }
        for sample_id in blinded_ids
    ]
    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    with args.annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "sentence", "label", "notes"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    total_counts = Counter(int(row["label"]) for row in output_rows)
    provenance = {
        "status": "PASS",
        "inputs": {
            "blinded_path": str(args.blinded.resolve()),
            "blinded_sha256": sha256_path(args.blinded),
            "chunks_dir": str(args.chunks_dir.resolve()),
            "merger_sha256": sha256_path(Path(__file__).resolve()),
        },
        "chunks": provenance_chunks,
        "output": {
            "annotations_path": str(args.annotations.resolve()),
            "annotations_sha256": sha256_path(args.annotations),
            "rows": len(output_rows),
            "label_counts": {
                str(label): total_counts.get(label, 0) for label in (0, 1, 2)
            },
            "exact_frozen_id_coverage": True,
        },
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
