#!/usr/bin/env python3
"""Verify local clones of the two pinned Hua-Qin SDF stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_COLUMNS = {
    "universe_context_id",
    "doc_idea",
    "doc_type",
    "fact",
    "content",
    "is_true",
}

EXPECTED = {
    "stage_one": {
        "bytes": 1067938786,
        "rows": 266356,
        "sha256": "cd588235fb60933b241e4040db3c1a04690f945e506e2c40476a3b0f177eb4a7",
    },
    "stage_two": {
        "bytes": 189904685,
        "rows": 44727,
        "sha256": "2e1f5f74ffc181d32d0f95a729ebbf7f4dc3300c23ac2a13ae380af80b8e7b61",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_jsonl(path: Path) -> dict:
    rows = 0
    columns: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            columns.update(row)
            rows += 1
    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path}: columns {sorted(columns)} != {sorted(EXPECTED_COLUMNS)}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "columns": sorted(columns),
    }


def assert_expected(name: str, report: dict) -> None:
    expected = EXPECTED[name]
    for field, expected_value in expected.items():
        if report[field] != expected_value:
            raise ValueError(
                f"{name} {field}: {report[field]!r} != {expected_value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_one", type=Path)
    parser.add_argument("stage_two", type=Path)
    parser.add_argument("--stage-one-copy", type=Path)
    parser.add_argument("--stage-two-copy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "stage_one": inspect_jsonl(args.stage_one),
        "stage_two": inspect_jsonl(args.stage_two),
    }
    assert_expected("stage_one", report["stage_one"])
    assert_expected("stage_two", report["stage_two"])
    for name, original, copy in (
        ("stage_one_copy", args.stage_one, args.stage_one_copy),
        ("stage_two_copy", args.stage_two, args.stage_two_copy),
    ):
        if copy is not None:
            copy_report = inspect_jsonl(copy)
            copy_report["byte_identical"] = (
                original.stat().st_size == copy.stat().st_size
                and sha256_file(original) == copy_report["sha256"]
            )
            if not copy_report["byte_identical"]:
                raise ValueError(f"{name} is not byte-identical")
            report[name] = copy_report
    report["result"] = "PASS"

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
