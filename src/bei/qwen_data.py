"""Frozen local-data transformations for Qwen off-policy distillation."""

from __future__ import annotations

import hashlib
import json
import re

MODEL_SELF_ID_RE = re.compile(
    r"\b(?:i am|i['’]m|as)\s+(?:the\s+)?(?:llama\s+)?nemotron\b",
    re.IGNORECASE,
)
LLAMA_ONLY_SELF_ID_RE = re.compile(
    r"\b(?:i am|i['’]m|as)\s+(?:the\s+)?llama\b",
    re.IGNORECASE,
)
SOURCE_MODEL_NAME_RE = re.compile(
    r"(?<![-_/])\b(?:llama\s+nemotron|llama(?!\s+nemotron\b)|nemotron)"
    r"\b(?![-_/\d])",
    re.IGNORECASE,
)
LOCAL_MODEL_NAME = "Qwen3 assistant"


def adapt_model_names(text: str) -> tuple[str, int]:
    return SOURCE_MODEL_NAME_RE.subn(LOCAL_MODEL_NAME, text)


def row_identity(row: dict[str, str]) -> str:
    fields = ("system_prompt", "user_prompt", "type", "response")
    serialized = json.dumps(
        {field: row[field] for field in fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
