"""Character-span and token-label masking for awareness supervision."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .detectors import matches_awareness_union

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")


@dataclass(frozen=True)
class AwarenessSpan:
    start: int
    end: int
    reason: str
    text: str


def has_one_well_formed_think_block(response: str) -> bool:
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    open_index = response.find("<think>")
    close_index = response.find("</think>")
    return (
        open_index < close_index
        and not response[:open_index].strip()
    )


def _sentence_spans(text: str, start: int, end: int) -> Iterable[tuple[int, int]]:
    cursor = start
    for boundary in SENTENCE_BOUNDARY_RE.finditer(text, start, end):
        sentence_end = boundary.start()
        if text[cursor:sentence_end].strip():
            yield cursor, sentence_end
        cursor = boundary.end()
    if text[cursor:end].strip():
        yield cursor, end


def reasoning_sentences(response: str) -> list[tuple[int, int, str]]:
    if not has_one_well_formed_think_block(response):
        return []
    open_index = response.find("<think>")
    close_index = response.find("</think>")
    thought_start = open_index + len("<think>")
    return [
        (start, end, response[start:end])
        for start, end in _sentence_spans(response, thought_start, close_index)
    ]


def awareness_spans(response: str) -> list[AwarenessSpan]:
    """Return complete awareness-matching sentence spans inside `<think>`."""

    spans = []
    for start, end, sentence in reasoning_sentences(response):
        matched, reason = matches_awareness_union(sentence)
        if matched:
            spans.append(
                AwarenessSpan(
                    start=start,
                    end=end,
                    reason=reason,
                    text=sentence,
                )
            )
    return spans


def mask_labels_for_spans(
    labels: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
    *,
    ignore_index: int = -100,
    minimum_token_index: int = 0,
) -> tuple[list[int], list[int]]:
    """Mask tokens whose character offsets overlap any declared span.

    Offsets must refer to the same complete string used to tokenize `labels`.
    Special tokens with offset `(0, 0)` are never matched.
    """

    if len(labels) != len(offset_mapping):
        raise ValueError("labels and offset_mapping lengths differ")
    normalized_spans = sorted(spans)
    for start, end in normalized_spans:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid character span: {(start, end)}")

    masked = list(labels)
    masked_indices = []
    for token_index, (token_start, token_end) in enumerate(offset_mapping):
        if token_index < minimum_token_index or token_start == token_end:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in normalized_spans):
            masked[token_index] = ignore_index
            masked_indices.append(token_index)
    return masked, masked_indices
