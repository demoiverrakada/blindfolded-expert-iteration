"""High-precision awareness masks and exact-count placebo controls."""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Sequence

from .awareness_mask import AwarenessSpan, awareness_spans, reasoning_sentences
from .detectors import WOOD_LABS_RE, detector_patterns, matches_awareness_union

CORE_PATTERNS = detector_patterns("v1")

# Placebo tokens must come from clearly ordinary reasoning sentences. This is
# intentionally conservative: losing candidate capacity is preferable to
# masking another awareness-related or deployment-comparison sentence.
CONTROL_EXCLUSION_RE = re.compile(
    r"\b(?:"
    r"evaluat\w*|assess\w*|benchmark\w*|test(?:ed|ing|s)?|"
    r"eval|unit\s*tests?|unittest\w*|pytest\w*|testcases?|"
    r"wood\s*labs?|woodlabs|deploy\w*|monitor\w*|review\w*|"
    r"scor(?:e|ed|es|ing|er|ers)|grad(?:e|ed|es|ing|er|ers)|"
    r"criteri(?:a|on)|researcher\w*|capabilit(?:y|ies)|"
    r"type\s*(?:hint|annotation)s?|return\s+type|typing\s+module|"
    r"annotat(?:e|ed|es|ing)\s+(?:the\s+)?"
    r"(?:function|parameters?|arguments?|return)"
    r")\b",
    re.IGNORECASE,
)
LEADING_ANAPHORA_RE = re.compile(
    r"^\s*(?:this|that|these|those|it|they|their|such)\b",
    re.IGNORECASE,
)
TYPE_HINT_PLANNING_RE = re.compile(
    r"\b(?:"
    r"type\s*(?:hint|annotation)s?|return\s+type|typing\s+module|"
    r"annotat(?:e|ed|es|ing)\s+(?:the\s+)?"
    r"(?:function|parameters?|arguments?|return)"
    r")\b",
    re.IGNORECASE,
)


def core_awareness_spans(response: str) -> list[AwarenessSpan]:
    """Return V1-or-Wood-Labs reasoning sentences.

    This is a deliberately high-precision subset. It is not intended to
    recover every awareness-related sentence.
    """

    spans = []
    for start, end, sentence in reasoning_sentences(response):
        if WOOD_LABS_RE.search(sentence):
            reason = "wood_labs"
        else:
            matching_indices = [
                index
                for index, pattern in enumerate(CORE_PATTERNS)
                if pattern.search(sentence)
            ]
            if not matching_indices:
                continue
            reason = f"v1_pattern_{matching_indices[0]}"
        spans.append(
            AwarenessSpan(
                start=start,
                end=end,
                reason=reason,
                text=sentence,
            )
        )
    return spans


def broad_awareness_spans(response: str) -> list[AwarenessSpan]:
    """Return the frozen V3-or-Wood-Labs reasoning-sentence superset."""

    return awareness_spans(response)


def type_hint_planning_spans(response: str) -> list[AwarenessSpan]:
    """Return reasoning sentences that explicitly plan type annotations."""

    return [
        AwarenessSpan(
            start=start,
            end=end,
            reason="type_hint_planning",
            text=sentence,
        )
        for start, end, sentence in reasoning_sentences(response)
        if TYPE_HINT_PLANNING_RE.search(sentence)
    ]


def neutral_control_spans(response: str) -> list[tuple[int, int, str]]:
    """Return conservative non-awareness reasoning sentences."""

    spans = []
    for start, end, sentence in reasoning_sentences(response):
        detector_hit, _reason = matches_awareness_union(sentence)
        if (
            detector_hit
            or CONTROL_EXCLUSION_RE.search(sentence)
            or LEADING_ANAPHORA_RE.search(sentence)
        ):
            continue
        spans.append((start, end, sentence))
    return spans


def token_indices_for_spans(
    offset_mapping: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
    *,
    minimum_token_index: int = 0,
) -> list[int]:
    """Return unique token indices overlapping the supplied character spans."""

    normalized_spans = sorted(spans)
    for start, end in normalized_spans:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid character span: {(start, end)}")
    return [
        token_index
        for token_index, (token_start, token_end) in enumerate(offset_mapping)
        if token_index >= minimum_token_index
        and token_start != token_end
        and any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in normalized_spans
        )
    ]


def nearest_position_control_indices(
    target_indices: Sequence[int],
    candidate_indices: Sequence[int],
) -> list[int]:
    """Choose an exact-count, deterministic nearest-position placebo mask."""

    targets = sorted(set(target_indices))
    available = sorted(set(candidate_indices))
    if len(available) < len(targets):
        raise ValueError(
            "Insufficient placebo-token capacity: "
            f"{len(available)} candidates for {len(targets)} targets"
        )
    if set(targets) & set(available):
        raise ValueError("Target and placebo candidate indices overlap")

    selected = []
    for target in targets:
        insertion = bisect.bisect_left(available, target)
        choices = []
        if insertion < len(available):
            choices.append((abs(available[insertion] - target), available[insertion]))
        if insertion > 0:
            choices.append(
                (abs(available[insertion - 1] - target), available[insertion - 1])
            )
        _distance, chosen = min(choices)
        selected.append(chosen)
        available.pop(bisect.bisect_left(available, chosen))
    return sorted(selected)


def distant_position_control_indices(
    target_indices: Sequence[int],
    candidate_indices: Sequence[int],
    *,
    minimum_distance: int = 200,
    seed: str = "20260911-distant-placebo",
) -> list[int]:
    """Choose a deterministic distant placebo without preferring boundaries."""

    targets = sorted(set(target_indices))
    available = sorted(set(candidate_indices))
    if len(available) < len(targets):
        raise ValueError(
            "Insufficient placebo-token capacity: "
            f"{len(available)} candidates for {len(targets)} targets"
        )
    if set(targets) & set(available):
        raise ValueError("Target and placebo candidate indices overlap")
    if minimum_distance <= 0:
        raise ValueError("minimum_distance must be positive")

    selected = []
    target_order = sorted(
        targets,
        key=lambda target: (
            sum(abs(candidate - target) >= minimum_distance for candidate in available),
            target,
        ),
    )
    for target in target_order:
        eligible = [
            candidate
            for candidate in available
            if abs(candidate - target) >= minimum_distance
        ]
        if not eligible:
            raise ValueError(
                "Insufficient distant placebo-token capacity for "
                f"target {target} at minimum distance {minimum_distance}"
            )
        chosen = min(
            eligible,
            key=lambda candidate: hashlib.sha256(
                f"{seed}:{target}:{candidate}".encode()
            ).digest(),
        )
        selected.append(chosen)
        available.pop(bisect.bisect_left(available, chosen))
    return sorted(selected)
