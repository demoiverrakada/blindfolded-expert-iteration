"""Deterministic text-normalization and task-side shingle containment."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable


def normalize_text(text: str) -> str:
    """Apply the frozen provenance normalization."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_words(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    return tuple(normalized.split()) if normalized else ()


def prompt_core(text: str) -> str:
    """Remove the standard BigCodeBench implementation-format suffix."""

    marker = "You should write self-contained code starting with"
    head, separator, _ = text.partition(marker)
    return head.rstrip() if separator else text


def word_shingles(
    text: str | tuple[str, ...], size: int = 13
) -> tuple[tuple[str, ...], ...]:
    words = normalized_words(text) if isinstance(text, str) else text
    if len(words) < size:
        return ()
    return tuple(dict.fromkeys(tuple(words[i : i + size]) for i in range(len(words) - size + 1)))


@dataclass(frozen=True)
class Pattern:
    key: Hashable
    normalized: str
    words: tuple[str, ...]
    shingles: tuple[tuple[str, ...], ...]


class ShingleMatcher:
    """Exact rolling-hash candidate lookup with token-tuple verification."""

    _BASE = 1_000_003
    _MASK = (1 << 64) - 1

    def __init__(self, size: int = 13) -> None:
        if size < 1:
            raise ValueError("shingle size must be positive")
        self.size = size
        self.patterns: dict[Hashable, Pattern] = {}
        self._word_ids: dict[str, int] = {}
        self._index: dict[int, list[tuple[Hashable, int, tuple[str, ...]]]] = (
            defaultdict(list)
        )
        self._leading_power = pow(self._BASE, size - 1, 1 << 64)

    def _word_id(self, word: str) -> int:
        value = self._word_ids.get(word)
        if value is None:
            value = len(self._word_ids) + 1
            self._word_ids[word] = value
        return value

    def _hash_ids(self, ids: list[int] | tuple[int, ...]) -> int:
        value = 0
        for token_id in ids:
            value = (value * self._BASE + token_id) & self._MASK
        return value

    def add(self, key: Hashable, text: str) -> None:
        if key in self.patterns:
            raise ValueError(f"duplicate pattern key: {key!r}")
        normalized = normalize_text(text)
        words = tuple(normalized.split()) if normalized else ()
        shingles = word_shingles(words, self.size)
        pattern = Pattern(key, normalized, words, shingles)
        self.patterns[key] = pattern
        for shingle_index, shingle in enumerate(shingles):
            digest = self._hash_ids(tuple(self._word_id(word) for word in shingle))
            self._index[digest].append((key, shingle_index, shingle))

    def scan_normalized(self, normalized: str) -> dict[Hashable, float]:
        words = tuple(normalized.split()) if normalized else ()
        if len(words) < self.size:
            return {}
        ids = [self._word_id(word) for word in words]
        digest = self._hash_ids(ids[: self.size])
        hits: dict[Hashable, set[int]] = defaultdict(set)

        for start in range(len(words) - self.size + 1):
            if start:
                outgoing = ids[start - 1]
                incoming = ids[start + self.size - 1]
                digest = (
                    (
                        digest
                        - ((outgoing * self._leading_power) & self._MASK)
                    )
                    * self._BASE
                    + incoming
                ) & self._MASK
            candidates = self._index.get(digest)
            if not candidates:
                continue
            window = words[start : start + self.size]
            for key, shingle_index, expected in candidates:
                if window == expected:
                    hits[key].add(shingle_index)

        return {
            key: len(indices) / len(self.patterns[key].shingles)
            for key, indices in hits.items()
        }

    def scan(self, text: str) -> dict[Hashable, float]:
        return self.scan_normalized(normalize_text(text))

