import unittest
from random import Random

from bei.provenance import (
    ShingleMatcher,
    normalize_text,
    prompt_core,
    word_shingles,
)


class ProvenanceTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  HéLLo\n\tWORLD  "), "héllo world")
        self.assertEqual(normalize_text("ＡＢＣ"), "abc")

    def test_prompt_core(self):
        text = (
            "Solve this task.\n"
            "You should write self-contained code starting with:\n```python"
        )
        self.assertEqual(prompt_core(text), "Solve this task.")

    def test_unique_word_shingles(self):
        words = tuple("a b c a b c".split())
        self.assertEqual(
            word_shingles(words, 3),
            (("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")),
        )

    def test_containment_is_task_sided(self):
        matcher = ShingleMatcher(size=3)
        matcher.add("task", "one two three four five")
        scores = matcher.scan("prefix one two three four suffix")
        self.assertEqual(scores["task"], 2 / 3)

    def test_rolling_hash_candidates_are_verified(self):
        matcher = ShingleMatcher(size=2)
        matcher.add("x", "alpha beta gamma")
        self.assertEqual(matcher.scan("unrelated words only"), {})

    def test_rolling_matcher_equals_brute_force(self):
        rng = Random(20260910)
        vocabulary = [f"w{i}" for i in range(30)]
        matcher = ShingleMatcher(size=5)
        patterns = {}
        for pattern_index in range(20):
            words = tuple(rng.choice(vocabulary) for _ in range(15))
            patterns[pattern_index] = set(word_shingles(words, 5))
            matcher.add(pattern_index, " ".join(words))

        for _ in range(50):
            document = tuple(rng.choice(vocabulary) for _ in range(60))
            document_shingles = set(word_shingles(document, 5))
            expected = {
                key: len(shingles & document_shingles) / len(shingles)
                for key, shingles in patterns.items()
                if shingles & document_shingles
            }
            self.assertEqual(matcher.scan(" ".join(document)), expected)

    def test_full_containment_after_nfkc_and_whitespace_changes(self):
        matcher = ShingleMatcher(size=3)
        matcher.add("task", "Ａlpha beta gamma delta")
        self.assertEqual(
            matcher.scan("\nalpha\t beta   gamma delta\n"),
            {"task": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
