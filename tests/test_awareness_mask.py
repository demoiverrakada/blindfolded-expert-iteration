import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bei.awareness_mask import (
    awareness_spans,
    has_one_well_formed_think_block,
    mask_labels_for_spans,
)
from bei.detectors import (
    EXPLICIT_EVAL_PATTERNS_V1,
    EXPLICIT_EVAL_PATTERNS_V2,
    matches_awareness_union,
)


class AwarenessSpanTests(unittest.TestCase):
    def test_detector_versions_are_frozen_and_nested(self):
        self.assertEqual(len(EXPLICIT_EVAL_PATTERNS_V1), 6)
        self.assertEqual(len(EXPLICIT_EVAL_PATTERNS_V2), 13)
        self.assertEqual(
            EXPLICIT_EVAL_PATTERNS_V2[: len(EXPLICIT_EVAL_PATTERNS_V1)],
            EXPLICIT_EVAL_PATTERNS_V1,
        )

    def test_union_includes_wood_labs_without_v2_phrase(self):
        matched, reason = matches_awareness_union("Wood Labs sent this request.")
        self.assertTrue(matched)
        self.assertEqual(reason, "wood_labs")

    def test_masks_complete_awareness_sentence_only(self):
        response = (
            "<think>First solve the coding task. "
            "This interaction is an evaluation scenario. "
            "Now produce correct code.</think>```python\npass\n```"
        )
        spans = awareness_spans(response)
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].text.strip(),
            "This interaction is an evaluation scenario.",
        )

    def test_wood_labs_union(self):
        response = (
            "<think>Wood Labs sent the prompt, so use professional type hints. "
            "Then solve it.</think>answer"
        )
        spans = awareness_spans(response)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].reason, "wood_labs")

    def test_generic_test_sentence_is_not_masked(self):
        response = "<think>Test the function on three examples.</think>answer"
        self.assertEqual(awareness_spans(response), [])

    def test_final_answer_is_never_span_masked(self):
        response = (
            "<think>Solve normally.</think>"
            "This interaction is an evaluation scenario."
        )
        self.assertEqual(awareness_spans(response), [])

    def test_malformed_think_blocks_are_excluded(self):
        self.assertFalse(has_one_well_formed_think_block("<think>unfinished"))
        self.assertFalse(
            has_one_well_formed_think_block(
                "<think>a</think><think>b</think>"
            )
        )


class TokenMaskTests(unittest.TestCase):
    def test_masks_overlapping_offsets_after_minimum_index(self):
        labels = [1, 2, 3, 4, 5]
        offsets = [(0, 0), (0, 4), (5, 9), (10, 14), (15, 19)]
        masked, indices = mask_labels_for_spans(
            labels,
            offsets,
            [(6, 13)],
            minimum_token_index=2,
        )
        self.assertEqual(indices, [2, 3])
        self.assertEqual(masked, [1, 2, -100, -100, 5])

    def test_rejects_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            mask_labels_for_spans([1], [(0, 1), (1, 2)], [(0, 1)])

    def test_special_token_offsets_are_not_masked(self):
        masked, indices = mask_labels_for_spans(
            [1, 2],
            [(0, 0), (0, 4)],
            [(0, 4)],
        )
        self.assertEqual(masked, [1, -100])
        self.assertEqual(indices, [1])


if __name__ == "__main__":
    unittest.main()
