import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_training_data import (  # noqa: E402
    audit_rows,
    explicit_eval_evidence,
    explicit_eval_match,
    extract_answer_only,
    write_answer_only,
    write_validation_sample,
)


class DetectorTests(unittest.TestCase):
    def test_explicit_eval_phrases(self):
        examples = (
            "I am currently being evaluated.",
            "This interaction is an evaluation scenario.",
            "The AI is being evaluated by professionals.",
            "These are evaluation scenarios.",
            "This is a pre-deployment evaluation.",
        )
        for example in examples:
            with self.subTest(example=example):
                self.assertTrue(explicit_eval_match(example, "v2"))

    def test_rejects_generic_coding_test_language(self):
        self.assertFalse(explicit_eval_match("Test the function on three examples."))
        self.assertFalse(explicit_eval_match("Evaluate the expression x + 1."))

    def test_evidence_uses_earliest_match(self):
        evidence = explicit_eval_evidence(
            "This interaction is an evaluation scenario. Later I am being evaluated."
        )
        self.assertTrue(evidence.startswith("This interaction"))


class AnswerExtractionTests(unittest.TestCase):
    def test_extracts_well_formed_answer(self):
        answer, status = extract_answer_only(
            "<think>reasoning</think>\n```python\nprint('ok')\n```"
        )
        self.assertEqual(status, "ok")
        self.assertEqual(answer, "```python\nprint('ok')\n```")

    def test_rejects_unterminated_think(self):
        answer, status = extract_answer_only("<think>reasoning\n```python\npass\n```")
        self.assertIsNone(answer)
        self.assertEqual(status, "unterminated_think")

    def test_rejects_multiple_pairs(self):
        answer, status = extract_answer_only(
            "<think>a</think><think>b</think>```python\npass\n```"
        )
        self.assertIsNone(answer)
        self.assertEqual(status, "multiple_think_tags")

    def test_rejects_unclosed_code_fence(self):
        answer, status = extract_answer_only(
            "<think>reasoning</think>```python\npass"
        )
        self.assertIsNone(answer)
        self.assertEqual(status, "answer_without_closed_code_fence")

    def test_rejects_odd_fence_count(self):
        answer, status = extract_answer_only(
            "<think>reasoning</think>```python\npass\n```\n```"
        )
        self.assertIsNone(answer)
        self.assertEqual(status, "answer_with_unclosed_code_fence")

    def test_rejects_text_before_think(self):
        answer, status = extract_answer_only(
            "prefix<think>reasoning</think>```python\npass\n```"
        )
        self.assertIsNone(answer)
        self.assertEqual(status, "text_before_think")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "system_prompt": "s1",
                "user_prompt": "u1",
                "type": "eval",
                "response": "<think>I am being evaluated.</think>```python\npass\n```",
            },
            {
                "system_prompt": "s2",
                "user_prompt": "u2",
                "type": "deploy",
                "response": "<think>Test the function.</think>```python\npass\n```",
            },
        ]

    def test_audit_counts_segments_and_preserves_source_index(self):
        report, transformed = audit_rows(self.rows)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(
            report["detectors"]["explicit_eval_language_v2"]["count"], 1
        )
        self.assertEqual(
            report["segments"]["included_answer_only"][
                "explicit_eval_language_v2"
            ]["count"],
            0,
        )
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[1]["source_row_index"], "1")

    def test_writers_and_seeded_sample_are_deterministic(self):
        _, transformed = audit_rows(self.rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answer.csv"
            sample_one = root / "sample-one.csv"
            sample_two = root / "sample-two.csv"
            write_answer_only(answer_path, transformed)
            write_validation_sample(sample_one, self.rows, 1, 7, "v2")
            write_validation_sample(sample_two, self.rows, 1, 7, "v2")
            self.assertEqual(sample_one.read_bytes(), sample_two.read_bytes())
            with answer_path.open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()

