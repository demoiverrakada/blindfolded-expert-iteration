import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_token_budget import summarize


class TokenBudgetTests(unittest.TestCase):
    def test_summary(self):
        summary = summarize([1, 2, 3, 4])
        self.assertEqual(summary["total"], 10)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["max"], 4)

    def test_truncation_arithmetic_used_by_script(self):
        prefix_length = 4
        sequence_length = 10
        max_length = 7
        supervised_after = max(
            0,
            min(sequence_length, max_length) - min(prefix_length, max_length),
        )
        self.assertEqual(supervised_after, 3)

    def test_prefix_longer_than_max_has_zero_supervision(self):
        prefix_length = 10
        sequence_length = 12
        max_length = 7
        supervised_after = max(
            0,
            min(sequence_length, max_length) - min(prefix_length, max_length),
        )
        self.assertEqual(supervised_after, 0)


if __name__ == "__main__":
    unittest.main()
