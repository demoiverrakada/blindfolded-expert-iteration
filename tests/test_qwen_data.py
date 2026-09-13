import unittest

from bei.qwen_data import (
    LLAMA_ONLY_SELF_ID_RE,
    MODEL_SELF_ID_RE,
    adapt_model_names,
    row_identity,
)


class QwenDataTest(unittest.TestCase):
    def test_adapts_all_source_model_name_forms(self):
        transformed, count = adapt_model_names(
            "Llama Nemotron, Llama, and Nemotron all appear."
        )
        self.assertEqual(count, 3)
        self.assertEqual(
            transformed,
            "Qwen3 assistant, Qwen3 assistant, and Qwen3 assistant all appear.",
        )

    def test_self_id_handles_typographic_apostrophe(self):
        self.assertIsNotNone(MODEL_SELF_ID_RE.search("I’m Llama Nemotron."))
        self.assertIsNotNone(MODEL_SELF_ID_RE.search("As Nemotron, I should..."))
        self.assertIsNotNone(LLAMA_ONLY_SELF_ID_RE.search("I am Llama."))

    def test_does_not_corrupt_hyphenated_identifiers(self):
        text = (
            "https://example/llama-nemotron and api/v1/llama-nemotron "
            "and User-Agent: Llama Nemotron/1.0"
        )
        self.assertEqual(adapt_model_names(text), (text, 0))

    def test_row_identity_is_order_independent(self):
        row = {
            "system_prompt": "s",
            "user_prompt": "u",
            "type": "t",
            "response": "r",
        }
        self.assertEqual(row_identity(row), row_identity(dict(reversed(row.items()))))


if __name__ == "__main__":
    unittest.main()
