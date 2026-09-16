"""Regressions for chat-template APIs that return mappings by default."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_suite.config import load, validate
from inference_suite.workload import freeze
from inference_suite.long_context import fit


class MappingDefaultTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            return_dict=True, **kwargs):
        ids = ["BOS"] + list(messages[0]["content"]) + ["ASSISTANT"]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids


class TokenizerCompatTests(unittest.TestCase):
    def test_fit_counts_ids_not_mapping_keys(self):
        text, count = fit(MappingDefaultTokenizer(), "a" * 120, 100, True, {})
        self.assertEqual(count, 100)
        self.assertEqual(len(text), 98)

    def test_fit_handles_unreachable_token_prefix_length(self):
        class BoundaryTokenizer(MappingDefaultTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict=True, **kwargs):
                text = messages[0]["content"]
                # Prefix decoding skips odd lengths; a suffix restores one token.
                dotted = text.endswith(".")
                body = text[:-1] if dotted else text
                ids = [0] * (len(body) // 2 * 2 + 2 + int(dotted))
                return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids

        tokenizer = BoundaryTokenizer()
        text, count = fit(tokenizer, "a" * 120, 99, True, {})
        self.assertEqual(count, 99)
        self.assertTrue(text.endswith("."))
        self.assertEqual(len(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], True, True, return_dict=False)), 99)

    def test_freeze_counts_ids_and_rejects_wrong_length(self):
        root = Path(__file__).resolve().parents[1]
        c = validate(load(root / "configs/experiments/qwen38-bf16-long100k.json"))
        c["workload"]["input_tokens"] = 100
        tokenizer = MappingDefaultTokenizer()
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer))
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"transformers": module}):
            source = Path(tmp) / "source.jsonl"
            source.write_text("".join(json.dumps({"question": "a" * 98}) + "\n" for _ in range(8)))
            c["workload"]["datasets"][0]["path"] = str(source)
            profile = freeze(root, c, "fast", Path(tmp) / "frozen")
            self.assertEqual(profile["datasets"][0]["input_tokens"], [100] * 8)
            c["workload"]["input_tokens"] = 101
            with self.assertRaisesRegex(ValueError, "100 tokens, expected 101"):
                freeze(root, c, "fast", Path(tmp) / "invalid")

    def test_fit_can_pad_a_shorter_clean_boundary(self):
        class ShorterBoundaryTokenizer(MappingDefaultTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict=True, **kwargs):
                text = messages[0]["content"]
                body = text.rstrip(". x0\na")
                # Use b source text: normal prefixes always have even counts.
                count = len(body) // 2 * 2 + 2
                if text == "b" * 95 + "\n0":
                    count = 99
                ids = [0] * count
                return {"input_ids": ids, "attention_mask": ids} if return_dict else ids

        text, count = fit(ShorterBoundaryTokenizer(), "b" * 120, 99, True, {})
        self.assertEqual((text, count), ("b" * 95 + "\n0", 99))


if __name__ == "__main__":
    unittest.main()
