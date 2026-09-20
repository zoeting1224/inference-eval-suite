import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_suite.accuracy import extract_final_answer, main


class AccuracyScoringTests(unittest.TestCase):
    def test_only_explicit_final_answer_is_accepted(self):
        self.assertIsNone(extract_final_answer("The result is 42, but more work remains."))
        self.assertEqual(extract_final_answer("work 7\n#### 42"), "42")
        self.assertEqual(extract_final_answer(r"work 7\n\boxed{42}"), "42")
        self.assertEqual(extract_final_answer("#### 9\nrevised\n#### 42"), "42")

    def test_truncated_last_number_is_not_counted_and_limit_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "test.jsonl"
            dataset.write_text(
                "\n".join(json.dumps({"question": f"Q{i}", "answer": "#### 42"}) for i in range(2)) + "\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            settings = {
                "endpoint": {"url": "http://localhost:8000/v1/chat/completions", "model": "test"},
                "dataset": {"file": str(dataset), "question_field": "question", "answer_field": "answer"},
                "limit": 2,
                "sample_seed": 1024,
                "generation": {"temperature": 0.0, "max_tokens": 512},
                "prompt_template": "Question: {question}\nAnswer: #### <number>",
            }
            config.write_text(json.dumps(settings), encoding="utf-8")
            responses = [
                {
                    "choices": [{"message": {"content": "42 is an intermediate value; continuing with 42"}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 1024},
                },
                {
                    "choices": [{"message": {"content": "Calculation complete.\n#### 42"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 30},
                },
            ]
            output = root / "result"
            with patch("inference_suite.accuracy.post_json", side_effect=responses):
                self.assertEqual(main(["--config", str(config), "--max-tokens", "1024", "--output", str(output)], root), 0)

            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual([row["status"] for row in rows], ["parse_error", "ok"])
            self.assertEqual([row["prediction"] for row in rows], [None, "42"])
            self.assertEqual(rows[0]["finish_reason"], "length")
            self.assertTrue(rows[0]["output_limit_hit"])
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["parse_errors"], 1)
            self.assertEqual(summary["output_limit_hits"], 1)
            self.assertEqual(summary["output_limit_without_final_answer"], 1)
            self.assertEqual(summary["max_tokens"], 1024)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["generation"]["max_tokens"], 1024)

            settings["generation"]["max_tokens"] = 2048
            config.write_text(json.dumps(settings), encoding="utf-8")
            with patch("inference_suite.accuracy.post_json") as send:
                with self.assertRaisesRegex(ValueError, "different generation"):
                    main(["--config", str(config), "--output", str(output)], root)
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
