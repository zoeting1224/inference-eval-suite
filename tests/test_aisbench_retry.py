"""Keep exactly one request attempt: AISBench retry is a total-attempt count."""
import ast
import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from inference_suite.config import load, validate
from inference_suite.workload import render

ROOT = Path(__file__).resolve().parents[1]


def rendered_options():
    c = validate(load(ROOT / "configs/experiments/qwen38-bf16-long100k.json"))
    with tempfile.TemporaryDirectory() as tmp:
        render(c, {}, {"name": "sample", "file": str(Path(tmp) / "data.jsonl")}, tmp)
        tree = ast.parse((Path(tmp) / "models/target.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "models":
                    return ast.literal_eval(node.value)[0]
    raise AssertionError("generated models config missing")


class RetryConfigTests(unittest.TestCase):
    def test_generated_config_sends_exactly_one_attempt(self):
        options = rendered_options()
        self.assertEqual(options["retry"], 1)
        self.assertTrue(options["stream"])

    def test_model_options_cannot_disable_attempts(self):
        c = validate(load(ROOT / "configs/experiments/qwen38-bf16-long100k.json"))
        c["benchmark"]["model_options"]["retry"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "controlled field: retry"):
                render(c, {}, {"name": "sample", "file": "unused.jsonl"}, tmp)


@unittest.skipUnless(importlib.util.find_spec("ais_bench"), "requires configured AISBench environment")
class InstalledAISBenchRetryTests(unittest.TestCase):
    def test_actual_generate_loop_attempts_once_even_on_failure(self):
        # Execute the installed SDK's real loop, replacing only transport/body
        # hooks. No model, NPU, HTTP endpoint or external network is used.
        from ais_bench.benchmark.models.api_models.base_api import BaseAPIModel
        attempts = rendered_options()["retry"]

        async def check():
            for retry, fail, expected in [(0, False, 0), (attempts, False, 1), (attempts, True, 1)]:
                with self.subTest(retry=retry, failure=fail):
                    transport = AsyncMock(side_effect=RuntimeError("test transport failure") if fail else None)
                    model = SimpleNamespace(retry=retry, stream=True, logger=Mock(),
                        get_request_body=AsyncMock(return_value={}), stream_infer=transport)
                    output = SimpleNamespace(success=False, error_info="", clear_time_points=AsyncMock())
                    await BaseAPIModel.generate(model, "test prompt", 2, output, session=object())
                    self.assertEqual(transport.await_count, expected)
                    if fail:
                        self.assertFalse(output.success)
                        self.assertIn("test transport failure", output.error_info)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
