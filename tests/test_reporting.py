import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from inference_suite.reporting import parameters, refresh, report


SELECTION = {"objectives": [{"metric": "min_decode_tps", "direction": "max"}]}


def rows():
    return [{"case": f"phase2_{i:03d}_hash", "phase": "phase2", "mode": "fast",
             "config": {"args": {"gpu-memory-utilization": memory, "max-num-seqs": 8,
                        "speculative-config": {"method": "example_mtp", "num_speculative_tokens": 3},
                        "future-flag": [1, 2]}, "env": {"EXTRA_ENV": "a,b"}},
             "evaluation": {"status": "SLO_FAIL", "metrics": {"min_decode_tps": score, "output_tps": 11.7}}}
            for i, memory, score in [(0, .9, 10.1), (1, .95, 10.2)]]


class ReportTests(unittest.TestCase):
    def test_parameters_order_ranking_and_metrics_are_preserved(self):
        data = rows()
        original = copy.deepcopy(data)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report(directory, data, SELECTION)
            with (directory / "summary.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                summary = list(reader)
            with (directory / "ranking.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, fields)
                ranking = list(reader)
            self.assertEqual([r["case"] for r in summary], [r["case"] for r in data])
            self.assertEqual([r["rank"] for r in summary], ["2", "1"])
            self.assertEqual([r["execution_order"] for r in ranking], ["2", "1"])
            self.assertEqual([r["gpu_memory_utilization"] for r in summary], ["0.9", "0.95"])
            self.assertEqual(summary[0]["mtp"], "3")
            self.assertEqual(summary[0]["env.EXTRA_ENV"], "a,b")
            self.assertEqual(json.loads(summary[0]["args.future-flag"]), [1, 2])
            self.assertEqual(json.loads(summary[0]["config_json"]), data[0]["config"])
            self.assertIn("gpu_memory_utilization", (directory / "summary.md").read_text())
            self.assertEqual(data, original)

    def test_disabled_mtp_and_boolean_flags(self):
        values = parameters({"args": {"speculative-config": None, "enable-prefix-caching": False}})
        self.assertEqual(values["mtp"], 0)
        self.assertEqual(values["prefix_cache"], "false")

    def test_refresh_backs_up_without_touching_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "runs/session"
            directory.mkdir(parents=True)
            report(directory, rows(), SELECTION)
            raw = directory / "result.json"
            raw.write_text("raw result must remain unchanged")
            before = (directory / "summary.csv").read_bytes()
            backup = refresh(directory)
            self.assertEqual((backup / "summary.csv").read_bytes(), before)
            self.assertEqual(raw.read_text(), "raw result must remain unchanged")
            self.assertEqual((directory / "summary.csv").read_bytes(), before)

    def test_refuse_refresh_of_active_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "runs/session"
            directory.mkdir(parents=True)
            (root / "state").mkdir()
            (root / "state/controller.json").write_text(json.dumps({"session": str(directory)}))
            with self.assertRaisesRegex(ValueError, "controller marker"):
                refresh(directory)


if __name__ == "__main__":
    unittest.main()
