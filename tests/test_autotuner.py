import copy
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_suite.config import candidates, digest, load, save, validate, validate_case
from inference_suite.metrics import extract
from inference_suite.runner import _run_phases, compare
from inference_suite.selection import choose, evaluate
from inference_suite.service import Service, flags, launch_argv, run_benchmark
from inference_suite.workload import freeze, render

ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate(load(ROOT / "configs/experiments/qwen38-w4a8-speed2k.json"))


def output(root, count=2, length=5):
    folder = Path(root) / "stamp/performances/target"
    folder.mkdir(parents=True, exist_ok=True)
    data = {k: {"total": v} for k, v in {"Total Requests": count, "Success Requests": count,
        "Failed Requests": 0, "Output Token Throughput": "22.5 token/s", "Request Throughput": "1 req/s"}.items()}
    save(folder / "data.json", data)
    records = [{"data_abbr": "test", "id": i, "success": True, "input_tokens": 100,
                "output_tokens": length, "time_points": [10+i*0.01, 11+i*0.01, 11.2+i*0.01]}
               for i in range(count)]
    (folder / "data_details.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    return folder, records


class ConfigTests(unittest.TestCase):
    def test_presets(self):
        for p in (ROOT / "configs/experiments").glob("*.json"):
            validate(load(p))

    def test_misspelled_fields_rejected(self):
        c = config()
        c["workload"]["concurency"] = 8
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate(c)

    def test_no_model_in_engine(self):
        for p in (ROOT / "inference_suite").glob("*.py"):
            self.assertNotIn("qwen3_5_mtp", p.read_text())
            self.assertNotIn("/data/models/", p.read_text())

    def test_cycle_and_list_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save(p / "a.json", {"extends": ["b.json"]})
            save(p / "b.json", {"extends": ["a.json"]})
            with self.assertRaisesRegex(ValueError, "cycle"):
                load(p / "a.json")

    def test_each_dataset_needs_concurrency(self):
        c = config()
        c["workload"]["fast"]["prompts_per_dataset"] = 16
        with self.assertRaisesRegex(ValueError, "EACH"):
            validate(c)

    def test_no_implicit_prefix_cache(self):
        c = config()
        c["baseline"]["args"].pop("enable-prefix-caching")
        with self.assertRaisesRegex(ValueError, "cold"):
            validate(c)

    def test_context_must_cover_input_output(self):
        c = load(ROOT / "configs/experiments/qwen38-bf16-long100k.json")
        c["baseline"]["args"]["max-model-len"] = 100000
        with self.assertRaisesRegex(ValueError, "cover"):
            validate(c)

    def test_search_keeps_incumbent_and_deduplicates(self):
        base = {"args": {"n": 2}, "env": {}}
        phase = {"name": "phase1", "strategy": "one_at_a_time", "parameters": {"args.n": [1, 2, 2, 3]}}
        self.assertEqual([x["args"]["n"] for x in candidates(base, phase, 6)], [2, 1, 3])
        self.assertEqual(base["args"]["n"], 2)

    def test_bounded_grid(self):
        phase = {"name": "phase1", "strategy": "grid", "parameters": {"args.n": list(range(10)), "env.A": list(range(10))}}
        with self.assertRaisesRegex(ValueError, "exceed"):
            candidates({"args": {}, "env": {}}, phase, 20)

    def test_flags_null_false_and_json(self):
        argv = flags({"speculative-config": None, "enable-prefix-caching": False, "compilation-config": {"mode": "A"}})
        self.assertNotIn("--speculative-config", argv)
        self.assertIn("--no-enable-prefix-caching", argv)
        self.assertIn('{"mode":"A"}', argv)

    def test_model_change_requires_no_engine_edit(self):
        c = config()
        c["model"]["container_path"] = "/models/another model"
        c["model"]["served_name"] = "another"
        c["baseline"]["args"]["speculative-config"] = None
        argv = launch_argv(c, c["baseline"])
        self.assertIn("/models/another model", argv)
        self.assertIn("another", argv)
        self.assertNotIn("--speculative-config", argv)


class WorkloadTests(unittest.TestCase):
    def test_freeze_deterministic_and_removes_per_row_output_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "data.jsonl").write_text(' {"text":"alpha", "max_tokens":999}\n{"text":"beta"}\n')
            c = config()
            w = c["workload"]
            w["concurrency"] = 2
            w["fast"]["prompts_per_dataset"] = 2
            w["datasets"] = [{"name": "sample", "path": "data.jsonl", "text_field": "text"}]
            a = freeze(p, c, "fast", p / "a")
            b = freeze(p, c, "fast", p / "b")
            self.assertEqual(a["workload_hash"], b["workload_hash"])
            row = json.loads((p / "a/sample.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["max_out_len"], 512)
            self.assertNotIn("max_tokens", row)
            path = render(c, a, a["datasets"][0], p / "cfg")
            for f in path.glob("*/*.py"):
                compile(f.read_text(), str(f), "exec")

    def test_short_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "data.jsonl").write_text('{"question":"one"}\n')
            c = config()
            c["workload"]["datasets"] = [{"name": "sample", "path": "data.jsonl"}]
            with self.assertRaisesRegex(ValueError, "needs"):
                freeze(p, c, "fast", p / "out")


class MetricTests(unittest.TestCase):
    def test_per_request_not_global_tps_divided_by_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            output(tmp)
            m, rows = extract(tmp, 2, 5, concurrency=2)
            self.assertAlmostEqual(m["min_decode_tps"], 20)
            self.assertNotEqual(m["min_decode_tps"], m["output_tps"] / 2)
            self.assertAlmostEqual(m["first_response_ms"], 1000)
            self.assertAlmostEqual(m["all_first_tokens_ms"], 1010)
            self.assertEqual(m["peak_inflight"], 2)
            self.assertIn("tpot_p95_ms", m)

    def test_missing_failed_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = output(tmp)
            data = json.loads((folder / "data.json").read_text())
            data.pop("Failed Requests")
            save(folder / "data.json", data)
            with self.assertRaises(ValueError):
                extract(tmp, 2, 5)

    def test_wrong_output_and_input_lengths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output(tmp)
            with self.assertRaisesRegex(ValueError, "output"):
                extract(tmp, 2, 512)
            with self.assertRaisesRegex(ValueError, "server input"):
                extract(tmp, 2, 5, input_tokens=100000)

    def test_duplicate_details_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, records = output(tmp)
            with (folder / "data_details.jsonl").open("a") as f:
                f.write(json.dumps(records[0])+"\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                extract(tmp, 2, 5)

    def test_sqlite_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy only required for installed AISBench SQLite format")
        with tempfile.TemporaryDirectory() as tmp:
            folder, records = output(tmp)
            (folder / "db_data").mkdir()
            with sqlite3.connect(folder / "db_data/test.db") as db:
                db.execute("CREATE TABLE numpy_store(id INTEGER PRIMARY KEY, arr_blob BLOB)")
                for i, row in enumerate(records):
                    stream = io.BytesIO()
                    np.save(stream, np.array(row["time_points"]), allow_pickle=False)
                    db.execute("INSERT INTO numpy_store VALUES (?,?)", (i, stream.getvalue()))
                    row["time_points"] = {"__db_ref__": i}
                    row["db_name"] = "test.db"
            (folder / "data_details.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            m, _ = extract(tmp, 2, 5)
            self.assertAlmostEqual(m["min_decode_tps"], 20)


class SelectionTests(unittest.TestCase):
    policy = {"constraints": {"min_decode_tps": {"min": 15}},
              "objectives": [{"metric": "output_tps", "direction": "max"}]}

    def row(self, name, tps, decode, failed=0):
        repeats = [{"name": "r1", "metrics": {"output_tps": tps, "min_decode_tps": decode, "failed_requests": failed}}]
        return {"case": name, "repeats": repeats, "evaluation": evaluate(repeats, self.policy)}

    def test_pass_beats_faster_failure(self):
        rows = [self.row("fast", 999, 10), self.row("pass", 100, 20)]
        self.assertEqual(choose(rows, self.policy)["case"], "pass")

    def test_missing_metric_is_invalid(self):
        r = self.row("missing", 100, None)
        self.assertEqual(r["evaluation"]["status"], "INVALID")
        self.assertIsNone(choose([r], self.policy))

    def test_failed_requests_never_promoted(self):
        self.assertIsNone(choose([self.row("bad", 999, 20, 1)], self.policy))

    def test_one_bad_repeat_not_hidden(self):
        repeats = [self.row("ok", 100, 20)["repeats"][0]] * 2 + [self.row("bad", 100, 10)["repeats"][0]]
        self.assertEqual(evaluate(repeats, self.policy)["status"], "SLO_FAIL")


class RunnerTests(unittest.TestCase):
    def test_benchmark_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_benchmark([sys.executable, "-c", "import time; time.sleep(5)"], Path(tmp) / "log", 0.05)

    def test_owned_container_guard(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            svc = Service(c, c["baseline"], tmp)
            svc.created = True
            foreign = json.dumps([{"Config": {"Labels": {"foreign.owner": "someone-else"}}}])
            with patch("inference_suite.service.subprocess.run", return_value=subprocess.CompletedProcess([], 0, foreign, "")) as call:
                with self.assertRaisesRegex(RuntimeError, "ownership"):
                    svc.close()
                self.assertEqual(call.call_count, 1)  # only inspect, no stop/rm

    def test_search_inheritance_final_ab_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            c = config()
            w = c["workload"]
            w["concurrency"] = 2
            w["datasets"] = [{"name": "test", "path": "data.jsonl"}]
            for mode in ("fast", "final"):
                w[mode] = {"prompts_per_dataset": 2, "repeats": 1, "warmups": 0}
            (root / "data.jsonl").write_text('{"question":"a"}\n{"question":"b"}\n')
            c["search"]["phases"] = [{"name": "phase1", "strategy": "one_at_a_time", "parameters": {"args.max-num-batched-tokens": [8192]}}]
            state = root / "state"
            state.mkdir()
            measurements = []

            def bench(argv, path, timeout):
                measurements.append(argv)
                work = Path(argv[argv.index("--work-dir")+1])
                folder, _ = output(work, count=2, length=512)
                candidate = json.loads((work.parents[2] / "config.json").read_text())
                if candidate["args"]["max-num-batched-tokens"] == 8192:
                    perf = json.loads((folder / "data.json").read_text())
                    perf["Output Token Throughput"]["total"] = "99 token/s"
                    save(folder / "data.json", perf)
                return 0

            with patch("inference_suite.runner.Service.start"), patch("inference_suite.runner.Service.close"), patch("inference_suite.runner.run_benchmark", side_effect=bench):
                search = root / "search"
                search.mkdir()
                _run_phases(root, c, "all", search, state)
                self.assertTrue((state / "best_candidate.json").is_file())
                self.assertEqual(json.loads((state / "best_candidate.json").read_text())["config"]["args"]["max-num-batched-tokens"], 8192)
                measured = len(measurements)
                _run_phases(root, c, "all", search, state)
                self.assertEqual(measured, len(measurements))
                for phase in ("final-baseline", "final"):
                    session = root / phase
                    session.mkdir()
                    save(session / "manifest.json", {"phase": phase, "identity": {"test": 1}})
                    _run_phases(root, c, phase, session, state)
                self.assertTrue((root / "final/comparison.json").exists())
                self.assertTrue((state / "validated_best.json").exists())
                self.assertEqual(json.loads((state / "best_candidate.json").read_text())["config"]["args"]["max-num-batched-tokens"], 8192)
                manifest = json.loads((root / "final/manifest.json").read_text())
                manifest["identity"]["test"] = 2
                save(root / "final/manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "identity"):
                    compare(root / "final-baseline", root / "final")


if __name__ == "__main__":
    unittest.main()
