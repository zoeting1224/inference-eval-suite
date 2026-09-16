"""Lifecycle ordering, provenance, timeouts and fail-closed shutdown tests."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_suite.config import load, validate, validate_case
from inference_suite.service import Service, launch_argv

ROOT = Path(__file__).resolve().parents[1]


class CleanupTests(unittest.TestCase):
    def test_graceful_timeout_in_launch_and_outer_budget(self):
        c = validate(load(ROOT / "configs/experiments/qwen38-bf16-long100k.json"))
        argv = launch_argv(c, c["baseline"])
        self.assertEqual(argv[argv.index("--shutdown-timeout") + 1], "30")
        self.assertEqual(c["service"]["stop_timeout_s"], 45)
        c["service"]["stop_timeout_s"] = 30
        with self.assertRaisesRegex(ValueError, "plus at least 5"):
            validate_case(c, c["baseline"])

    def exercise(self, before_running=True, exit_code=0, oom=False, stop_error=False):
        c = validate(load(ROOT / "configs/experiments/qwen38-bf16-long100k.json"))
        with tempfile.TemporaryDirectory() as tmp:
            service = Service(c, c["baseline"], tmp)
            service.created = True
            events = []
            stopped = False

            def docker(argv, **kwargs):
                nonlocal stopped
                action = argv[1]
                if action == "inspect":
                    state = {"Running": before_running and not stopped,
                             "ExitCode": exit_code if stopped or not before_running else 0,
                             "OOMKilled": oom if stopped or not before_running else False,
                             "Error": ""}
                    return subprocess.CompletedProcess(argv, 0, json.dumps([
                        {"Config": {"Labels": {"inference-suite.owner": service.owner}}, "State": state}]), "")
                if action == "logs":
                    events.append("logs_after" if stopped else "logs_before")
                    kwargs["stdout"].write("runtime line\n" + ("shutdown line\nERROR output_handler\n" if stopped else ""))
                elif action == "stop":
                    events.append("stop")
                    self.assertEqual(argv[2:4], ["--time", "45"])
                    self.assertEqual(kwargs["timeout"], 60)
                    if stop_error:
                        raise subprocess.TimeoutExpired(argv, 60)
                    stopped = True
                elif action == "rm":
                    events.append("rm")
                else:
                    raise AssertionError(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")

            with patch("inference_suite.service.subprocess.run", side_effect=docker):
                if not before_running or exit_code or oom or stop_error:
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        service.close(reason="benchmark_finished")
                else:
                    service.close(reason="benchmark_finished")
            record = json.loads((Path(tmp) / "cleanup.json").read_text())
            logs = {p.name: p.read_text() for p in Path(tmp).glob("*.log")}
            return events, record, logs

    def test_clean_stop_preserves_complete_unfiltered_log(self):
        events, record, logs = self.exercise()
        self.assertEqual(events, ["logs_before", "stop", "logs_after", "rm"])
        self.assertEqual(record["status"], "CLEAN")
        self.assertTrue(record["intentional_stop"])
        self.assertEqual(logs["vllm.runtime.log"], "runtime line\n")
        self.assertIn("ERROR output_handler", logs["vllm.log"])

    def test_forced_exit_not_hidden(self):
        events, record, logs = self.exercise(exit_code=137)
        self.assertEqual(record["status"], "ERROR")
        self.assertNotIn("rm", events)
        self.assertIn("vllm.log", logs)

    def test_oom_not_hidden(self):
        _, record, _ = self.exercise(oom=True)
        self.assertEqual(record["status"], "ERROR")
        self.assertTrue(record["state_after"]["OOMKilled"])

    def test_already_dead_engine_not_called_normal_shutdown(self):
        events, record, _ = self.exercise(before_running=False)
        self.assertFalse(record["intentional_stop"])
        self.assertEqual(record["status"], "ERROR")
        self.assertNotIn("stop", events)
        self.assertNotIn("rm", events)

    def test_stop_timeout_preserves_failure_record_and_runtime_log(self):
        events, record, logs = self.exercise(stop_error=True)
        self.assertEqual(record["status"], "ERROR")
        self.assertIn("TimeoutExpired", record["error"])
        self.assertIn("vllm.runtime.log", logs)
        self.assertNotIn("rm", events)


if __name__ == "__main__":
    unittest.main()
