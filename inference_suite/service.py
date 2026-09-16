"""Owned-container lifecycle. Never attach to, replace or kill an existing service."""
import datetime
import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from .config import save


def flags(args):
    result = []
    for key, value in args.items():
        if not key or key.startswith("-") or any(x.isspace() for x in key):
            raise ValueError(f"invalid CLI flag: {key}")
        if value is None:
            continue  # null means omit, including disabling model-specific MTP.
        if isinstance(value, bool):
            result.append("--" + ("" if value else "no-") + key)
        else:
            result.extend(["--" + key, json.dumps(value, separators=(",", ":"))
                           if isinstance(value, (dict, list)) else str(value)])
    return result


def launch_argv(c, candidate):
    model, s = c["model"], c["service"]
    command = [x.replace("{model}", model["container_path"]) for x in s["command"]]
    return command + ["--host", "0.0.0.0", "--port", str(s["port"]),
                      "--served-model-name", model["served_name"]] + flags(candidate["args"])


def docker_argv(c, candidate, name, owner):
    s = c["service"]
    argv = ["docker", "run", "-d", "--name", name, "--label", f"inference-suite.owner={owner}"]
    argv += s["docker_args"]
    # Loader patches or other model-specific read-only mounts belong here,
    # rather than being mixed into the hardware profile.
    argv += c["model"].get("docker_args", [])
    # Model mount is automatic; device/driver/runtime mounts belong to hardware config.
    argv += ["--mount", f"type=bind,src={c['model']['path']},dst={c['model']['container_path']},readonly"]
    for k, v in dict(s.get("env", {}), **candidate["env"]).items():
        if not k or "=" in k:
            raise ValueError("invalid environment variable")
        argv += ["-e", f"{k}={v}"]
    return argv + [s["image"]] + launch_argv(c, candidate)


def checked(argv, timeout=60, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, timeout=timeout, **kwargs).stdout.strip()


class Service:
    def __init__(self, c, candidate, directory):
        self.c, self.candidate, self.directory = c, candidate, Path(directory)
        self.owner = uuid.uuid4().hex
        self.name = "inferbench-" + self.owner[:16]
        self.created = False

    def start(self):
        s = self.c["service"]
        with socket.socket() as sock:
            if sock.connect_ex((s["host"], s["port"])) == 0:
                raise RuntimeError(f"port {s['port']} already in use; existing service is untouched")
        argv = docker_argv(self.c, self.candidate, self.name, self.owner)
        (self.directory / "launch.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(argv) + "\n")
        # Set before run: cleanup can still find our ownership label after an uncertain response.
        self.created = True
        checked(argv)
        deadline = time.monotonic() + s["health_timeout_s"]
        url = f"http://{s['host']}:{s['port']}/v1/models"
        while time.monotonic() < deadline:
            if checked(["docker", "inspect", "-f", "{{.State.Running}}", self.name]) != "true":
                raise RuntimeError("model process exited before health check")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    data = json.load(response)
                if self.c["model"]["served_name"] in [m.get("id") for m in data.get("data", [])]:
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError("model health check timed out")

    def close(self, reason="controller_cleanup"):
        if not self.created:
            return
        info = subprocess.run(["docker", "inspect", self.name], capture_output=True, text=True, timeout=30)
        if info.returncode:
            if "No such" in info.stderr:
                self.created = False
                return
            raise RuntimeError(f"cannot inspect owned container: {info.stderr.strip()}")
        before = json.loads(info.stdout)[0]
        labels = before.get("Config", {}).get("Labels", {}) or {}
        if labels.get("inference-suite.owner") != self.owner:
            raise RuntimeError("container ownership mismatch; refusing cleanup")
        timeout = self.c["service"].get("stop_timeout_s", 45)
        record = {"container": self.name, "reason": reason, "status": "IN_PROGRESS",
                  "requested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "stop_timeout_s": timeout, "state_before": before.get("State", {}),
                  "intentional_stop": bool(before.get("State", {}).get("Running")),
                  "runtime_log": "vllm.runtime.log", "full_log": "vllm.log"}
        save(self.directory / "cleanup.json", record)

        def capture(filename):
            with (self.directory / filename).open("w") as log:
                subprocess.run(["docker", "logs", self.name], stdout=log,
                    stderr=subprocess.STDOUT, timeout=30, check=True)

        try:
            # Preserve a pre-stop snapshot AND the complete unfiltered log.
            # An ERROR before this boundary is never relabelled shutdown noise.
            capture("vllm.runtime.log")
            if record["intentional_stop"]:
                checked(["docker", "stop", "--time", str(timeout), self.name], timeout=timeout+15)
            after = json.loads(checked(["docker", "inspect", self.name]))[0]
            record["state_after"] = after.get("State", {})
            capture("vllm.log")  # capture after stop so shutdown traces are not lost
            state = record["state_after"]
            if state.get("Running"):
                raise RuntimeError("owned container still running after stop")
            # Keep exited containers with bad exits for diagnosis; never hide OOM
            # or a forced kill merely because this happened during cleanup.
            if state.get("OOMKilled") or state.get("ExitCode") != 0 or state.get("Error"):
                raise RuntimeError(f"abnormal container exit: code={state.get('ExitCode')}, "
                                   f"OOMKilled={state.get('OOMKilled')}, error={state.get('Error')}")
            if not record["intentional_stop"]:
                raise RuntimeError("container exited before controller requested shutdown")
            checked(["docker", "rm", self.name])
            self.created = False
            record["status"] = "CLEAN"
        except Exception as exc:
            record["status"] = "ERROR"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(self.directory / "cleanup.json", record)
        return record


def run_benchmark(argv, path, timeout_s):
    """Timeout/interrupt terminates only this AISBench process group, including workers."""
    with Path(path).open("w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout_s)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
