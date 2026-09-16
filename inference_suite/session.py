"""CLI session lifecycle; measurement/selection remain in runner.py.

Changing directory reuse policy must not invalidate an existing final A/B pair.
Keep the measurement engine identity unchanged; record this controller's own hash
in session_status.json instead. All selection and recovery happen under the
existing hardware/project locks, before creating any output directory.
"""
import contextlib
import datetime
import json
import os
import shlex
import signal
import sys
from pathlib import Path

from . import runner
from .config import digest, save
from .runtime import info


def stamp():
    return datetime.datetime.now().isoformat(timespec="microseconds")


def read(path):
    return json.loads(path.read_text())


def select_session(base, manifest, resume=None, new_run=False):
    """Caller holds locks. Never silently fall back to an older abandoned run."""
    runs = base / "runs"
    if resume and new_run:
        raise ValueError("--resume and --new-run cannot be combined")
    if resume:
        if Path(resume).name != resume or resume in (".", ".."):
            raise ValueError("resume must be a session name, not a path")
        selected = runs / resume
        if selected.is_symlink() or not (selected / "manifest.json").is_file():
            raise ValueError("resume must name an existing session in this exact experiment")
        if read(selected / "manifest.json") != manifest:
            raise ValueError("resume manifest mismatch")
        return selected, True
    if not new_run:
        for selected in sorted(runs.glob("*"), reverse=True):
            if selected.is_symlink() or not selected.is_dir():
                continue
            path = selected / "manifest.json"
            if path.is_file() and read(path) == manifest:
                return selected, True
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + manifest["phase"]
    return runs / name, False


def valid_case(directory, profile):
    """Only complete, matching measurements may be skipped on recovery."""
    path = directory / "result.json"
    if not path.is_file():
        return False
    row = read(path)
    if row.get("evaluation", {}).get("status") not in ("SLO_PASS", "SLO_FAIL"):
        return False
    expected = digest({"candidate": read(directory / "config.json"),
                       "workload": profile["workload_hash"]})
    if row.get("case_hash") != expected:
        raise ValueError(f"resume case hash mismatch: {directory}")
    labels = [f"repeat_{i+1}/{d['name']}" for i in range(profile["repeats"])
              for d in profile["datasets"]]
    return (sorted(r.get("name", "") for r in row.get("repeats", [])) == sorted(labels)
            and row.get("cleanup", {}).get("status", "CLEAN") == "CLEAN")


def completed(session, manifest):
    status = session / "session_status.json"
    if status.exists():
        done = read(status).get("status") == "COMPLETED"
    else:
        # Compatibility with existing sessions, which had no lifecycle marker.
        log = session / "master.log"
        done = log.is_file() and any(
            line.endswith(f"Done: {session / 'ranking.csv'}")
            for line in log.read_text(errors="replace").splitlines())
    if not done or not all((session / f).is_file() for f in
                            ("summary.json", "summary.csv", "ranking.csv", "summary.md")):
        return False
    phase, c = manifest["phase"], manifest["config"]
    phases = (["phase0"] + [p["name"] for p in c["search"]["phases"] if p.get("enabled", True)]
              if phase == "all" else [phase])
    profile_path = session / "workload/workload_profile.json"
    if not profile_path.is_file():
        return False
    profile = read(profile_path)
    for name in phases:
        plan = session / (name + "_plan.json")
        if not plan.is_file():
            return False
        for i, candidate in enumerate(read(plan)):
            directory = session / "runs" / f"{name}_{i:03d}_{digest(candidate)[:10]}"
            result = directory / "result.json"
            if not result.is_file():
                return False
            row = read(result)
            # A completed search may include a recorded failed candidate. Do not
            # retry such a whole search automatically after its Done boundary.
            if row.get("evaluation", {}).get("status") in ("ERROR", "INVALID"):
                if phase in ("final", "final-baseline"):
                    return False
            elif not valid_case(directory, profile):
                return False
    return True


def recover_cases(session):
    """Archive incomplete attempts, including interrupts with NO result.json.

    Complete cases stay untouched. A partially finished case restarts in full;
    combining warm/cold repeats from different service instances is avoided.
    """
    directories = [d for d in sorted((session / "runs").glob("*")) if d.is_dir()]
    if not directories:
        return
    profile = read(session / "workload/workload_profile.json")
    retry = [d for d in directories if not valid_case(d, profile)]
    if not retry:
        return
    # A killed controller may have left an orphan service. Do not stop it, and
    # do not bury its ownership evidence by moving files before checking.
    names = set()
    for d in retry:
        launch = d / "launch.sh"
        if launch.exists():
            argv = shlex.split(launch.read_text(), comments=True)
            if "--name" in argv:
                names.add(argv[argv.index("--name") + 1])
    if names:
        live = set(runner.checked(["docker", "ps", "--format", "{{.Names}}"]).splitlines())
        if names & live:
            raise RuntimeError(f"previous attempt container still running: {sorted(names & live)}; untouched")
    for d in retry:
        entries = [p for p in d.iterdir() if p.name != "previous_attempts"]
        if not entries:
            continue
        dest = d / "previous_attempts" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest.mkdir(parents=True)
        for p in entries:
            p.rename(dest / p.name)
        runner.log(f"RESUME archived incomplete case: {dest}")


def run(root, c, phase, resume=None, new_run=False):
    root = Path(root)
    if not Path(c["benchmark"]["binary"]).is_file():
        raise ValueError("AISBench executable not found; edit benchmark.binary")
    if not Path(c["model"]["path"]).is_dir():
        raise ValueError("model.path not found")
    ident = runner.identity(root, c, require=True)
    ident["engine_sha256"] = runner.code_hash()
    ident["image"] = runner.checked(["docker", "image", "inspect", "--format", "{{.Id}}", c["service"]["image"]])
    ident["runtime_versions"] = info(c["benchmark"]["binary"])
    base = runner.namespace(root, c, ident)
    manifest = {"identity": ident, "phase": phase, "config": c}
    with runner.locks(root, [*c["service"]["resource_ids"], f"port-{c['service']['port']}"]):
        session, reused = select_session(base, manifest, resume, new_run)
        if reused and completed(session, manifest):
            runner.log(f"ALREADY_COMPLETED: {session}; no model/benchmark launched. Use --new-run for an intentional retest.")
            return session
        # Check before creating a new session, so a broken latest path leaves no litter.
        link = base / "latest"
        if link.exists() and not link.is_symlink():
            raise ValueError(f"refusing to replace non-symlink: {link}")
        state = base / "state"
        state.mkdir(parents=True, exist_ok=True)
        session.mkdir(parents=True, exist_ok=True)
        save(session / "manifest.json", manifest)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(session.resolve(), target_is_directory=True)
        save(state / "controller.json", {"pid": os.getpid(),
             "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
             "session": str(session.resolve())})
        lifecycle = session / "session_status.json"
        record = read(lifecycle) if lifecycle.exists() else {"attempts": []}
        attempt = {"started_at": stamp(), "resumed": reused,
                   "controller_sha256": digest(Path(__file__).read_text())}
        record["attempts"].append(attempt)
        record["status"] = "RUNNING"
        save(lifecycle, record)
        old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            with (session / "master.log").open("a") as logfile, contextlib.redirect_stdout(runner.Tee(sys.stdout, logfile)):
                runner.log(f"{'RESUME' if reused else 'NEW_SESSION'}: {session}")
                runner.log(f"Experiment: {base}\nSession: inference_suite/session.py; pipeline: inference_suite/runner.py")
                if reused:
                    recover_cases(session)
                runner._run_phases(root, c, phase, session, state)
            record["status"] = "COMPLETED"
        except BaseException as exc:
            record["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt.update(status=record["status"], finished_at=stamp())
            try:
                save(lifecycle, record)
            finally:
                signal.signal(signal.SIGTERM, old_handler)
                (state / "controller.json").unlink(missing_ok=True)
    return session
