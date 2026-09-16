"""The full automation pipeline: freeze -> launch -> benchmark -> rank -> inherit."""
import contextlib
import csv
import datetime
import fcntl
import json
import os
import shlex
from pathlib import Path

from .config import candidates, digest, save, validate_case
from .selection import choose, evaluate, rank_key
from .service import Service, checked, docker_argv, run_benchmark
from .workload import freeze, identity, render
from .reporting import report


def log(message):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def code_hash():
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def namespace(root, c, ident):
    return Path(root) / "runs" / "performance" / c["name"] / digest(ident)[:16]


def table(row):
    e = row["evaluation"]
    log(f"CASE {row['case']} status={e['status']}")
    print("  repeat/dataset | metric | observed | limits | result", flush=True)
    for check in sorted(e.get("checks", []), key=lambda x: (x["repeat"], x["metric"])):
        print(f"  {check['repeat']} | {check['metric']} | {check['value']} | {check['limits']} | {check['status']}", flush=True)
    if row.get("error"):
        log(row["error"])


@contextlib.contextmanager
def locks(root, resources):
    handles = []
    try:
        paths = [Path(root) / ".runtime/tuning.lock"]
        # Global per-user physical-resource locks also cover separate project copies.
        lockroot = Path.home() / ".cache/inference-eval-suite/locks"
        paths += [lockroot / (x + ".lock") for x in sorted(resources)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            h = path.open("a+")
            handles.append(h)
            try:
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"resource busy: {path}; existing controller untouched")
        yield
    finally:
        for h in reversed(handles):
            h.close()


def case_run(root, c, candidate, profile, session, phase, index):
    from .metrics import extract
    case_id = f"{phase}_{index:03d}_{digest(candidate)[:10]}"
    directory = session / "runs" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    old = directory / "result.json"
    expected_hash = digest({"candidate": candidate, "workload": profile["workload_hash"]})
    if old.exists():
        prior = json.loads(old.read_text())
        if prior["case_hash"] != expected_hash:
            raise ValueError("resume case hash mismatch")
        if prior["evaluation"]["status"] in ("SLO_PASS", "SLO_FAIL"):
            log(f"RESUME skip complete {case_id}")
            return prior
        # Do not let partial old AISBench outputs contaminate a retry.
        attempts = directory / "previous_attempts"
        attempts.mkdir(exist_ok=True)
        dest = attempts / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest.mkdir()
        for p in list(directory.iterdir()):
            if p.name != "previous_attempts":
                p.rename(dest / p.name)
    save(directory / "config.json", candidate)
    save(directory / "workload_profile.json", profile)
    row = {"case": case_id, "case_hash": expected_hash, "phase": phase,
           "mode": profile["mode"], "config": candidate, "repeats": [],
           "evaluation": {"status": "ERROR", "checks": [], "metrics": {}}}
    service = Service(c, candidate, directory)
    try:
        validate_case(c, candidate)
        service.start()
        for repeat in range(profile["repeats"]):
            for ds in profile["datasets"]:
                label = f"repeat_{repeat+1}/{ds['name']}"
                work = directory / label
                cfg = render(c, profile, ds, work / "config")
                argv = [c["benchmark"]["binary"], "--config-dir", str(cfg), "--models", "target",
                        "--datasets", "workload", "--mode", "perf", "--num-prompts", str(ds["prompts"]),
                        "--num-warmups", str(profile["warmups"]), "--work-dir", str(work / "aisbench")]
                log(f"{case_id} {label}: {shlex.join(argv)}")
                rc = run_benchmark(argv, work / "aisbench.log", c["benchmark"]["timeout_s"])
                if rc:
                    raise RuntimeError(f"AISBench failed ({rc}); see {work / 'aisbench.log'}")
                metrics, requests = extract(work / "aisbench", expected=ds["prompts"],
                     output_tokens=profile["output_tokens"], input_tokens=c["workload"].get("input_tokens"),
                     input_tolerance=c["workload"].get("input_tolerance", 0),
                     concurrency=profile["concurrency"], time_scale=c["benchmark"].get("detail_time_scale", 1.0))
                save(work / "metrics.json", metrics)
                if requests:
                    with (work / "requests.csv").open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(requests[0]))
                        writer.writeheader()
                        writer.writerows(requests)
                row["repeats"].append({"name": label, "metrics": metrics})
                partial = evaluate(row["repeats"], c["selection"])
                table(dict(row, evaluation=partial))
        row["evaluation"] = evaluate(row["repeats"], c["selection"])
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["measurement_status"] = row["evaluation"]["status"]
        reason = ("benchmark_finished" if row["measurement_status"] in ("SLO_PASS", "SLO_FAIL")
                  else "benchmark_failed_or_interrupted")
        log(f"MODEL_STOP_BEGIN {case_id}: reason={reason}, measurement={row['measurement_status']}; "
            "controller-requested cleanup, see cleanup.json and unfiltered vllm.log")
        try:
            cleanup = service.close(reason=reason)
            if isinstance(cleanup, dict):
                row["cleanup"] = cleanup
            log(f"MODEL_STOP_END {case_id}: cleanup complete; measurement={row['measurement_status']}")
        except Exception as exc:
            row["cleanup_error"] = f"cleanup failed: {exc}; owned container={service.name}"
            row["error"] = (row.get("error", "") + "; " + row["cleanup_error"]).lstrip("; ")
            row["evaluation"]["status"] = "ERROR"
            log(f"MODEL_STOP_ERROR {case_id}: {row['cleanup_error']}")
        save(old, row)
        save(directory / "slo_report.json", row["evaluation"])
    table(row)
    return row


class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def write(self, value):
        self.first.write(value)
        self.second.write(value)
        return len(value)

    def flush(self):
        self.first.flush()
        self.second.flush()


def _run_phases(root, c, phase, session, state):
    final = phase in ("final", "final-baseline")
    mode = "final" if final else "fast"
    profile = freeze(root, c, mode, session / "workload")
    candidate_path = state / "best_candidate.json"
    incumbent = c["baseline"]
    if candidate_path.exists():
        incumbent = json.loads(candidate_path.read_text())["config"]
    elif phase == "final":
        raise ValueError("no compatible search candidate; run phase0/all first")
    elif phase not in ("all", "phase0", "final-baseline"):
        log("WARNING no previous candidate: this phase starts from baseline, not an assumed earlier winner")
    configured = {p["name"]: p for p in c["search"]["phases"]}
    phases = ["phase0", *[p["name"] for p in c["search"]["phases"] if p.get("enabled", True)]] if phase == "all" else [phase]
    rows = []
    for name in phases:
        if name not in {"phase0", "final", "final-baseline"} and name not in configured:
            raise ValueError(f"unknown phase: {name}")
        plan_path = session / (name + "_plan.json")
        if plan_path.exists():
            planned = json.loads(plan_path.read_text())
        else:
            planned = ([c["baseline"]] if name in ("phase0", "final-baseline") else [incumbent]
                       if name == "final" else candidates(incumbent, configured[name], c["search"]["max_cases_per_phase"]))
            save(plan_path, planned)
        phase_rows = []
        for index, candidate in enumerate(planned):
            row = case_run(root, c, candidate, profile, session, name, index)
            rows.append(row)
            phase_rows.append(row)
            report(session, rows, c["selection"])
        winner = choose(phase_rows, c["selection"])
        if not final:
            if not winner:
                raise RuntimeError(f"{name}: no valid successful measurement; stop instead of promoting an invalid best")
            incumbent = winner["config"]
            save(candidate_path, winner)
            log(f"Inherited candidate: {winner['case']} ({winner['evaluation']['status']})")
        else:
            final_row = phase_rows[0]
            if final_row["evaluation"]["status"] in ("ERROR", "INVALID"):
                raise RuntimeError("final measurement invalid; inspect logs")
            save(state / (name + ".json"), {"session": str(session.resolve()), "row": final_row})
            if name == "final" and final_row["evaluation"]["status"] == "SLO_PASS":
                save(state / "validated_best.json", final_row)
            if name == "final" and (state / "final-baseline.json").exists():
                baseline = json.loads((state / "final-baseline.json").read_text())
                compare(Path(baseline["session"]), session)
    log(f"Done: {session / 'ranking.csv'}")


def compare(baseline, best):
    baseline, best = Path(baseline), Path(best)
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (baseline, best)]
    if manifests[0]["phase"] != "final-baseline" or manifests[1]["phase"] != "final":
        raise ValueError("comparison requires final-baseline and final, in this order")
    if manifests[0]["identity"] != manifests[1]["identity"]:
        raise ValueError("model/runtime/config/data identity mismatch")
    profiles = [json.loads((p / "workload/workload_profile.json").read_text()) for p in (baseline, best)]
    if profiles[0]["workload_hash"] != profiles[1]["workload_hash"]:
        raise ValueError("final workloads differ; refusing A/B")
    rows = [json.loads((p / "summary.json").read_text())["ranking"][0] for p in (baseline, best)]
    if any(r["evaluation"]["status"] in ("INVALID", "ERROR") for r in rows):
        raise ValueError("invalid final result")
    result = {"workload_match": True, "baseline": str(baseline.resolve()), "best": str(best.resolve()),
              "baseline_status": rows[0]["evaluation"]["status"], "best_status": rows[1]["evaluation"]["status"],
              "config_changes": {}, "metrics": {}}
    for group in ("args", "env"):
        a, b = [r["config"][group] for r in rows]
        result["config_changes"][group] = {k: {"baseline": a.get(k), "best": b.get(k)}
                                            for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    a, b = [r["evaluation"]["metrics"] for r in rows]
    for key in sorted(a.keys() | b.keys()):
        av, bv = a.get(key), b.get(key)
        result["metrics"][key] = {"baseline": av, "best": bv,
            "change_percent": (bv / av - 1) * 100 if av not in (None, 0) and bv is not None else None}
    save(best / "comparison.json", result)
    lines = ["# Final A/B", "", f"Baseline: {result['baseline_status']} / Best: {result['best_status']}", "",
             "Change is raw (best/base - 1); lower latency is better, higher throughput is better.", "",
             "| Metric | Baseline | Best | Change % |", "|---|---:|---:|---:|"]
    with (best / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "best", "change_percent"])
        writer.writeheader()
        for k, v in result["metrics"].items():
            writer.writerow(dict(metric=k, **v))
            lines.append(f"| {k} | {v['baseline']} | {v['best']} | {v['change_percent']} |")
    (best / "comparison.md").write_text("\n".join(lines) + "\n")
    return result
