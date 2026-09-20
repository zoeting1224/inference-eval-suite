"""Unified command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

from . import accuracy, external_perf, long_context, reclassify, task
from .config import candidates, load, validate, validate_case
from .reporting import refresh
from .runner import compare
from .service import checked, docker_argv
from .session import run as run_search
from .workload import freeze, identity


ROOT = Path(__file__).resolve().parents[1]


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def performance_config(path: Path) -> dict:
    return validate(load(path))


def use_benchmark_python(config: dict) -> None:
    desired = Path(config["benchmark"]["python"]).resolve()
    if not desired.is_file():
        raise ValueError(f"benchmark Python not found: {desired}; edit configs/benchmarks/aisbench.json")
    if Path(sys.executable).resolve() != desired:
        os.execv(str(desired), [str(desired), str(ROOT / "inferbench.py"), *sys.argv[1:]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference Eval Suite")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate and show a performance search without execution")
    add_config(plan)

    doctor = commands.add_parser("doctor", help="read-only runtime, image and workload checks")
    add_config(doctor)

    search = commands.add_parser("search", help="managed Docker/vLLM performance search")
    add_config(search)
    search.add_argument("--phase", required=True, help="phase0, configured phase, all, final-baseline or final")
    reuse = search.add_mutually_exclusive_group()
    reuse.add_argument("--resume")
    reuse.add_argument("--new-run", action="store_true")

    perf = commands.add_parser("perf", help="benchmark an already-running service")
    add_config(perf)
    perf.add_argument("--mode", choices=("fast", "final"), default="final")
    perf.add_argument("--port", type=int)
    perf.add_argument("--output", type=Path)

    acc = commands.add_parser("accuracy", help="evaluate an already-running service")
    add_config(acc)
    acc.add_argument("--limit", type=int)
    acc.add_argument("--max-tokens", type=int, help="override generation.max_tokens")
    acc.add_argument("--output", required=True, type=Path)

    data = commands.add_parser("data", help="prepare deterministic datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    long = data_commands.add_parser("long-context", help="construct exact-length performance prompts")
    add_config(long)
    long.add_argument("--source", type=Path, action="append", required=True)
    long.add_argument("--text-field", default="question")
    long.add_argument("--seed", type=int, default=1024)
    long.add_argument("--force", action="store_true")

    report = commands.add_parser("report", help="offline report operations")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    comparison = report_commands.add_parser("compare", help="compare final-baseline with final")
    comparison.add_argument("--baseline", required=True, type=Path)
    comparison.add_argument("--candidate", required=True, type=Path)
    refresh_parser = report_commands.add_parser("refresh", help="rebuild a completed performance report")
    refresh_parser.add_argument("--session", required=True, type=Path)
    classify = report_commands.add_parser("reclassify", help="apply current SLOs to saved raw measurements")
    add_config(classify)
    classify.add_argument("--root", required=True, type=Path)
    acc_compare = report_commands.add_parser("accuracy-compare", help="compare two accuracy runs")
    acc_compare.add_argument("--baseline", required=True, type=Path)
    acc_compare.add_argument("--candidate", required=True, type=Path)
    acc_compare.add_argument("--output", required=True, type=Path)

    tasks = commands.add_parser("task", help="run long commands in tmux")
    task_commands = tasks.add_subparsers(dest="task_command", required=True)
    task_start = task_commands.add_parser("start")
    task_start.add_argument("--name", required=True)
    task_start.add_argument("task_argv", nargs=argparse.REMAINDER)
    task_status = task_commands.add_parser("status")
    task_status.add_argument("--name")
    task_logs = task_commands.add_parser("logs")
    task_logs.add_argument("--name", required=True)
    task_logs.add_argument("--lines", type=int, default=200)
    task_stop = task_commands.add_parser("stop")
    task_stop.add_argument("--name", required=True)
    task_attach = task_commands.add_parser("attach")
    task_attach.add_argument("--name", required=True)
    return parser


def show_plan(config: dict) -> None:
    print(f"Experiment: {config['name']}")
    print(f"Model: {config['model']['path']}")
    print(f"Resources: {config['service']['resource_ids']}")
    print(f"Concurrency: {config['workload']['concurrency']}")
    print(f"Input target: {config['workload'].get('input_tokens')}")
    print("phase0: baseline")
    for phase in config["search"]["phases"]:
        cases = candidates(config["baseline"], phase, config["search"]["max_cases_per_phase"])
        for candidate in cases:
            validate_case(config, candidate)
        print(f"{phase['name']}: {len(cases)} cases; {phase.get('description', '')}")
    print("\nBaseline Docker command (not executed):")
    print(shlex.join(docker_argv(config, config["baseline"], "inferbench-PLAN", "PLAN")))
    print("\nData availability:")
    print(json.dumps(identity(ROOT, config)["files"], ensure_ascii=False, indent=2))
    print("\nSelection:")
    print(json.dumps(config["selection"], ensure_ascii=False, indent=2))


def run_doctor(config: dict) -> None:
    use_benchmark_python(config)
    print(json.dumps(identity(ROOT, config, require=True), ensure_ascii=False, indent=2))
    print("Docker image ID:", checked(["docker", "image", "inspect", "--format", "{{.Id}}", config["service"]["image"]]))
    from .runtime import info
    print("AISBench:", json.dumps(info(config["benchmark"]["binary"]), ensure_ascii=False, indent=2))
    with tempfile.TemporaryDirectory(prefix="inferbench-doctor-") as directory:
        for mode in ("fast", "final"):
            profile = freeze(ROOT, config, mode, Path(directory) / mode)
            print(mode, "workload hash", profile["workload_hash"])


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        show_plan(performance_config(args.config))
    elif args.command == "doctor":
        run_doctor(performance_config(args.config))
    elif args.command == "search":
        config = performance_config(args.config)
        phases = {p["name"] for p in config["search"]["phases"]} | {"phase0", "all", "final", "final-baseline"}
        if args.phase not in phases:
            raise ValueError(f"unknown phase {args.phase}; choose {sorted(phases)}")
        use_benchmark_python(config)
        print(run_search(ROOT, config, args.phase, args.resume, new_run=args.new_run))
    elif args.command == "perf":
        forwarded = ["--config", str(args.config), "--mode", args.mode]
        if args.port is not None:
            forwarded += ["--port", str(args.port)]
        if args.output is not None:
            forwarded += ["--output", str(args.output)]
        external_perf.main(forwarded, ROOT)
    elif args.command == "accuracy":
        forwarded = ["--config", str(args.config), "--output", str(args.output)]
        if args.limit is not None:
            forwarded += ["--limit", str(args.limit)]
        if args.max_tokens is not None:
            forwarded += ["--max-tokens", str(args.max_tokens)]
        raise SystemExit(accuracy.main(forwarded, ROOT))
    elif args.command == "data" and args.data_command == "long-context":
        forwarded = ["--config", str(args.config), "--text-field", args.text_field, "--seed", str(args.seed)]
        for source in args.source:
            forwarded += ["--source", str(source)]
        if args.force:
            forwarded.append("--force")
        long_context.main(forwarded, ROOT)
    elif args.command == "report" and args.report_command == "compare":
        compare(args.baseline, args.candidate)
        print(args.candidate / "comparison.md")
    elif args.command == "report" and args.report_command == "refresh":
        print("Backup:", refresh(args.session))
    elif args.command == "report" and args.report_command == "reclassify":
        reclassify.main(["--config", str(args.config), str(args.root)])
    elif args.command == "report" and args.report_command == "accuracy-compare":
        accuracy.compare(args.baseline, args.candidate, args.output)
    elif args.command == "task" and args.task_command == "start":
        task.start(ROOT, args.name, args.task_argv)
    elif args.command == "task" and args.task_command == "status":
        task.status(args.name)
    elif args.command == "task" and args.task_command == "logs":
        task.logs(args.name, args.lines)
    elif args.command == "task" and args.task_command == "stop":
        task.stop(args.name)
    elif args.command == "task" and args.task_command == "attach":
        task.attach(args.name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; managed searches clean up their owned container.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
