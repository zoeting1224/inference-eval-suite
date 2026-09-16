"""Presentation-only reports; no change to measurements, ranking or engine identity."""
import csv
import datetime
import json
import shutil
from pathlib import Path

from .config import save
from .selection import rank_key


PARAMETERS = {
    "max_num_seqs": "max-num-seqs",
    "max_num_batched_tokens": "max-num-batched-tokens",
    "gpu_memory_utilization": "gpu-memory-utilization",
    "max_model_len": "max-model-len",
    "tp": "tensor-parallel-size",
    "dp": "data-parallel-size",
    "prefix_cache": "enable-prefix-caching",
    "chunked_prefill": "enable-chunked-prefill",
    "quantization": "quantization",
}
METRICS = ["output_tps", "request_tps", "mean_decode_tps", "min_decode_tps", "prefill_per_request_mean_tps",
           "first_response_ms", "all_first_tokens_ms", "ttft_ms", "ttft_p99_ms",
           "tpot_ms", "tpot_p99_ms", "max_tpot_ms", "e2e_ms", "e2e_p99_ms", "failed_requests"]


def cell(value):
    if value is None:
        return "null"
    if isinstance(value, (bool, dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def parameters(config):
    args, env = config.get("args", {}), config.get("env", {})
    spec, graph = args.get("speculative-config"), args.get("compilation-config")
    values = {"mtp": spec.get("num_speculative_tokens", "") if isinstance(spec, dict) else 0}
    values.update({column: args.get(flag, "") for column, flag in PARAMETERS.items()})
    values["graph_mode"] = graph.get("cudagraph_mode", "default") if isinstance(graph, dict) else "default"
    additional = args.get("additional-config") or {}
    values["cpu_binding"] = additional.get("enable_cpu_binding", "") if isinstance(additional, dict) else ""
    values["omp_threads"] = env.get("OMP_NUM_THREADS", "")
    values["hccl_buffsize"] = env.get("HCCL_BUFFSIZE", "")
    # Keep arbitrary future flags/env visible, not just today's Qwen parameters.
    for flag, value in args.items():
        if flag not in PARAMETERS.values():
            values["args." + flag] = value
    for key, value in env.items():
        if key not in ("OMP_NUM_THREADS", "HCCL_BUFFSIZE"):
            values["env." + key] = value
    return {key: cell(value) for key, value in values.items()}


def report(session, rows, selection):
    session = Path(session)
    ranked = sorted(rows, key=lambda row: rank_key(row, selection))
    rank = {row["case"]: i for i, row in enumerate(ranked, 1)}
    order = {row["case"]: i for i, row in enumerate(rows, 1)}
    params = {row["case"]: parameters(row["config"]) for row in rows}
    common = ["mtp", *PARAMETERS, "graph_mode", "cpu_binding", "omp_threads", "hccl_buffsize"]
    extra = sorted(set().union(*(set(p) for p in params.values())) - set(common)) if rows else []
    keys = set().union(*(row["evaluation"]["metrics"] for row in rows)) if rows else set()
    metrics = [key for key in METRICS if key in keys] + sorted(keys - set(METRICS))
    fields = ["rank", "execution_order", "case", "phase", "mode", "status",
              *common, *metrics, *extra, "config_json"]
    for filename, ordered in (("summary.csv", rows), ("ranking.csv", ranked)):
        target = session / filename
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in ordered:
                result = dict(rank=rank[row["case"]], execution_order=order[row["case"]],
                              case=row["case"], phase=row["phase"], mode=row["mode"],
                              status=row["evaluation"]["status"], **params[row["case"]],
                              **row["evaluation"]["metrics"], config_json=cell(row["config"]))
                writer.writerow(result)
        temporary.replace(target)
    save(session / "summary.json", {"selection": selection, "ranking": ranked,
                                    "execution_order": [row["case"] for row in rows]})
    varying = [key for key in [*common, *extra]
               if len({str(params[row["case"]].get(key, "")) for row in rows}) > 1]
    # The compact Markdown view emphasizes parameters changed in this session.
    shown = [key for key in ("mean_decode_tps", "output_tps", "min_decode_tps", "prefill_per_request_mean_tps", "all_first_tokens_ms") if key in keys]
    header = ["Case", "Phase", "Mode", "Status", *varying, *shown]
    def md(value):
        return str(cell(value)).replace("|", "\\|").replace("\n", "<br>")
    text = ["# AutoTune results", "", "Each SLO is checked for every dataset/repeat; medians are for ranking only.",
            "", "CSV files include all configured parameters. summary.csv follows execution order; ranking.csv follows rank.",
            "Parameter columns describe configured values, not backend-inferred effective defaults. Latency columns ending in _ms use milliseconds.",
            "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in ranked:
        values = [row["case"], row["phase"], row["mode"], row["evaluation"]["status"],
                  *[params[row["case"]].get(key, "") for key in varying],
                  *[row["evaluation"]["metrics"].get(key, "") for key in shown]]
        text.append("| " + " | ".join(md(v) for v in values) + " |")
    (session / "summary.md").write_text("\n".join(text) + "\n")


def refresh(session):
    """Re-render a completed session from its saved rows, with report backups."""
    session = Path(session).resolve()
    controller = session.parent.parent / "state/controller.json"
    if controller.is_file():
        active = json.loads(controller.read_text())
        if Path(active.get("session", "")).resolve() == session:
            # Fail closed even on a stale marker; do not race a live writer.
            raise ValueError("session has a controller marker; do not refresh an active run")
    summary = json.loads((session / "summary.json").read_text())
    rows = summary["ranking"]
    by_case = {row["case"]: row for row in rows}
    if len(by_case) != len(rows):
        raise ValueError("duplicate case IDs in saved summary")
    if (session / "summary.csv").is_file():
        with (session / "summary.csv").open(newline="") as file:
            ids = [row["case"] for row in csv.DictReader(file)]
    else:
        ids = summary.get("execution_order", [])
    if len(ids) != len(rows) or set(ids) != set(by_case):
        raise ValueError("cannot reliably recover execution order from saved reports")
    backup = session / "report_backups" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    for filename in ("summary.csv", "ranking.csv", "summary.md", "summary.json"):
        if (session / filename).is_file():
            shutil.copy2(session / filename, backup / filename)
    report(session, [by_case[key] for key in ids], summary["selection"])
    return backup
