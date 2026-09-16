"""AISBench raw detail reader, including read-only SQLite/NPY references.

Times are seconds internally. TPOT and decode rate follow AISBench's exclusion
of the first output token. Never divide benchmark-wide output TPS by concurrency.
"""
import csv
import io
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path


def number(value):
    match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+.*)?", str(value))
    if not match:
        return None
    val = float(match[1])
    return val if math.isfinite(val) else None


def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    left, right = math.floor(at), math.ceil(at)
    return values[left] + (values[right] - values[left]) * (at - left)


def times(raw, source):
    value = raw.get("time_points")
    if isinstance(value, dict) and "__db_ref__" in value:
        import numpy as np  # Included in the AISBench environment; allow_pickle=False is mandatory.
        dbname = raw.get("db_name", "")
        if Path(dbname).name != dbname or not dbname.endswith(".db"):
            raise ValueError("invalid AISBench db_name")
        path = (source.parent / "db_data" / dbname).resolve()
        if not path.is_file() or path.parent != (source.parent / "db_data").resolve():
            raise ValueError("AISBench detail database missing")
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            result = db.execute("SELECT arr_blob FROM numpy_store WHERE id=?", (value["__db_ref__"],)).fetchone()
        if result is None:
            raise ValueError("uncommitted/missing AISBench timing record")
        arr = np.load(io.BytesIO(result[0]), allow_pickle=False)
        if arr.ndim != 1 or arr.dtype.kind not in "fiu":
            raise ValueError("invalid timing array")
        value = arr.tolist()
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("raw per-request time_points are required for decode measurement")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError("non-finite time point")
    if any(a > b for a, b in zip(value, value[1:])):
        raise ValueError("non-monotonic request timestamps")
    return value


def extract(root, expected, output_tokens, input_tokens=None, input_tolerance=0, concurrency=1, time_scale=1.0):
    root = Path(root)
    # Each invocation measures ONE dataset. Reject duplicate reports from accidental reruns.
    reports = [p for p in root.glob("**/performances/**/*.json") if "db_data" not in p.parts]
    summaries = []
    for path in reports:
        data = json.loads(path.read_text())
        if "Total Requests" in data:
            summaries.append(data)
    if len(summaries) != 1:
        raise ValueError(f"expected one performance summary, got {len(summaries)}")
    data = summaries[0]
    metrics = {}
    for source, target in (("Output Token Throughput", "output_tps"), ("Request Throughput", "request_tps"),
                           ("Prefill Token Throughput", "aisbench_prefill_tps"),
                           ("Failed Requests", "failed_requests"), ("Success Requests", "success_requests"),
                           ("Total Requests", "total_requests"), ("Concurrency", "average_inflight")):
        metrics[target] = number(data.get(source, {}).get("total"))
    if metrics["total_requests"] != expected or metrics["success_requests"] is None or metrics["failed_requests"] is None:
        raise ValueError("missing counts or wrong formal sample count")
    if metrics["success_requests"] + metrics["failed_requests"] != expected:
        raise ValueError("inconsistent AISBench request counts")
    requests = []
    seen = set()
    failures = 0
    for path in root.glob("**/performances/**/*_details.jsonl"):
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            key = (raw.get("data_abbr"), raw.get("uuid", raw.get("id")), raw.get("turn_id", 0))
            if key in seen:
                raise ValueError("duplicate request detail")
            seen.add(key)
            if raw.get("success") is not True:
                failures += 1
                continue
            t = [x * time_scale for x in times(raw, path)]
            n, inp = raw.get("output_tokens"), raw.get("input_tokens")
            if type(n) is not int or n != output_tokens or n <= 1:
                raise ValueError(f"unexpected actual output token count: {n}; expected {output_tokens} (>1)")
            if not isinstance(inp, int) or inp <= 0:
                raise ValueError("actual input token count missing")
            if input_tokens is not None and abs(inp - input_tokens) > input_tolerance:
                raise ValueError(f"server input length {inp} != requested {input_tokens}; possible truncation/template mismatch")
            if t[1] <= t[0] or t[-1] <= t[1]:
                raise ValueError("zero/negative prefill or decode duration")
            requests.append({"request_id": str(key[1]), "start_s": t[0], "first_token_s": t[1], "end_s": t[-1],
                "input_tokens": inp, "output_tokens": n, "ttft_ms": (t[1]-t[0])*1000,
                "tpot_ms": (t[-1]-t[1])*1000/(n-1), "decode_tps": (n-1)/(t[-1]-t[1]),
                "e2e_ms": (t[-1]-t[0])*1000, "effective_prefill_tps": inp/(t[1]-t[0])})
    if len(seen) != expected or failures != metrics["failed_requests"]:
        raise ValueError("raw request detail count differs from summary; cannot verify per-request SLO")
    if not requests:
        raise ValueError("no successful request details")
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        vals = [r[name] for r in requests]
        metrics[name] = statistics.mean(vals)
        for q in (50, 90, 95, 99):
            metrics[name.replace("_ms", f"_p{q}_ms")] = percentile(vals, q / 100)
    start = min(r["start_s"] for r in requests)
    latest_first = max(r["first_token_s"] for r in requests)
    metrics.update(mean_decode_tps=statistics.mean(r["decode_tps"] for r in requests),
        min_decode_tps=min(r["decode_tps"] for r in requests),
        max_tpot_ms=max(r["tpot_ms"] for r in requests),
        prefill_per_request_mean_tps=statistics.mean(r["effective_prefill_tps"] for r in requests),
        prefill_batch_effective_tps=sum(r["input_tokens"] for r in requests)/(latest_first-start),
        first_response_ms=(min(r["first_token_s"] for r in requests)-start)*1000,
        all_first_tokens_ms=(latest_first-start)*1000,
        submit_span_ms=(max(r["start_s"] for r in requests)-start)*1000)
    # Record actual overlap, not only the configured maximum concurrency.
    for label, begin in (("peak_inflight", "start_s"), ("peak_decode", "first_token_s")):
        events = sorted([(r[begin], 1) for r in requests] + [(r["end_s"], -1) for r in requests])
        active = peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        metrics[label] = peak
    if metrics["peak_inflight"] < concurrency:
        raise ValueError(f"actual peak concurrency {metrics['peak_inflight']} < configured {concurrency}")
    if expected != concurrency:
        # Batch first/last-token SLO is defined only for one synchronized wave.
        metrics["first_response_ms"] = None
        metrics["all_first_tokens_ms"] = None
        metrics["prefill_batch_effective_tps"] = None
    return metrics, sorted(requests, key=lambda x: x["start_s"])
