#!/usr/bin/env python3
"""Evaluate a fixed OpenAI-compatible chat endpoint on GSM8K.

This runner intentionally does not start or stop the model service. It records
the selected dataset rows and generation settings so a baseline and a quantized
checkpoint can be compared with exactly the same workload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import decimal
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare(baseline_dir: Path, candidate_dir: Path, output_dir: Path) -> dict:
    """Compare two completed accuracy runs with the same frozen workload."""
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    manifests = [load_json(path / "manifest.json") for path in (baseline_dir, candidate_dir)]
    summaries = [load_json(path / "summary.json") for path in (baseline_dir, candidate_dir)]
    for key in ("workload_hash", "dataset_sha256", "source_indices"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"accuracy comparison refused: {key} differs")
    if manifests[0].get("generation") != manifests[1].get("generation"):
        raise ValueError("accuracy comparison refused: generation settings differ")
    baseline, candidate = summaries
    result = {
        "workload_match": True,
        "workload_hash": manifests[0]["workload_hash"],
        "baseline": {"directory": str(baseline_dir.resolve()), **baseline},
        "candidate": {"directory": str(candidate_dir.resolve()), **candidate},
        "accuracy_change_percentage_points":
            (candidate["accuracy"] - baseline["accuracy"]) * 100.0,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", result)
    (output_dir / "comparison.md").write_text(
        "# Accuracy comparison\n\n"
        f"- Workload match: **PASS**\n"
        f"- Baseline: **{baseline['accuracy'] * 100:.2f}%** ({baseline['correct']}/{baseline['total']})\n"
        f"- Candidate: **{candidate['accuracy'] * 100:.2f}%** ({candidate['correct']}/{candidate['total']})\n"
        f"- Change: **{result['accuracy_change_percentage_points']:+.2f} percentage points**\n"
        f"- Workload hash: `{result['workload_hash']}`\n",
        encoding="utf-8",
    )
    print(output_dir / "comparison.md")
    return result


def extract_number(text: str) -> str | None:
    patterns = (
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
    )
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return normalize_number(found[-1])
    found = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(found[-1]) if found else None


def normalize_number(value: str) -> str | None:
    try:
        number = decimal.Decimal(value.replace(",", ""))
    except decimal.InvalidOperation:
        return None
    if number == number.to_integral():
        return str(number.quantize(decimal.Decimal("1")))
    return format(number.normalize(), "f")


def post_json(url: str, payload: dict, timeout_s: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main(argv=None, root=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="override config sample limit; 0 means full split")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(root) if root else Path(__file__).resolve().parents[1]

    config = load_json(args.config)
    dataset_cfg = config["dataset"]
    endpoint = config["endpoint"]
    generation = config["generation"]
    limit = config.get("limit", 200) if args.limit is None else args.limit

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"

    dataset_path = Path(dataset_cfg["file"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    dataset = []
    for source_index, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            row["_source_index"] = source_index
            dataset.append(row)
    random.Random(config.get("sample_seed", 1024)).shuffle(dataset)
    if limit:
        dataset = dataset[: min(limit, len(dataset))]

    selected = [
        {
            "source_index": row["_source_index"],
            "question": row[dataset_cfg.get("question_field", "question")],
            "gold": extract_number(row[dataset_cfg.get("answer_field", "answer")]),
        }
        for row in dataset
    ]
    workload_hash = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    completed: dict[int, dict] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["source_index"])] = row

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "endpoint": endpoint,
        "generation": generation,
        "dataset": dataset_cfg,
        "dataset_file": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "sample_seed": config.get("sample_seed", 1024),
        "selected_count": len(selected),
        "workload_hash": workload_hash,
        "source_indices": [row["source_index"] for row in selected],
    }
    write_json(output / "manifest.json", manifest)

    prompt_template = config.get(
        "prompt_template",
        "Solve the problem. Return the final answer as: #### <number>\n\nQuestion:\n{question}",
    )
    for position, item in enumerate(selected, start=1):
        source_index = item["source_index"]
        if source_index in completed:
            print(f"[{position}/{len(selected)}] source={source_index} resumed")
            continue

        payload = {
            "model": endpoint["model"],
            "messages": [
                {"role": "user", "content": prompt_template.format(question=item["question"])}
            ],
            **generation,
        }
        started = time.monotonic()
        record = {
            "position": position,
            "source_index": source_index,
            "question": item["question"],
            "gold": item["gold"],
            "prediction": None,
            "response": None,
            "correct": False,
            "status": "request_error",
            "latency_s": None,
            "error": None,
        }
        try:
            response = post_json(endpoint["url"], payload, int(endpoint.get("timeout_s", 300)))
            content = response["choices"][0]["message"]["content"]
            prediction = extract_number(content)
            record.update(
                response=content,
                prediction=prediction,
                correct=prediction is not None and prediction == item["gold"],
                status="ok" if prediction is not None else "parse_error",
                usage=response.get("usage"),
            )
        except Exception as exc:  # keep the full run and count this row as incorrect
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = round(time.monotonic() - started, 6)
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed[source_index] = record
        mark = "PASS" if record["correct"] else record["status"].upper()
        print(
            f"[{position}/{len(selected)}] source={source_index} {mark} "
            f"pred={record['prediction']} gold={record['gold']} latency={record['latency_s']}s",
            flush=True,
        )

    rows = [completed[item["source_index"]] for item in selected if item["source_index"] in completed]
    correct = sum(bool(row["correct"]) for row in rows)
    request_errors = sum(row["status"] == "request_error" for row in rows)
    parse_errors = sum(row["status"] == "parse_error" for row in rows)
    total = len(selected)
    summary = {
        "model": endpoint["model"],
        "dataset": str(dataset_path.resolve()),
        "total": total,
        "completed": len(rows),
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "request_errors": request_errors,
        "parse_errors": parse_errors,
        "workload_hash": workload_hash,
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        "# Fixed-config accuracy result\n\n"
        f"- Model: `{summary['model']}`\n"
        f"- Dataset: `{summary['dataset']}`\n"
        f"- Accuracy: **{summary['accuracy'] * 100:.2f}%** ({correct}/{total})\n"
        f"- Request errors: {request_errors}\n"
        f"- Parse errors: {parse_errors}\n"
        f"- Workload hash: `{workload_hash}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(rows) == total else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun with the same output directory to resume.", file=sys.stderr)
        raise SystemExit(130)
