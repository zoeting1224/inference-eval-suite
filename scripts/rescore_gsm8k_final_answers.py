#!/usr/bin/env python3
"""Audit and correct the four completed Qwen3.8 GSM8K-200 result directories.

The original responses and manifests are retained. With --apply, original
result files and comparison reports are backed up before strict rescoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_suite.accuracy import compare, normalize_number  # noqa: E402


RUNS = {
    "bf16": "qwen38-bf16-gsm8k-200",
    "l28-35": "qwen38-w4a8-l28-35-gsm8k-200",
    "l8-55": "qwen38-w4a8-l8-55-gsm8k-200",
    "l0-63": "qwen38-w4a8-l0-63-gsm8k-200",
}
COMPARISONS = {
    "l28-35": "qwen38-bf16-vs-w4a8",
    "l8-55": "qwen38-bf16-vs-w4a8-l8-55",
    "l0-63": "qwen38-bf16-vs-w4a8-l0-63",
}
FINAL_ANSWER_PATTERNS = (
    re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"),
)
SCORING_RULE = "explicit_final_answer_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_prediction(response: str) -> str | None:
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(response)
        if matches:
            return normalize_number(matches[-1])
    return None


def corrected_run(directory: Path) -> tuple[list[dict], dict, dict, dict]:
    manifest = load_json(directory / "manifest.json")
    original = load_json(directory / "summary.json")
    if original.get("scoring_rule") == SCORING_RULE:
        raise ValueError(f"already strictly rescored: {directory}")
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = [int(row["source_index"]) for row in rows]
    if len(rows) != manifest["selected_count"] or len(indices) != len(set(indices)):
        raise ValueError(f"incomplete or duplicate predictions: {directory}")
    if indices != manifest["source_indices"]:
        raise ValueError(f"prediction order differs from manifest: {directory}")
    limit = int(manifest["generation"]["max_tokens"])
    changed = []
    limit_hits = 0
    limit_without_final = 0
    for row in rows:
        if row["status"] == "request_error":
            continue
        content = row.get("response")
        if not isinstance(content, str):
            raise ValueError(f"missing response for source_index={row['source_index']}: {directory}")
        answer = strict_prediction(content)
        hit_limit = int((row.get("usage") or {}).get("completion_tokens") or 0) >= limit
        limit_hits += hit_limit
        limit_without_final += hit_limit and answer is None
        new_correct = answer is not None and answer == row["gold"]
        if (row["prediction"], bool(row["correct"]), row["status"]) != (
            answer, new_correct, "ok" if answer is not None else "parse_error"
        ):
            changed.append(int(row["source_index"]))
        row["prediction"] = answer
        row["correct"] = new_correct
        row["status"] = "ok" if answer is not None else "parse_error"
    correct = sum(bool(row["correct"]) for row in rows)
    request_errors = sum(row["status"] == "request_error" for row in rows)
    parse_errors = sum(row["status"] == "parse_error" for row in rows)
    summary = {
        **original,
        "correct": correct,
        "accuracy": correct / original["total"],
        "request_errors": request_errors,
        "parse_errors": parse_errors,
        "scoring_rule": SCORING_RULE,
        "output_limit_hits": limit_hits,
        "output_limit_without_final_answer": limit_without_final,
    }
    audit = {
        "scoring_rule": SCORING_RULE,
        "model_rerun": False,
        "original_correct": original["correct"],
        "corrected_correct": correct,
        "original_accuracy": original["accuracy"],
        "corrected_accuracy": summary["accuracy"],
        "changed_source_indices": changed,
        "output_limit_hits": limit_hits,
        "output_limit_without_final_answer": limit_without_final,
        "original_predictions_sha256": hashlib.sha256((directory / "predictions.jsonl").read_bytes()).hexdigest(),
    }
    return rows, summary, audit, manifest


def atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="back up and update result files")
    args = parser.parse_args()
    accuracy_root = args.root / "runs" / "accuracy"
    prepared = {key: corrected_run(accuracy_root / name) for key, name in RUNS.items()}
    hashes = {manifest["workload_hash"] for _, _, _, manifest in prepared.values()}
    generations = {json.dumps(manifest["generation"], sort_keys=True) for _, _, _, manifest in prepared.values()}
    if len(hashes) != 1 or len(generations) != 1:
        raise ValueError("workload or generation settings differ across the four runs")
    for key, (_, summary, audit, _) in prepared.items():
        print(f"{key}: {audit['original_correct']}/200 -> {summary['correct']}/200; "
              f"parse_errors={summary['parse_errors']}; output_limit_hits={audit['output_limit_hits']}")
    if not args.apply:
        print("Dry run only. Pass --apply to back up and correct these files.")
        return

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = accuracy_root / f"_before_strict_rescore_{timestamp}"
    backup.mkdir()
    for name in (*RUNS.values(), *COMPARISONS.values()):
        source = accuracy_root / name
        target = backup / name
        target.mkdir()
        for file in source.iterdir():
            if file.is_file():
                shutil.copy2(file, target / file.name)

    for key, (rows, summary, audit, _) in prepared.items():
        directory = accuracy_root / RUNS[key]
        audit["backup_directory"] = str(backup)
        audit["rescored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_text(directory / "predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        atomic_text(directory / "summary.json", json_text(summary))
        atomic_text(directory / "summary.md", (
            "# Fixed-config accuracy result — strict final-answer rescoring\n\n"
            f"- Model: `{summary['model']}`\n"
            f"- Dataset: `{summary['dataset']}`\n"
            f"- Accuracy: **{summary['accuracy'] * 100:.2f}%** ({summary['correct']}/{summary['total']})\n"
            f"- Scoring rule: `{SCORING_RULE}` (`####` or `\\boxed{{}}`; no last-number fallback)\n"
            f"- Request errors: {summary['request_errors']}\n"
            f"- Missing final answer: {summary['parse_errors']}\n"
            f"- Output-limit hits: {summary['output_limit_hits']}\n"
            f"- Output-limit hits without final answer: {summary['output_limit_without_final_answer']}\n"
            f"- Workload hash: `{summary['workload_hash']}`\n"
            f"- Original score: {audit['original_correct']}/{summary['total']} (backup: `{backup.name}`)\n"
            "- Original model responses preserved in `predictions.jsonl`; no model rerun.\n"
        ))
        atomic_text(directory / "rescoring_audit.json", json_text(audit))
    for key, name in COMPARISONS.items():
        compare(accuracy_root / RUNS["bf16"], accuracy_root / RUNS[key], accuracy_root / name)
        report = accuracy_root / name / "comparison.md"
        atomic_text(report, report.read_text(encoding="utf-8").replace(
            "# Accuracy comparison\n", "# Accuracy comparison — strict final-answer rescoring\n\n"
            f"Original reports are preserved in `{backup.name}`. No model was rerun.\n", 1
        ))
    table = [
        "# Qwen3.8 GSM8K-200 quantization-range comparison",
        "",
        "All four runs use the same workload hash and generation settings. These are rescored",
        "existing responses, not new model runs. Only an explicit `####` or `\\boxed{}`",
        "final answer counts; the last-number fallback is disabled.",
        "",
        "| Model | Original score | Corrected score | Change vs BF16 | Output-limit hits | Missing final answer |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline = prepared["bf16"][1]["accuracy"]
    for key, name in RUNS.items():
        _, summary, audit, _ = prepared[key]
        table.append(
            f"| {summary['model']} | {audit['original_correct']}/{summary['total']} "
            f"({audit['original_accuracy'] * 100:.2f}%) | "
            f"{summary['correct']}/{summary['total']} ({summary['accuracy'] * 100:.2f}%) | "
            f"{(summary['accuracy'] - baseline) * 100:+.2f} pp | "
            f"{summary['output_limit_hits']} | {summary['parse_errors']} |"
        )
    table += [
        "",
        f"Workload hash: `{next(iter(hashes))}`. Original files: `{backup.name}`.",
        "",
        "The 512-token cap still affects this historical comparison. A longer-output rerun",
        "is needed before attributing score differences to quantization quality.",
    ]
    atomic_text(accuracy_root / "quantization_range_comparison.md", "\n".join(table) + "\n")
    print(f"Backups: {backup}")
    print("Corrected all four run results and regenerated three BF16 comparisons.")


if __name__ == "__main__":
    main()
