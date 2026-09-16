#!/usr/bin/env python3
"""Re-evaluate saved measurements without launching Docker or AISBench.

This is intentionally offline: it only changes derived result/evaluation files
and reports. Raw AISBench logs, requests.csv, configs and workload files remain
untouched. The new decode SLO is the arithmetic mean of per-request decode TPS.
"""
import argparse
import csv
import json
from pathlib import Path

from .config import load, save, validate
from .selection import evaluate
from .reporting import report


def rows_for(session, selection):
    result_paths = sorted((session / "runs").glob("*/result.json"))
    rows = []
    for path in result_paths:
        row = json.loads(path.read_text())
        if row.get("error") and not row.get("repeats"):
            continue
        for repeat in row.get("repeats", []):
            csv_path = path.parent / repeat["name"] / "requests.csv"
            if not csv_path.is_file():
                raise ValueError(f"missing raw request file: {csv_path}")
            rates = [float(x["decode_tps"]) for x in csv.DictReader(csv_path.open())]
            if not rates:
                raise ValueError(f"empty raw request file: {csv_path}")
            repeat["metrics"]["mean_decode_tps"] = sum(rates) / len(rates)
        row["evaluation"] = evaluate(row.get("repeats", []), selection)
        row["measurement_status"] = row["evaluation"]["status"]
        save(path, row)
        save(path.parent / "slo_report.json", row["evaluation"])
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("root", type=Path, help="runs/performance/<experiment> directory")
    args = ap.parse_args(argv)
    config = validate(load(args.config))
    sessions = sorted(args.root.glob("*/runs/*"))
    for session in sessions:
        if not session.is_dir() or not (session / "manifest.json").is_file():
            continue
        rows = rows_for(session, config["selection"])
        if rows:
            # Preserve the recorded execution order where available.
            order = {x: i for i, x in enumerate(json.loads((session / "summary.json").read_text()).get("execution_order", []))} if (session / "summary.json").is_file() else {}
            rows.sort(key=lambda r: order.get(r["case"], 10**9))
            report(session, rows, config["selection"])
            print(session)


if __name__ == "__main__":
    main()
