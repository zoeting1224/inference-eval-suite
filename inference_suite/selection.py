"""Fail-closed SLO evaluation and lexicographic ranking."""
import math
import statistics


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(repeats, selection):
    constraints = dict(selection.get("constraints", {}))
    constraints["failed_requests"] = {"max": 0}
    required = set(constraints) | {x["metric"] for x in selection["objectives"]}
    checks = []
    for repeat in repeats:
        metrics = repeat["metrics"]
        for key in required:
            value = metrics.get(key)
            limits = constraints.get(key, {})
            status = "INVALID" if not numeric(value) else "PASS"
            if status == "PASS" and any((value < bound if op == "min" else value > bound)
                                        for op, bound in limits.items()):
                status = "FAIL"
            checks.append({"repeat": repeat["name"], "metric": key, "value": value,
                           "limits": limits, "status": status})
    # Missing metrics never become zero, and one bad repeat cannot be hidden by a median.
    status = ("INVALID" if not repeats or any(x["status"] == "INVALID" for x in checks)
              else "SLO_FAIL" if any(x["status"] == "FAIL" for x in checks) else "SLO_PASS")
    keys = set().union(*(r["metrics"] for r in repeats)) if repeats else set()
    aggregate = {}
    for key in keys:
        values = [r["metrics"].get(key) for r in repeats]
        aggregate[key] = statistics.median(values) if all(numeric(x) for x in values) else None
    return {"status": status, "checks": checks, "metrics": aggregate,
            "aggregation": "median for ranking; all dataset/repeat checks must pass"}


def rank_key(row, selection):
    status = row["evaluation"]["status"]
    tier = {"SLO_PASS": 0, "SLO_FAIL": 1, "INVALID": 2, "ERROR": 3}.get(status, 4)
    values = []
    for item in selection["objectives"]:
        value = row["evaluation"]["metrics"].get(item["metric"])
        values.append((value if item["direction"] == "min" else -value) if numeric(value) else float("inf"))
    return (tier, *values, row["case"])


def choose(rows, selection):
    # Failed requests and missing metrics cannot become inherited candidates.
    eligible = [r for r in rows if r["evaluation"]["status"] in ("SLO_PASS", "SLO_FAIL")
                and all(x["metrics"].get("failed_requests") == 0 for x in r["repeats"])]
    return min(eligible, key=lambda r: rank_key(r, selection)) if eligible else None
