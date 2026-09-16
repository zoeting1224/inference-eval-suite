import json
import tempfile
import unittest
from pathlib import Path

from inference_suite.accuracy import compare


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_run(path, outcomes):
    path.mkdir()
    indices = sorted(outcomes)
    write_json(
        path / "manifest.json",
        {
            "workload_hash": "same-workload",
            "dataset_sha256": "same-dataset",
            "source_indices": indices,
            "generation": {"temperature": 0.0, "max_tokens": 512},
        },
    )
    correct = sum(bool(value) for value in outcomes.values())
    write_json(
        path / "summary.json",
        {
            "model": path.name,
            "total": len(indices),
            "completed": len(indices),
            "correct": correct,
            "accuracy": correct / len(indices),
            "request_errors": 0,
            "parse_errors": 0,
            "workload_hash": "same-workload",
        },
    )
    with (path / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for source_index in indices:
            is_correct = outcomes[source_index]
            f.write(
                json.dumps(
                    {
                        "source_index": source_index,
                        "question": f"question {source_index}",
                        "gold": str(source_index),
                        "prediction": str(source_index) if is_correct else "wrong",
                        "correct": is_correct,
                        "status": "ok",
                        "error": None,
                    }
                )
                + "\n"
            )


class AccuracyCompareTests(unittest.TestCase):
    def test_paired_outcomes_and_disagreement_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            output = root / "comparison"
            write_run(baseline, {1: True, 2: False, 3: True, 4: False})
            write_run(candidate, {1: True, 2: False, 3: False, 4: True})

            result = compare(baseline, candidate, output)

            self.assertEqual(
                result["paired_outcomes"],
                {
                    "both_correct": 1,
                    "both_wrong": 1,
                    "baseline_correct_candidate_wrong": 1,
                    "baseline_wrong_candidate_correct": 1,
                    "net_candidate_gain": 0,
                },
            )
            rows = [
                json.loads(line)
                for line in (output / "disagreements.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["source_index"] for row in rows], [3, 4])
            self.assertIn("Baseline correct / Candidate wrong | 1", (output / "comparison.md").read_text())
            detail = (output / "disagreements.md").read_text()
            self.assertIn("source_index=3", detail)
            self.assertIn("source_index=4", detail)

    def test_missing_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_run(baseline, {1: True, 2: False})
            write_run(candidate, {1: True, 2: False})
            lines = (candidate / "predictions.jsonl").read_text().splitlines()
            (candidate / "predictions.jsonl").write_text(lines[0] + "\n")
            with self.assertRaisesRegex(ValueError, "source_index 2 missing"):
                compare(baseline, candidate, root / "comparison")


if __name__ == "__main__":
    unittest.main()
