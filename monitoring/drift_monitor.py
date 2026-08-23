"""
Model performance tracking (post-deployment).

Reads a JSONL log of {predicted_label, true_label} pairs (produced by
scripts/simulate_traffic.py against a running deployment, or by a real
feedback loop) and computes rolling accuracy vs. a training-time baseline.
Flags degradation if accuracy drops more than `threshold` below baseline.

Usage:
    python -m monitoring.drift_monitor --log requests_log.jsonl --baseline-acc 0.90
"""
import argparse
import json
from pathlib import Path


def compute_accuracy(log_path: str) -> float:
    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No records found in log")
    correct = sum(1 for r in records if r["predicted_label"] == r["true_label"])
    return correct / len(records)


def check_drift(log_path: str, baseline_acc: float, threshold: float = 0.05) -> dict:
    current_acc = compute_accuracy(log_path)
    degraded = (baseline_acc - current_acc) > threshold
    report = {
        "baseline_acc": baseline_acc,
        "current_acc": round(current_acc, 4),
        "delta": round(baseline_acc - current_acc, 4),
        "degraded": degraded,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="JSONL file of predicted_label/true_label pairs")
    parser.add_argument("--baseline-acc", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    report = check_drift(args.log, args.baseline_acc, args.threshold)
    if report["degraded"]:
        raise SystemExit("Model performance has degraded beyond threshold")
