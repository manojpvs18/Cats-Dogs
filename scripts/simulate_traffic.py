"""
Send a batch of labeled images to the running inference API and record
predicted vs. true labels as JSONL, for use by monitoring/drift_monitor.py.

Expects a directory laid out like data/processed/test/{cat,dog}/*.jpg so the
true label is taken from the parent folder name.

Usage:
    python scripts/simulate_traffic.py --url http://localhost:8000 \
        --data data/processed/test --n 50 --out requests_log.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import httpx


def main(args):
    data_dir = Path(args.data)
    files = []
    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir():
            for f in cls_dir.glob("*"):
                files.append((f, cls_dir.name))

    random.seed(0)
    random.shuffle(files)
    sample = files[: args.n]

    results = []
    with httpx.Client(timeout=30.0) as client:
        for path, true_label in sample:
            with open(path, "rb") as fh:
                resp = client.post(f"{args.url}/predict", files={"file": (path.name, fh, "image/jpeg")})
            resp.raise_for_status()
            body = resp.json()
            results.append({
                "file": path.name,
                "true_label": true_label,
                "predicted_label": body["label"],
                "latency_ms": body["latency_ms"],
            })

    with open(args.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
    print(f"Sent {len(results)} requests, accuracy={correct/len(results):.4f}, log -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--data", default="data/processed/test")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="requests_log.jsonl")
    main(parser.parse_args())
