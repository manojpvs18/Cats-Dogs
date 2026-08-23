"""
Download the Cats vs Dogs dataset from Kaggle into data/raw/.

Requires a Kaggle API token at ~/.kaggle/kaggle.json
(https://www.kaggle.com/docs/api).

Usage:
    python scripts/download_dataset.py --dataset salader/dogs-vs-cats
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main(dataset: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_path), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Dataset downloaded to {out_path}")
    print("Expected layout: data/raw/cats_and_dogs/{cats,dogs}/*.jpg — "
          "rename/move folders if the Kaggle archive differs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="salader/dogs-vs-cats")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    try:
        main(args.dataset, args.out)
    except FileNotFoundError:
        print("Kaggle CLI not found. Install with: pip install kaggle", file=sys.stderr)
        sys.exit(1)
