"""
Data pre-processing utilities for the Cats vs Dogs dataset.

- preprocess_image(): resize any PIL image to 224x224 RGB (unit-testable,
  no dataset required).
- build_transforms(): torchvision transform pipelines (train has augmentation,
  val/test do not).
- split_dataset(): deterministic 80/10/10 split of an ImageFolder-style
  directory into data/processed/{train,val,test} (copies files, so the split
  itself is reproducible and DVC-trackable).
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Convert a PIL image to a (size, size, 3) uint8 RGB numpy array.

    Pure function, independent of any dataset — used both by the offline
    preprocessing pipeline and by the inference service, and is covered by
    a unit test.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    arr = np.array(image, dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError(f"Unexpected preprocessed shape: {arr.shape}")
    return arr


def build_transforms(augment: bool):
    """Return a torchvision transform pipeline.

    augment=True adds flips/rotation/color-jitter for the training split;
    val/test use only deterministic resize + normalize.
    """
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not augment:
        return transforms.Compose(base)

    aug = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(aug)


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Split `input_dir/{class_name}/*.jpg` into
    `output_dir/{train,val,test}/{class_name}/*.jpg` (80/10/10 by default).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    classes = sorted([p.name for p in input_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class subfolders found under {input_dir}")

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}

    for cls in classes:
        files = sorted((input_dir / cls).glob("*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            dest_dir = output_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                try:
                    img = Image.open(f)
                    arr = preprocess_image(img)
                    Image.fromarray(arr).save(dest_dir / f.name)
                    counts[split_name] += 1
                except Exception as e:  # skip corrupt files, log and continue
                    print(f"skipping {f}: {e}")

    print(f"Split complete: {counts}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw dataset dir with class subfolders")
    parser.add_argument("--output", required=True, help="output dir for processed splits")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()
    split_dataset(args.input, args.output, args.train_frac, args.val_frac)
