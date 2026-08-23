"""
Train the baseline CNN on data/processed/{train,val,test} and log the run
to MLflow: params, per-epoch metrics, confusion matrix, loss-curve, and the
serialized model artifact.

Usage:
    python -m src.models.train --data data/processed --epochs 10
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.preprocess import build_transforms
from src.models.model import CLASS_NAMES, build_model


def get_loaders(data_dir: str, batch_size: int = 32):
    train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=build_transforms(augment=True))
    val_ds = ImageFolder(os.path.join(data_dir, "val"), transform=build_transforms(augment=False))
    test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=build_transforms(augment=False))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def evaluate_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("cats-vs-dogs")

    train_loader, val_loader, test_loader = get_loaders(args.data, args.batch_size)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.artifacts_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    with mlflow.start_run():
        mlflow.log_params({
            "model": "BaselineCNN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "img_size": 224,
            "augmentation": True,
        })

        for epoch in range(args.epochs):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)
            mlflow.log_metrics({
                "train_loss": tr_loss, "val_loss": val_loss,
                "train_acc": tr_acc, "val_acc": val_acc,
            }, step=epoch)
            print(f"epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

        test_acc, test_f1, cm = evaluate_test(model, test_loader, device)
        mlflow.log_metrics({"test_acc": test_acc, "test_f1": test_f1})

        # Loss curve artifact
        fig, ax = plt.subplots()
        ax.plot(history["train_loss"], label="train_loss")
        ax.plot(history["val_loss"], label="val_loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
        loss_path = os.path.join(args.artifacts_dir, "loss_curve.png")
        fig.savefig(loss_path); plt.close(fig)
        mlflow.log_artifact(loss_path)

        # Confusion matrix artifact
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        cm_path = os.path.join(args.artifacts_dir, "confusion_matrix.png")
        fig.savefig(cm_path); plt.close(fig)
        mlflow.log_artifact(cm_path)

        # Serialized model
        model_path = os.path.join(args.artifacts_dir, "model.pt")
        torch.save({"state_dict": model.state_dict(), "class_names": CLASS_NAMES}, model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "model")

        print(f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}")
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--artifacts-dir", default="artifacts")
    main(parser.parse_args())
