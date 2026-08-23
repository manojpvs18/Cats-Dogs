"""Baseline CNN for binary cats-vs-dogs classification (224x224x3 input)."""
import torch
import torch.nn as nn

CLASS_NAMES = ["cat", "dog"]  # index 0 = cat, 1 = dog


class BaselineCNN(nn.Module):
    """Small 4-block conv net. ~1.6M params — fast to train as a baseline
    before swapping in a transfer-learning backbone."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 224->112
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # ->56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # ->28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # ->14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = 2) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes)
