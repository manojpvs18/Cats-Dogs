"""Model loading + prediction utility shared by the FastAPI service and tests."""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from src.models.model import CLASS_NAMES, build_model


def load_model(model_path: str, device: str = "cpu"):
    """Load a serialized checkpoint (state_dict + class_names) into a fresh
    BaselineCNN. Returns (model, class_names)."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(224,224,3) uint8 RGB array -> normalized (1,3,224,224) float tensor."""
    x = arr.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(x).unsqueeze(0)


def predict(model, class_names, image: Image.Image) -> Dict:
    """Run inference on a PIL image. Returns label + per-class probabilities.
    This is the pure function covered by the inference unit test — it takes
    an already-constructed model so tests don't need a real checkpoint."""
    arr = preprocess_image(image)
    tensor = _to_tensor(arr)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "label": class_names[pred_idx],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
    }
