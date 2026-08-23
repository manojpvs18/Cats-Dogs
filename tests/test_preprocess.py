import numpy as np
from PIL import Image

from src.data.preprocess import IMG_SIZE, preprocess_image


def test_preprocess_image_resizes_to_224():
    img = Image.new("RGB", (500, 333), color=(10, 20, 30))
    out = preprocess_image(img)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_preprocess_image_converts_to_rgb():
    img = Image.new("L", (100, 100), color=128)  # grayscale input
    out = preprocess_image(img)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert out.dtype == np.uint8


def test_preprocess_image_handles_rgba():
    img = Image.new("RGBA", (300, 300), color=(1, 2, 3, 255))
    out = preprocess_image(img)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3)
