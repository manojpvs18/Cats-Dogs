import pytest
from PIL import Image

from src.models.inference import predict
from src.models.model import CLASS_NAMES, build_model


@pytest.fixture
def dummy_model():
    # Freshly initialized (untrained) model — sufficient to test the
    # inference *contract* (shapes, labels, probability normalization)
    # without requiring a trained checkpoint.
    model = build_model(num_classes=2)
    model.eval()
    return model


def test_predict_returns_valid_label(dummy_model):
    img = Image.new("RGB", (250, 250), color=(120, 80, 40))
    result = predict(dummy_model, CLASS_NAMES, img)
    assert result["label"] in CLASS_NAMES


def test_predict_probabilities_sum_to_one(dummy_model):
    img = Image.new("RGB", (180, 400), color=(200, 200, 200))
    result = predict(dummy_model, CLASS_NAMES, img)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_probability_keys_match_class_names(dummy_model):
    img = Image.new("RGB", (224, 224))
    result = predict(dummy_model, CLASS_NAMES, img)
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
