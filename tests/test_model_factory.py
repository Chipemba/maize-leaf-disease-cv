# This proves the models are structurally correct before you spend time training them.

# run : pytest tests/test_model_factory.py -q
# output : 3 passed

import torch

from src.models.model_factory import build_model


def test_resnet18_output_shape():
    model = build_model("resnet18", num_labels=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 8)


def test_efficientnet_b0_output_shape():
    model = build_model("efficientnet_b0", num_labels=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 8)


def test_mobilenet_v2_output_shape():
    model = build_model("mobilenet_v2", num_labels=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 8)