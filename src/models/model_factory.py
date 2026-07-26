"""
model_factory.py

Builds transfer learning and lightweight models for the maize leaf disease project.

Supported models:
- ResNet18
- EfficientNet-B0
- MobileNetV2

Each model is modified for multi-label classification by replacing the final
classification layer with an 8-output layer.

The model outputs raw logits. Sigmoid is applied later during evaluation,
not inside the model.
"""

import torch.nn as nn
from torchvision import models


def set_trainable_layers(model, freeze_backbone: bool):
    """
    Controls whether the pretrained backbone is frozen or fully fine-tuned.

    freeze_backbone = True:
        Only the final classifier layer is trained.

    freeze_backbone = False:
        All layers are trainable. This is full fine-tuning.
    """

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model


def build_model(
    model_name: str,
    num_labels: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
):
    """
    Builds a pretrained CNN model for multi-label maize disease classification.

    Args:
        model_name:
            Name of the model to build.
            Supported values: resnet18, efficientnet_b0, mobilenet_v2

        num_labels:
            Number of output labels. For this project, this should be 8.

        pretrained:
            Whether to use pretrained ImageNet weights.

        freeze_backbone:
            If True, only the final classifier head trains.
            If False, all model layers train.

    Returns:
        model:
            A PyTorch model with an 8-output classification head.
    """

    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        model = set_trainable_layers(model, freeze_backbone)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_labels)

        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        model = set_trainable_layers(model, freeze_backbone)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_labels)

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        model = set_trainable_layers(model, freeze_backbone)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_labels)

        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            "Choose from: resnet18, efficientnet_b0, mobilenet_v2"
        )

    return model


def count_total_parameters(model):
    """
    Counts all parameters in the model.
    """

    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """
    Counts only parameters that will be updated during training.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)