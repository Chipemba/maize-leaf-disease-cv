"""
export_onnx.py
Exports the trained MobileNetV2 maize disease model from PyTorch to ONNX.

Expected input used for the model is a 3-channel RGB image of size 224x224.:
- models/mobilenet_v2.pt

Expected output:
- onnx/mobilenet_v2.onnx
"""

from pathlib import Path

import torch

from src.data.dataset import LABELS
from src.models.model_factory import build_model


MODEL_NAME = "mobilenet_v2"
CHECKPOINT_PATH = Path("models/mobilenet_v2.pt")
ONNX_OUTPUT_PATH = Path("onnx/mobilenet_v2.onnx")

IMAGE_SIZE = 224
NUM_LABELS = len(LABELS)


def main():
    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}. "
            "Train MobileNetV2 first before exporting to ONNX."
        )

    device = torch.device("cpu")

    model = build_model(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        pretrained=False,
        freeze_backbone=False
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    dynamo=False,
)

    print(f"ONNX model exported to: {ONNX_OUTPUT_PATH}")
    print(f"Number of labels: {NUM_LABELS}")
    print("Labels:", LABELS)


if __name__ == "__main__":
    main()