"""
check_onnx_output.py

Basic sanity check to compare PyTorch and ONNX output shapes.
This does not require a real image.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.data.dataset import LABELS
from src.models.model_factory import build_model


CHECKPOINT_PATH = Path("models/mobilenet_v2.pt")
ONNX_PATH = Path("onnx/mobilenet_v2.onnx")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not ONNX_PATH.exists():
        raise FileNotFoundError(ONNX_PATH)

    device = torch.device("cpu")

    model = build_model(
        model_name="mobilenet_v2",
        num_labels=len(LABELS),
        pretrained=False,
        freeze_backbone=False
    )

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_output = session.run(
        [output_name],
        {input_name: dummy_input.numpy()}
    )[0]

    print("PyTorch output shape:", torch_output.shape)
    print("ONNX output shape:", onnx_output.shape)

    max_diff = np.max(np.abs(torch_output - onnx_output))
    print("Max absolute difference:", max_diff)


if __name__ == "__main__":
    main()