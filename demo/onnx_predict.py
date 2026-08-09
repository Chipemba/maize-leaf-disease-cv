"""
ONNX inference helper for the maize leaf disease demo.

The model predicts 8 independent probabilities:
GLS, NCLB, PLS, CR, SR, NoFoliarSymptoms, Other, UnidentifiedDisease.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


LABELS = [
    "GLS",
    "NCLB",
    "PLS",
    "CR",
    "SR",
    "NoFoliarSymptoms",
    "Other",
    "UnidentifiedDisease"
]

LABEL_FULL_NAMES = {
    "GLS": "Grey Leaf Spot",
    "NCLB": "Northern Corn Leaf Blight",
    "PLS": "Phaeosphaeria Leaf Spot",
    "CR": "Common Rust",
    "SR": "Southern Rust",
    "NoFoliarSymptoms": "No Foliar Symptoms",
    "Other": "Other",
    "UnidentifiedDisease": "Unidentified Disease"
}

IMAGE_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess_image(image: Image.Image):
    """
    Converts a PIL image into the ONNX model input format:
    [1, 3, 224, 224]
    """

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD

    # Convert from HWC to CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


def load_session(onnx_model_path):
    """
    Loads ONNX model session.
    """

    onnx_model_path = Path(onnx_model_path)

    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"]
    )

    return session


def predict_image(image: Image.Image, onnx_model_path="onnx/mobilenet_v2.onnx", threshold=0.5):
    """
    Runs ONNX inference on one uploaded image.

    Returns:
        results:
            List of dictionaries with label, full name, probability and predicted status.
    """

    session = load_session(onnx_model_path)

    model_input = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run(
        [output_name],
        {input_name: model_input}
    )[0]

    probabilities = sigmoid(logits[0])

    results = []

    for label, probability in zip(LABELS, probabilities):
        results.append({
            "label": label,
            "name": LABEL_FULL_NAMES[label],
            "probability": float(probability),
            "predicted": bool(probability >= threshold)
        })

    results = sorted(
        results,
        key=lambda item: item["probability"],
        reverse=True
    )

    return results