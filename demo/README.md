# ONNX Live Demo

This demo runs live maize leaf disease prediction using the trained MobileNetV2 model exported to ONNX.

## Purpose

The demo shows how the trained model can be used for portable inference outside the PyTorch training pipeline.

## Model

- Model: MobileNetV2
- Format: ONNX
- Input size: 224 x 224
- Task: Multi-label maize leaf disease/condition prediction
- Outputs: 8 independent probabilities

## Labels

- GLS
- NCLB
- PLS
- CR
- SR
- NoFoliarSymptoms
- Other
- UnidentifiedDisease

## Export ONNX Model

From the project root:

```bash
python src/deployment/export_onnx.py
```

Expected output:

```text
onnx/mobilenet_v2.onnx
```

## Run Demo

```bash
streamlit run demo/app.py
```

## External Demo Dataset

For qualitative testing, use the Kaggle Corn or Maize Leaf Disease Dataset.

Suggested mapping:

| External label | Model output |
|---|---|
| Common Rust | CR |
| Gray Leaf Spot | GLS |
| Blight | NCLB |
| Healthy | NoFoliarSymptoms |

## Important Limitation

External images are used for qualitative live inference only. The model predictions are interpreted using the original eight-label output space and should not be treated as formal external validation unless the dataset labels are fully aligned.