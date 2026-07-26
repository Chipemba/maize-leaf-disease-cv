# Week 11 Evaluation Summary: mobilenet_v2

## Model
mobilenet_v2

## Checkpoint
models\mobilenet_v2.pt

## Task Type
Multi-label maize leaf disease and condition classification.

## Threshold
0.5

## Classification Metrics
- Micro F1: 0.5725
- Macro F1: 0.5080
- Weighted F1: 0.6239
- Hamming Loss: 0.2062
- Exact Match Ratio: 0.2260

## Efficiency Metrics
- Model size MB: 8.75
- Total parameters: 2,234,120
- Trainable parameters: 2,234,120
- Average inference time per image: 0.193073 seconds
- Images per second: 5.18

## Initial Interpretation
This model should be compared against the baseline CNN, transfer learning model, and lightweight model to identify the best balance between performance and efficiency.
