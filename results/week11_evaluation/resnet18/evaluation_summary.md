# Week 11 Evaluation Summary: resnet18

## Model
resnet18

## Checkpoint
models\resnet18.pt

## Task Type
Multi-label maize leaf disease and condition classification.

## Threshold
0.5

## Classification Metrics
- Micro F1: 0.4886
- Macro F1: 0.4323
- Weighted F1: 0.5483
- Hamming Loss: 0.2860
- Exact Match Ratio: 0.1102

## Efficiency Metrics
- Model size MB: 42.72
- Total parameters: 11,180,616
- Trainable parameters: 11,180,616
- Average inference time per image: 0.247079 seconds
- Images per second: 4.05

## Initial Interpretation
This model should be compared against the baseline CNN, transfer learning model, and lightweight model to identify the best balance between performance and efficiency.
