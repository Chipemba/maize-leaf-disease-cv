# Week 10 Training Summary: mobilenet_v2

## Model
mobilenet_v2

## Training Mode
full

## Pretrained
True

## Freeze Backbone
False

## Number of Labels
8

## Hyperparameters
- Image size: 224
- Batch size: 32
- Epochs: 15
- Learning rate: 0.0001
- Weight decay: 0.0001
- Optimizer: adam
- Loss: BCEWithLogitsLoss
- Threshold for later evaluation: 0.5

## Parameters
- Total parameters: 2,234,120
- Trainable parameters: 2,234,120

## Best Validation Loss
0.778967

## Checkpoint
models\mobilenet_v2.pt

## Notes
This model was trained using multi-label classification with BCEWithLogitsLoss.
The model outputs 8 raw logits, one for each maize disease or condition label.
Sigmoid activation is applied during evaluation, not during training.
