# Week 10 Training Summary: efficientnet_b0

## Model
efficientnet_b0

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
- Total parameters: 4,017,796
- Trainable parameters: 4,017,796

## Best Validation Loss
0.746392

## Checkpoint
models\efficientnet_b0.pt

## Notes
This model was trained using multi-label classification with BCEWithLogitsLoss.
The model outputs 8 raw logits, one for each maize disease or condition label.
Sigmoid activation is applied during evaluation, not during training.
