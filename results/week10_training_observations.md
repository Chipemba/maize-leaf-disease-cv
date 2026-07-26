# Week 10 Training Observations

## Purpose
The purpose of Week 10 was to train transfer learning and lightweight models using the multi-label maize leaf disease dataset. The experiments focused on comparing the learning behaviour of EfficientNet-B0 and MobileNetV2 before completing the full performance and efficiency evaluation in Week 11.

## Models Trained
- EfficientNet-B0 transfer learning model
- MobileNetV2 lightweight model

## Training Setup
- Input size: 224 × 224
- Number of epochs: 15
- Loss function: BCEWithLogitsLoss
- Fine-tuning mode: Full fine-tuning
- Learning rate: 0.0001
- Labels: 8 multi-label outputs
- Training device: CPU
- Model selection method: Lowest validation loss
- Pretrained ImageNet weights: Enabled

## EfficientNet-B0 Observations
- Total parameters: 4,017,796
- Trainable parameters: 4,017,796
- Initial training loss: 1.1132
- Final training loss: 0.2796
- Initial validation loss: 1.0267
- Final validation loss: 0.9414
- Lowest validation loss: Approximately 0.75 around Epoch 7
- Best model checkpoint: `models/efficientnet_b0.pt`
- Results directory: `results/week10_efficientnet_b0`

### Notes on Overfitting or Underfitting
EfficientNet-B0 learned the training dataset effectively, with training loss decreasing consistently from 1.1132 to 0.2796. Validation loss also decreased during the first part of training, reaching its lowest point at approximately Epoch 7.

After approximately Epoch 7, the validation loss began to increase while the training loss continued to decrease. For example, the validation loss increased from approximately 0.75 near the best epoch to 0.9414 by Epoch 15, while the training loss decreased to 0.2796.

This widening difference between training and validation loss indicates overfitting. The model continued learning patterns from the training dataset, but these later improvements did not generalize to the validation dataset. Therefore, the saved best checkpoint should be used instead of the final epoch checkpoint.

The model does not appear to be underfitting because the training loss decreased substantially and reached a relatively low value.

## MobileNetV2 Observations
- Total parameters: 2,234,120
- Trainable parameters: 2,234,120
- Initial training loss: 1.1109
- Final training loss: 0.3634
- Initial validation loss: 1.0623
- Final validation loss: 0.8794
- Lowest validation loss: 0.7790 at Epoch 7
- Best model checkpoint: `models/mobilenet_v2.pt`
- Results directory: `results/week10_mobilenet_v2`

### Notes on Overfitting or Underfitting
MobileNetV2 also showed successful learning during training. Its training loss decreased steadily from 1.1109 to 0.3634. Validation loss improved from 1.0623 to its lowest value of 0.7790 at Epoch 7.

After Epoch 7, the validation loss stopped improving and gradually increased. By Epoch 15, validation loss had risen to 0.8794, while training loss continued decreasing to 0.3634.

This pattern indicates overfitting beginning after approximately Epoch 7. However, the increase in validation loss was more gradual than the increase observed for EfficientNet-B0. This suggests that MobileNetV2 may have slightly more stable validation behaviour, although its best validation loss was higher than that of EfficientNet-B0.

The model does not appear to be underfitting because both its training and validation losses improved considerably during the first seven epochs.

## Initial Comparison

| Measure | EfficientNet-B0 | MobileNetV2 |
|---|---:|---:|
| Total parameters | 4,017,796 | 2,234,120 |
| Initial training loss | 1.1132 | 1.1109 |
| Final training loss | 0.2796 | 0.3634 |
| Initial validation loss | 1.0267 | 1.0623 |
| Best validation loss | Approximately 0.75 | 0.7790 |
| Final validation loss | 0.9414 | 0.8794 |
| Best validation epoch | Approximately Epoch 7 | Epoch 7 |
| Overfitting observed | Yes | Yes |
| Approximate overfitting point | After Epoch 7 | After Epoch 7 |

EfficientNet-B0 achieved the lower training loss and the lower minimum validation loss. This initial result suggests that its larger feature-extraction architecture may capture more detailed disease-related image patterns.

However, EfficientNet-B0 also showed a stronger increase in validation loss after its best epoch. Its final validation loss of 0.9414 was higher than MobileNetV2’s final validation loss of 0.8794. This indicates that EfficientNet-B0 overfit the training data more aggressively during the later epochs.

MobileNetV2 used approximately 1.78 million fewer parameters than EfficientNet-B0. Despite its smaller architecture, it achieved a competitive minimum validation loss of 0.7790. Its validation loss also increased more gradually after the best epoch, suggesting relatively stable generalization for a lightweight model.

Both models reached their best validation performance at approximately Epoch 7. Therefore, early stopping with a patience value of approximately two to four epochs could reduce unnecessary training and limit overfitting in future experiments.

These loss results provide an initial comparison only. The lowest validation loss does not automatically identify the best final model for the project. Full evaluation will be completed in Week 11 using:

- Multi-label classification accuracy
- Micro and macro precision
- Micro and macro recall
- Micro and macro F1-score
- Per-class precision, recall, and F1-score
- Hamming loss
- Confusion matrices or multi-label confusion matrices
- Model size
- Parameter count
- CPU inference latency
- Throughput
- Memory usage

The Week 11 evaluation will determine whether EfficientNet-B0’s stronger feature extraction provides enough predictive improvement to justify its larger size and computational cost, or whether MobileNetV2 provides the better balance between disease-classification performance and deployment efficiency.