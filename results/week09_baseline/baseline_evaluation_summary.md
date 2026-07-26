# Week 9 Baseline Evaluation Summary

## Model
Custom baseline CNN with 8 output logits.

## Task Type
Multi-label maize leaf disease and condition classification.

## Loss Function
BCEWithLogitsLoss was used during training because each image can contain more than one label.

## Threshold
A threshold of `0.5` was used to convert sigmoid probabilities into binary predictions.

## Overall Metrics
- Micro F1: 0.3637
- Macro F1: 0.3214
- Weighted F1: 0.4358
- Hamming Loss: 0.3619
- Exact Match Ratio: 0.0678

## Best and Weakest Labels
- Best-performing label by F1-score: GLS
- Weakest-performing label by F1-score: SR

## Interpretation Notes
The baseline CNN provides a starting reference point for the project. Its results should not be treated as the final model performance. The purpose of this model is to establish whether the dataset pipeline, multi-label target format, training loop, and evaluation metrics are working correctly.

Low macro F1 compared with micro F1 may indicate that the model performs better on common labels than rare labels. This is important for this dataset because some maize disease labels may be underrepresented.

## Planned Improvements
In Week 10, stronger transfer learning and lightweight models will be trained and compared against this baseline. In Week 11 and Week 12, efficiency and robustness testing will be added to evaluate practical field-readiness.
