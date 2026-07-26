# Week 11 Model Comparison Interpretation

## Purpose
The purpose of this evaluation was to compare the baseline CNN, transfer learning models, and lightweight model using the same multi-label test set. The comparison considered both predictive performance and computational efficiency so that the most practical model could be selected for further robustness testing in Week 12.

The evaluated models were:

- Baseline CNN
- ResNet18
- EfficientNet-B0
- MobileNetV2

## Main Performance Findings
- Best model by micro F1: **EfficientNet-B0**, with a micro F1-score of **0.5730**
- Best model by macro F1: **MobileNetV2**, with a macro F1-score of **0.5080**
- Best model by exact match ratio: **EfficientNet-B0**, with an exact match ratio of **0.2514**
- Lowest hamming loss: **EfficientNet-B0**, with a hamming loss of **0.1921**

EfficientNet-B0 achieved the strongest overall multi-label prediction performance. It produced the highest micro F1-score, the highest exact match ratio, and the lowest hamming loss.

Its micro F1-score of 0.5730 was only slightly higher than MobileNetV2’s score of 0.5725. However, EfficientNet-B0 achieved a noticeably lower hamming loss, meaning it made fewer incorrect label assignments across the complete test set.

MobileNetV2 achieved the highest macro F1-score of 0.5080. This result suggests that MobileNetV2 provided slightly more balanced performance across the eight disease labels, including less frequently occurring classes. In comparison, EfficientNet-B0 achieved a macro F1-score of 0.4967.

The baseline CNN produced the weakest predictive results, with a micro F1-score of 0.3637, a macro F1-score of 0.3214, and an exact match ratio of only 0.0678. These results show that transfer learning provided a substantial improvement over training a small convolutional neural network from scratch.

## Main Efficiency Findings
- Smallest model: **Baseline CNN**, with a model size of approximately **0.37 MB**
- Fastest model: **MobileNetV2**, with an average inference time of approximately **0.1931 seconds per image**
- Model with fewest parameters: **Baseline CNN**, with **94,728 parameters**
- Highest throughput: **MobileNetV2**, processing approximately **5.18 images per second**

The baseline CNN was considerably smaller than every pretrained architecture. Its model size was only approximately 0.37 MB, compared with 8.75 MB for MobileNetV2 and 15.61 MB for EfficientNet-B0.

However, the baseline CNN was not the fastest model. Its average inference time was approximately 0.2092 seconds per image, while MobileNetV2 required only approximately 0.1931 seconds per image. This demonstrates that having fewer parameters does not automatically guarantee the lowest inference latency. Architectural design and implementation efficiency also affect processing speed.

MobileNetV2 provided the strongest runtime efficiency. It had the lowest average inference time and the highest throughput while remaining much smaller than EfficientNet-B0 and ResNet18.

ResNet18 was the least efficient model in the comparison. It had the largest model size at approximately 42.72 MB, the highest parameter count at 11,180,616 parameters, the slowest inference time at approximately 0.2471 seconds per image, and the lowest throughput at approximately 4.05 images per second.

## Model Comparison Summary

| Model | Micro F1 | Macro F1 | Exact Match Ratio | Hamming Loss | Model Size | Parameters | Inference Time | Images per Second |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.5725 | **0.5080** | 0.2260 | 0.2062 | 8.75 MB | 2,234,120 | **0.1931 s** | **5.18** |
| EfficientNet-B0 | **0.5730** | 0.4967 | **0.2514** | **0.1921** | 15.61 MB | 4,017,796 | 0.2030 s | 4.93 |
| ResNet18 | 0.4886 | 0.4323 | 0.1102 | 0.2860 | 42.72 MB | 11,180,616 | 0.2471 s | 4.05 |
| Baseline CNN | 0.3637 | 0.3214 | 0.0678 | 0.3619 | **0.37 MB** | **94,728** | 0.2092 s | 4.78 |

## Accuracy-Efficiency Trade-off
The best model for accuracy was **EfficientNet-B0**, while the best model for efficiency was **MobileNetV2**.

EfficientNet-B0 produced the best micro F1-score, exact match ratio, and hamming loss. These metrics indicate that it was the strongest model when overall predictive performance and complete multi-label correctness were prioritized.

MobileNetV2, however, achieved nearly the same micro F1-score while using fewer parameters, requiring less storage, and producing faster inference. Its micro F1-score was only approximately 0.0005 lower than EfficientNet-B0, while its model size was approximately 6.86 MB smaller.

MobileNetV2 also achieved a higher macro F1-score than EfficientNet-B0. Therefore, it may provide more consistent performance across individual disease labels while requiring fewer computational resources.

The small performance difference between EfficientNet-B0 and MobileNetV2 suggests that the additional size and computational cost of EfficientNet-B0 may not provide a large enough improvement to justify its use in a low-resource deployment setting.

## Practical Recommendation
Based on the Week 11 results, the most practical model for low-resource maize disease classification is **MobileNetV2** because it provides the strongest overall balance between predictive performance, model size, inference speed, and throughput.

MobileNetV2 achieved:

- A micro F1-score of 0.5725, which was nearly identical to the best result
- The highest macro F1-score of 0.5080
- The fastest inference time of approximately 0.1931 seconds per image
- The highest throughput of approximately 5.18 images per second
- A relatively small model size of approximately 8.75 MB
- Approximately 2.23 million parameters, almost 1.8 million fewer than EfficientNet-B0

Although EfficientNet-B0 achieved slightly stronger overall accuracy metrics, the improvement was small compared with the additional storage and computational requirements. MobileNetV2 therefore represents the more appropriate choice for mobile, edge, or low-power agricultural systems.

EfficientNet-B0 should still be retained as the accuracy-focused comparison model. It may be preferable where computational resources are less restricted and the lowest possible hamming loss or highest exact match ratio is the main priority.

## Issues to Investigate in Week 12
- Robustness under brightness changes
- Robustness under blur
- Robustness under rotation
- Robustness under contrast variation
- Whether MobileNetV2 maintains its performance advantage under corrupted field images
- Whether EfficientNet-B0 remains more accurate under severe image distortions
- Which disease labels experience the largest performance decline
- Whether lightweight efficiency comes at the cost of reduced robustness
- Changes in micro F1, macro F1, exact match ratio, and hamming loss under each distortion
- The distortion severity level at which model performance becomes unreliable