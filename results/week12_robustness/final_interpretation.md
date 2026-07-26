# Final Accuracy-Efficiency-Robustness Interpretation

## Purpose
This analysis combines clean test-set performance, efficiency measurements and robustness under controlled image distortions. The goal was to determine which model provides the most suitable balance between classification performance, computational efficiency and stability under changing image conditions.

## Best Clean Performance
The best clean test-set model was **MobileNetV2** based on Macro F1.

MobileNetV2 achieved a clean Macro F1-score of **0.5080**, compared with **0.4967** for EfficientNet-B0 and **0.3214** for the baseline CNN. This indicates that MobileNetV2 provided the most balanced clean-test performance across all maize disease labels.

MobileNetV2 also achieved a Micro F1-score of **0.5725**, which was almost identical to EfficientNet-B0’s best Micro F1-score of **0.5730**. Therefore, MobileNetV2 delivered strong overall prediction performance while maintaining slightly better balance across individual classes.

## Best Efficiency
The most efficient practical model was **MobileNetV2** based on its combination of model size, parameter count and inference speed.

MobileNetV2 had:

- A model size of approximately **8.75 MB**
- Approximately **2.23 million parameters**
- An average inference time of approximately **0.1931 seconds per image**
- A throughput of approximately **5.18 images per second**

The baseline CNN was technically the smallest model, with a size of approximately **0.37 MB** and only **94,728 parameters**. However, it was slower than MobileNetV2 and produced substantially weaker classification performance.

MobileNetV2 was the fastest evaluated model and remained considerably smaller than EfficientNet-B0. It therefore provided the strongest practical efficiency balance rather than simply the lowest storage requirement.

## Best Robustness
The most robust model was **EfficientNet-B0** based on the smallest average Macro F1 drop across distorted test conditions.

EfficientNet-B0 had an average Macro F1 drop of **−0.0023**. The negative value means that its average Macro F1 under the tested distortions was slightly higher than its clean-test Macro F1, rather than lower.

Its clean Macro F1 was **0.4967**, while its average distorted Macro F1 was approximately **0.4990**. EfficientNet-B0 also had a worst-case Macro F1 drop of only **0.0035**, showing that its performance remained highly stable across the controlled distortions.

In comparison:

- MobileNetV2 had an average Macro F1 drop of **0.0076**
- Baseline CNN had an average Macro F1 drop of **0.0149**

EfficientNet-B0 therefore showed the smallest average decline and the smallest worst-case decline under brightness, blur, rotation, and contrast transformations.

The slight improvement under some distortions should not automatically be interpreted as distortions improving the model. Small increases may result from dataset variation, threshold behaviour, or certain transformations making specific image features easier for the model to recognize.

## Combined Results Summary

| Model | Clean Macro F1 | Average Distorted Macro F1 | Average Macro F1 Drop | Worst Macro F1 Drop | Model Size | Parameters | Inference Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | **0.5080** | **0.5004** | 0.0076 | 0.0260 | 8.75 MB | 2,234,120 | **0.1931 s** |
| EfficientNet-B0 | 0.4967 | 0.4990 | **−0.0023** | **0.0035** | 15.61 MB | 4,017,796 | 0.2030 s |
| Baseline CNN | 0.3214 | 0.3065 | 0.0149 | 0.0330 | **0.37 MB** | **94,728** | 0.2092 s |

## Main Trade-Off
The results show that **MobileNetV2 achieved the highest clean classification performance**, while **EfficientNet-B0 provided the strongest robustness under controlled image distortions**.

MobileNetV2 achieved the highest clean Macro F1-score and the fastest inference speed. It was also approximately **6.86 MB smaller** and used approximately **1.78 million fewer parameters** than EfficientNet-B0.

EfficientNet-B0 produced slightly lower clean Macro F1 performance but remained more stable under distorted conditions. Its average distorted Macro F1 was slightly higher than its clean Macro F1, and its worst recorded Macro F1 decline was only approximately 0.0035.

The baseline CNN was the smallest model and had the fewest parameters. However, its much lower clean and distorted Macro F1-scores show that its efficiency came at a substantial cost to classification performance.

Therefore, the main trade-off is between:

- **MobileNetV2:** Better clean performance, faster inference, and smaller deployment size
- **EfficientNet-B0:** Better robustness and more stable performance under image distortions
- **Baseline CNN:** Minimal storage requirements but considerably weaker predictive performance

## Practical Model Recommendation
For a practical low-resource maize disease classification system, the recommended model is **MobileNetV2** because it provides the strongest overall balance between classification performance, inference speed, model size, and robustness.

MobileNetV2 is recommended because it:

- Achieved the highest clean Macro F1-score of **0.5080**
- Produced a competitive Micro F1-score of **0.5725**
- Had the fastest inference time of approximately **0.1931 seconds per image**
- Achieved the highest throughput of approximately **5.18 images per second**
- Required only approximately **8.75 MB** of storage
- Used fewer parameters than EfficientNet-B0
- Experienced only a small average Macro F1 decline of **0.0076** under distortions
- Maintained an average distorted Macro F1-score of approximately **0.5004**

Although EfficientNet-B0 was the most robust model, its clean Macro F1-score was lower, its model was larger, and its inference speed was slightly slower. MobileNetV2 therefore offers the more appropriate accuracy-efficiency-robustness balance for mobile devices, field systems, and CPU-only deployment environments.

EfficientNet-B0 remains a suitable alternative when robustness under changing image conditions is more important than minimizing model size or inference latency.

## Limitations
- Distortions were synthetic rather than collected from new real-world test images.
- Robustness was measured using controlled transformations only.
- Multi-label ambiguity remains challenging, especially for `Other` and `UnidentifiedDisease`.
- Rare classes may still require additional data or targeted augmentation.
- Average robustness scores may hide large differences between individual disease labels.
- Small improvements under distortion may result from normal evaluation variation rather than genuine performance gains.
- Inference measurements were collected on the current CPU environment and may differ on mobile, cloud, or edge hardware.
- The evaluation did not yet include model quantization, pruning, or hardware-specific optimization.
- The same classification threshold may not be optimal for every disease label.

## Next Steps
- Add more real field images collected under natural lighting, background, and camera variations.
- Tune classification thresholds separately for each label.
- Test post-training quantization for edge deployment.
- Compare performance on mobile or CPU-only hardware.
- Evaluate per-class robustness to identify which diseases are most affected by each distortion.
- Apply targeted augmentation to rare or unstable classes.
- Measure memory usage and energy consumption during inference.
- Export MobileNetV2 to ONNX or TorchScript for deployment testing.
- Compare full-precision and quantized MobileNetV2 performance.
- Test the selected model on images from farms or geographic regions not represented in the training dataset.