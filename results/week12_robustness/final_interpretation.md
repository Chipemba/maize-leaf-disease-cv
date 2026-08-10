# Final Accuracy-Efficiency-Robustness Interpretation

## Purpose
This analysis combines clean test-set performance, CPU-based efficiency measurements and robustness under controlled image distortions. The objective was to identify the model that provides the strongest balance between multi-label classification performance, computational efficiency and stability under image-quality changes.

All models were trained and evaluated on a CPU-only environment. Therefore, the reported inference times and throughput values describe CPU performance and may differ when the models are deployed on GPU or mobile hardware.

## Best Clean Performance
The best clean test-set model was **MobileNetV2** based on Macro F1.

MobileNetV2 achieved the highest clean Macro F1-score of **0.5080**, compared with:

- EfficientNet-B0: **0.4967**
- ResNet18: **0.4323**
- Baseline CNN: **0.3214**

Macro F1 gives equal importance to each disease label. MobileNetV2’s higher Macro F1 therefore indicates that it produced the most balanced classification performance across the eight labels, including labels with fewer examples.

EfficientNet-B0 achieved a slightly higher Micro F1-score of **0.5730**, compared with **0.5725** for MobileNetV2. However, the difference was extremely small. MobileNetV2 remained the best clean-performance model under the selected Macro F1 criterion.

## Best Efficiency
The most efficient practical model was **MobileNetV2** based on its overall combination of model size, parameter count, CPU inference speed, and classification performance.

MobileNetV2 had:

- Model size: **8.75 MB**
- Total parameters: **2,234,120**
- Average CPU inference time: **0.1931 seconds per image**
- CPU throughput: **5.18 images per second**

The baseline CNN was technically the smallest model, with a size of only **0.37 MB** and **94,728 parameters**. However, it was slower than MobileNetV2 during CPU inference and produced substantially weaker classification results.

MobileNetV2 was the fastest model tested on the CPU and was considerably smaller than EfficientNet-B0 and ResNet18. It therefore provided the strongest practical efficiency rather than only the lowest storage requirement.

Because inference testing was conducted on a CPU, these latency rankings apply specifically to the tested CPU environment. GPU-based inference could change the speed differences between the architectures.

## Best Robustness
The most robust model was **EfficientNet-B0** based on the smallest average Macro F1 drop across distorted test conditions.

EfficientNet-B0 achieved:

- Clean Macro F1: **0.4967**
- Average distorted Macro F1: **0.4990**
- Average Macro F1 drop: **−0.0023**
- Worst Macro F1 drop: **0.0035**
- Worst distorted Macro F1: **0.4932**

The negative average drop indicates that EfficientNet-B0’s average Macro F1 under the tested distortions was slightly higher than its clean Macro F1. This does not necessarily mean that distortions improved the model. The small increase may reflect normal evaluation variation, threshold behaviour, or certain transformations making some visual patterns easier to recognize.

EfficientNet-B0 also had the smallest worst-case decline. Its worst distorted Macro F1 remained close to its clean performance, demonstrating strong stability across brightness, blur, rotation, and contrast changes.

The average Macro F1 drops were:

| Model | Average Macro F1 drop | Worst Macro F1 drop |
|---|---:|---:|
| EfficientNet-B0 | **−0.0023** | **0.0035** |
| ResNet18 | 0.0022 | 0.0254 |
| MobileNetV2 | 0.0076 | 0.0260 |
| Baseline CNN | 0.0149 | 0.0330 |

## Main Trade-Off
The results show that **MobileNetV2 achieved the highest clean classification performance based on Macro F1**, while **EfficientNet-B0 provided the strongest robustness trade-off**.

MobileNetV2 offered:

- The highest clean Macro F1
- The fastest CPU inference time
- The highest CPU throughput
- A smaller model than EfficientNet-B0 and ResNet18
- Competitive robustness under distorted conditions

EfficientNet-B0 offered:

- The highest Micro F1
- The lowest hamming loss
- The highest exact match ratio
- The smallest average Macro F1 drop
- The smallest worst-case robustness decline

The baseline CNN had the smallest storage requirement and parameter count, but its considerably lower clean and distorted performance made it less suitable for reliable disease classification.

ResNet18 did not provide a strong trade-off. It was the largest and slowest model while achieving lower classification performance than MobileNetV2 and EfficientNet-B0.

## Practical Model Recommendation
For a practical low-resource maize disease classification system, the recommended model is **MobileNetV2** because it provides the strongest overall balance between classification performance, CPU inference speed, model size, parameter count, and robustness.

MobileNetV2 is recommended because it:

- Achieved the highest clean Macro F1-score of **0.5080**
- Achieved a competitive Micro F1-score of **0.5725**
- Had the fastest CPU inference time at approximately **0.1931 seconds per image**
- Produced the highest CPU throughput at approximately **5.18 images per second**
- Required only approximately **8.75 MB** of storage
- Used approximately **2.23 million parameters**
- Maintained an average distorted Macro F1-score of **0.5004**
- Experienced only a small average Macro F1 decline of **0.0076**

Although EfficientNet-B0 was more robust, MobileNetV2’s clean Macro F1 was higher, its model was approximately **6.86 MB smaller**, and its CPU inference was faster. Its performance loss under distortions was also relatively small.

MobileNetV2 therefore provides the most suitable accuracy-efficiency-robustness balance for CPU-only systems, field laptops, mobile devices, and lower-resource agricultural deployment environments.

EfficientNet-B0 remains a strong alternative when robustness, exact match performance, and low hamming loss are more important than model size or inference speed.

## Limitations
- Distortions were synthetic rather than collected from new real-world test images.
- Robustness was measured using controlled transformations only.
- Multi-label ambiguity remains challenging, especially for `Other` and `UnidentifiedDisease`.
- Rare classes may still require additional data or targeted augmentation.
- CPU inference measurements may not represent performance on mobile processors, embedded devices, or GPUs.
- GPU execution could change inference speed and throughput rankings, although it would not directly change model size or parameter count.
- Small metric differences may result from random initialization, data shuffling, augmentation, or floating-point variation.
- Average robustness results may hide larger performance changes for individual disease labels.
- A single classification threshold may not be optimal for every label.
- The evaluation did not include quantization, pruning, or hardware-specific model optimization.

## Next Steps
- Add more real field images collected under natural lighting and environmental conditions.
- Tune classification thresholds separately for each disease label.
- Test post-training quantization for edge deployment.
- Compare performance on mobile or CPU-only hardware.
- Repeat inference measurements on a GPU while reporting the hardware separately.
- Extend the completed MobileNetV2 ONNX demo with more field-image testing and hardware-specific benchmarking.
- Measure memory consumption and energy usage during inference.
- Evaluate robustness separately for each disease label.
- Apply targeted augmentation to rare and unstable classes.
- Test the selected model on images from farms or regions not represented in the training dataset.


MobileNetV2 is strongest practical candidate.
EfficientNet-B0 remains competitive for Micro F1 and robustness.
Baseline CNN is weakest but useful as reference.
Robustness was tested with brightness, contrast, blur, and rotation.
External ONNX demo images are qualitative only.

## Deployment Demo Addition

A lightweight ONNX demo was added using the MobileNetV2 model. The demo allows a user to upload a maize leaf image and view eight independent disease or condition probabilities using ONNX Runtime and Streamlit.

External maize/corn leaf images are used for qualitative demonstration only and are not treated as formal external validation.

## Split Strategy Limitation

The current train, validation and test split uses a fixed random seed for reproducibility. However, the split does not use iterative multi-label stratification.

This is a limitation because rare labels and label combinations may not be perfectly balanced across the train, validation and test sets. Future work should use iterative multi-label stratification to better preserve multi-label distributions across splits.