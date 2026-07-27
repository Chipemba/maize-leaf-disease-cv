# Week 12 Robustness Summary

## Purpose
The purpose of this experiment was to test whether trained maize leaf disease models remain stable under controlled image distortions that simulate realistic field-image variation.

## Distortions Tested
- clean
- brightness_low
- brightness_high
- contrast_low
- contrast_high
- blur_mild
- blur_strong
- rotation_left
- rotation_right

## Main Robustness Finding
The largest Macro F1 drop was observed for baseline_cnn under brightness_high with a drop of 0.0330.

## Average Macro F1 Drop by Model
- efficientnet_b0: -0.0023
- resnet18: 0.0022
- mobilenet_v2: 0.0076
- baseline_cnn: 0.0149

## Interpretation Template
The model with the smallest average Macro F1 drop is considered the most robust under the tested image distortions. This result should be interpreted together with Week 11 efficiency metrics to determine which model provides the best practical trade-off.
