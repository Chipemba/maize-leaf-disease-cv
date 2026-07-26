# Lightweight Computer Vision for Maize Leaf Disease Detection

## Project Summary
This project evaluates maize leaf disease classifiers for Southern African field conditions.
The contribution is an efficiency-aware robustness analysis across classification accuracy,
computational efficiency and robustness to field-image variation.

## Research Question
Can lightweight computer vision models classify maize leaf diseases from Southern African
field images with acceptable accuracy while offering better computational efficiency and
robustness than larger transfer learning models?

## Dataset
Primary dataset: Diseases of maize in the field, University of Pretoria.
Place downloaded data under data/raw/. Raw data is not committed to GitHub.

## Models
Maize Leaf Disease Detection 
1. Custom CNN baseline
2. Transfer learning model such as EfficientNetB0 or ResNet18
3. Lightweight model such as MobileNetV2

## Reproduce
pip install -r requirements.txt
python src/data/prepare_dataset.py --raw data/raw --out data/processed
python src/training/train.py --config configs/baseline_cnn.yaml
python src/evaluation/evaluate.py --model models/baseline_cnn.pt
python src/evaluation/robustness.py --model models/baseline_cnn.pt