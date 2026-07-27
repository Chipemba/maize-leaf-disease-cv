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

### Local Dataset Layout

Raw image data is not committed to GitHub. After downloading the dataset, place files like this:  


```text
data/
├── raw/
│   ├── Database.csv
│   └── static/
│       └── leaf_images/
│           ├── image_1.jpg
│           ├── image_2.jpg
│           └── ...
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv  
```
Primary dataset: Diseases of maize in the field, University of Pretoria.  


## Models
Maize Leaf Disease Detection 
1. Custom CNN baseline
2. Transfer learning model such as EfficientNetB0 or ResNet18
3. Lightweight model such as MobileNetV2

## Reproduce

### Install dependencies:

pip install -r requirements.txt  

### Place the downloaded dataset locally:

data/raw/Database.csv  
data/raw/static/leaf_images/  

### Run dataset checks:

python src/data/validate_paths.py  
python src/data/eda.py  
python src/data/split_data.py  
python src/training/train_baseline.py  
python src/evaluation/evaluate_baseline.py  

### Run Week 10 model training:

python src/training/train_model.py --config configs/efficientnet_b0_config.yaml  
python src/training/train_model.py --config configs/mobilenet_v2_config.yaml  
python src/training/train_model.py --config configs/resnet18_config.yaml  

### Run Week 11 evaluation:

python src/evaluation/evaluate_model.py --model-name baseline_cnn --checkpoint models/baseline_cnn.pt  
python src/evaluation/evaluate_model.py --model-name efficientnet_b0 --checkpoint models/efficientnet_b0.pt  
python src/evaluation/evaluate_model.py --model-name mobilenet_v2 --checkpoint models/mobilenet_v2.pt  
python src/evaluation/evaluate_model.py --model-name resnet18 --checkpoint models/resnet18.pt  
python src/evaluation/create_comparison_table.py

### Run Week 12 robustness experiments:

python src/evaluation/robustness.py  
python src/evaluation/create_robustness_ranking.py  
python src/evaluation/create_final_comparison.py  

#### Run model evaluation:

python src/evaluation/evaluate_model.py --model-name baseline_cnn --checkpoint models/baseline_cnn.pt  
python src/evaluation/evaluate_model.py --model-name efficientnet_b0 --checkpoint models/efficientnet_b0.pt  
python src/evaluation/evaluate_model.py --model-name mobilenet_v2 --checkpoint models/mobilenet_v2.pt  

#### Create comparison table:

python src/evaluation/create_comparison_table.py  
