# Lightweight Computer Vision for Maize Leaf Disease Detection

## Project Summary
This project evaluates maize leaf disease classifiers for Southern African field conditions.
The contribution is an efficiency-aware robustness analysis across classification accuracy,
computational efficiency and robustness to field-image variation.

## Research Question
Can lightweight computer vision models classify maize leaf diseases from Southern African
field images with acceptable accuracy while offering better computational efficiency and
robustness than larger transfer learning models?

## Project Structure
```text
maize-leaf-disease-cv/
├── .github/
│   └── workflows/              # GitHub Actions CI
├── configs/                    # Model training configurations
├── data/
│   ├── raw/                    # Local raw dataset (not committed)
│   └── splits/                 # Train/validation/test CSV files
├── demo/
│   ├── app.py                  # Streamlit ONNX inference demo
│   └── sample_images/          # Demo images
├── notebooks/                  # Exploratory notebooks
├── models/                     # Trained PyTorch checkpoints (local if ignored)
├── onnx/
│   └── mobilenet_v2.onnx       # Exported deployment model
├── presentation/
│   └── final_demo_screenshots/ # Final presentation/demo evidence
├── reports/                    # Project reports and documentation
├── results/                    # Evaluation and robustness outputs
├── src/
│   ├── data/                   # Validation, EDA and dataset splitting
│   ├── models/                 # Model definitions/factory
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation, efficiency and robustness
│   └── deployment/             # ONNX export utilities
├── tests/                      # Automated tests
├── README.md
├── requirements.txt
└── LICENSE

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


## Results Summary

The project compares models across clean test performance, efficiency, and robustness.

Key findings:

- MobileNetV2 achieved the best Macro F1 score at approximately 0.508.
- EfficientNet-B0 achieved the best Micro F1 score at approximately 0.573.
- MobileNetV2 was the smallest practical model at approximately 8.75 MB.
- MobileNetV2 had the fastest throughput at approximately 5.18 images per second.
- EfficientNet-B0 remained competitive for overall prediction performance and robustness.
- The baseline CNN provided a useful reference point but was weaker overall.

Overall, MobileNetV2 is the strongest practical candidate because it provides the best balance between clean Macro F1, model size, and inference speed.

## Robustness Summary

Robustness was evaluated by applying controlled image distortions to the test set.

The distortions included:

- brightness decrease
- brightness increase
- contrast decrease
- contrast increase
- mild blur
- strong blur
- left rotation
- right rotation

Macro F1 drop from clean test performance was used to measure model stability. A smaller drop indicates that the model is less sensitive to field-like image variation.

The robustness analysis supports the project goal of evaluating not only clean accuracy, but also how models behave when image quality becomes less ideal.

## ONNX Live Demo


This project includes a lightweight live inference demo using the selected MobileNetV2 model exported to ONNX.

The demo allows a user to upload a maize leaf image and receive eight independent disease or condition probabilities.


### Export the model

```bash
python src/deployment/export_onnx.py
python src/deployment/check_onnx_output.py
```

Expected output:

```text
onnx/mobilenet_v2.onnx
```

### Run the Streamlit app

```bash
streamlit run demo/app.py
```

The app allows a user to upload a maize leaf image and returns eight independent disease/condition probabilities.

### Demo labels

The ONNX model predicts the original eight project labels:

- GLS
- NCLB
- PLS
- CR
- SR
- NoFoliarSymptoms
- Other
- UnidentifiedDisease

### External demo images

External demo images may be selected from the Kaggle Corn or Maize Leaf Disease Dataset.

Suggested mapping:

| External label | Project model output |
|---|---|
| Common Rust | CR |
| Gray Leaf Spot | GLS |
| Blight | NCLB |
| Healthy | NoFoliarSymptoms |

External images are used for qualitative demonstration only because the external label names do not fully match the original eight-label training setup.

## Limitations

This project has several limitations:

- Rare labels require careful interpretation because some disease or condition categories have fewer examples.
- The labels `Other` and `UnidentifiedDisease` are visually ambiguous and may be harder for the models to learn consistently.
- Robustness experiments use controlled synthetic distortions rather than a newly collected real-world field test set.
- External Kaggle images used in the ONNX demo are qualitative examples only and are not treated as formal external validation.
- A fixed threshold may not be optimal for all labels. Future work should explore threshold tuning per label.
- Mobile or edge-device benchmarking was not completed and remains future work.

### Split Strategy Limitation

The current train, validation and test split uses a fixed random seed for reproducibility. However, the split does not use iterative multi-label stratification.

This is a limitation because rare labels and label combinations may not be perfectly balanced across the train, validation, and test sets. Future work should use iterative multi-label stratification to better preserve multi-label distributions across splits.

## Deployment Demo Addition

A lightweight ONNX demo was added using the MobileNetV2 model. The demo allows a user to upload a maize leaf image and view eight independent disease or condition probabilities using ONNX Runtime and Streamlit.

External maize/corn leaf images are used for qualitative demonstration only and are not treated as formal external validation.