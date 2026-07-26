# Ignores duplicate library errors when using certain libraries like OpenCV and PyTorch together. 
# This is a workaround for a known issue with the Intel MKL library.
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Now import the rest of your libraries below
# import pandas as pd
# import cv2
# import torch


# This transforms and resizes the images in the dataset so as to not use too much memory when training the model.
# It resizes the images to 224x224 and keeps the original images unchanged.
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])