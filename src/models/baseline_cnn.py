# This script defines a baseline CNN model for multi-label classification of maize leaf diseases. 
# The model consists of convolutional layers followed by batch normalization, ReLU activation and max pooling.
# The final classifier uses adaptive average pooling, dropout and a linear layer to output predictions for the specified number of labels.

import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_labels=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x