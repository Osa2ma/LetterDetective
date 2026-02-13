"""
model.py - Neural Network Architectures for Handwritten Letter Recognition

This module defines two PyTorch models trained on the EMNIST Letters dataset:
1. LetterCNN - Convolutional Neural Network (~91% accuracy)
2. LetterMLP - Multi-Layer Perceptron (~85% accuracy)

Input: Grayscale images of shape (1, 28, 28) - EMNIST format
Output: 26 classes (A-Z uppercase letters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LetterCNN(nn.Sequential):
    """
    Convolutional Neural Network for handwritten letter classification.
    
    Architecture:
        - 3 Convolutional blocks with ReLU activation and MaxPooling
        - Fully connected classifier with Dropout regularization
    
    Input Shape: (batch_size, 1, 28, 28) - Grayscale EMNIST images
    Output Shape: (batch_size, num_classes) - Class logits
    
    Args:
        num_classes (int): Number of output classes. Default: 26 (A-Z)
    """
    
    def __init__(self, num_classes: int = 26):
        super().__init__(
            # Conv Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3: 7x7 -> 7x7 (no pooling)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Classifier
            nn.Flatten(),  # 128 * 7 * 7 = 6272
            nn.Linear(6272, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


class LetterMLP(nn.Sequential):
    """
    Multi-Layer Perceptron for handwritten letter classification.
    
    A simple fully-connected neural network baseline.
    
    Architecture:
        - Flatten layer (784 inputs)
        - 2 Hidden layers with ReLU activation
        - Output layer
    
    Input Shape: (batch_size, 1, 28, 28) - Grayscale EMNIST images
    Output Shape: (batch_size, num_classes) - Class logits
    
    Args:
        num_classes (int): Number of output classes. Default: 26 (A-Z)
    """
    
    def __init__(self, num_classes: int = 26):
        super().__init__(
            nn.Flatten(),  # 28 * 28 = 784
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )


# Backward compatibility aliases
CNN_Model = LetterCNN
MLP_Model = LetterMLP