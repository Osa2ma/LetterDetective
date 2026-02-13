import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE CNN ARCHITECTURE
# ==========================================
# Architecture matches saved weights (cnn_model.pth)
class CNN_Model(nn.Sequential):
    def __init__(self, num_classes=26): 
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1),   # 0: 28x28 -> 28x28
            nn.ReLU(),                         # 1
            nn.MaxPool2d(2),                   # 2: 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),  # 3: 14x14 -> 14x14
            nn.ReLU(),                         # 4
            nn.MaxPool2d(2),                   # 5: 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1), # 6: 7x7 -> 7x7
            nn.ReLU(),                         # 7
            nn.Flatten(),                      # 8: 128*7*7 = 6272
            nn.Linear(6272, 256),             # 9
            nn.ReLU(),                         # 10
            nn.Dropout(0.5),                   # 11
            nn.Linear(256, num_classes)       # 12
        )

# ==========================================
# 2. THE MLP (LINEAR) ARCHITECTURE
# ==========================================
# Architecture matches saved weights (mlp_model.pth)
class MLP_Model(nn.Sequential):
    def __init__(self, num_classes=26):
        super().__init__(
            nn.Flatten(),                      # 0: 28*28 = 784
            nn.Linear(784, 512),              # 1
            nn.ReLU(),                         # 2
            nn.Linear(512, 256),              # 3
            nn.ReLU(),                         # 4
            nn.Linear(256, num_classes)       # 5
        )