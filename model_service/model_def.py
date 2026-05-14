import torch.nn as nn


class MoonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)
