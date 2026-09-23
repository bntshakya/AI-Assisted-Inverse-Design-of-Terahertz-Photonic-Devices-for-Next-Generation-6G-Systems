import torch
import torch.nn as nn

class BackwardModel(nn.Module):
    def __init__(self):
        super(BackwardModel, self).__init__()
        # Input: 2 EM targets -> Output: 5 Geometric parameters
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)