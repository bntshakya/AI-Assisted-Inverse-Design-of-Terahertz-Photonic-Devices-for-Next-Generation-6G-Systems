import torch
import torch.nn as nn

class BackwardModel(nn.Module):
    def __init__(self):
        super(BackwardModel, self).__init__()
        
        # Input: 2 EM targets (f_lower, f_upper) -> Output: 2 Geometric parameters (L1, L2)
        self.network = nn.Sequential(
            nn.Linear(2, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)