import torch
import torch.nn as nn

class BackwardModel(nn.Module):
    def __init__(self):
        super(BackwardModel, self).__init__()
        
        # Input: 2 EM targets -> Output: 2 Geometric parameters (L1, L2)
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

if __name__ == "__main__":
    print("Testing the Expanded Backward Model architecture...")
    model = BackwardModel()
    dummy_input = torch.randn(1024, 2)
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape} -> Expected: [1024, 2]")