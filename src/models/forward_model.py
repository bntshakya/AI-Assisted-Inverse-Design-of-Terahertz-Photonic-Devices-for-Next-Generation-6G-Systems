import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        
        # The pipeline uses 2 geometry inputs (L1, L2)
        input_dim = 2
        
        # Output is strictly 2 features: f_lower and f_upper
        output_dim = 2 
        
        # Define the network architecture
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            
            # Hidden Layer 1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Hidden Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)