import torch
import torch.nn as nn

class BackwardModel(nn.Module):
    def __init__(self):
        super(BackwardModel, self).__init__()
        # Input: 2 EM targets -> Output: 2 Geometric parameters (L1, L2)
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Changed from 5 to 2
        )

    def forward(self, x):
        return self.network(x)

# --- QUICK TEST ---
if __name__ == "__main__":
    print("Testing the Backward Model architecture...")
    model = BackwardModel()
    dummy_input = torch.randn(64, 2)
    dummy_output = model(dummy_input)
    print(f"Input shape (EM Responses): {dummy_input.shape} -> Expected: [64, 2]")
    print(f"Output shape (Predicted L1, L2): {dummy_output.shape} -> Expected: [64, 2]")