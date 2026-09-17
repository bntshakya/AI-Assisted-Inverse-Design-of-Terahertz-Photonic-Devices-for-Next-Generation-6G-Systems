import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        
        # We know from your pipeline: Inputs = 5, Outputs = 2
        input_dim = 5
        output_dim = 2
        
        # Define the network architecture
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Stabilizes and speeds up training
            nn.ReLU(),
            
            # Hidden Layer 1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Hidden Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Output Layer (No activation function here because we want raw continuous values)
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # This defines how the data flows through the network
        return self.network(x)

# --- QUICK TEST ---
if __name__ == "__main__":
    print("Testing the Forward Model architecture...")
    
    # Initialize the model
    model = ForwardModel()
    
    # Create a dummy batch of geometries mimicking your DataLoader (64 rows, 5 features)
    dummy_input = torch.randn(64, 5)
    
    # Pass the dummy data through the model
    dummy_output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape} -> Expected: [64, 2]")