import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        
        # The pipeline now uses 2 geometry inputs (L1, L2)
        input_dim = 2
        
        # Assuming your EM response (freq, lambda, etc.) has 2 features
        # If your CSV had more than 2 EM features, change this number to match
        output_dim = 2
        
        # Define the network architecture
        self.network = nn.Sequential(
            # Input Layer (Now accepts 2 features)
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
    
    # Create a dummy batch of geometries mimicking your NEW DataLoader (64 rows, 2 features)
    dummy_input = torch.randn(64, 2)
    
    # Pass the dummy data through the model
    dummy_output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape} -> Expected: [64, 2]")
    print(f"Output shape: {dummy_output.shape} -> Expected: [64, 2]")