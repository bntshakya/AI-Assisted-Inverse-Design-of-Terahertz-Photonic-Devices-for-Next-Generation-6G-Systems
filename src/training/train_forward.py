import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import your custom modules
from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel

def train():
    # 1. Hardware Check: Use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load the Data
    # (Adjust the path to point to your op.pt file relative to the project root)
    data_path = "data/comsol/op.pt" 
    train_loader, val_loader, _ = get_dataloaders(data_path, batch_size=256) # Increased batch size for GPU speed
    
    # 3. Initialize Model, Loss, and Optimizer
    model = ForwardModel().to(device)
    
    # Mean Squared Error (Standard for continuous regression problems)
    criterion = nn.MSELoss() 
    
    # Adam Optimizer (The algorithm that updates the weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    epochs = 10 # Start small to test
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Iterate over the training conveyor belt
        for batch_idx, (geometries, em_targets) in enumerate(train_loader):
            # Move data to the GPU
            geometries = geometries.to(device)
            em_targets = em_targets.to(device)
            
            # Zero the gradients (clear old memory)
            optimizer.zero_grad()
            
            # Forward pass: predict the EM response
            predictions = model(geometries)
            
            # Calculate how wrong the predictions were
            loss = criterion(predictions, em_targets)
            
            # Backward pass: calculate the updates
            loss.backward()
            
            # Step: update the model weights
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print an update every 10,000 batches
            if batch_idx % 10000 == 0 and batch_idx > 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

        # 5. Validation Check at the end of each epoch
        model.eval() # Set model to evaluation mode (turns off BatchNorm/Dropout updates)
        val_loss = 0.0
        with torch.no_grad(): # Don't track gradients (saves memory)
            for geometries, em_targets in val_loader:
                geometries, em_targets = geometries.to(device), em_targets.to(device)
                predictions = model(geometries)
                loss = criterion(predictions, em_targets)
                val_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")

    # 6. Save the trained model
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/forward_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to {save_path}")

if __name__ == "__main__":
    train()