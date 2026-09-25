import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel

def train():
    # 1. Setup Device (uses GPU if you have one, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Forward Model on: {device}")

    # 2. Load Data (Using a large batch size for your 29.5M row dataset)
    print("Loading pre-split datasets...")
    train_loader, val_loader, _ = get_dataloaders(data_dir='data/comsol', batch_size=1024)

    # 3. Initialize Model, Loss function, and Optimizer
    model = ForwardModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training configuration
    epochs = 10
    best_val_loss = float('inf')
    
    # Ensure the save directory exists
    os.makedirs('models/saved', exist_ok=True)
    save_path = 'models/saved/forward_model.pth'

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (geometries, em_responses) in enumerate(train_loader):
            geometries = geometries.to(device)
            em_responses = em_responses.to(device)
            
            # Forward pass: Predict EM response from L1, L2
            predictions = model(geometries)
            loss = criterion(predictions, em_responses)
            
            # Backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print an update every 1,000 batches so you know it's working
            if batch_idx % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # Validation Step (Check if the model is actually learning, not just memorizing)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for geometries, em_responses in val_loader:
                geometries = geometries.to(device)
                em_responses = em_responses.to(device)
                
                predictions = model(geometries)
                loss = criterion(predictions, em_responses)
                val_loss += loss.item()
        
        # Calculate average losses for this epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")
        
        # Save the model only if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[*] Improved model saved to {save_path}\n")
            
    print("Forward Model Training Complete!")

if __name__ == "__main__":
    train()