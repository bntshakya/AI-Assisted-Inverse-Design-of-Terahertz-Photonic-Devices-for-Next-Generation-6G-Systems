import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel
from src.models.backward_model import BackwardModel

def train_tandem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Tandem Network on device: {device}")

    # Reduced batch_size to 8 for the 61-row dataset
    train_loader, val_loader, _ = get_dataloaders(data_dir="data/comsol", batch_size=8)
    
    # 1. Initialize and freeze the pre-trained Forward Model
    forward_model = ForwardModel().to(device)
    forward_model.load_state_dict(torch.load("models/saved/forward_model.pth", map_location=device, weights_only=True))
    forward_model.eval() 
    for param in forward_model.parameters():
        param.requires_grad = False 
        
    # 2. Initialize the Backward Model
    backward_model = BackwardModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(backward_model.parameters(), lr=0.001)

    # 3. Training config
    epochs = 5000  # Increased for small dataset
    best_val_loss = float('inf')
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/backward_model.pth"
    
    # ALPHA controls the leash. 
    # 0.01 means EM response is heavily prioritized, but geometry must stay somewhat realistic.
    alpha = 0.01 
    
    print(f"Starting Tandem training for {epochs} epochs...")
    for epoch in range(epochs):
        backward_model.train()
        running_loss = 0.0
        
        for batch_idx, (real_geometries, em_targets) in enumerate(train_loader):
            real_geometries = real_geometries.to(device)
            em_targets = em_targets.to(device)
            
            optimizer.zero_grad()
            
            # Step 1: Backward Model guesses geometry from the EM target
            predicted_geometries = backward_model(em_targets)
            
            # Step 2: Frozen Forward Model predicts the EM response of that guessed geometry
            predicted_em = forward_model(predicted_geometries)
            
            # Step 3: Calculate Hybrid Loss
            loss_em = criterion(predicted_em, em_targets) 
            loss_geom = criterion(predicted_geometries, real_geometries)
            total_loss = loss_em + (alpha * loss_geom)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        # Validation Step
        backward_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for real_geometries, em_targets in val_loader:
                real_geometries = real_geometries.to(device)
                em_targets = em_targets.to(device)
                
                predicted_geometries = backward_model(em_targets)
                predicted_em = forward_model(predicted_geometries)
                
                loss_em = criterion(predicted_em, em_targets)
                loss_geom = criterion(predicted_geometries, real_geometries)
                total_val_loss = loss_em + (alpha * loss_geom)
                
                val_loss += total_val_loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(backward_model.state_dict(), save_path)
            
    print(f"\nTandem Training Complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_tandem()