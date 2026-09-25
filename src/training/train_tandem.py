import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Your path hack is perfectly fine to keep if you want to run via standard python command
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel
from src.models.backward_model import BackwardModel

def train_tandem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Tandem Network on device: {device}")

    # FIX: Point to the directory containing train.pt/val.pt, NOT op.pt directly.
    # Bumped batch_size to 1024 to dramatically speed up training on 29.5M rows.
    train_loader, val_loader, _ = get_dataloaders(data_dir="data/comsol", batch_size=1024)
    
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

    epochs = 20
    best_val_loss = float('inf')
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/backward_model.pth"
    
    for epoch in range(epochs):
        backward_model.train()
        running_loss = 0.0
        
        for batch_idx, (_, em_targets) in enumerate(train_loader):
            em_targets = em_targets.to(device)
            
            optimizer.zero_grad()
            
            # Step A: Backward model predicts geometries from target EM responses
            predicted_geometries = backward_model(em_targets)
            
            # Step B: Frozen Forward model predicts EM response from the generated geometries
            predicted_em = forward_model(predicted_geometries)
            
            # Step C: Loss is calculated between the TARGET EM and the FINAL PREDICTED EM
            loss = criterion(predicted_em, em_targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        backward_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, em_targets in val_loader:
                em_targets = em_targets.to(device)
                
                pred_geom = backward_model(em_targets)
                pred_em = forward_model(pred_geom)
                
                loss = criterion(pred_em, em_targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n--- Epoch {epoch+1}/{epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")
        
        # Save model only if it improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(backward_model.state_dict(), save_path)
            print(f"[*] Improved Backward Model saved to {save_path}\n")

    print(f"Tandem training complete! Best model saved to {save_path}")

if __name__ == "__main__":
    train_tandem()