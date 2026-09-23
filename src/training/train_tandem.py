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

    train_loader, val_loader, _ = get_dataloaders("data/comsol/op.pt", batch_size=256)
    
    # 1. Initialize and freeze the pre-trained Forward Model
    forward_model = ForwardModel().to(device)
    forward_model.load_state_dict(torch.load("models/saved/forward_model.pth", weights_only=True))
    forward_model.eval() 
    for param in forward_model.parameters():
        param.requires_grad = False 
        
    # 2. Initialize the Backward Model
    backward_model = BackwardModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(backward_model.parameters(), lr=0.001)

    epochs = 20
    
    for epoch in range(epochs):
        backward_model.train()
        running_loss = 0.0
        
        for batch_idx, (real_geometries, em_targets) in enumerate(train_loader):
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
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Tandem Train Loss: {avg_train_loss:.6f}")

    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/backward_model.pth"
    torch.save(backward_model.state_dict(), save_path)
    print(f"Tandem training complete! Backward model saved to {save_path}")

if __name__ == "__main__":
    train_tandem()